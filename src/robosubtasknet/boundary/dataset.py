"""PyTorch Dataset for the Stage-1 boundary segmenter.

Stage 1 of the 2-stage pipeline is a *class-agnostic* boundary detector trained
on binary labels: ``1`` at frames where consecutive ``action_text_id`` values
in the source LeRobot parquet change, ``0`` elsewhere. The labels are computed
once and cached on disk by ``scripts/extract_features_boundary.py`` (a
different agent's scope), together with the I3D RGB and Flow features for the
same episode at the same temporal stride.

This module is the read side of that cache: it presents one ``.npz`` per
episode as a ``Dataset`` sample of ``(rgb, flow, boundaries, mask)``, and a
collate function that right-pads variable-length episodes into a batch.

Design notes
------------
* Labels are **binary** (``0``/``1``), not multi-class. Padding is therefore
  done with ``0`` (background), and the accompanying ``mask`` tells the loss
  module which frames to actually score. Cross-entropy-style ``ignore_index``
  is intentionally avoided here so the downstream Gaussian-soft-labelling and
  focal/BCE losses can work on a dense binary tensor.
* The on-disk ``labels`` array is ``int32`` (per the extractor contract); we
  cast to ``int64`` here because PyTorch's loss functions and indexing helpers
  expect ``LongTensor`` targets.
* Smoothing the hard 0/1 labels into Gaussian-soft targets is **not** done
  here; it belongs to the loss module.
* Temporal alignment is defensive: ``rgb``, ``flow``, and ``labels`` are
  truncated to their common length to absorb the occasional off-by-one
  introduced by I3D's temporal stride and label downsampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "BoundaryDataset",
    "pad_collate_boundary",
]


class BoundaryDataset(Dataset):
    """Per-episode dataset of ``(rgb, flow, boundaries, mask)`` tensors.

    The dataset reads pre-extracted ``.npz`` files from ``feature_dir``. Each
    file is one LeRobot episode and must contain the following keys (the
    contract enforced by ``scripts/extract_features_boundary.py``):

    * ``rgb``: ``float16`` array of shape ``[T_feat, feature_dim]``.
    * ``flow``: ``float16`` array of shape ``[T_feat, feature_dim]``.
    * ``labels``: ``int32`` array of shape ``[T_feat]`` with values in
      ``{0, 1}`` (1 = boundary at this feature timestep).
    * ``meta``: optional JSON string with provenance information. Loaded but
      not parsed by this class.

    Parameters
    ----------
    feature_dir
        Directory containing ``<episode_id>.npz`` files. All ``.npz`` files in
        the directory are included; no split file is required at this layer —
        train/val/test splitting is the caller's responsibility (typically by
        passing different ``feature_dir``\\s or by wrapping with
        ``torch.utils.data.Subset``).
    feature_dim
        Per-stream feature dimensionality. Defaults to 1024 (I3D).
    """

    def __init__(
        self,
        feature_dir: Path,
        feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.feature_dir: Path = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(
                f"feature_dir does not exist or is not a directory: "
                f"{self.feature_dir}"
            )
        self.feature_dim: int = int(feature_dim)

        self.video_ids: List[str] = sorted(
            p.stem for p in self.feature_dir.glob("*.npz")
        )
        if not self.video_ids:
            raise RuntimeError(
                f"No .npz feature files found in {self.feature_dir}."
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        rgb, flow, boundaries = self._load_npz(video_id)

        # float16 storage -> float32 compute; int32 labels -> int64.
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        flow_t = torch.from_numpy(np.ascontiguousarray(flow)).float()
        boundaries_t = torch.from_numpy(
            np.ascontiguousarray(boundaries)
        ).long()

        # Defensive temporal alignment: feature extraction and label
        # downsampling can disagree by a frame or two due to I3D's temporal
        # stride. Truncate to the common length so losses see consistent
        # shapes.
        T = min(rgb_t.shape[0], flow_t.shape[0], boundaries_t.shape[0])
        rgb_t = rgb_t[:T]
        flow_t = flow_t[:T]
        boundaries_t = boundaries_t[:T]

        return {
            "rgb": rgb_t,
            "flow": flow_t,
            "boundaries": boundaries_t,
            "video_id": video_id,
            "length": int(T),
        }

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def _load_npz(
        self, video_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and validate one ``.npz`` file produced by the extractor."""
        path = self.feature_dir / f"{video_id}.npz"
        with np.load(path, allow_pickle=True) as data:
            for key in ("rgb", "flow", "labels"):
                if key not in data.files:
                    raise KeyError(
                        f"{path}: missing required key {key!r} "
                        f"(found {data.files})."
                    )
            rgb = np.asarray(data["rgb"])
            flow = np.asarray(data["flow"])
            labels = np.asarray(data["labels"])

        if rgb.ndim != 2 or rgb.shape[1] != self.feature_dim:
            raise ValueError(
                f"{path}: rgb must have shape [T, {self.feature_dim}], "
                f"got {rgb.shape}."
            )
        if flow.shape != rgb.shape:
            raise ValueError(
                f"{path}: flow shape {flow.shape} != rgb shape {rgb.shape}."
            )
        if labels.ndim != 1:
            raise ValueError(
                f"{path}: labels must be 1-D, got shape {labels.shape}."
            )
        # Cast int32 -> int64 for PyTorch; copy=False avoids the alloc when
        # the array already happens to be int64.
        return rgb, flow, labels.astype(np.int64, copy=False)


def pad_collate_boundary(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Right-pad variable-length boundary samples into a batch.

    Mirrors :func:`robosubtasknet.data.dataset.pad_collate` but for the binary
    boundary task: padded label frames are set to ``0`` (non-boundary) rather
    than to ``-100``, and the accompanying ``mask`` is what downstream losses
    use to mask out padding. This matters because the Stage-1 loss is BCE /
    focal-on-Gaussian-soft-labels, not multi-class cross-entropy with an
    ``ignore_index``.

    Parameters
    ----------
    batch
        Sequence of sample dicts from :meth:`BoundaryDataset.__getitem__`.

    Returns
    -------
    dict
        ``rgb``: ``FloatTensor [B, T_max, D]`` zero-padded.
        ``flow``: ``FloatTensor [B, T_max, D]`` zero-padded.
        ``boundaries``: ``LongTensor [B, T_max]`` zero-padded (0 = non-boundary).
        ``mask``: ``FloatTensor [B, T_max]`` with ``1.0`` for valid frames and
        ``0.0`` for padding. Losses must multiply by this mask.
        ``lengths``: ``LongTensor [B]`` with the original per-sample length.
        ``video_ids``: ``list[str]`` preserving the order of the batch.
    """
    if not batch:
        raise ValueError("pad_collate_boundary received an empty batch.")

    lengths = [int(b["length"]) for b in batch]
    T_max = max(lengths)
    B = len(batch)
    D = int(batch[0]["rgb"].shape[1])

    rgb = torch.zeros(B, T_max, D, dtype=torch.float32)
    flow = torch.zeros(B, T_max, D, dtype=torch.float32)
    # Pad with 0 (non-boundary) — mask is the authoritative signal for losses.
    boundaries = torch.zeros(B, T_max, dtype=torch.long)
    mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i, sample in enumerate(batch):
        L = lengths[i]
        rgb[i, :L] = sample["rgb"][:L]
        flow[i, :L] = sample["flow"][:L]
        boundaries[i, :L] = sample["boundaries"][:L]
        mask[i, :L] = 1.0

    return {
        "rgb": rgb,
        "flow": flow,
        "boundaries": boundaries,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "video_ids": [str(b["video_id"]) for b in batch],
    }
