"""PyTorch Dataset for RoboSubtaskNet.

Supports two storage formats:

- ``mode="mstcn"``: the published MS-TCN feature layout where each video is a
  single ``[T, 2048]`` ``.npy`` array (first 1024 dims = RGB, last 1024 = Flow)
  and ground-truth labels are one-per-line text files in a parallel
  ``groundTruth/`` directory. Used to reproduce GTEA / Breakfast numbers
  (Section 5.1 of the plan).
- ``mode="npz"``: the format written by our own extractor (Section 6.3), one
  ``.npz`` per video with keys ``rgb``, ``flow``, ``labels``, ``meta``.

The dataset deliberately does **not** apply any temporal cropping or
augmentation — that is the responsibility of an external ``transform`` callable
and/or the dedicated ``augmentations`` module. Variable-length videos are
handled at collate time via :func:`pad_collate`, following the MS-TCN
convention of batch size 1 (or bucketed batches).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "RoboSubtaskDataset",
    "pad_collate",
    "load_mapping",
    "parse_split",
]

# Pad value for labels so that ``nn.CrossEntropyLoss(ignore_index=-100)``
# silently skips padded frames during training.
LABEL_PAD_VALUE: int = -100


def load_mapping(path: str | Path) -> Dict[str, int]:
    """Parse an MS-TCN ``mapping.txt`` (one ``"<int> <name>"`` per line).

    Lines that are blank or start with ``#`` are skipped. The integer is the
    class index; the name is the sub-task label.

    Returns a dict ``{name -> int_index}``. To get the inverse, do
    ``{v: k for k, v in mapping.items()}``.
    """
    mapping: Dict[str, int] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed mapping line in {p}: {raw!r} "
                    "(expected '<int> <name>')"
                )
            idx_str, name = parts
            try:
                idx = int(idx_str)
            except ValueError as e:
                raise ValueError(
                    f"Non-integer class id in {p}: {raw!r}"
                ) from e
            mapping[name] = idx
    return mapping


def parse_split(path: str | Path) -> List[str]:
    """Read an MS-TCN-style split bundle.

    Each non-blank line is a single filename, e.g. ``S1_Cheese_C1.txt`` for
    GTEA. Comments (``#``) and blank lines are ignored. The trailing extension
    is preserved so callers can decide whether to strip it.
    """
    p = Path(path)
    names: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


class RoboSubtaskDataset(Dataset):
    """Per-video dataset of (rgb, flow, labels) tensors.

    Parameters
    ----------
    feature_dir
        Directory containing feature files. For ``mode="mstcn"`` these are
        ``<video_id>.npy`` arrays of shape ``[T, 2 * feature_dim]``. For
        ``mode="npz"`` they are ``<video_id>.npz`` files with keys ``rgb``,
        ``flow``, ``labels``, ``meta``.
    split_file
        Optional MS-TCN-style bundle file listing which videos to include
        (filenames as written in the bundle, typically with a ``.txt``
        extension that refers to the groundTruth file). If ``None``, every
        feature file in ``feature_dir`` is used.
    mapping_file
        Optional ``mapping.txt`` defining ``name -> int`` for the sub-task
        vocabulary. Required when ``mode="mstcn"`` because groundTruth files
        store class names, not integers. For ``mode="npz"`` the labels are
        already integers, but providing the mapping is still useful for
        downstream code (e.g. the grammar module).
    mode
        ``"mstcn"`` to load published MS-TCN features (split a single ``.npy``
        into RGB+Flow halves and read groundTruth ``.txt`` files);
        ``"npz"`` to load our own extractor's ``.npz`` files.
    feature_dim
        Per-stream feature dimensionality. Defaults to 1024 (I3D). For
        ``mode="mstcn"`` the ``.npy`` files must therefore be of shape
        ``[T, 2 * feature_dim]``.
    transform
        Optional callable applied to the sample dict immediately before
        return. Must return a dict with the same keys.
    """

    def __init__(
        self,
        feature_dir: Path,
        split_file: Optional[Path] = None,
        mapping_file: Optional[Path] = None,
        mode: Literal["mstcn", "npz"] = "npz",
        feature_dim: int = 1024,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        if mode not in ("mstcn", "npz"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'mstcn' or 'npz'."
            )
        self.feature_dir: Path = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(
                f"feature_dir does not exist or is not a directory: "
                f"{self.feature_dir}"
            )

        self.mode: Literal["mstcn", "npz"] = mode
        self.feature_dim: int = int(feature_dim)
        self.transform = transform

        # Load mapping (name -> int). Mandatory for mstcn mode because we have
        # to translate textual class labels in groundTruth files to indices.
        self.mapping: Optional[Dict[str, int]] = (
            load_mapping(mapping_file) if mapping_file is not None else None
        )
        if self.mode == "mstcn" and self.mapping is None:
            raise ValueError(
                "mode='mstcn' requires mapping_file (textual labels in "
                "groundTruth/ must be translated to integer ids)."
            )

        # Optional groundTruth directory parallel to features/, MS-TCN
        # convention: data/<dataset>/groundTruth/<video_id>.txt.
        self.gt_dir: Optional[Path] = None
        if self.mode == "mstcn":
            candidate = self.feature_dir.parent / "groundTruth"
            if candidate.is_dir():
                self.gt_dir = candidate

        self.split_file: Optional[Path] = (
            Path(split_file) if split_file is not None else None
        )
        self.video_ids: List[str] = self._discover_videos()
        if not self.video_ids:
            raise RuntimeError(
                f"No videos discovered in {self.feature_dir} "
                f"(split_file={self.split_file})."
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_videos(self) -> List[str]:
        """Resolve which video IDs this dataset will serve.

        The split file (if present) is authoritative. Filenames in the bundle
        may have any extension; we strip it to obtain a stable ``video_id``.
        If no split file is given, we enumerate the feature directory.
        """
        if self.split_file is not None:
            entries = parse_split(self.split_file)
            return [Path(e).stem for e in entries]

        suffix = ".npz" if self.mode == "npz" else ".npy"
        return sorted(p.stem for p in self.feature_dir.glob(f"*{suffix}"))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]

        if self.mode == "mstcn":
            rgb, flow, labels = self._load_mstcn(video_id)
        else:
            rgb, flow, labels = self._load_npz(video_id)

        # Cast features from float16 (storage) to float32 (compute).
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        flow_t = torch.from_numpy(np.ascontiguousarray(flow)).float()
        labels_t = torch.from_numpy(np.ascontiguousarray(labels)).long()

        # Defensive temporal alignment: features and labels can disagree by a
        # frame or two due to I3D's temporal stride / annotation rounding;
        # truncate to the shorter to keep losses honest.
        T = min(rgb_t.shape[0], flow_t.shape[0], labels_t.shape[0])
        rgb_t = rgb_t[:T]
        flow_t = flow_t[:T]
        labels_t = labels_t[:T]

        sample: Dict[str, Any] = {
            "rgb": rgb_t,
            "flow": flow_t,
            "labels": labels_t,
            "video_id": video_id,
            "length": int(T),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # ------------------------------------------------------------------
    # Loaders per mode
    # ------------------------------------------------------------------
    def _load_mstcn(
        self, video_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single MS-TCN feature file and the matching groundTruth."""
        feat_path = self.feature_dir / f"{video_id}.npy"
        feats = np.load(feat_path)  # may be [T, 2D] or [2D, T]
        feats = self._mstcn_to_TC(feats)

        expected = 2 * self.feature_dim
        if feats.shape[1] != expected:
            raise ValueError(
                f"{feat_path}: expected last dim {expected} "
                f"(2 * feature_dim={self.feature_dim}), got {feats.shape[1]}."
            )
        rgb = feats[:, : self.feature_dim]
        flow = feats[:, self.feature_dim :]

        labels = self._load_mstcn_labels(video_id, T=feats.shape[0])
        return rgb, flow, labels

    @staticmethod
    def _mstcn_to_TC(feats: np.ndarray) -> np.ndarray:
        """Normalize MS-TCN features to ``[T, C]`` layout.

        The original MS-TCN repository saves features as ``[2048, T]``
        (channel-first). Some published copies use ``[T, 2048]`` instead. We
        detect orientation by the longer axis being time — MS-TCN features
        always have C=2048 while T varies per video.
        """
        if feats.ndim != 2:
            raise ValueError(
                f"MS-TCN feature must be 2-D, got shape {feats.shape}."
            )
        # Heuristic: feature dim is small relative to typical T (a 1-minute
        # GTEA video at I3D stride is hundreds of frames). If axis 0 is 2048
        # exactly, we assume channel-first and transpose.
        if feats.shape[0] in (2048, 2 * 1024) and feats.shape[1] != feats.shape[0]:
            feats = feats.T
        return feats

    def _load_mstcn_labels(self, video_id: str, T: int) -> np.ndarray:
        """Read groundTruth/<video_id>.txt (one class name per frame).

        If no groundTruth directory is configured (e.g. test split with hidden
        labels) we return an all-background array as a placeholder, which is
        compatible with ``ignore_index=-100`` once flagged by the caller.
        """
        if self.gt_dir is None:
            return np.zeros(T, dtype=np.int64)

        # MS-TCN bundles list ``<video_id>.txt`` directly, but our discovery
        # already strips the extension; add it back here.
        gt_path = self.gt_dir / f"{video_id}.txt"
        if not gt_path.is_file():
            return np.zeros(T, dtype=np.int64)

        assert self.mapping is not None  # enforced in __init__ for mstcn
        with gt_path.open("r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]

        try:
            ids = [self.mapping[n] for n in names]
        except KeyError as e:
            raise KeyError(
                f"Sub-task name {e.args[0]!r} from {gt_path} is missing from "
                f"mapping file."
            ) from e
        return np.asarray(ids, dtype=np.int64)

    def _load_npz(
        self, video_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single ``.npz`` written by our own extractor (Section 6.3)."""
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
        return rgb, flow, labels.astype(np.int64, copy=False)


def pad_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length samples by right-padding to the batch maximum.

    Parameters
    ----------
    batch
        Sequence of sample dicts as returned by :meth:`RoboSubtaskDataset.__getitem__`.

    Returns
    -------
    dict
        ``rgb``: ``FloatTensor [B, T_max, D]`` zero-padded.
        ``flow``: ``FloatTensor [B, T_max, D]`` zero-padded.
        ``labels``: ``LongTensor [B, T_max]`` padded with ``-100`` so that
        ``CrossEntropyLoss(ignore_index=-100)`` skips pad frames.
        ``mask``: ``FloatTensor [B, T_max]`` with ``1.0`` for valid frames and
        ``0.0`` for padding. Use this in T-MSE / transition losses so they
        ignore padding too.
        ``lengths``: ``LongTensor [B]`` with the original per-sample length.
        ``video_ids``: ``list[str]`` preserving the order of the batch.
    """
    if not batch:
        raise ValueError("pad_collate received an empty batch.")

    lengths = [int(b["length"]) for b in batch]
    T_max = max(lengths)
    B = len(batch)
    D = int(batch[0]["rgb"].shape[1])

    rgb = torch.zeros(B, T_max, D, dtype=torch.float32)
    flow = torch.zeros(B, T_max, D, dtype=torch.float32)
    labels = torch.full(
        (B, T_max), fill_value=LABEL_PAD_VALUE, dtype=torch.long
    )
    mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i, sample in enumerate(batch):
        L = lengths[i]
        rgb[i, :L] = sample["rgb"][:L]
        flow[i, :L] = sample["flow"][:L]
        labels[i, :L] = sample["labels"][:L]
        mask[i, :L] = 1.0

    return {
        "rgb": rgb,
        "flow": flow,
        "labels": labels,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "video_ids": [str(b["video_id"]) for b in batch],
    }
