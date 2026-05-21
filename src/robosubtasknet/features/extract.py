"""Feature extraction orchestration.

Pure-function API. The companion CLI lives in ``scripts/extract_features.py``.

Pipeline (per video):

1. Read frames with ``decord`` (lazy import).
2. Optionally compute optical flow with :mod:`robosubtasknet.features.flow`.
3. Slide a ``window``-sized window with the given ``stride`` over the frames,
   feed each clip to an :class:`I3DFeatureExtractor` (RGB and/or flow), and
   collect 1024-d features per window.
4. Save as a per-video ``.npz`` with float16 features and int32 labels, plus
   a ``meta`` JSON-serializable dict.

Storage format follows Section 6.3 of IMPLEMENTATION_PLAN.md::

    data/features/<dataset>/<video_id>.npz
    +-- rgb:    float16 [T_feat, 1024]
    +-- flow:   float16 [T_feat, 1024]
    +-- labels: int32   [T_feat]
    +-- meta:   {fps, original_T, video_path, hash}

float16 reduces disk usage by 2x; :func:`load_features_npz` casts back to
float32.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


__all__ = [
    "extract_features_for_video",
    "save_features_npz",
    "load_features_npz",
]


# --------------------------------------------------------------------------- #
# Lazy import helpers (keep optional heavy deps out of import-time path).
# --------------------------------------------------------------------------- #


def _import_decord():
    try:
        import decord  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading video frames requires `decord`. Install with "
            "`pip install decord` (or `pip install -e .[video]` once the "
            "extra is configured)."
        ) from exc
    return decord


def _import_torch():
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Feature extraction requires PyTorch. Install with "
            "`pip install torch torchvision`."
        ) from exc
    return torch


def _import_i3d():
    """Lazy import the I3D wrapper (owned by another module)."""
    try:
        from robosubtasknet.features.i3d import I3DFeatureExtractor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "I3DFeatureExtractor is missing. Ensure "
            "`robosubtasknet.features.i3d` is installed and that "
            "`pytorchvideo` is available (`pip install pytorchvideo`)."
        ) from exc
    return I3DFeatureExtractor


def _import_cv2():
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optical-flow extraction requires OpenCV. Install with "
            "`pip install opencv-contrib-python`."
        ) from exc
    return cv2


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def extract_features_for_video(
    video_path: Path,
    *,
    modality: Literal["rgb", "flow", "both"] = "both",
    flow_method: str = "tvl1",
    window: int = 16,
    stride: int = 8,
    device: str = "cuda",
) -> dict:
    """Extract per-window I3D features for a single video.

    Parameters
    ----------
    video_path : Path
        Path to the source video (anything ``decord`` can read).
    modality : {"rgb", "flow", "both"}, default "both"
        Which feature stream(s) to compute. ``"both"`` returns separate RGB
        and flow arrays; downstream code is expected to fuse them (see
        :class:`robosubtasknet.models.fusion.AttentionFusion`).
    flow_method : str, default "tvl1"
        Optical-flow estimator name, forwarded to
        :func:`robosubtasknet.features.flow.compute_flow_from_frames`. Use
        ``"tvl1"`` for Kinetics-distributional fidelity; ``"raft"`` is
        higher quality but distributionally mismatched (see Section 6.2).
    window : int, default 16
        Number of frames per I3D clip. Standard Kinetics-I3D setting.
    stride : int, default 8
        Frame stride between consecutive clip starts. With ``window=16``
        and ``stride=8`` we get one feature per ~8 raw frames, matching
        I3D's natural temporal downsampling.
    device : str, default "cuda"
        Torch device for the I3D forward pass. Falls back to CPU if CUDA
        is unavailable.

    Returns
    -------
    dict
        ``{"rgb": np.ndarray | None, "flow": np.ndarray | None,
           "meta": {...}}``. RGB / flow arrays are float32 of shape
        ``[T_feat, 1024]``; ``None`` when that modality was not requested.
        ``meta`` includes ``fps``, ``original_T``, ``video_path``,
        ``hash``, ``window``, ``stride``, ``flow_method``, ``T_feat``.

    Notes
    -----
    The number of feature windows ``T_feat`` is
    ``max(0, 1 + floor((T_eff - window) / stride))`` where ``T_eff = T``
    for the RGB stream and ``T_eff = T - 1`` for the flow stream (one
    fewer frame because flow is a between-frame quantity). For consistent
    ``T_feat`` across modalities, the RGB stream is right-truncated to
    ``T - 1`` frames when both modalities are requested.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    modality_lc = modality.lower()
    if modality_lc not in {"rgb", "flow", "both"}:
        raise ValueError(
            f"modality must be 'rgb', 'flow', or 'both'; got {modality!r}."
        )
    if window <= 0 or stride <= 0:
        raise ValueError(
            f"window and stride must be positive; got window={window}, "
            f"stride={stride}."
        )

    torch = _import_torch()
    decord = _import_decord()

    # ----- Read frames ------------------------------------------------------
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path))
    original_T = len(vr)
    try:
        fps = float(vr.get_avg_fps())
    except Exception:  # pragma: no cover - decord version dependent
        fps = 0.0
    frames = vr.get_batch(list(range(original_T))).asnumpy()  # [T, H, W, 3] uint8
    del vr  # release file handle promptly

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"decord returned unexpected frame shape {frames.shape}; "
            "expected [T, H, W, 3]."
        )

    # Align lengths so RGB and flow yield the same T_feat.
    need_rgb = modality_lc in {"rgb", "both"}
    need_flow = modality_lc in {"flow", "both"}
    if need_rgb and need_flow:
        rgb_frames = frames[:-1]  # drop last to match flow's [T-1] length
    else:
        rgb_frames = frames

    # ----- Resolve device ---------------------------------------------------
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    # ----- I3D RGB stream ---------------------------------------------------
    rgb_features = None
    if need_rgb:
        I3DFeatureExtractor = _import_i3d()
        rgb_model = I3DFeatureExtractor(modality="rgb", pretrained=True).to(
            torch_device
        )
        rgb_model.eval()
        rgb_features = _i3d_sliding_window(
            rgb_model,
            rgb_frames,
            channels=3,
            window=window,
            stride=stride,
            device=torch_device,
        )

    # ----- I3D flow stream --------------------------------------------------
    flow_features = None
    if need_flow:
        from robosubtasknet.features.flow import compute_flow_from_frames

        flow = compute_flow_from_frames(frames, method=flow_method)  # [T-1, H, W, 2]
        I3DFeatureExtractor = _import_i3d()
        flow_model = I3DFeatureExtractor(modality="flow", pretrained=True).to(
            torch_device
        )
        flow_model.eval()
        flow_features = _i3d_sliding_window(
            flow_model,
            flow,
            channels=2,
            window=window,
            stride=stride,
            device=torch_device,
            already_normalized=True,
        )

    # ----- Derive T_feat for meta ------------------------------------------
    if rgb_features is not None and flow_features is not None:
        T_feat = min(rgb_features.shape[0], flow_features.shape[0])
        rgb_features = rgb_features[:T_feat]
        flow_features = flow_features[:T_feat]
    elif rgb_features is not None:
        T_feat = rgb_features.shape[0]
    elif flow_features is not None:
        T_feat = flow_features.shape[0]
    else:  # pragma: no cover - modality validation above prevents this
        T_feat = 0

    meta = {
        "fps": float(fps),
        "original_T": int(original_T),
        "video_path": str(video_path),
        "hash": _file_md5(video_path),
        "window": int(window),
        "stride": int(stride),
        "flow_method": str(flow_method) if need_flow else None,
        "modality": modality_lc,
        "T_feat": int(T_feat),
    }

    return {"rgb": rgb_features, "flow": flow_features, "meta": meta}


def save_features_npz(
    out_path: Path,
    rgb: np.ndarray | None,
    flow: np.ndarray | None,
    labels: np.ndarray | None,
    meta: dict[str, Any],
) -> None:
    """Save per-video features to a single ``.npz`` (Section 6.3 format).

    Features are stored as ``float16`` to halve disk usage; labels as
    ``int32``; ``meta`` is JSON-serialized to a 0-d string array.

    Parameters
    ----------
    out_path : Path
        Destination ``.npz`` path. Parent directories are created.
    rgb, flow : np.ndarray or None
        Float features of shape ``[T_feat, 1024]``. ``None`` is stored as a
        zero-length float16 array of shape ``(0, 0)`` so consumers can
        still find the key but recognize it as absent.
    labels : np.ndarray or None
        Frame-level integer labels of shape ``[T_feat]``. May be ``None``
        for unlabeled videos; stored as an empty int32 array in that case.
    meta : dict
        JSON-serializable metadata.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_arr = (
        rgb.astype(np.float16, copy=False)
        if rgb is not None
        else np.empty((0, 0), dtype=np.float16)
    )
    flow_arr = (
        flow.astype(np.float16, copy=False)
        if flow is not None
        else np.empty((0, 0), dtype=np.float16)
    )
    labels_arr = (
        labels.astype(np.int32, copy=False)
        if labels is not None
        else np.empty((0,), dtype=np.int32)
    )

    try:
        meta_str = json.dumps(meta, default=str)
    except TypeError as exc:
        raise ValueError(
            f"meta must be JSON-serializable; failed to encode: {exc}"
        ) from exc

    np.savez_compressed(
        out_path,
        rgb=rgb_arr,
        flow=flow_arr,
        labels=labels_arr,
        meta=np.array(meta_str),
    )


def load_features_npz(path: Path) -> dict:
    """Load a per-video feature ``.npz`` produced by :func:`save_features_npz`.

    Returns
    -------
    dict
        ``{"rgb": np.ndarray | None, "flow": np.ndarray | None,
           "labels": np.ndarray, "meta": dict}``. ``rgb`` / ``flow`` are
        cast back to ``float32``; ``labels`` is ``int64`` (PyTorch-friendly).
        Modalities saved as empty placeholders are returned as ``None``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        rgb_raw = npz["rgb"] if "rgb" in npz.files else None
        flow_raw = npz["flow"] if "flow" in npz.files else None
        labels_raw = (
            npz["labels"] if "labels" in npz.files else np.empty((0,), dtype=np.int32)
        )
        meta_raw = npz["meta"] if "meta" in npz.files else np.array("{}")

    rgb = _restore_features(rgb_raw)
    flow = _restore_features(flow_raw)
    labels = np.asarray(labels_raw, dtype=np.int64)

    meta_str = meta_raw.item() if isinstance(meta_raw, np.ndarray) else meta_raw
    try:
        meta = json.loads(meta_str) if meta_str else {}
    except (TypeError, json.JSONDecodeError):
        meta = {}

    return {"rgb": rgb, "flow": flow, "labels": labels, "meta": meta}


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _restore_features(arr: np.ndarray | None) -> np.ndarray | None:
    """Cast features back to float32, or return ``None`` if placeholder."""
    if arr is None:
        return None
    if arr.size == 0 or arr.ndim < 2:
        return None
    return np.asarray(arr, dtype=np.float32)


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    """Stream MD5 hash of a file (used for cache invalidation)."""
    h = hashlib.md5()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _i3d_sliding_window(
    model,
    frames: np.ndarray,
    *,
    channels: int,
    window: int,
    stride: int,
    device,
    already_normalized: bool = False,
) -> np.ndarray:
    """Run an I3D model over ``frames`` with a (window, stride) sliding window.

    Parameters
    ----------
    model : nn.Module
        An :class:`I3DFeatureExtractor` (already on ``device`` and in eval).
    frames : np.ndarray
        For ``channels=3``: RGB ``uint8`` frames of shape ``[T, H, W, 3]``.
        For ``channels=2``: flow ``float32`` frames of shape ``[T, H, W, 2]``
        already in ``[-1, 1]``.
    channels : int
        Number of input channels expected by the model (3 for RGB,
        2 for flow).
    window : int
        Number of frames per clip.
    stride : int
        Frame stride between clips.
    device : torch.device
    already_normalized : bool, default False
        If True, skip the ``frames / 255 -> [-1, 1]`` rescaling (used for
        flow inputs, which are already in [-1, 1]).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[T_feat, 1024]``.
    """
    torch = _import_torch()

    if frames.ndim != 4 or frames.shape[-1] != channels:
        raise ValueError(
            f"Expected frames of shape [T, H, W, {channels}]; got {frames.shape}."
        )

    T = frames.shape[0]
    if T < window:
        # Not enough frames for even one window — produce an empty result
        # rather than crashing. Downstream code can decide what to do.
        return np.zeros((0, 1024), dtype=np.float32)

    starts = list(range(0, T - window + 1, stride))
    feats = []

    # Move-once-to-device strategy: convert the whole stack once, then slice.
    frames_t = torch.from_numpy(np.ascontiguousarray(frames))  # [T, H, W, C]
    if already_normalized:
        frames_t = frames_t.to(torch.float32)
    else:
        # uint8 RGB -> float32 in [-1, 1].
        frames_t = frames_t.to(torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
    # Reorder to [T, C, H, W] for slicing into clips.
    frames_t = frames_t.permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        for start in starts:
            clip = frames_t[start : start + window]  # [window, C, H, W]
            # I3D expects [B, C, T, H, W].
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            out = model(clip)  # [1, T_out, 1024] or [1, 1024]
            vec = _reduce_to_vector(out)
            feats.append(vec.detach().cpu().to(torch.float32).numpy())

    return np.stack(feats, axis=0).astype(np.float32, copy=False)


def _reduce_to_vector(out):
    """Reduce an I3D output to a single 1024-d vector per clip.

    The plan's :class:`I3DFeatureExtractor` strips the classification head
    but may still return ``[B, T_out, 1024]`` or ``[B, 1024]`` depending on
    whether the final pooling collapses time. Either way, average over any
    remaining temporal dim and squeeze the batch.
    """
    if out.dim() == 5:
        # [B, C, T, H, W] -> spatially+temporally pool.
        out = out.mean(dim=(2, 3, 4))
    elif out.dim() == 4:
        # [B, C, H, W] -> spatial pool.
        out = out.mean(dim=(2, 3))
    elif out.dim() == 3:
        # [B, T_out, 1024] -> average over T_out.
        out = out.mean(dim=1)
    elif out.dim() != 2:
        raise RuntimeError(
            f"Unexpected I3D output rank {out.dim()} with shape {tuple(out.shape)}."
        )
    return out.squeeze(0)
