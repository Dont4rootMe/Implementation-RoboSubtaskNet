"""Feature-extraction CLI.

Iterate over the raw videos in a directory, run the I3D RGB and optical-flow
streams through ``robosubtasknet.features.extract.extract_features_for_video``,
and persist the resulting per-video ``.npz`` to disk in the layout described in
Section 6.3 of ``IMPLEMENTATION_PLAN.md``.

Per-video output (``data/features/<dataset>/<video_id>.npz``):

* ``rgb``    -- float16 ``[T_feat, 1024]``
* ``flow``   -- float16 ``[T_feat, 1024]``
* ``labels`` -- int32   ``[T_feat]`` (zeros when no annotations are supplied)
* ``meta``   -- JSON ``{fps, original_T, video_path, hash, window, stride, ...}``

CLI matches Section 6.4 of the plan::

    python scripts/extract_features.py \\
        --videos data/raw/robosubtask/ \\
        --annotations data/annotations/robosubtask/ \\
        --output data/features/robosubtask/ \\
        --flow tvl1 \\
        --window 16 --stride 8

Annotations are optional; the CLI tries to pair each video with a
``<video_id>.txt`` (MS-TCN) or ``<video_id>.csv`` (RoboSubtask format) under
``--annotations`` and downsamples them to the ``T_feat`` produced by the
sliding-window extractor.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# Make the in-repo ``src/`` importable when the package isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robosubtasknet.features.extract import (  # noqa: E402
    extract_features_for_video,
    save_features_npz,
)


# Video file extensions ``decord`` is happy to read. Lowercased; the discovery
# loop matches on the suffix in a case-insensitive manner.
_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and return the CLI namespace.

    Defined as a module-level function so the tests (and any later Hydra-based
    wrapper) can call into it without invoking ``main`` end-to-end.
    """
    parser = argparse.ArgumentParser(
        description="Extract I3D RGB + optical-flow features per video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="Directory containing the input videos.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Optional directory of per-video annotation files. Looks for "
            "<video_id>.txt (MS-TCN) or <video_id>.csv (frame_idx,subtask)."
        ),
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help=(
            "Optional label-name -> integer mapping (MS-TCN mapping.txt format). "
            "Required when annotations carry textual labels."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write per-video .npz feature files into.",
    )
    parser.add_argument(
        "--flow",
        choices=("tvl1", "raft"),
        default="tvl1",
        help="Optical-flow estimator to use (Section 6.2 of the plan).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="I3D input window length (frames).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between consecutive sliding-window starts (frames).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the I3D forward pass (auto-falls back to CPU).",
    )
    parser.add_argument(
        "--modality",
        choices=("rgb", "flow", "both"),
        default="both",
        help="Which feature streams to extract.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if an output .npz already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, stop after this many videos (useful for smoke tests).",
    )
    return parser.parse_args(argv)


def _discover_videos(videos_dir: Path) -> list[Path]:
    """Return all video files under ``videos_dir`` sorted by name."""
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"--videos directory does not exist: {videos_dir}")
    found: list[Path] = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
            found.append(p)
    return found


def _load_label_mapping(mapping_file: Optional[Path]) -> Optional[dict[str, int]]:
    """Lazy-load the label-name -> int mapping; ``None`` if not supplied."""
    if mapping_file is None:
        return None
    from robosubtasknet.data.dataset import load_mapping

    return load_mapping(mapping_file)


def _resolve_labels(
    video_id: str,
    annotations_dir: Optional[Path],
    mapping: Optional[dict[str, int]],
    t_feat: int,
    feature_stride: int,
) -> np.ndarray:
    """Look up a per-frame annotation file and downsample it to ``T_feat``.

    Returns an all-zero int32 array when no annotation is found (so the call
    site can still write a well-formed ``.npz``). Down-sampling is a simple
    "take every ``feature_stride``-th frame" because the I3D extractor produces
    one feature per ``stride`` raw frames.
    """
    if t_feat <= 0:
        return np.empty((0,), dtype=np.int32)
    if annotations_dir is None:
        return np.zeros((t_feat,), dtype=np.int32)

    txt_path = annotations_dir / f"{video_id}.txt"
    csv_path = annotations_dir / f"{video_id}.csv"

    # Import lazily so missing optional helpers don't break the CLI for users
    # who never pass --annotations.
    from robosubtasknet.data.annotation import (
        labels_to_indices,
        parse_per_frame_labels_csv,
        parse_per_frame_labels_txt,
    )

    if txt_path.exists():
        names = parse_per_frame_labels_txt(txt_path)
    elif csv_path.exists():
        names = parse_per_frame_labels_csv(csv_path)
    else:
        return np.zeros((t_feat,), dtype=np.int32)

    if mapping is not None:
        indices = labels_to_indices(names, mapping)
    else:
        # No mapping supplied: best-effort interpretation as integers.
        try:
            indices = [int(n) for n in names]
        except ValueError as exc:
            raise ValueError(
                f"Annotation file for {video_id} has textual labels but no "
                "--mapping-file was provided."
            ) from exc

    # Downsample by selecting one index per feature window. Roughly matches the
    # MS-TCN convention of subsampling labels at the I3D feature stride.
    frame_arr = np.asarray(indices, dtype=np.int32)
    if frame_arr.size == 0:
        return np.zeros((t_feat,), dtype=np.int32)

    sample_idx = np.linspace(
        0, frame_arr.size - 1, num=t_feat, dtype=np.int64
    )
    return frame_arr[sample_idx].astype(np.int32, copy=False)


def main() -> int:
    """Entry-point used by ``python -m scripts.extract_features``."""
    args = parse_args()

    videos_dir: Path = args.videos
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _discover_videos(videos_dir)
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        print(f"[extract] No videos discovered under {videos_dir}", file=sys.stderr)
        return 1

    mapping = _load_label_mapping(args.mapping_file)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for video_path in videos:
        video_id = video_path.stem
        out_path = output_dir / f"{video_id}.npz"

        if out_path.exists() and not args.overwrite:
            print(f"[extract] skip (exists): {video_id}")
            n_skip += 1
            continue

        print(f"[extract] {video_id}  <- {video_path}")
        try:
            result = extract_features_for_video(
                video_path,
                modality=args.modality,
                flow_method=args.flow,
                window=args.window,
                stride=args.stride,
                device=args.device,
            )
        except Exception as exc:  # noqa: BLE001 - we want to keep going.
            print(
                f"[extract] FAILED {video_id}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            n_fail += 1
            continue

        rgb = result.get("rgb")
        flow = result.get("flow")
        meta = dict(result.get("meta", {}))
        meta.setdefault("video_id", video_id)

        t_feat = int(meta.get("T_feat", 0) or 0)
        labels = _resolve_labels(
            video_id=video_id,
            annotations_dir=args.annotations,
            mapping=mapping,
            t_feat=t_feat,
            feature_stride=int(args.stride),
        )

        save_features_npz(out_path, rgb=rgb, flow=flow, labels=labels, meta=meta)
        n_ok += 1
        print(f"[extract] saved  {out_path}  (T_feat={t_feat})")

    print(
        f"[extract] done: {n_ok} ok, {n_skip} skipped, {n_fail} failed "
        f"(total {len(videos)})"
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
