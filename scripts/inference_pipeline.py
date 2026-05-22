#!/usr/bin/env python
"""End-to-end auto-labeling CLI: boundary segmenter + VLM labeler → new LeRobot dataset.

Thin wrapper around :func:`robosubtasknet.pipeline.run_inference_pipeline`.

The input LeRobot dataset is never modified; a new dataset is written to
``--output`` with predicted segmentation/labels. The boundary segmenter
proposes temporal cut points and the (optional) VLM labels each segment.
If ``--vlm-checkpoint`` is empty/missing, the VLM stage falls back to
zero-shot prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the 2-stage inference pipeline (boundary segmenter + VLM labeler) "
            "over a LeRobot dataset and write a new auto-labeled LeRobot dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input LeRobot root (never modified).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output LeRobot root (new auto-labeled dataset).",
    )
    p.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera key driving prediction (e.g. observation.images.head_left_eye).",
    )
    p.add_argument(
        "--boundary-checkpoint",
        type=Path,
        required=True,
        help="Trained boundary segmenter checkpoint (e.g. ckpts/boundary/final.pt).",
    )
    p.add_argument(
        "--vlm-checkpoint",
        type=Path,
        default=None,
        help=(
            "VLM LoRA adapter directory (e.g. ckpts/vlm/lora_adapter). "
            "Leave empty for zero-shot VLM prediction."
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.5,
        help="Threshold on boundary probability for peak acceptance.",
    )
    p.add_argument(
        "--min-distance",
        type=int,
        default=3,
        help="Peak NMS minimum distance in feature frames.",
    )
    p.add_argument(
        "--min-segment-length",
        type=int,
        default=5,
        help="Minimum segment length in feature frames; shorter segments are merged.",
    )
    p.add_argument(
        "--prominence",
        type=float,
        default=0.1,
        help="Minimum peak prominence for boundary detection.",
    )
    p.add_argument(
        "--n-frames-per-segment",
        type=int,
        default=8,
        help="Number of frames sampled per segment when querying the VLM.",
    )

    predict_group = p.add_mutually_exclusive_group()
    predict_group.add_argument(
        "--predict-task-if-missing",
        dest="predict_task_if_missing",
        action="store_true",
        help=(
            "Use the VLM to predict the high-level task when episode metadata "
            "lacks one (default behavior)."
        ),
    )
    predict_group.add_argument(
        "--no-predict-task",
        dest="predict_task_if_missing",
        action="store_false",
        help="Disable VLM task prediction when metadata is missing a task.",
    )
    p.set_defaults(predict_task_if_missing=True)

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N episodes (smoke test).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.is_dir():
        print(f"[infer] error: input dataset not found: {args.input}", file=sys.stderr)
        return 1
    if not args.boundary_checkpoint.exists():
        print(
            f"[infer] error: boundary checkpoint not found: {args.boundary_checkpoint}",
            file=sys.stderr,
        )
        return 1

    vlm_ckpt: Path | None = args.vlm_checkpoint
    if vlm_ckpt is not None:
        # Treat empty string / "none" sentinel as zero-shot.
        if str(vlm_ckpt) in ("", ".", "none", "None"):
            vlm_ckpt = None
        elif not vlm_ckpt.exists():
            print(
                f"[infer] error: VLM checkpoint not found: {vlm_ckpt}",
                file=sys.stderr,
            )
            return 1

    args.output.mkdir(parents=True, exist_ok=True)

    from robosubtasknet.pipeline import run_inference_pipeline

    result = run_inference_pipeline(
        input_root=args.input,
        output_root=args.output,
        camera_key=args.camera,
        boundary_checkpoint=args.boundary_checkpoint,
        vlm_checkpoint=vlm_ckpt,
        device=args.device,
        boundary_threshold=args.boundary_threshold,
        min_distance=args.min_distance,
        min_segment_length=args.min_segment_length,
        prominence=args.prominence,
        n_frames_per_segment=args.n_frames_per_segment,
        predict_task_if_missing=args.predict_task_if_missing,
        limit=args.limit,
    )

    if isinstance(result, dict):
        n_eps = int(
            result.get(
                "num_episodes",
                result.get("n_episodes", result.get("episodes", 0)),
            )
        )
    elif isinstance(result, int):
        n_eps = result
    else:
        n_eps = 0

    print(f"[infer] done. {n_eps} episodes → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
