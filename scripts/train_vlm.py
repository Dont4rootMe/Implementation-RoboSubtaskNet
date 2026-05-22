#!/usr/bin/env python
"""Fine-tune Qwen2-VL with LoRA on robot subtask labeling.

LoRA fine-tunes Qwen2-VL-2B on ``(video segment, task instruction, subtask label)``
triplets sourced from one or more LeRobot dataset roots. The fine-tuned LoRA
adapter is saved to ``<output>/lora_adapter/`` and is loaded at e2e inference.

Pipeline:

1. Seed RNGs for reproducibility.
2. Broadcast/validate ``--camera`` against ``--input``.
3. Build :class:`VLMSegmentDataset` over the (root, camera) pairs.
4. Build :class:`SegmentLabelerVLM` with a LoRA adapter attached.
5. Build :class:`VLMSegmentCollator` for chat-format batching.
6. Compose ``transformers.TrainingArguments`` via
   :func:`build_hf_training_args` and hand everything off to
   :func:`train_lora`.
7. Write a ``metadata.json`` sidecar with the run configuration so the
   adapter can be reloaded at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2-VL on robot subtask labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more LeRobot dataset roots. Each root must expose either "
            "``action_config`` or ``action_text_id`` + ``action_text.json``."
        ),
    )
    p.add_argument(
        "--camera",
        nargs="+",
        type=str,
        required=True,
        help="Camera key(s) per --input (1 broadcasts or one-to-one).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (LoRA adapter + checkpoints + logs).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Hugging Face model id to fine-tune.",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--n-frames",
        type=int,
        default=8,
        help="Frames sampled per segment.",
    )
    p.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("H", "W"),
        help="Frame resize target as (H, W).",
    )
    p.add_argument(
        "--max-segments-per-episode",
        type=int,
        default=None,
        help="Optional cap on segments drawn per episode (default: no cap).",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Gradient checkpointing: default ON, --no-gc disables.
    p.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (default: on).",
    )
    p.add_argument(
        "--no-gc",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )

    # bf16: default ON, --no-bf16 disables.
    p.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        default=True,
        help="Use bf16 mixed precision (default: on).",
    )
    p.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable bf16 mixed precision.",
    )
    return p.parse_args()


def _broadcast_cameras(inputs: list[Path], cameras: list[str]) -> list[str]:
    if len(cameras) == 1:
        return list(cameras) * len(inputs)
    if len(cameras) == len(inputs):
        return list(cameras)
    raise SystemExit(
        f"--camera must have 1 or {len(inputs)} values; got {len(cameras)}."
    )


def main() -> int:
    args = parse_args()

    inputs: list[Path] = list(args.input)
    camera_keys = _broadcast_cameras(inputs, list(args.camera))
    pairs = list(zip(inputs, camera_keys))

    args.output.mkdir(parents=True, exist_ok=True)

    # Imports deferred so ``--help`` is fast and doesn't require heavy deps.
    from robosubtasknet.vlm import SegmentLabelerVLM
    from robosubtasknet.vlm.collator import VLMSegmentCollator
    from robosubtasknet.vlm.dataset import VLMSegmentDataset
    from robosubtasknet.vlm.training import build_hf_training_args, train_lora
    from robosubtasknet.training import set_seed

    set_seed(args.seed)

    frame_size = tuple(args.frame_size)
    print(
        f"[train-vlm] inputs={len(pairs)} cameras={camera_keys} "
        f"n_frames={args.n_frames} frame_size={frame_size}"
    )

    dataset = VLMSegmentDataset(
        pairs,
        n_frames=args.n_frames,
        frame_size=frame_size,
        max_segments_per_episode=args.max_segments_per_episode,
        seed=args.seed,
    )
    print(f"[train-vlm] dataset: {len(dataset)} segments")

    vlm = SegmentLabelerVLM.from_pretrained(
        args.model_name,
        lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype="bfloat16",
    )

    collator = VLMSegmentCollator(
        vlm.processor,
        vlm.tokenizer,
        max_length=args.max_length,
    )

    training_args = build_hf_training_args(
        output_dir=args.output,
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        seed=args.seed,
    )

    result = train_lora(
        vlm,
        dataset,
        collator,
        args.output,
        training_args=training_args,
    )

    # Metadata sidecar — mirrors what the inference path needs to reload
    # the LoRA adapter alongside its base model.
    metadata = {
        "model_name": args.model_name,
        "camera_keys": camera_keys,
        "input_paths": [str(p) for p in inputs],
        "n_frames": args.n_frames,
        "frame_size": list(frame_size),
        "max_segments_per_episode": args.max_segments_per_episode,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "training": {
            "epochs": args.epochs,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16": args.bf16,
            "seed": args.seed,
        },
        "adapter_dir": "lora_adapter",
    }
    metadata_path = args.output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[train-vlm] metadata sidecar → {metadata_path}")

    adapter_dir = args.output / "lora_adapter"
    print(f"[train-vlm] LoRA adapter expected at → {adapter_dir}")
    if isinstance(result, dict):
        best = result.get("best_metric")
        if best is not None:
            print(f"[train-vlm] best metric: {best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
