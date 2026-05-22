#!/usr/bin/env python
"""Auto-label a LeRobot dataset using a trained RoboSubtaskNet model.

Produces a new LeRobot v2.1 dataset at ``--output``:

- Per-episode parquet: source parquet copied with ``action_text_id`` column
  replaced by model predictions (using the MODEL's vocabulary).
- Video mp4s: **hard-linked** from the input (no copy). All camera streams
  are linked, not just the one used for prediction, so the output is a
  complete LeRobot dataset.
- Meta files: ``info.json``, ``tasks.jsonl``, ``episodes_stats.jsonl`` copied;
  ``episodes.jsonl`` rewritten with derived ``action_config`` per episode;
  ``action_text.json`` replaced with the model's vocabulary.

The model's vocabulary OVERRIDES the input's. Predictions are only meaningful
in the model's class space; output IDs are model class indices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-label a LeRobot dataset → new dataset with hard-linked videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Input LeRobot root.")
    p.add_argument("--output", type=Path, required=True, help="Output LeRobot root.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Trained .pt file.")
    p.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera key driving predictions (e.g. observation.images.head_left_eye).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--flow",
        choices=["tvl1", "raft"],
        default=None,
        help="Override flow method; defaults to value stored in checkpoint.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N episodes (for testing).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# LeRobot metadata I/O
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_chunk_idx(ep_idx: int, info: dict) -> int:
    return ep_idx // int(info.get("chunks_size", 1000))


def resolve_video_path(
    root: Path, info: dict, ep_idx: int, camera_key: str
) -> Path:
    chunk_idx = get_chunk_idx(ep_idx, info)
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return root / template.format(
        episode_chunk=chunk_idx,
        video_key=camera_key,
        episode_index=ep_idx,
    )


def resolve_parquet_path(root: Path, info: dict, ep_idx: int) -> Path:
    chunk_idx = get_chunk_idx(ep_idx, info)
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return root / template.format(
        episode_chunk=chunk_idx,
        episode_index=ep_idx,
    )


def list_camera_keys(root: Path, info: dict, ep_idx: int) -> list[str]:
    """List camera subdirectories that have a video for this episode."""
    chunk_idx = get_chunk_idx(ep_idx, info)
    videos_dir = root / "videos" / f"chunk-{chunk_idx:03d}"
    if not videos_dir.exists():
        return []
    return sorted(d.name for d in videos_dir.iterdir() if d.is_dir())


# --------------------------------------------------------------------------- #
# Metadata extraction from checkpoint
# --------------------------------------------------------------------------- #


def extract_metadata(ckpt: dict, checkpoint_path: Path) -> dict:
    """Pull ``label_map`` / ``action_text`` / ``model_config`` / etc. from a
    checkpoint, falling back to a ``metadata.json`` sidecar next to it.
    """
    required = ("label_map", "model_config")
    if all(k in ckpt for k in required):
        return ckpt
    # Try config_snapshot (Trainer.fit stores this)
    cs = ckpt.get("config_snapshot")
    if isinstance(cs, dict) and all(k in cs for k in required):
        return cs
    # Fall back to metadata.json sidecar
    sidecar = checkpoint_path.parent / "metadata.json"
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise RuntimeError(
        f"Checkpoint {checkpoint_path} has no embedded metadata (label_map, "
        f"model_config). Pair it with a metadata.json sidecar in the same "
        f"directory, or use the final.pt produced by train_lerobot.py."
    )


def coerce_action_text(action_text_raw: Any) -> dict[int, str]:
    """Accept dict with int or str keys; return int-keyed dict."""
    if not isinstance(action_text_raw, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in action_text_raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Prediction post-processing
# --------------------------------------------------------------------------- #


def upsample_predictions(predictions: np.ndarray, target_length: int) -> np.ndarray:
    """Upsample short prediction sequence to ``target_length`` via nearest-neighbor."""
    n = len(predictions)
    if target_length <= 0:
        return np.zeros(0, dtype=np.int64)
    if n == 0:
        return np.zeros(target_length, dtype=np.int64)
    if n == target_length:
        return predictions.astype(np.int64)
    if target_length == 1:
        return np.asarray([predictions[0]], dtype=np.int64)
    indices = np.linspace(0, n - 1, num=target_length).round().astype(np.int64)
    indices = np.clip(indices, 0, n - 1)
    return predictions[indices].astype(np.int64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input dataset not found: {args.input}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Load checkpoint + metadata
    print(f"[infer] loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = extract_metadata(ckpt, args.checkpoint)

    label_map: dict[str, int] = metadata["label_map"]
    action_text = coerce_action_text(
        metadata.get("action_text", {v: k for k, v in label_map.items()})
    )
    model_config: dict[str, Any] = metadata["model_config"]
    feat_cfg: dict[str, Any] = metadata.get(
        "feature_extraction",
        {"window": 16, "stride": 8, "flow_method": "tvl1"},
    )
    if args.flow is not None:
        feat_cfg = dict(feat_cfg)
        feat_cfg["flow_method"] = args.flow

    # Build model
    from robosubtasknet.data.lerobot_writer import (
        hardlink_or_fallback,
        segments_from_labels,
        write_episode_parquet,
        write_meta_files,
    )
    from robosubtasknet.features.extract import extract_features_for_video
    from robosubtasknet.models import RoboSubtaskNet

    model = RoboSubtaskNet(**model_config).to(device)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        state_dict = ckpt  # bare state_dict checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(
        f"[infer] model loaded ({model.count_parameters():,} params), "
        f"num_classes={model_config['num_classes']}"
    )

    # Read input dataset metadata
    input_info = load_json(args.input / "meta" / "info.json")
    if input_info is None:
        raise SystemExit(f"Missing {args.input / 'meta' / 'info.json'}")
    tasks = load_jsonl(args.input / "meta" / "tasks.jsonl")
    input_episodes = load_jsonl(args.input / "meta" / "episodes.jsonl")
    input_stats = load_jsonl(args.input / "meta" / "episodes_stats.jsonl")

    if args.limit is not None:
        input_episodes = input_episodes[: args.limit]

    # Validate camera presence on first available chunk
    if input_episodes:
        sample_ep = int(input_episodes[0]["episode_index"])
        present = list_camera_keys(args.input, input_info, sample_ep)
        if args.camera not in present:
            raise SystemExit(
                f"--camera={args.camera!r} not present in input dataset. "
                f"Found: {present}"
            )

    args.output.mkdir(parents=True, exist_ok=True)

    # Process each episode
    output_episodes: list[dict] = []
    total = len(input_episodes)
    n_hardlink = n_symlink = n_skipped = 0

    for idx, ep in enumerate(input_episodes):
        ep_idx = int(ep["episode_index"])
        chunk_idx = get_chunk_idx(ep_idx, input_info)

        video_path = resolve_video_path(args.input, input_info, ep_idx, args.camera)
        parquet_path = resolve_parquet_path(args.input, input_info, ep_idx)

        if not video_path.exists():
            print(f"  [skip] ep {ep_idx}: missing video {video_path}")
            continue
        if not parquet_path.exists():
            print(f"  [skip] ep {ep_idx}: missing parquet {parquet_path}")
            continue

        print(f"[infer] ({idx + 1}/{total}) ep {ep_idx}...")

        # Feature extraction
        try:
            features = extract_features_for_video(
                video_path,
                modality="both",
                flow_method=str(feat_cfg.get("flow_method", "tvl1")),
                window=int(feat_cfg.get("window", 16)),
                stride=int(feat_cfg.get("stride", 8)),
                device=device_str,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] ep {ep_idx} feature extraction: {exc}")
            continue

        if features["meta"]["T_feat"] == 0:
            print(f"  [skip] ep {ep_idx}: too short for window")
            continue

        rgb_t = (
            torch.from_numpy(features["rgb"]).float().unsqueeze(0).to(device)
        )  # [1, T, D]
        flow_t = (
            torch.from_numpy(features["flow"]).float().unsqueeze(0).to(device)
        )

        with torch.no_grad():
            preds = model.predict(rgb_t, flow_t)  # [1, T_feat]
        preds_np = preds.cpu().numpy()[0].astype(np.int64)

        target_length = int(features["meta"]["original_T"])
        labels_per_frame = upsample_predictions(preds_np, target_length)

        # Write parquet
        output_parquet = (
            args.output / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        )
        try:
            write_episode_parquet(output_parquet, parquet_path, labels_per_frame)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] ep {ep_idx} parquet write: {exc}")
            continue

        # Hard-link videos for ALL cameras (preserve the full dataset).
        cam_keys = list_camera_keys(args.input, input_info, ep_idx)
        for cam in cam_keys:
            src = resolve_video_path(args.input, input_info, ep_idx, cam)
            if not src.exists():
                continue
            dst = (
                args.output / "videos" / f"chunk-{chunk_idx:03d}" / cam / f"episode_{ep_idx:06d}.mp4"
            )
            kind = hardlink_or_fallback(src, dst)
            if kind == "hardlink":
                n_hardlink += 1
            elif kind == "symlink":
                n_symlink += 1
            else:
                n_skipped += 1

        # Build updated episode meta with new action_config from RLE.
        segments = segments_from_labels(labels_per_frame, action_text)
        new_ep = dict(ep)
        new_ep["action_config"] = segments
        new_ep["length"] = int(target_length)
        output_episodes.append(new_ep)

    # Finalize meta files
    output_info = dict(input_info)
    output_info["total_episodes"] = len(output_episodes)
    output_info["total_frames"] = int(
        sum(ep.get("length", 0) for ep in output_episodes)
    )
    output_info["splits"] = {"train": f"0:{len(output_episodes)}"}

    write_meta_files(
        args.output,
        info=output_info,
        tasks=tasks,
        episodes=output_episodes,
        episodes_stats=input_stats,
        action_text=action_text,
    )

    print(
        f"[infer] done. {len(output_episodes)}/{total} episodes written. "
        f"videos: hardlink={n_hardlink} symlink={n_symlink} skipped={n_skipped}"
    )
    print(f"[infer] output → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
