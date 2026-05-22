#!/usr/bin/env python
"""Extract I3D RGB + Flow features from one or more LeRobot datasets.

Outputs:

- ``<cache>/<dataset>__episode_<M:06d>.npz``  per-episode features + labels
  (schema compatible with ``RoboSubtaskDataset(mode="npz")``).
- ``<label_map>``  unified vocabulary across all input datasets
  ``{action_text_name: class_idx}``. ``"background"`` reserved as class 0;
  remaining names sorted alphabetically.
- ``<cache>/manifest.json``  list of cached episode filenames.

Idempotent: skips episodes whose ``.npz`` already exists.

Usage:
  python scripts/extract_features_lerobot.py \\
      --input /data/lerobot_d1 /data/lerobot_d2 \\
      --camera observation.images.head_left_eye \\
      --cache ./features \\
      --label-map ./features/label_map.json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract I3D features from LeRobot datasets and build a union "
            "label vocabulary for RoboSubtaskNet training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more LeRobot dataset roots.",
    )
    p.add_argument(
        "--camera",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Camera key(s). Either 1 value (applied to every input) or one per "
            "input position. Example: observation.images.head_left_eye"
        ),
    )
    p.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Output features cache directory.",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="Output JSON file: {action_text_name: class_idx}.",
    )
    p.add_argument("--flow", choices=["tvl1", "raft"], default="tvl1")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N episodes per dataset (smoke test).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# LeRobot metadata I/O
# --------------------------------------------------------------------------- #


def load_episodes_meta(root: Path) -> list[dict]:
    p = root / "meta" / "episodes.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    episodes: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                episodes.append(json.loads(s))
    return episodes


def load_action_text(root: Path) -> dict[int, str]:
    """Load ``meta/action_text.json`` → {int_id: name}.

    Returns empty dict if file is missing.
    """
    p = root / "meta" / "action_text.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Path resolution (templates from info.json)
# --------------------------------------------------------------------------- #


def _format_template(template: str, **kwargs: Any) -> str:
    """Like ``str.format`` but ignore unused placeholders gracefully."""
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(
            f"Template {template!r} missing key {exc.args[0]!r}; "
            f"available={list(kwargs.keys())}"
        ) from exc


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
    rel = _format_template(
        template,
        episode_chunk=chunk_idx,
        video_key=camera_key,
        episode_index=ep_idx,
    )
    return root / rel


def resolve_parquet_path(root: Path, info: dict, ep_idx: int) -> Path:
    chunk_idx = get_chunk_idx(ep_idx, info)
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = _format_template(
        template,
        episode_chunk=chunk_idx,
        episode_index=ep_idx,
    )
    return root / rel


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


def build_union_label_map(input_roots: list[Path]) -> dict[str, int]:
    """Collect every subtask name across all input datasets' action_text.json
    files; assign deterministic class indices.

    Convention:

    - ``"background"`` → 0 (always; inserted even if absent from inputs).
    - All other names sorted alphabetically → 1, 2, ...
    """
    names: set[str] = set()
    for root in input_roots:
        action_text = load_action_text(root)
        names.update(action_text.values())
    names.discard("background")
    label_map: dict[str, int] = {"background": 0}
    for name in sorted(names):
        label_map[name] = len(label_map)
    return label_map


# --------------------------------------------------------------------------- #
# Label downsampling
# --------------------------------------------------------------------------- #


def downsample_labels(
    labels: np.ndarray, T_feat: int, stride: int
) -> np.ndarray:
    """Downsample dense per-frame labels to ``T_feat`` I3D windows.

    Uses linear-spaced indexing so the first/last windows are anchored to
    the start/end of the original sequence; intermediate windows pick the
    label at the nearest original frame.

    Returns int64 of length exactly ``T_feat``. If labels is empty or
    T_feat is 0, returns zeros.
    """
    if T_feat <= 0:
        return np.zeros(0, dtype=np.int64)
    if len(labels) == 0:
        return np.zeros(T_feat, dtype=np.int64)
    if T_feat == 1:
        return np.asarray([labels[0]], dtype=np.int64)
    indices = np.linspace(0, len(labels) - 1, num=T_feat).round().astype(np.int64)
    indices = np.clip(indices, 0, len(labels) - 1)
    return labels[indices].astype(np.int64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()

    # Resolve --camera vs --input pairing
    if len(args.camera) == 1 and len(args.input) >= 1:
        cameras = list(args.camera) * len(args.input)
    elif len(args.camera) == len(args.input):
        cameras = list(args.camera)
    else:
        raise SystemExit(
            f"--camera must have either 1 value (broadcast) or "
            f"{len(args.input)} values (one per --input); got {len(args.camera)}."
        )

    args.cache.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: union vocabulary
    print(f"[extract] building union vocabulary from {len(args.input)} dataset(s)...")
    label_map = build_union_label_map(args.input)
    args.label_map.parent.mkdir(parents=True, exist_ok=True)
    with args.label_map.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print(
        f"[extract] vocabulary: {len(label_map)} classes "
        f"(background + {len(label_map) - 1}) → {args.label_map}"
    )

    # ----- Step 2: per-episode feature extraction
    # Local import: pulls in torch/decord/pytorchvideo lazily.
    import pandas as pd
    from robosubtasknet.features.extract import (
        extract_features_for_video,
        save_features_npz,
    )

    manifest: list[str] = []
    total_episodes = 0
    total_ok = 0
    total_skipped = 0

    for dataset_root, camera_key in zip(args.input, cameras):
        print(f"[extract] processing {dataset_root.name} (camera={camera_key})...")
        info = load_info(dataset_root)
        episodes = load_episodes_meta(dataset_root)
        if args.limit is not None:
            episodes = episodes[: args.limit]
        local_action_text = load_action_text(dataset_root)

        # Build local action_text_id → class_idx via name lookup.
        local_id_to_class: dict[int, int] = {}
        for local_id, name in local_action_text.items():
            local_id_to_class[int(local_id)] = label_map.get(name, 0)

        for ep in episodes:
            total_episodes += 1
            ep_idx = int(ep["episode_index"])
            cache_name = f"{dataset_root.name}__episode_{ep_idx:06d}.npz"
            cache_path = args.cache / cache_name

            if cache_path.exists():
                manifest.append(cache_name)
                total_skipped += 1
                continue

            video_path = resolve_video_path(dataset_root, info, ep_idx, camera_key)
            parquet_path = resolve_parquet_path(dataset_root, info, ep_idx)

            if not video_path.exists():
                print(f"  [skip] ep {ep_idx}: missing video {video_path}")
                continue
            if not parquet_path.exists():
                print(f"  [skip] ep {ep_idx}: missing parquet {parquet_path}")
                continue

            try:
                features = extract_features_for_video(
                    video_path,
                    modality="both",
                    flow_method=args.flow,
                    window=args.window,
                    stride=args.stride,
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] ep {ep_idx} feature extraction: {exc}")
                continue

            T_feat = int(features["meta"]["T_feat"])
            if T_feat == 0:
                print(f"  [skip] ep {ep_idx}: too short for window={args.window}")
                continue

            # Read labels from parquet (action_text_id per original frame)
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] ep {ep_idx} parquet read: {exc}")
                continue

            if "action_text_id" in df.columns:
                raw_labels = df["action_text_id"].values.astype(np.int64)
                mapped = np.array(
                    [local_id_to_class.get(int(lid), 0) for lid in raw_labels],
                    dtype=np.int64,
                )
            else:
                # Unlabeled — fill with background. Training on such data is
                # pointless but kept for symmetry with inference inputs.
                mapped = np.zeros(features["meta"]["original_T"], dtype=np.int64)

            labels_downsampled = downsample_labels(mapped, T_feat, args.stride)

            meta = dict(features["meta"])
            meta["dataset"] = dataset_root.name
            meta["episode_index"] = ep_idx
            meta["camera_key"] = camera_key

            save_features_npz(
                cache_path,
                rgb=features["rgb"],
                flow=features["flow"],
                labels=labels_downsampled,
                meta=meta,
            )
            manifest.append(cache_name)
            total_ok += 1
            print(f"  [ok] ep {ep_idx} → {cache_name} (T_feat={T_feat})")

    # ----- Step 3: manifest
    manifest_path = args.cache / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": manifest,
                "label_map": str(args.label_map),
                "num_classes": len(label_map),
                "feature_extraction": {
                    "window": args.window,
                    "stride": args.stride,
                    "flow_method": args.flow,
                },
            },
            f,
            indent=2,
        )

    print(
        f"[extract] done. {total_episodes} episodes seen, "
        f"{total_ok} new, {total_skipped} cached, "
        f"{total_episodes - total_ok - total_skipped} skipped."
    )
    print(f"[extract] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
