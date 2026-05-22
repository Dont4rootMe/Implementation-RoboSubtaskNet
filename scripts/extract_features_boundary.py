#!/usr/bin/env python
"""Extract I3D features + binary boundary labels from one or more LeRobot datasets.

Outputs:
    <cache>/<dataset>__episode_<M:06d>.npz   per-episode features + binary boundaries
    <cache>/manifest.json                     list of cached episodes + extraction settings

Boundary derivation: at I3D feature stride 8, label[t] = 1 iff the underlying
frame-level ``action_text_id`` array contains a transition somewhere in the
``[t*stride, (t+1)*stride)`` window. This handles intra-window boundaries
correctly.

Idempotent: skips episodes whose ``.npz`` already exists.

Episodes whose parquet lacks the ``action_text_id`` column are skipped — we
cannot derive boundary supervision without it.

Usage:
  python scripts/extract_features_boundary.py \\
      --input /data/lerobot_d1 /data/lerobot_d2 \\
      --camera observation.images.head_left_eye \\
      --cache ./features_boundary
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
            "Extract I3D features and BINARY boundary labels from LeRobot "
            "datasets (class-agnostic; no vocabulary needed)."
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
# LeRobot metadata I/O (copied from scripts/extract_features_lerobot.py)
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
# Path resolution (templates from info.json) — copied
# --------------------------------------------------------------------------- #


def _format_template(template: str, **kwargs: Any) -> str:
    """Like ``str.format`` but raises with a clear message for missing keys."""
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
# Boundary derivation
# --------------------------------------------------------------------------- #


def derive_boundaries(
    action_text_ids: np.ndarray, T_feat: int, stride: int
) -> np.ndarray:
    """For each output frame t, set ``boundary[t]=1`` iff
    ``action_text_ids[t*stride:(t+1)*stride]`` contains at least one
    transition (i.e. consecutive values differ).

    Returns int32 of length ``T_feat``.
    """
    boundary = np.zeros(T_feat, dtype=np.int32)
    N = len(action_text_ids)
    for t in range(T_feat):
        a = t * stride
        b = min((t + 1) * stride, N)
        if b - a < 2:
            continue
        chunk = action_text_ids[a:b]
        if np.any(np.diff(chunk) != 0):
            boundary[t] = 1
    return boundary


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

    # Local import: pulls in torch/decord/pytorchvideo lazily so --help stays fast.
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

            # Read parquet first so we can skip unlabeled episodes BEFORE
            # spending GPU time on feature extraction.
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] ep {ep_idx} parquet read: {exc}")
                continue

            if "action_text_id" not in df.columns:
                print(
                    f"  [skip] ep {ep_idx}: parquet lacks 'action_text_id' column "
                    f"(cannot derive boundaries)"
                )
                continue

            action_text_ids = df["action_text_id"].values.astype(np.int64)

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

            # Derive binary boundaries at feature stride.
            boundary = derive_boundaries(action_text_ids, T_feat, args.stride)

            # Truncate to the shorter of T_feat / len(boundary). With the
            # derivation above ``len(boundary) == T_feat`` by construction,
            # but we keep the guard for symmetry / future changes.
            T_final = int(min(T_feat, len(boundary)))
            rgb = features["rgb"][:T_final]
            flow = features["flow"][:T_final]
            boundary = boundary[:T_final]

            meta = dict(features["meta"])
            meta["dataset"] = dataset_root.name
            meta["episode_index"] = ep_idx
            meta["camera_key"] = camera_key
            meta["T_feat"] = T_final
            meta["label_kind"] = "binary_boundary"

            save_features_npz(
                cache_path,
                rgb=rgb,
                flow=flow,
                labels=boundary.astype(np.int32),
                meta=meta,
            )
            manifest.append(cache_name)
            total_ok += 1
            n_pos = int(boundary.sum())
            print(
                f"  [ok] ep {ep_idx} → {cache_name} "
                f"(T_feat={T_final}, boundaries={n_pos})"
            )

    # ----- Manifest
    manifest_path = args.cache / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": manifest,
                "label_kind": "binary_boundary",
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
