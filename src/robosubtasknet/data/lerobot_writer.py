"""Writer for LeRobot v2.1 dataset format.

At inference, RoboSubtaskNet produces a new LeRobot dataset at a user-specified
output location. To avoid duplicating multi-gigabyte video files, mp4s are
hard-linked (``os.link``) into the output tree. Parquet files are reread from
the input and rewritten with predicted ``action_text_id`` labels.

Cross-filesystem hard-links fail with ``OSError(EXDEV)``; we fall back to
symlink with a warning rather than silently copying.
"""

from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


__all__ = [
    "hardlink_or_fallback",
    "segments_from_labels",
    "write_meta_files",
    "write_episode_parquet",
]

logger = logging.getLogger(__name__)


def hardlink_or_fallback(
    src: Path, dst: Path
) -> Literal["hardlink", "symlink", "skipped"]:
    """Hard-link ``src`` to ``dst``. Fall back to symlink on cross-filesystem.

    Idempotent: returns ``"skipped"`` if ``dst`` already exists (no overwrite).

    Returns
    -------
    str
        ``"hardlink"`` on success, ``"symlink"`` if fell back across filesystems,
        ``"skipped"`` if dst already exists.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "skipped"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e:
        if e.errno == errno.EXDEV:
            logger.warning(
                "Cross-filesystem hard-link from %s to %s not allowed; "
                "falling back to symlink (output dataset will require the "
                "original videos to remain present).",
                src,
                dst,
            )
            os.symlink(str(src.resolve()), str(dst))
            return "symlink"
        raise


def segments_from_labels(
    labels: np.ndarray | Iterable[int],
    action_text: dict[int, str],
) -> list[dict]:
    """RLE-encode per-frame labels into segment dicts.

    Returns a list of ``{"start_frame", "end_frame", "english_action_text",
    "action_text_id"}`` entries with HALF-OPEN intervals (end_frame is
    exclusive). Compatible with the converter scripts' ``action_config``
    format in ``meta/episodes.jsonl``.

    Empty input returns ``[]``.
    """
    labels_arr = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels)
    if labels_arr.size == 0:
        return []
    segments: list[dict] = []
    start = 0
    current = int(labels_arr[0])
    for i in range(1, len(labels_arr)):
        v = int(labels_arr[i])
        if v != current:
            segments.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(i),
                    "english_action_text": action_text.get(
                        current, f"class_{current}"
                    ),
                    "action_text_id": int(current),
                }
            )
            start = i
            current = v
    segments.append(
        {
            "start_frame": int(start),
            "end_frame": int(len(labels_arr)),
            "english_action_text": action_text.get(current, f"class_{current}"),
            "action_text_id": int(current),
        }
    )
    return segments


def write_meta_files(
    output_root: Path,
    info: dict,
    tasks: list[dict],
    episodes: list[dict],
    episodes_stats: list[dict],
    action_text: dict[int, str],
) -> None:
    """Write all ``meta/*.json[l]`` files for the output LeRobot dataset.

    Parameters
    ----------
    output_root : Path
        Root of the output dataset (``meta/`` is created inside).
    info : dict
        Contents of ``meta/info.json`` (codebase_version, fps, features, paths).
    tasks : list[dict]
        Contents of ``meta/tasks.jsonl``, one dict per line.
    episodes : list[dict]
        Contents of ``meta/episodes.jsonl``, one dict per line.
    episodes_stats : list[dict]
        Contents of ``meta/episodes_stats.jsonl``, one dict per line.
    action_text : dict[int, str]
        Subtask vocabulary {class_idx: english_name}. Written to
        ``meta/action_text.json`` with string-keyed IDs (LeRobot convention).
    """
    output_root = Path(output_root)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for s in episodes_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    action_text_str = {str(k): v for k, v in action_text.items()}
    with (meta_dir / "action_text.json").open("w", encoding="utf-8") as f:
        json.dump(action_text_str, f, indent=4, ensure_ascii=False)


def write_episode_parquet(
    output_path: Path,
    source_parquet: Path,
    predicted_labels: np.ndarray,
) -> None:
    """Read source parquet, replace ``action_text_id`` column with predictions,
    write to output.

    If the source parquet lacks an ``action_text_id`` column it is added.
    If predicted_labels length differs from frame count, it's resized
    (pad with last value or truncate).
    """
    import pandas as pd  # local import: pandas is heavy, only needed here

    df = pd.read_parquet(source_parquet)
    n_frames = len(df)
    pred = np.asarray(predicted_labels).astype(np.int64)
    if len(pred) != n_frames:
        if len(pred) < n_frames:
            fill = int(pred[-1]) if len(pred) > 0 else 0
            pad = np.full(n_frames - len(pred), fill, dtype=np.int64)
            pred = np.concatenate([pred, pad])
        else:
            pred = pred[:n_frames]
    df["action_text_id"] = pred

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
