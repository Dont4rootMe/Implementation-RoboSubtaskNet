"""End-to-end inference pipeline: boundary segmentation + VLM labeling.

This module orchestrates the full two-stage RoboSubtaskNet inference pass:

1. Read an input LeRobot v2.1 dataset (untouched).
2. For each episode:
   a. Extract I3D RGB + flow features for the driving camera video.
   b. Run the Stage-1 boundary segmenter to obtain per-frame boundary
      probabilities, then post-process them into discrete ``(start, end)``
      segments in original-frame coordinates.
   c. Run the Stage-2 Qwen2-VL labeler (optionally LoRA-adapted) on each
      segment, conditioned on the episode's task instruction, to obtain a
      short natural-language subtask label.
   d. Deduplicate the predicted phrases into a *globally* shared output
      vocabulary and build a per-frame ``action_text_id`` array.
3. Write a fresh LeRobot dataset at the output path:
   - Per-episode parquet: source parquet copied with ``action_text_id``
     replaced by predictions.
   - Per-camera mp4s: hard-linked from the input (cross-FS symlink fallback).
   - Meta files: ``info.json`` / ``tasks.jsonl`` / ``episodes_stats.jsonl``
     copied from input; ``episodes.jsonl`` rewritten with derived
     ``action_config`` per episode; ``action_text.json`` built fresh from
     the incrementally constructed output vocabulary.

The input dataset is treated as read-only; ``meta/action_text.json`` on the
output side is *constructed* from predictions, not copied from the input.

All heavy dependencies (``torch``, ``transformers``, ``peft``, ``decord``,
``cv2``) are imported lazily inside the function body so that simply
importing this module is cheap.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np


__all__ = ["run_inference_pipeline"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vocabulary helpers
# --------------------------------------------------------------------------- #
_BACKGROUND_TEXT = "background"
_BACKGROUND_ID = 0


def _build_output_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Initialise the output subtask vocabulary.

    Returns a pair of mappings: ``text -> id`` and ``id -> text``. The
    vocabulary is seeded with the reserved ``background`` class at id ``0``,
    used for frames that fall outside any predicted segment.
    """
    name_to_id: dict[str, int] = {_BACKGROUND_TEXT: _BACKGROUND_ID}
    id_to_name: dict[int, str] = {_BACKGROUND_ID: _BACKGROUND_TEXT}
    return name_to_id, id_to_name


def _ensure_text_id(
    text: str,
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> int:
    """Look up (or insert) ``text`` in the running output vocabulary.

    Normalises whitespace and case-folds the key so that minor stylistic
    drift in VLM outputs (``"Reach for the cup."`` vs ``"reach for the cup"``)
    collapses to the same class id. Empty / whitespace-only strings fall back
    to the reserved background class so that the per-frame label stream stays
    well-formed even when the VLM emits nothing for a segment.

    Parameters
    ----------
    text
        Predicted subtask phrase. Stripped of leading / trailing whitespace
        and lowercased before being used as the dictionary key.
    name_to_id
        Running ``text -> id`` mapping. Mutated in place when ``text`` is new.
    id_to_name
        Running ``id -> text`` mapping. Mutated in place when ``text`` is new.

    Returns
    -------
    int
        Stable class id for ``text``.
    """
    key = (text or "").strip()
    if not key:
        return _BACKGROUND_ID
    norm = key.lower()
    if norm in name_to_id:
        return name_to_id[norm]
    new_id = max(id_to_name) + 1 if id_to_name else 0
    name_to_id[norm] = new_id
    # Preserve the *original* (non-lowercased) phrase for human-readable
    # ``meta/action_text.json``; we only lowercase for dedup, not display.
    id_to_name[new_id] = key
    return new_id


# --------------------------------------------------------------------------- #
# LeRobot metadata helpers
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict | None:
    """Read a JSON file; return ``None`` if the file is missing."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; return ``[]`` when the file is missing."""
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                out.append(json.loads(stripped))
    return out


def _chunk_idx(ep_idx: int, info: dict) -> int:
    """Translate an episode index to its containing chunk number."""
    return ep_idx // int(info.get("chunks_size", 1000))


def _resolve_video_path(
    root: Path, info: dict, ep_idx: int, camera_key: str
) -> Path:
    """Resolve the mp4 path for ``(ep_idx, camera_key)`` under ``root``.

    Uses the ``video_path`` template from ``info.json`` when present, falling
    back to the LeRobot v2.1 default layout.
    """
    chunk = _chunk_idx(ep_idx, info)
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return root / template.format(
        episode_chunk=chunk,
        video_key=camera_key,
        episode_index=ep_idx,
    )


def _resolve_parquet_path(root: Path, info: dict, ep_idx: int) -> Path:
    """Resolve the parquet path for episode ``ep_idx`` under ``root``."""
    chunk = _chunk_idx(ep_idx, info)
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return root / template.format(
        episode_chunk=chunk,
        episode_index=ep_idx,
    )


def _list_camera_keys(root: Path, info: dict, ep_idx: int) -> list[str]:
    """Return every camera subdirectory present for ``ep_idx`` under ``root``.

    A LeRobot dataset stores one mp4 per camera per episode; we hard-link
    *all* of them into the output (not just the driving camera) so the output
    dataset is complete and self-contained.
    """
    chunk = _chunk_idx(ep_idx, info)
    videos_dir = root / "videos" / f"chunk-{chunk:03d}"
    if not videos_dir.exists():
        return []
    return sorted(d.name for d in videos_dir.iterdir() if d.is_dir())


def _extract_metadata(ckpt: dict, checkpoint_path: Path) -> dict:
    """Pull boundary checkpoint metadata, falling back to a sidecar.

    Mirrors the convention used by ``scripts/train_boundary.py`` and
    ``scripts/inference_lerobot.py``: checkpoints may carry their
    ``model_config`` / ``feature_extraction`` directly, nested under
    ``config_snapshot``, or in a ``metadata.json`` next to the ``.pt``.
    """
    required = ("model_config",)
    if all(k in ckpt for k in required):
        return ckpt
    cs = ckpt.get("config_snapshot")
    if isinstance(cs, dict) and all(k in cs for k in required):
        return cs
    sidecar = checkpoint_path.parent / "metadata.json"
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise RuntimeError(
        f"Boundary checkpoint {checkpoint_path} has no embedded metadata "
        f"(model_config). Pair it with a metadata.json sidecar in the same "
        f"directory, or use the final.pt produced by scripts/train_boundary.py."
    )


def _resolve_task_text(
    ep: dict,
    tasks: list[dict],
) -> str:
    """Recover the episode's task instruction from LeRobot metadata.

    LeRobot stores task references on the episode as either a list of
    free-form strings (``"tasks"``) or as integer ids into ``tasks.jsonl``
    (``"tasks"`` containing ints, or a dedicated ``"task_index"`` field).
    We try, in order:

    1. ``ep["tasks"]`` as a list of strings — return the first.
    2. ``ep["tasks"]`` as a list of ints — look the first up in
       ``tasks.jsonl``.
    3. ``ep["task_index"]`` — look it up in ``tasks.jsonl``.
    4. ``ep["task"]`` as a plain string.

    Returns the empty string when no task can be recovered; the caller is
    expected to decide whether to fall back to VLM-based task prediction.
    """
    tasks_field = ep.get("tasks")
    if isinstance(tasks_field, list) and tasks_field:
        first = tasks_field[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (int, np.integer)):
            for entry in tasks:
                if int(entry.get("task_index", -1)) == int(first):
                    return str(entry.get("task", ""))
    task_idx = ep.get("task_index")
    if isinstance(task_idx, (int, np.integer)):
        for entry in tasks:
            if int(entry.get("task_index", -1)) == int(task_idx):
                return str(entry.get("task", ""))
    task_field = ep.get("task")
    if isinstance(task_field, str):
        return task_field
    return ""


def _per_frame_label_array(
    segments: list[tuple[int, int]],
    subtask_texts: list[str],
    total_frames: int,
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> np.ndarray:
    """Expand ``(segment, subtask_text)`` pairs into a per-frame label array.

    Frames outside any segment (gaps before the first / after the last /
    between segments) inherit the reserved ``background`` class id 0. Each
    new subtask text discovered is appended to the running output vocabulary
    via :func:`_ensure_text_id`.

    Parameters
    ----------
    segments
        Half-open ``(start_frame, end_frame)`` intervals in original-frame
        coordinates.
    subtask_texts
        VLM-predicted phrases, one per segment.
    total_frames
        Length of the per-frame array to materialise (matches the parquet
        frame count).
    name_to_id, id_to_name
        Running output vocabulary mappings.

    Returns
    -------
    np.ndarray
        Int64 array of shape ``[total_frames]`` carrying class ids in
        ``[0, len(id_to_name))``.
    """
    labels = np.zeros(int(total_frames), dtype=np.int64)
    if total_frames <= 0:
        return labels
    for (start, end), text in zip(segments, subtask_texts):
        lo = max(0, int(start))
        hi = min(int(total_frames), int(end))
        if hi <= lo:
            continue
        cls_id = _ensure_text_id(text, name_to_id, id_to_name)
        labels[lo:hi] = cls_id
    return labels


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_inference_pipeline(
    input_root: Path,
    output_root: Path,
    *,
    boundary_checkpoint: Path,
    vlm_checkpoint: Path | None,
    camera_key: str,
    device: str = "cuda",
    boundary_threshold: float = 0.5,
    min_distance: int = 3,
    min_segment_length: int = 5,
    prominence: float = 0.1,
    n_frames_per_segment: int = 8,
    predict_task_if_missing: bool = True,
    limit: int | None = None,
) -> dict:
    """Run the full two-stage inference pipeline on a LeRobot dataset.

    The input dataset at ``input_root`` is read-only. A fresh LeRobot v2.1
    dataset is materialised at ``output_root`` with:

    - Hard-linked (or symlinked, on cross-FS) per-camera mp4s.
    - Per-episode parquet files with predicted ``action_text_id`` columns.
    - ``meta/action_text.json`` built incrementally from VLM outputs.
    - ``meta/episodes.jsonl`` rewritten with derived ``action_config``.
    - ``meta/info.json`` / ``tasks.jsonl`` / ``episodes_stats.jsonl`` copied
      with ``total_episodes`` / ``total_frames`` updated to reflect the
      number actually processed.

    Parameters
    ----------
    input_root
        Root of the input LeRobot dataset.
    output_root
        Destination directory for the new LeRobot dataset. Created on demand.
    boundary_checkpoint
        Path to the Stage-1 boundary segmenter ``.pt`` (as produced by
        ``scripts/train_boundary.py``). Must carry ``model_config`` and
        ``feature_extraction`` keys, either directly or via a
        ``metadata.json`` sidecar.
    vlm_checkpoint
        Directory containing a PEFT-LoRA adapter for the Stage-2 VLM. When
        ``None``, the base Qwen2-VL model is used zero-shot (no adapter).
    camera_key
        Camera stream used to drive predictions (e.g.
        ``"observation.images.head_left_eye"``). All camera mp4s present in
        the input are still hard-linked into the output.
    device
        Torch device string for both stages. Falls back to ``"cpu"`` if
        ``"cuda"`` is requested but unavailable.
    boundary_threshold, min_distance, min_segment_length, prominence
        Post-processing knobs for
        :func:`robosubtasknet.boundary.postprocess.segment_video_from_probs`.
    n_frames_per_segment
        Number of frames the Stage-2 labeler samples from each segment.
    predict_task_if_missing
        When ``True`` and the episode's task instruction is unavailable from
        LeRobot metadata, query the VLM for a one-shot task description
        before labeling segments.
    limit
        If set, process only the first ``limit`` episodes (useful for
        smoke tests).

    Returns
    -------
    dict
        Summary statistics::

            {
                "n_episodes_input": int,
                "n_episodes_processed": int,
                "n_episodes_skipped": int,
                "n_segments_total": int,
                "n_videos_hardlinked": int,
                "n_videos_symlinked": int,
                "n_videos_skipped": int,
                "action_text": dict[int, str],
                "per_episode": [
                    {
                        "episode_index": int,
                        "n_segments": int,
                        "n_frames": int,
                        "task_text": str,
                        "subtask_texts": list[str],
                    },
                    ...
                ],
                "wall_time_sec": float,
            }
    """
    # Deferred heavy imports keep ``import robosubtasknet.pipeline`` cheap.
    import torch

    from robosubtasknet.boundary import BoundarySegmenter
    from robosubtasknet.boundary.postprocess import segment_video_from_probs
    from robosubtasknet.data.lerobot_writer import (
        hardlink_or_fallback,
        segments_from_labels,
        write_episode_parquet,
        write_meta_files,
    )
    from robosubtasknet.features.extract import extract_features_for_video
    from robosubtasknet.vlm import SegmentLabelerVLM
    from robosubtasknet.vlm.inference import label_segments, predict_task_text

    input_root = Path(input_root)
    output_root = Path(output_root)
    boundary_checkpoint = Path(boundary_checkpoint)
    vlm_checkpoint_path = (
        Path(vlm_checkpoint) if vlm_checkpoint is not None else None
    )

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")
    if not boundary_checkpoint.exists():
        raise FileNotFoundError(
            f"Boundary checkpoint not found: {boundary_checkpoint}"
        )

    # ----- Resolve device --------------------------------------------------- #
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    torch_device = torch.device(device)

    # ----- Stage 1: boundary segmenter ------------------------------------- #
    logger.info("Loading boundary checkpoint from %s", boundary_checkpoint)
    ckpt = torch.load(
        boundary_checkpoint, map_location=torch_device, weights_only=False
    )
    metadata = _extract_metadata(ckpt, boundary_checkpoint)
    model_config: dict[str, Any] = dict(metadata["model_config"])
    feat_cfg: dict[str, Any] = dict(
        metadata.get(
            "feature_extraction",
            {"window": 16, "stride": 8, "flow_method": "tvl1"},
        )
    )

    boundary = BoundarySegmenter(**model_config).to(torch_device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    boundary.load_state_dict(state_dict, strict=False)
    boundary.eval()

    # ----- Stage 2: VLM labeler -------------------------------------------- #
    logger.info(
        "Loading VLM%s",
        f" with LoRA adapter at {vlm_checkpoint_path}"
        if vlm_checkpoint_path is not None
        else " (zero-shot, no LoRA)",
    )
    vlm = SegmentLabelerVLM.from_pretrained(
        lora_path=vlm_checkpoint_path,
        device=device,
    )

    # ----- Read input LeRobot metadata ------------------------------------- #
    meta_dir = input_root / "meta"
    info = _load_json(meta_dir / "info.json")
    if info is None:
        raise FileNotFoundError(f"Missing {meta_dir / 'info.json'}")
    tasks = _load_jsonl(meta_dir / "tasks.jsonl")
    input_episodes = _load_jsonl(meta_dir / "episodes.jsonl")
    input_stats = _load_jsonl(meta_dir / "episodes_stats.jsonl")

    if limit is not None:
        input_episodes = input_episodes[: int(limit)]

    if input_episodes:
        sample_ep = int(input_episodes[0]["episode_index"])
        present = _list_camera_keys(input_root, info, sample_ep)
        if present and camera_key not in present:
            raise ValueError(
                f"camera_key={camera_key!r} not present in input dataset; "
                f"available cameras for episode {sample_ep}: {present}"
            )

    output_root.mkdir(parents=True, exist_ok=True)

    # ----- Output vocabulary (built incrementally) ------------------------- #
    name_to_id, id_to_name = _build_output_vocab()

    # ----- Per-episode processing ----------------------------------------- #
    output_episodes: list[dict] = []
    per_ep_summary: list[dict[str, Any]] = []
    n_hardlink = n_symlink = n_skipped_video = 0
    n_processed = 0
    n_skipped_episodes = 0
    n_segments_total = 0
    start_t = time.time()

    total = len(input_episodes)
    for idx, ep in enumerate(input_episodes):
        ep_idx = int(ep["episode_index"])
        chunk = _chunk_idx(ep_idx, info)

        video_path = _resolve_video_path(input_root, info, ep_idx, camera_key)
        parquet_path = _resolve_parquet_path(input_root, info, ep_idx)

        if not video_path.exists():
            logger.warning("ep %d: missing video %s -- skipping", ep_idx, video_path)
            n_skipped_episodes += 1
            continue
        if not parquet_path.exists():
            logger.warning(
                "ep %d: missing parquet %s -- skipping", ep_idx, parquet_path
            )
            n_skipped_episodes += 1
            continue

        logger.info("(%d/%d) processing episode %d", idx + 1, total, ep_idx)

        # --- (b) Feature extraction --------------------------------------- #
        try:
            features = extract_features_for_video(
                video_path,
                modality="both",
                flow_method=str(feat_cfg.get("flow_method", "tvl1")),
                window=int(feat_cfg.get("window", 16)),
                stride=int(feat_cfg.get("stride", 8)),
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ep %d: feature extraction failed: %s", ep_idx, exc)
            n_skipped_episodes += 1
            continue

        meta_info = features["meta"]
        if meta_info["T_feat"] == 0:
            logger.warning("ep %d: video too short for I3D window -- skipping", ep_idx)
            n_skipped_episodes += 1
            continue

        rgb_np = features["rgb"]
        flow_np = features["flow"]
        if rgb_np is None or flow_np is None:
            logger.warning(
                "ep %d: missing RGB or flow features -- skipping", ep_idx
            )
            n_skipped_episodes += 1
            continue

        # --- (c) Boundary probabilities + segment post-processing -------- #
        rgb_t = (
            torch.from_numpy(rgb_np).float().unsqueeze(0).to(torch_device)
        )  # [1, T_feat, D]
        flow_t = (
            torch.from_numpy(flow_np).float().unsqueeze(0).to(torch_device)
        )

        with torch.no_grad():
            probs = boundary.predict_probs(rgb_t, flow_t)  # [1, T_feat]
        probs_np = probs.detach().cpu().numpy()[0]

        original_T = int(meta_info["original_T"])
        stride = int(feat_cfg.get("stride", 8))
        # --- (d) Segments in original-frame coordinates ------------------ #
        segments = segment_video_from_probs(
            probs_np,
            original_T,
            stride=stride,
            threshold=float(boundary_threshold),
            min_distance=int(min_distance),
            min_segment_length=int(min_segment_length),
            prominence=float(prominence),
        )

        # --- (e) Task text (or VLM-predicted fallback) ------------------- #
        task_text = _resolve_task_text(ep, tasks)
        if not task_text and predict_task_if_missing:
            try:
                task_text = predict_task_text(
                    vlm, video_path, n_frames=int(n_frames_per_segment)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ep %d: task prediction failed (%s); using empty task text",
                    ep_idx,
                    exc,
                )
                task_text = ""

        # --- (f) Per-segment VLM labels --------------------------------- #
        try:
            subtask_texts = label_segments(
                vlm,
                video_path,
                segments,
                task_text,
                n_frames=int(n_frames_per_segment),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ep %d: segment labeling failed: %s", ep_idx, exc)
            n_skipped_episodes += 1
            continue

        if len(subtask_texts) != len(segments):
            # Defensive: pad / truncate so downstream code never indexes OOB.
            logger.warning(
                "ep %d: VLM returned %d labels for %d segments; aligning.",
                ep_idx,
                len(subtask_texts),
                len(segments),
            )
            if len(subtask_texts) < len(segments):
                subtask_texts = list(subtask_texts) + [""] * (
                    len(segments) - len(subtask_texts)
                )
            else:
                subtask_texts = list(subtask_texts)[: len(segments)]

        # --- (g) Per-frame action_text_id array -------------------------- #
        labels_per_frame = _per_frame_label_array(
            segments,
            subtask_texts,
            total_frames=original_T,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
        )

        # --- (h) Write per-episode parquet ------------------------------- #
        output_parquet = (
            output_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{ep_idx:06d}.parquet"
        )
        try:
            write_episode_parquet(output_parquet, parquet_path, labels_per_frame)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ep %d: parquet write failed: %s", ep_idx, exc)
            n_skipped_episodes += 1
            continue

        # --- (i) Hard-link mp4s for every camera ------------------------ #
        for cam in _list_camera_keys(input_root, info, ep_idx):
            src = _resolve_video_path(input_root, info, ep_idx, cam)
            if not src.exists():
                continue
            dst = (
                output_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / cam
                / f"episode_{ep_idx:06d}.mp4"
            )
            kind = hardlink_or_fallback(src, dst)
            if kind == "hardlink":
                n_hardlink += 1
            elif kind == "symlink":
                n_symlink += 1
            else:
                n_skipped_video += 1

        # --- (j) Build the episode meta with derived action_config ------- #
        ep_segments = segments_from_labels(labels_per_frame, id_to_name)
        new_ep = dict(ep)
        new_ep["action_config"] = ep_segments
        new_ep["length"] = int(original_T)
        output_episodes.append(new_ep)

        per_ep_summary.append(
            {
                "episode_index": ep_idx,
                "n_segments": int(len(segments)),
                "n_frames": int(original_T),
                "task_text": task_text,
                "subtask_texts": list(subtask_texts),
            }
        )
        n_segments_total += len(segments)
        n_processed += 1

    # ----- Finalise meta files -------------------------------------------- #
    output_info = dict(info)
    output_info["total_episodes"] = len(output_episodes)
    output_info["total_frames"] = int(
        sum(ep.get("length", 0) for ep in output_episodes)
    )
    output_info["splits"] = {"train": f"0:{len(output_episodes)}"}

    write_meta_files(
        output_root,
        info=output_info,
        tasks=tasks,
        episodes=output_episodes,
        episodes_stats=input_stats,
        action_text=id_to_name,
    )

    wall_time = time.time() - start_t
    logger.info(
        "Pipeline complete: %d/%d episodes, %d segments, %.1fs wall-clock.",
        n_processed,
        total,
        n_segments_total,
        wall_time,
    )

    return {
        "n_episodes_input": int(total),
        "n_episodes_processed": int(n_processed),
        "n_episodes_skipped": int(n_skipped_episodes),
        "n_segments_total": int(n_segments_total),
        "n_videos_hardlinked": int(n_hardlink),
        "n_videos_symlinked": int(n_symlink),
        "n_videos_skipped": int(n_skipped_video),
        "action_text": dict(id_to_name),
        "per_episode": per_ep_summary,
        "wall_time_sec": float(wall_time),
    }
