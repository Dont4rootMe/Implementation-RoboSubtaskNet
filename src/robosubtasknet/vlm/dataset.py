"""Stage-2 VLM fine-tuning dataset.

This module produces ``(frames, task_text, subtask_text)`` triplets from one or
more LeRobot v2.1 datasets so that a vision-language model (Qwen2-VL with LoRA
in our setup) can be fine-tuned to caption short video segments with the
sub-task they correspond to.

Pipeline (per dataset root)
---------------------------
1. Read ``meta/episodes.jsonl`` and ``meta/info.json``.
2. For each episode:
   * Extract the *task text* (the top-level natural-language instruction)
     from ``episode["tasks"][0]``. Episodes without ``tasks`` are skipped —
     there is no usable supervision target for the task field.
   * Extract the per-segment *subtask text* from one of two sources:

     - **Preferred:** ``episode["action_config"]``, an explicit list of
       ``{"start_frame", "end_frame", "english_action_text",
       "action_text_id"}`` entries with half-open intervals — the format
       written by :func:`robosubtasknet.data.lerobot_writer.segments_from_labels`
       and by the upstream LeRobot converter scripts.
     - **Fallback:** when ``action_config`` is absent, the per-frame
       ``action_text_id`` column of the episode's parquet is RLE-encoded
       into the same shape, resolving names via ``meta/action_text.json``
       (LeRobot stores keys as strings; we coerce to int).
3. Filter segments:
   * length ``< 4`` frames (too short to sample ``n_frames`` distinct
     positions meaningfully, and uninformative for a VLM caption);
   * label equal to ``"background"`` (no useful supervision — this matches
     the "background" class-0 convention used elsewhere in the codebase).

Each surviving segment becomes one training record. ``__getitem__`` uses
``decord.VideoReader`` (lazy import, GPU/CPU agnostic) to read ``n_frames``
RGB frames uniformly spaced inside ``[start_frame, end_frame)``, resizes them
to ``frame_size`` via OpenCV, and returns a dict the collator (a sibling
module not in this agent's scope) can pack into Qwen2-VL processor inputs.

Why a flat list of (root, episode, segment) records?
----------------------------------------------------
A single LeRobot episode typically contains 5-30 sub-task segments; sampling
*by episode* would bias the data loader toward long episodes and toward
whichever sub-task happens to come first. Flattening to segment-level records
makes :class:`torch.utils.data.RandomSampler` give the fine-tune uniform
exposure across sub-task classes (modulo natural frequency), and lets us
honor ``max_segments_per_episode`` to keep extreme outliers from dominating.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "VLMSegmentDataset",
    "discover_segments",
    "derive_segments_from_action_text_id",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


#: Minimum segment length (in original-video frames) to keep. Segments shorter
#: than this cannot host ``n_frames=8`` meaningfully and are noise for a VLM
#: caption objective.
MIN_SEGMENT_FRAMES: int = 4

#: Label string that means "no useful supervision" (matches the class-0
#: convention in :mod:`scripts.extract_features_lerobot.build_union_label_map`).
BACKGROUND_LABEL: str = "background"


# --------------------------------------------------------------------------- #
# LeRobot metadata I/O (kept local: avoids pulling pandas/scripts into the
# import graph of an `import robosubtasknet.vlm` from a fine-tune script).
# --------------------------------------------------------------------------- #


def _load_episodes_jsonl(root: Path) -> list[dict[str, Any]]:
    """Read ``<root>/meta/episodes.jsonl`` into a list of dicts.

    Raises :class:`FileNotFoundError` if the file is missing — the caller is
    expected to validate dataset roots before constructing the dataset.
    """
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot episode metadata: {path}")
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def _load_info(root: Path) -> dict[str, Any]:
    """Read ``<root>/meta/info.json``.

    ``info.json`` carries the path templates we use to locate the per-episode
    ``.mp4`` and ``.parquet`` files. Missing file → ``FileNotFoundError``.
    """
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_action_text(root: Path) -> dict[int, str]:
    """Read ``<root>/meta/action_text.json`` → ``{int_id: name}``.

    Returns an empty dict if the file is absent — some LeRobot exports use
    only ``action_config`` and have no separate text table.
    """
    path = root / "meta" / "action_text.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def _resolve_template(
    template: str, episode_index: int, chunk_size: int, **extra: Any
) -> str:
    """Format a LeRobot ``info.json`` path template.

    LeRobot stores videos and parquets under chunked directories; the chunk
    index is ``episode_index // chunk_size``. Templates use
    ``{episode_chunk:03d}``, ``{episode_index:06d}``, ``{video_key}``.
    """
    chunk_idx = episode_index // max(int(chunk_size), 1)
    try:
        return template.format(
            episode_chunk=chunk_idx,
            episode_index=episode_index,
            **extra,
        )
    except KeyError as exc:
        raise KeyError(
            f"Path template {template!r} requires {exc.args[0]!r}; "
            f"got episode_index={episode_index}, chunk_size={chunk_size}, "
            f"extra={extra}."
        ) from exc


def _video_path(
    root: Path, info: dict[str, Any], episode_index: int, camera_key: str
) -> Path:
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4",
    )
    rel = _resolve_template(
        template,
        episode_index=episode_index,
        chunk_size=int(info.get("chunks_size", 1000)),
        video_key=camera_key,
    )
    return root / rel


def _parquet_path(
    root: Path, info: dict[str, Any], episode_index: int
) -> Path:
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = _resolve_template(
        template,
        episode_index=episode_index,
        chunk_size=int(info.get("chunks_size", 1000)),
    )
    return root / rel


# --------------------------------------------------------------------------- #
# Segment derivation
# --------------------------------------------------------------------------- #


def derive_segments_from_action_text_id(
    parquet_path: Path,
    action_text: dict[int, str],
) -> list[dict[str, Any]]:
    """Run-length encode the parquet's ``action_text_id`` column into segments.

    Used when an episode lacks an explicit ``action_config`` block in
    ``episodes.jsonl``. The output shape matches the ``action_config`` schema
    so downstream code can be agnostic about the source.

    Parameters
    ----------
    parquet_path
        Path to the episode parquet (LeRobot v2.1 layout). Must contain an
        ``action_text_id`` column; if absent, an empty list is returned (the
        caller treats that as "skip this episode").
    action_text
        ``{int_id: english_name}`` from ``meta/action_text.json``. Used to fill
        ``english_action_text`` for each segment. Missing IDs fall back to a
        ``"class_{id}"`` placeholder, matching
        :func:`robosubtasknet.data.lerobot_writer.segments_from_labels`.

    Returns
    -------
    list[dict]
        One dict per run, with keys ``start_frame``, ``end_frame``
        (half-open), ``english_action_text``, ``action_text_id``. Empty list
        if the parquet is empty or has no ``action_text_id`` column.
    """
    import pandas as pd  # local import: pandas is heavy.

    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        return []

    df = pd.read_parquet(parquet_path, columns=None)
    if "action_text_id" not in df.columns or len(df) == 0:
        return []

    labels = np.asarray(df["action_text_id"].values, dtype=np.int64)
    segments: list[dict[str, Any]] = []
    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        v = int(labels[i])
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
            "end_frame": int(len(labels)),
            "english_action_text": action_text.get(current, f"class_{current}"),
            "action_text_id": int(current),
        }
    )
    return segments


def discover_segments(root: Path, camera: str) -> list[dict[str, Any]]:
    """Enumerate VLM training records for one LeRobot dataset root.

    Walks every episode in ``meta/episodes.jsonl`` and emits one record per
    eligible segment. Segments are sourced from ``episode["action_config"]``
    when present; otherwise from the parquet's per-frame ``action_text_id``
    via :func:`derive_segments_from_action_text_id`.

    Filters (in this order, so reporting/debugging is easy to reason about):

    1. Drop episodes whose ``tasks`` field is missing or empty — no task text.
    2. Drop segments whose ``end_frame - start_frame < MIN_SEGMENT_FRAMES``.
    3. Drop segments whose ``english_action_text == "background"``.

    Parameters
    ----------
    root
        LeRobot dataset root (the directory containing ``meta/``).
    camera
        Camera key to associate with each record. We don't open videos here —
        but we *do* validate the camera template resolves to a real path on
        disk for at least one episode, so a typo'd camera surfaces early
        instead of at the first ``__getitem__`` call.

    Returns
    -------
    list[dict]
        Each dict has::

            {
                "dataset_root": Path,
                "camera_key": str,
                "episode_index": int,
                "video_path": Path,
                "parquet_path": Path,
                "start_frame": int,
                "end_frame": int,        # half-open
                "subtask_text": str,
                "task_text": str,
                "action_text_id": int,
            }
    """
    root = Path(root)
    info = _load_info(root)
    episodes = _load_episodes_jsonl(root)
    action_text = _load_action_text(root)

    records: list[dict[str, Any]] = []
    for ep in episodes:
        if "episode_index" not in ep:
            # Defensive: LeRobot v2.1 always sets this; bail rather than guess.
            continue
        ep_idx = int(ep["episode_index"])

        # --- task text ---------------------------------------------------- #
        tasks = ep.get("tasks") or []
        if not tasks:
            # Skip the whole episode: no usable instruction-text supervision.
            continue
        task_text = str(tasks[0]).strip()
        if not task_text:
            continue

        # --- segments ----------------------------------------------------- #
        segments = ep.get("action_config")
        if not segments:
            # Fallback path: RLE the parquet's action_text_id column.
            parquet_path = _parquet_path(root, info, ep_idx)
            segments = derive_segments_from_action_text_id(
                parquet_path, action_text
            )
        if not segments:
            continue

        video_path = _video_path(root, info, ep_idx, camera)
        parquet_path = _parquet_path(root, info, ep_idx)

        for seg in segments:
            start = int(seg["start_frame"])
            end = int(seg["end_frame"])
            if end - start < MIN_SEGMENT_FRAMES:
                continue
            subtask_text = str(
                seg.get("english_action_text", "")
            ).strip()
            if not subtask_text or subtask_text == BACKGROUND_LABEL:
                continue
            records.append(
                {
                    "dataset_root": root,
                    "camera_key": camera,
                    "episode_index": ep_idx,
                    "video_path": video_path,
                    "parquet_path": parquet_path,
                    "start_frame": start,
                    "end_frame": end,
                    "subtask_text": subtask_text,
                    "task_text": task_text,
                    "action_text_id": int(seg.get("action_text_id", -1)),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Frame sampling helpers
# --------------------------------------------------------------------------- #


def _uniform_frame_indices(
    start: int, end: int, n_frames: int, rng: random.Random
) -> list[int]:
    """Pick ``n_frames`` indices uniformly inside ``[start, end)``.

    Strategy: divide the half-open interval into ``n_frames`` equal bins and
    pick one index per bin. With a deterministic RNG seeded per-record we get
    reproducible mini-jitter without ever collapsing to a single frame on
    long segments (which a strict linspace would do for tiny n_frames).

    For very short segments (``end - start <= n_frames``) we just take a
    linearly spaced sample with replacement allowed at the boundaries; this
    is only reachable when the caller bypasses the ``MIN_SEGMENT_FRAMES``
    filter.
    """
    length = end - start
    if length <= 0:
        # Defensive: degenerate segment; repeat ``start`` so downstream code
        # still gets ``n_frames`` items rather than an empty list.
        return [start] * n_frames
    if length <= n_frames:
        # linspace, clipped to [start, end-1]; some frames may repeat.
        idx = np.linspace(start, end - 1, num=n_frames).round().astype(np.int64)
        return [int(i) for i in idx]

    # Per-bin random pick. Bin edges are spaced linearly across the segment.
    bins = np.linspace(start, end, num=n_frames + 1).astype(np.int64)
    out: list[int] = []
    for i in range(n_frames):
        lo = int(bins[i])
        hi = int(bins[i + 1])
        if hi <= lo:
            out.append(lo)
        else:
            out.append(rng.randrange(lo, hi))
    return out


def _resize_rgb(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB array to ``frame_size`` (H, W) via OpenCV.

    OpenCV is the dependency the codebase already uses (see requirements.txt);
    we import lazily so importing this module from non-training contexts (e.g.
    docs / tests that mock the dataset) doesn't pay for the cv2 import.
    """
    import cv2  # local import — see docstring.

    H, W = int(frame_size[0]), int(frame_size[1])
    if frame.shape[0] == H and frame.shape[1] == W:
        return frame
    # cv2.resize takes (width, height) — note the order.
    resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    # cv2.resize on an HxWx3 uint8 RGB array preserves the channel order; it
    # treats channels as opaque. We never go through cv2.imread/imwrite here,
    # so no BGR↔RGB swap is needed at this step.
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8, copy=False)
    return np.ascontiguousarray(resized)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class VLMSegmentDataset(torch.utils.data.Dataset):
    """Yields ``{frames, task_text, subtask_text}`` triplets for VLM LoRA
    fine-tuning.

    Each ``__getitem__`` returns one segment: ``n_frames`` RGB frames decoded
    from the chosen camera's ``.mp4`` between ``start_frame`` (inclusive) and
    ``end_frame`` (exclusive), the top-level episode instruction
    (``task_text``), and the sub-task name (``subtask_text``).

    Parameters
    ----------
    inputs
        List of ``(lerobot_root, camera_key)`` pairs. Multiple roots are
        flattened into a single index space; ``dataset_idx`` in the returned
        sample tells the caller which input the record came from.
    n_frames
        Number of frames to sample per segment (default 8 — Qwen2-VL's
        nominal video budget without aggressive token compression).
    frame_size
        ``(H, W)`` to resize frames to (default ``(224, 224)``).
    max_segments_per_episode
        Optional cap on the number of segments kept per episode. When a
        single episode has dozens of sub-task slots (e.g. long pick-and-place
        sequences), uncapped sampling can let one episode dominate a mini-
        batch. We drop the tail beyond this many segments per episode (after
        the standard filters). ``None`` means no cap.
    seed
        Base seed for deterministic per-record frame sampling. Each record's
        sampling RNG is seeded with ``hash((seed, dataset_idx, episode_index,
        start_frame))``, so a given record yields the same frame indices
        across epochs / DataLoader workers — useful for caching and for
        making validation losses reproducible.

    Notes
    -----
    * Heavy dependencies (decord, OpenCV, pandas) are imported lazily on
      first use so that ``import robosubtasknet.vlm.dataset`` itself stays
      cheap (matters for CLI ``--help`` latency and test collection).
    * Frames are returned as a list (not stacked into a ``(T, H, W, 3)``
      ndarray) because Qwen2-VL's processor expects an iterable of PIL/np
      images, and stacking + unstacking would just churn memory.
    """

    def __init__(
        self,
        inputs: list[tuple[Path, str]],
        n_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        max_segments_per_episode: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not inputs:
            raise ValueError(
                "VLMSegmentDataset requires at least one (root, camera) pair."
            )
        if int(n_frames) < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}.")
        if (
            len(frame_size) != 2
            or int(frame_size[0]) < 1
            or int(frame_size[1]) < 1
        ):
            raise ValueError(
                f"frame_size must be a (H, W) pair of positive ints, "
                f"got {frame_size!r}."
            )

        self.inputs: list[tuple[Path, str]] = [
            (Path(root), str(cam)) for root, cam in inputs
        ]
        self.n_frames: int = int(n_frames)
        self.frame_size: tuple[int, int] = (
            int(frame_size[0]),
            int(frame_size[1]),
        )
        self.max_segments_per_episode: int | None = (
            int(max_segments_per_episode)
            if max_segments_per_episode is not None
            else None
        )
        self.seed: int = int(seed)

        # ------------------------------------------------------------------ #
        # Discovery (eager: cheap-enough O(#episodes) and front-loads errors).
        # ------------------------------------------------------------------ #
        self.records: list[dict[str, Any]] = []
        for dataset_idx, (root, camera) in enumerate(self.inputs):
            per_dataset = discover_segments(root, camera)
            if self.max_segments_per_episode is not None:
                per_dataset = self._cap_per_episode(
                    per_dataset, self.max_segments_per_episode
                )
            for rec in per_dataset:
                rec["dataset_idx"] = dataset_idx
                self.records.append(rec)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cap_per_episode(
        records: list[dict[str, Any]], cap: int
    ) -> list[dict[str, Any]]:
        """Keep at most ``cap`` segments per ``episode_index``.

        Order is preserved (sub-tasks within an episode are temporally
        meaningful and the caller may rely on it for debugging). We drop the
        *tail* rather than shuffling-and-truncating because shuffling here
        would break determinism for a fixed ``seed`` and conflict with the
        DataLoader's own sampler.
        """
        seen: dict[int, int] = {}
        out: list[dict[str, Any]] = []
        for rec in records:
            ep = int(rec["episode_index"])
            n = seen.get(ep, 0)
            if n >= cap:
                continue
            seen[ep] = n + 1
            out.append(rec)
        return out

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.records):
            raise IndexError(
                f"VLMSegmentDataset index {idx} out of range "
                f"(len={len(self.records)})."
            )
        rec = self.records[idx]

        # Deterministic per-record RNG. Hashing into Python's int is fine here
        # because we only ever feed it to ``random.Random``; we avoid
        # ``hash(...)`` which is salted per-process and would break repro.
        seed_tuple = (
            int(self.seed),
            int(rec["dataset_idx"]),
            int(rec["episode_index"]),
            int(rec["start_frame"]),
        )
        rng_seed = (
            (seed_tuple[0] & 0xFFFF) << 48
            | (seed_tuple[1] & 0xFF) << 40
            | (seed_tuple[2] & 0xFFFFFF) << 16
            | (seed_tuple[3] & 0xFFFF)
        )
        rng = random.Random(rng_seed)

        indices = _uniform_frame_indices(
            int(rec["start_frame"]),
            int(rec["end_frame"]),
            self.n_frames,
            rng,
        )

        frames = self._read_frames(Path(rec["video_path"]), indices)

        video_id = (
            f"{Path(rec['dataset_root']).name}"
            f"__ep{int(rec['episode_index']):06d}"
            f"__{rec['camera_key']}"
            f"__{int(rec['start_frame']):06d}_{int(rec['end_frame']):06d}"
        )

        return {
            "frames": frames,
            "task_text": str(rec["task_text"]),
            "subtask_text": str(rec["subtask_text"]),
            "video_id": video_id,
            "dataset_idx": int(rec["dataset_idx"]),
        }

    # ------------------------------------------------------------------ #
    # Video I/O
    # ------------------------------------------------------------------ #

    def _read_frames(
        self, video_path: Path, indices: list[int]
    ) -> list[np.ndarray]:
        """Decode the requested frame indices from ``video_path`` and resize.

        Uses ``decord.VideoReader`` for sequential random access; decord
        returns RGB by default (``bridge='native'``) so no BGR↔RGB swap is
        needed. Out-of-range indices are clamped to the last decodable frame
        — preferable to crashing because LeRobot occasionally records a
        parquet that runs a frame or two past the mp4 (encoder rounding).
        """
        import decord  # local import: heavy, GPU/CPU dispatch on first call.

        if not video_path.is_file():
            raise FileNotFoundError(
                f"VLMSegmentDataset: video not found: {video_path}"
            )

        # ``num_threads=1`` keeps DataLoader workers from oversubscribing CPU
        # when n_workers > 1; decord's default is num_cpus() which is wrong
        # under torch DataLoader.
        vr = decord.VideoReader(str(video_path), num_threads=1)
        total = len(vr)
        if total == 0:
            raise RuntimeError(
                f"VLMSegmentDataset: empty video file {video_path}."
            )
        clamped = [max(0, min(int(i), total - 1)) for i in indices]

        # ``get_batch`` returns a decord NDArray that supports .asnumpy().
        # Note: we ask for a single batched read instead of looping with
        # __getitem__ because decord's batched path is the fast one.
        batch = vr.get_batch(clamped)
        arr = batch.asnumpy()  # (T, H, W, 3), uint8, RGB
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"VLMSegmentDataset: unexpected decord output shape "
                f"{arr.shape} from {video_path}."
            )

        return [_resize_rgb(arr[i], self.frame_size) for i in range(arr.shape[0])]
