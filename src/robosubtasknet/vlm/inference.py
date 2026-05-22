"""End-to-end inference helpers for the Stage-2 VLM segment labeler.

This module is the glue between the Stage-1 boundary segmenter and the
Stage-2 :class:`SegmentLabelerVLM` (defined in
:mod:`robosubtasknet.vlm.model`). Given an episode video and a list of
``(start_frame, end_frame)`` segments produced by the boundary detector,
it samples a fixed number of frames per segment, builds the Qwen2-VL chat
payload via :func:`robosubtasknet.vlm.model.make_segment_chat`, and calls
the VLM's ``generate`` method to obtain the predicted subtask text label.

The VLM itself is treated opaquely (``Any``) so this file does not pull
``transformers`` / ``peft`` / ``torch`` into the import-time path; the
``vlm`` object only has to satisfy a small duck-typed contract:

* ``vlm.generate(messages_or_batched_messages, *, max_new_tokens, do_sample)
  -> str | list[str]`` — accepts a single chat (``list[dict]``) or a list
  of chats and returns the decoded assistant turn(s).

This keeps the e2e pipeline thin: callers construct one ``SegmentLabelerVLM``
and pass it through :func:`label_segments` once per episode.

Conventions:
    * Frames are returned as a plain Python ``list`` of ``np.ndarray``
      ``[H, W, 3]`` ``uint8`` RGB images. This matches what
      ``make_segment_chat`` and the Qwen2-VL processor expect.
    * Resizing uses ``cv2.INTER_AREA`` (good default for downsampling,
      acceptable for upsampling) to keep the dependency surface minimal.
    * All heavy imports (``decord``, ``cv2``) are lazy so unit tests that
      do not exercise inference do not need them installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "sample_segment_frames",
    "label_segments",
    "predict_task_text",
]


# --------------------------------------------------------------------------- #
# Lazy import helpers.
# --------------------------------------------------------------------------- #


def _import_decord():
    try:
        import decord  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Sampling video frames requires `decord`. Install with "
            "`pip install decord`."
        ) from exc
    return decord


def _import_cv2():
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Resizing video frames requires OpenCV. Install with "
            "`pip install opencv-python`."
        ) from exc
    return cv2


# --------------------------------------------------------------------------- #
# Internal helpers.
# --------------------------------------------------------------------------- #


def _sanitize_output(text: Any) -> str:
    """Return a clean, non-empty subtask label.

    The Qwen2-VL processor sometimes returns the full chat string including
    leading whitespace, repeated ``assistant`` markers, or trailing EOS
    tokens. We strip these and fall back to ``"unknown"`` when the model
    produced no usable content.
    """
    if text is None:
        return "unknown"
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # pragma: no cover - defensive
            return "unknown"
    text = text.strip()
    # Strip a leading "assistant" role marker if the processor leaked it.
    for marker in ("assistant\n", "assistant:", "ASSISTANT:"):
        if text.lower().startswith(marker.lower()):
            text = text[len(marker):].strip()
    # Strip stray Qwen-style special tokens that occasionally survive decode.
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(tok, "")
    text = text.strip()
    if not text:
        return "unknown"
    return text


def _build_segment_chat(
    frames: list[np.ndarray], task_text: str
) -> list[dict]:
    """Construct the chat payload for one segment.

    Prefer :func:`robosubtasknet.vlm.model.make_segment_chat` when the model
    module is importable; fall back to an in-place builder so this module
    stays usable in isolation (e.g. during unit tests with a mocked VLM).
    """
    try:
        from robosubtasknet.vlm.model import make_segment_chat  # noqa: WPS433
    except ImportError:
        make_segment_chat = None  # type: ignore[assignment]

    if make_segment_chat is not None:
        return make_segment_chat(frames=frames, task_text=task_text)

    # Fallback: minimal Qwen2-VL chat schema. Each frame is one image item
    # so the processor can apply per-frame patch embeddings; the final text
    # item carries the task instruction the model must condition on.
    user_content: list[dict] = [{"type": "image", "image": f} for f in frames]
    user_content.append(
        {
            "type": "text",
            "text": (
                f"The robot is performing: '{task_text}'. "
                "Identify the current subtask in a single short phrase."
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You label short clips of robot demonstrations with the "
                "subtask being executed. Reply with a single concise phrase."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _build_task_summary_chat(frames: list[np.ndarray]) -> list[dict]:
    """Construct the chat payload for whole-video task summarization."""
    user_content: list[dict] = [{"type": "image", "image": f} for f in frames]
    user_content.append(
        {
            "type": "text",
            "text": (
                "You will see frames sampled across an entire robot "
                "demonstration. Describe the overall task in a single short "
                'instruction (e.g., "pick and place the cup", "wipe the '
                'table").'
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You summarize robot demonstrations into a single short "
                "task instruction. Reply with one short imperative phrase."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _generate_one(
    vlm: Any,
    messages: list[dict],
    *,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    """Call ``vlm.generate`` for a single chat with defensive error handling."""
    try:
        out = vlm.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    except TypeError:
        # VLM may not accept kwargs; try positional fallback.
        out = vlm.generate(messages)
    if isinstance(out, (list, tuple)):
        out = out[0] if out else ""
    return _sanitize_output(out)


def _generate_batch(
    vlm: Any,
    batched_messages: list[list[dict]],
    *,
    max_new_tokens: int,
    do_sample: bool,
) -> list[str]:
    """Call ``vlm.generate`` for a batch of chats, falling back per-sample."""
    if not batched_messages:
        return []
    try:
        out = vlm.generate(
            batched_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if isinstance(out, str):
            # VLM returned a single string for a list input: assume length-1.
            out = [out]
        if not isinstance(out, (list, tuple)):
            raise TypeError(
                f"Expected list output from batched generate; got {type(out)!r}."
            )
        if len(out) != len(batched_messages):
            raise RuntimeError(
                "Batched generate returned "
                f"{len(out)} outputs for {len(batched_messages)} inputs."
            )
        return [_sanitize_output(t) for t in out]
    except Exception:
        # Fall back to per-sample inference. Slower but always correct.
        return [
            _generate_one(
                vlm,
                msg,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            for msg in batched_messages
        ]


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def sample_segment_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    n_frames: int = 8,
    frame_size: tuple[int, int] = (224, 224),
) -> list:
    """Uniformly sample ``n_frames`` frames from ``[start_frame, end_frame)``.

    Parameters
    ----------
    video_path : Path
        Source video file (anything ``decord`` can read).
    start_frame : int
        Inclusive lower bound of the sampling window. Clipped to ``[0, T)``.
    end_frame : int
        Exclusive upper bound of the sampling window. Clipped to ``(start, T]``.
    n_frames : int, default 8
        Number of frames to return. Must be ``> 0``.
    frame_size : tuple[int, int], default (224, 224)
        ``(height, width)`` to which each frame is resized via OpenCV.

    Returns
    -------
    list[np.ndarray]
        A list of ``n_frames`` arrays of shape ``[H, W, 3]``, ``uint8``, RGB.

    Notes
    -----
    * If the segment is shorter than ``n_frames`` (including the degenerate
      single-frame case), sampling is done **with replacement** so the
      output length is always exactly ``n_frames``.
    * Out-of-range bounds are clipped; if no usable interval remains after
      clipping the function raises ``ValueError``.
    * ``decord``'s ``native`` bridge is used to return numpy arrays directly,
      avoiding a torch dependency in this module.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive; got {n_frames}.")
    height, width = int(frame_size[0]), int(frame_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"frame_size must be positive (H, W); got {frame_size!r}."
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    decord = _import_decord()
    cv2 = _import_cv2()

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        del vr
        raise RuntimeError(f"Video has zero frames: {video_path}")

    # Clip the window to the video's actual frame range.
    start = max(0, int(start_frame))
    end = min(total, int(end_frame))
    if end <= start:
        # Caller asked for an empty / out-of-range window. Snap to a single
        # frame at the (clipped) start so we still return something usable.
        if start >= total:
            start = total - 1
        end = start + 1

    span = end - start  # >= 1 here

    if span >= n_frames:
        # Even spacing within [start, end). Use endpoint=False so we never
        # land on `end` (which is exclusive); pick midpoints for stability.
        # linspace with endpoint=False on integers via float arithmetic.
        offsets = np.linspace(0, span, num=n_frames, endpoint=False)
        # Center within each bin so we sample the middle of equal partitions.
        bin_width = span / n_frames
        offsets = offsets + bin_width / 2.0
        indices = (start + offsets).astype(np.int64)
        indices = np.clip(indices, start, end - 1)
    else:
        # Span shorter than requested: sample with replacement, evenly
        # spread across the available window.
        # Use linspace on the integer indices then round to ints.
        raw = np.linspace(start, end - 1, num=n_frames)
        indices = np.round(raw).astype(np.int64)
        indices = np.clip(indices, start, end - 1)

    indices_list = [int(i) for i in indices.tolist()]
    try:
        batch = vr.get_batch(indices_list)
    finally:
        del vr  # release file handle promptly

    # Convert decord NDArray (native bridge) to numpy.
    if hasattr(batch, "asnumpy"):
        frames = batch.asnumpy()
    else:  # pragma: no cover - decord version dependent
        frames = np.asarray(batch)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"decord returned unexpected frame shape {frames.shape}; "
            "expected [N, H, W, 3]."
        )
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    # Resize each frame to (height, width). cv2.resize wants (W, H).
    out: list[np.ndarray] = []
    src_h, src_w = frames.shape[1], frames.shape[2]
    if src_h == height and src_w == width:
        for i in range(frames.shape[0]):
            out.append(np.ascontiguousarray(frames[i]))
    else:
        for i in range(frames.shape[0]):
            resized = cv2.resize(
                frames[i], (width, height), interpolation=cv2.INTER_AREA
            )
            out.append(np.ascontiguousarray(resized))
    return out


def label_segments(
    vlm: Any,
    video_path: Path,
    segments: list[tuple[int, int]],
    task_text: str,
    *,
    n_frames: int = 8,
    frame_size: tuple[int, int] = (224, 224),
    max_new_tokens: int = 64,
    do_sample: bool = False,
    batch_size: int = 1,
) -> list[str]:
    """Predict a subtask text label for every ``(start, end)`` segment.

    Parameters
    ----------
    vlm : SegmentLabelerVLM
        Stage-2 VLM wrapper. Must expose a ``generate(messages, *,
        max_new_tokens, do_sample) -> str | list[str]`` method.
    video_path : Path
        Source video for the episode.
    segments : list[tuple[int, int]]
        ``(start_frame, end_frame)`` pairs with ``end_frame`` exclusive.
        May be empty.
    task_text : str
        Episode-level task instruction (e.g. ``"pick and place the cup"``)
        the VLM conditions on when labeling subtasks.
    n_frames : int, default 8
        Frames sampled uniformly per segment.
    frame_size : tuple[int, int], default (224, 224)
        ``(H, W)`` to which each frame is resized.
    max_new_tokens : int, default 64
        Generation budget per segment.
    do_sample : bool, default False
        ``True`` enables sampling; ``False`` is greedy (the default at
        inference for reproducibility).
    batch_size : int, default 1
        Group ``batch_size`` segments per VLM call when ``> 1``. If the VLM
        does not support batched input, falls back to per-segment calls.

    Returns
    -------
    list[str]
        One predicted label per input segment, in the same order. If
        ``video_path`` does not exist this returns ``[]``. If the VLM
        produces empty / garbage output for a segment, that segment's
        label is ``"unknown"``.

    Notes
    -----
    The function intentionally swallows per-segment failures (returning
    ``"unknown"``) so that one bad segment cannot abort labeling for the
    whole episode. Hard failures (missing video file, malformed segment
    list types) still raise.
    """
    if not segments:
        return []

    video_path = Path(video_path)
    if not video_path.exists():
        # Spec: missing video => return empty list, callers handle.
        return []

    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive; got {n_frames}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")

    # ----- Sample frames for every segment up front. ---------------------- #
    # We sample sequentially because each call re-opens the video; doing so
    # at most ``len(segments)`` times is acceptable for typical episode sizes
    # (~tens of segments) and keeps memory bounded.
    per_segment_frames: list[list[np.ndarray] | None] = []
    for seg in segments:
        if not isinstance(seg, (tuple, list)) or len(seg) != 2:
            raise TypeError(
                f"Each segment must be (start, end); got {seg!r}."
            )
        s, e = int(seg[0]), int(seg[1])
        try:
            frames = sample_segment_frames(
                video_path,
                s,
                e,
                n_frames=n_frames,
                frame_size=frame_size,
            )
        except Exception:
            # Defensive: bad indices, codec error, etc. We mark the segment
            # as unsampleable and emit "unknown" later.
            frames = None
        per_segment_frames.append(frames)

    # ----- Build chat payloads for sampleable segments. ------------------- #
    chats: list[list[dict] | None] = []
    for frames in per_segment_frames:
        if frames is None or len(frames) == 0:
            chats.append(None)
        else:
            chats.append(_build_segment_chat(frames, task_text))

    # ----- Run the VLM. --------------------------------------------------- #
    labels: list[str] = ["unknown"] * len(segments)

    # Indices of segments we actually need to run the VLM on.
    runnable: list[int] = [i for i, c in enumerate(chats) if c is not None]
    if not runnable:
        return labels

    if batch_size == 1:
        for i in runnable:
            chat = chats[i]
            assert chat is not None  # for type-checkers
            try:
                labels[i] = _generate_one(
                    vlm,
                    chat,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            except Exception:
                labels[i] = "unknown"
    else:
        for start in range(0, len(runnable), batch_size):
            chunk = runnable[start : start + batch_size]
            batched = [chats[i] for i in chunk]
            # Filter Nones defensively (shouldn't happen given `runnable`).
            batched_nonnull: list[list[dict]] = [
                c for c in batched if c is not None
            ]
            try:
                outs = _generate_batch(
                    vlm,
                    batched_nonnull,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            except Exception:
                outs = ["unknown"] * len(batched_nonnull)
            for j, idx in enumerate(chunk):
                labels[idx] = outs[j] if j < len(outs) else "unknown"

    return labels


def predict_task_text(
    vlm: Any,
    video_path: Path,
    *,
    n_frames: int = 16,
    frame_size: tuple[int, int] = (224, 224),
    max_new_tokens: int = 64,
) -> str:
    """Ask the VLM to summarize an entire video into a task instruction.

    Used at inference time when the input LeRobot dataset lacks an
    episode-level task string. Samples ``n_frames`` uniformly across the
    full video and prompts the VLM for a single short imperative phrase.

    Parameters
    ----------
    vlm : SegmentLabelerVLM
        Stage-2 VLM wrapper (same duck-typed contract as
        :func:`label_segments`).
    video_path : Path
        Source video for the episode.
    n_frames : int, default 16
        Whole-video summary uses a denser sample than per-segment labeling
        (16 vs. 8) because the temporal span is much larger.
    frame_size : tuple[int, int], default (224, 224)
        ``(H, W)`` to which each frame is resized.
    max_new_tokens : int, default 64
        Generation budget for the summary.

    Returns
    -------
    str
        A single short imperative phrase, e.g. ``"pick and place the cup"``.
        Returns ``"unknown"`` if the VLM yields nothing usable, or if the
        video file is missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return "unknown"
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive; got {n_frames}.")

    # Determine the video length once and span the whole thing.
    decord = _import_decord()
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    del vr
    if total <= 0:
        return "unknown"

    try:
        frames = sample_segment_frames(
            video_path,
            0,
            total,
            n_frames=n_frames,
            frame_size=frame_size,
        )
    except Exception:
        return "unknown"

    if not frames:
        return "unknown"

    chat = _build_task_summary_chat(frames)
    try:
        return _generate_one(
            vlm,
            chat,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # task summary is always greedy / deterministic
        )
    except Exception:
        return "unknown"
