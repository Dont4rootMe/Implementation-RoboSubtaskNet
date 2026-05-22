"""Post-processing for Stage 1 boundary segmenter outputs.

Converts per-frame boundary probabilities into discrete segment intervals via
peak detection (NMS / min-distance suppression), minimum-segment-length
filtering, and nearest-neighbor upsampling from the feature stride back to the
original frame rate. Pure-NumPy with a lazy ``scipy.signal`` import.
"""

from __future__ import annotations

import numpy as np


def _find_peaks_fallback(
    probs: np.ndarray,
    min_distance: int,
    prominence: float,
) -> np.ndarray:
    """NumPy-only peak detector used when ``scipy`` is unavailable.

    Picks strict local maxima, enforces a minimum inter-peak distance via
    greedy NMS on descending probability, and applies a crude prominence
    filter (peak value minus the minimum value in a ``min_distance``-radius
    window around the peak).
    """
    T = int(probs.shape[0])
    if T == 0:
        return np.empty(0, dtype=np.int64)

    # Strict local maxima with plateau-safe handling: a frame is a candidate
    # if it is >= both neighbors and strictly greater than at least one.
    candidates: list[int] = []
    for i in range(T):
        left = probs[i - 1] if i - 1 >= 0 else -np.inf
        right = probs[i + 1] if i + 1 < T else -np.inf
        if probs[i] >= left and probs[i] >= right and (probs[i] > left or probs[i] > right):
            candidates.append(i)
    if not candidates:
        return np.empty(0, dtype=np.int64)

    cand = np.asarray(candidates, dtype=np.int64)

    # Prominence filter: peak minus min over a local window.
    if prominence > 0.0:
        radius = max(1, int(min_distance))
        keep_mask = np.zeros(cand.shape[0], dtype=bool)
        for k, idx in enumerate(cand):
            lo = max(0, int(idx) - radius)
            hi = min(T, int(idx) + radius + 1)
            local_min = float(np.min(probs[lo:hi]))
            if float(probs[idx]) - local_min >= prominence:
                keep_mask[k] = True
        cand = cand[keep_mask]
        if cand.size == 0:
            return np.empty(0, dtype=np.int64)

    # Greedy NMS: keep highest, suppress neighbors within ``min_distance``.
    order = np.argsort(-probs[cand], kind="stable")
    suppressed = np.zeros(cand.shape[0], dtype=bool)
    kept: list[int] = []
    for j in order:
        if suppressed[j]:
            continue
        kept.append(int(cand[j]))
        suppressed |= np.abs(cand - cand[j]) < max(1, int(min_distance))
    kept_arr = np.asarray(sorted(kept), dtype=np.int64)
    return kept_arr


def detect_boundaries(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 3,
    prominence: float = 0.1,
) -> np.ndarray:
    """Return sorted array of boundary frame indices.

    ``probs`` is a 1-D array of length ``T`` with values in ``[0, 1]``. Peaks
    are found via ``scipy.signal.find_peaks`` (lazy-imported) with the given
    ``min_distance`` and ``prominence``; if SciPy is unavailable, a pure-NumPy
    fallback is used. The ``threshold`` filters out weak peaks.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return np.empty(0, dtype=np.int64)

    try:  # Lazy import so the module stays usable without SciPy.
        from scipy.signal import find_peaks  # type: ignore[import-not-found]

        peaks, _props = find_peaks(
            probs,
            distance=max(1, int(min_distance)),
            prominence=float(prominence) if prominence > 0.0 else None,
        )
        peaks = np.asarray(peaks, dtype=np.int64)
    except Exception:
        peaks = _find_peaks_fallback(
            probs,
            min_distance=max(1, int(min_distance)),
            prominence=float(prominence),
        )

    if peaks.size == 0:
        return peaks
    mask = probs[peaks] >= float(threshold)
    peaks = peaks[mask]
    peaks.sort()
    return peaks.astype(np.int64, copy=False)


def boundaries_to_segments(
    boundaries: np.ndarray,
    total_length: int,
    min_segment_length: int = 5,
) -> list[tuple[int, int]]:
    """Convert boundary frame indices to half-open ``(start, end)`` segments.

    The first segment always starts at ``0`` and the last ends at
    ``total_length``. Segments shorter than ``min_segment_length`` are merged
    into a neighbor (the shorter adjacent segment wins; if both are equal,
    the left neighbor wins) until either no short segments remain or only one
    segment is left.
    """
    total_length = int(total_length)
    if total_length <= 0:
        return []

    b = np.asarray(boundaries, dtype=np.int64).reshape(-1)
    # Clip into the valid interior range (a boundary at 0 or T is redundant).
    b = b[(b > 0) & (b < total_length)]
    b = np.unique(b)

    edges: list[int] = [0, *b.tolist(), total_length]
    segments: list[tuple[int, int]] = [
        (int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)
    ]

    min_len = max(1, int(min_segment_length))
    if min_len <= 1 or len(segments) <= 1:
        return segments

    # Repeatedly merge the shortest segment that violates ``min_len`` into
    # the smaller of its neighbors until none remain.
    while True:
        if len(segments) == 1:
            break
        lengths = np.asarray([e - s for s, e in segments], dtype=np.int64)
        bad = np.where(lengths < min_len)[0]
        if bad.size == 0:
            break
        # Merge the shortest violator first for stable behavior.
        i = int(bad[np.argmin(lengths[bad])])
        if i == 0:
            merge_into = 1
        elif i == len(segments) - 1:
            merge_into = i - 1
        else:
            left_len = lengths[i - 1]
            right_len = lengths[i + 1]
            merge_into = i - 1 if left_len <= right_len else i + 1

        s_i, e_i = segments[i]
        s_j, e_j = segments[merge_into]
        merged = (min(s_i, s_j), max(e_i, e_j))
        lo, hi = (i, merge_into) if i < merge_into else (merge_into, i)
        segments = segments[:lo] + [merged] + segments[hi + 1 :]

    return segments


def upsample_probs(probs: np.ndarray, target_length: int) -> np.ndarray:
    """Upsample a 1-D probability sequence to ``target_length`` via nearest-neighbor.

    Used to map predictions made at the feature stride (e.g. I3D stride 8)
    back to the original frame index space.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    T_in = int(probs.shape[0])
    T_out = int(target_length)
    if T_out <= 0:
        return np.empty(0, dtype=np.float64)
    if T_in == 0:
        return np.zeros(T_out, dtype=np.float64)
    if T_in == T_out:
        return probs.astype(np.float64, copy=True)

    # Nearest-neighbor sampling at evenly spaced cell centers.
    out_centers = (np.arange(T_out, dtype=np.float64) + 0.5) * (T_in / T_out)
    src_idx = np.clip(np.floor(out_centers).astype(np.int64), 0, T_in - 1)
    return probs[src_idx]


def segment_video_from_probs(
    probs: np.ndarray,
    original_T: int,
    stride: int = 8,
    threshold: float = 0.5,
    min_distance: int = 3,
    min_segment_length: int = 5,
    prominence: float = 0.1,
) -> list[tuple[int, int]]:
    """End-to-end helper: upsample ``probs`` to ``original_T`` and segment it.

    ``probs`` is the per-feature-step output of Stage 1; ``stride`` is the
    feature stride relative to the original frame rate (informational only --
    upsampling targets ``original_T`` directly). Returns half-open
    ``(start, end)`` segments in original-frame coordinates.
    """
    del stride  # Kept for API clarity; ``original_T`` already encodes the scale.
    up = upsample_probs(np.asarray(probs, dtype=np.float64), int(original_T))
    peaks = detect_boundaries(
        up,
        threshold=threshold,
        min_distance=min_distance,
        prominence=prominence,
    )
    return boundaries_to_segments(
        peaks,
        total_length=int(original_T),
        min_segment_length=min_segment_length,
    )
