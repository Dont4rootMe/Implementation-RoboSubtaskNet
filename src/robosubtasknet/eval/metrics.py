"""Segment-level evaluation metrics for temporal action segmentation.

Faithful re-implementation of the helpers used by MS-TCN's ``eval.py``
(https://github.com/yabufarha/ms-tcn). The plan explicitly anchors the
evaluation semantics to MS-TCN (see Section 11 of ``IMPLEMENTATION_PLAN.md``):

    "Use the MS-TCN evaluation script as the reference implementation;
    do not write your own from scratch first — verify your eval against
    theirs on a small example."

The function names, signatures, and segment-matching semantics here
mirror that script exactly so unit tests can compare numbers against the
upstream reference without translation.

Public surface:
    * ``get_labels_start_end_time`` — frame-label sequence → contiguous segments.
    * ``levenstein`` — Levenshtein distance over label sequences (kept under
      its original (mis)spelling to match MS-TCN's API).
    * ``edit_score`` — normalised Levenshtein edit score in ``[0, 100]``.
    * ``f_score`` — per-video ``(TP, FP, FN)`` at a given IoU threshold.
    * ``frame_accuracy`` — per-frame accuracy with an ignore-index mask.
    * ``aggregate_f1`` — fold ``(TP, FP, FN)`` totals into ``(P, R, F1)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


__all__ = [
    "get_labels_start_end_time",
    "levenstein",
    "edit_score",
    "f_score",
    "frame_accuracy",
    "aggregate_f1",
]


def get_labels_start_end_time(
    frame_labels: Sequence[int],
    bg_class: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Convert a per-frame label sequence to contiguous segments.

    Mirrors MS-TCN's ``get_labels_start_end_time``: walks the sequence
    once and emits a new segment whenever the label changes. Frames
    whose label is in ``bg_class`` are skipped — they are treated as
    "between segments" rather than as a segment of their own. This is
    important so that segment-level F1 / Edit do not reward correctly
    labelled background frames.

    Args:
        frame_labels: Per-frame integer labels of length ``T``.
        bg_class: Optional list of background label ids to exclude.
            When ``None`` no labels are excluded (every distinct stretch
            becomes a segment).

    Returns:
        Tuple ``(labels, starts, ends)`` of equal-length lists, where
        each segment spans frames ``[starts[i], ends[i])`` and carries
        label ``labels[i]``.
    """
    if bg_class is None:
        bg_class = []
    bg_set = set(bg_class)

    labels: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    n = len(frame_labels)
    if n == 0:
        return labels, starts, ends

    # Match MS-TCN: only "open" a segment if the first frame is non-bg.
    last_label = frame_labels[0]
    if last_label not in bg_set:
        labels.append(int(last_label))
        starts.append(0)

    for i in range(1, n):
        cur = frame_labels[i]
        if cur != last_label:
            # Close the previous segment if it was a non-bg one.
            if last_label not in bg_set:
                ends.append(i)
            # Open a new segment if the new label is non-bg.
            if cur not in bg_set:
                labels.append(int(cur))
                starts.append(i)
            last_label = cur

    # Close a still-open trailing segment.
    if last_label not in bg_set:
        ends.append(n)

    return labels, starts, ends


def levenstein(
    pred: list[int],
    gt: list[int],
    norm: bool = False,
) -> float:
    """Classic Levenshtein (edit) distance between two label sequences.

    Implementation matches MS-TCN's ``levenstein``: full DP table with
    insertion / deletion / substitution costs of 1 each. The spelling
    ``levenstein`` (missing an "h") is preserved deliberately so users
    porting code from MS-TCN do not have to rename anything.

    Args:
        pred: Predicted label sequence.
        gt: Ground-truth label sequence.
        norm: If ``True``, normalise the raw distance by
            ``max(len(pred), len(gt))`` and rescale to ``[0, 100]``.
            This is the form ``edit_score`` consumes.

    Returns:
        Edit distance as a ``float`` (raw integer count when ``norm=False``,
        a value in ``[0, 100]`` when ``norm=True``).
    """
    m_row = len(pred)
    n_col = len(gt)

    # DP table padded with the trivial base cases on both axes.
    D = np.zeros((m_row + 1, n_col + 1), dtype=float)
    D[:, 0] = np.arange(m_row + 1)
    D[0, :] = np.arange(n_col + 1)

    for i in range(1, m_row + 1):
        for j in range(1, n_col + 1):
            if pred[i - 1] == gt[j - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(
                    D[i - 1, j] + 1,      # deletion
                    D[i, j - 1] + 1,      # insertion
                    D[i - 1, j - 1] + 1,  # substitution
                )

    if norm:
        denom = max(m_row, n_col)
        if denom == 0:
            return 0.0
        return float(D[-1, -1]) / denom * 100.0
    return float(D[-1, -1])


def edit_score(
    pred_labels: Sequence[int],
    gt_labels: Sequence[int],
    bg_class: list[int] | None = None,
) -> float:
    r"""Normalised Levenshtein edit score over segment-label sequences.

    Computes

    .. math::
        \mathrm{Edit}
            = \Bigl(1 - \frac{\mathrm{Lev}(\hat{\mathbf{s}}, \mathbf{s})}
                              {\max(|\hat{\mathbf{s}}|, |\mathbf{s}|)}\Bigr)
              \times 100,

    where :math:`\hat{\mathbf{s}}, \mathbf{s}` are the *segment* label
    sequences (one label per contiguous predicted/ground-truth segment).
    Sensitive to over-segmentation: every spurious extra segment incurs
    at least one insertion.

    Args:
        pred_labels: Per-frame predicted labels.
        gt_labels: Per-frame ground-truth labels.
        bg_class: Optional list of background label ids excluded from
            segmentation (passed through to ``get_labels_start_end_time``).

    Returns:
        Edit score in ``[0, 100]`` (higher is better). Returns ``100.0``
        when both segment sequences are empty (perfectly matching
        all-background videos).
    """
    p_segs, _, _ = get_labels_start_end_time(pred_labels, bg_class)
    g_segs, _, _ = get_labels_start_end_time(gt_labels, bg_class)

    denom = max(len(p_segs), len(g_segs))
    if denom == 0:
        # Both sides are empty → trivially perfect.
        return 100.0

    # ``levenstein`` with norm=True already returns Lev / denom * 100.
    return 100.0 - levenstein(p_segs, g_segs, norm=True)


def f_score(
    pred_labels: Sequence[int],
    gt_labels: Sequence[int],
    overlap: float,
    bg_class: list[int] | None = None,
) -> tuple[int, int, int]:
    """Per-video ``(TP, FP, FN)`` at a given IoU threshold.

    Matches MS-TCN's ``f_score``: for each predicted segment, find the
    ground-truth segment with the same label and maximum IoU. If that
    IoU is at least ``overlap`` *and* that GT segment has not already
    been matched, count a true positive; otherwise a false positive.
    Any ground-truth segment that never gets matched is a false negative.

    Returns the raw counts so they can be summed across a corpus and
    funnelled through ``aggregate_f1`` once at the end (which is how
    the original script accumulates the dataset-level F1).

    Args:
        pred_labels: Per-frame predicted labels.
        gt_labels: Per-frame ground-truth labels.
        overlap: IoU threshold in ``[0, 1]`` (e.g. ``0.1``, ``0.25``, ``0.5``).
        bg_class: Optional list of background label ids excluded from
            segmentation.

    Returns:
        ``(tp, fp, fn)`` integer counts for this video.
    """
    p_labels, p_starts, p_ends = get_labels_start_end_time(pred_labels, bg_class)
    g_labels, g_starts, g_ends = get_labels_start_end_time(gt_labels, bg_class)

    tp = 0
    fp = 0
    # ``hits[j] == 1`` once GT segment ``j`` has been claimed by a TP.
    hits = np.zeros(len(g_labels), dtype=int)

    for i in range(len(p_labels)):
        # IoU of predicted segment i against every GT segment.
        # intersection = min(ends) - max(starts), clamped at 0.
        # union       = max(ends) - min(starts).
        intersection = np.minimum(p_ends[i], g_ends) - np.maximum(p_starts[i], g_starts)
        union = np.maximum(p_ends[i], g_ends) - np.minimum(p_starts[i], g_starts)
        # Only GT segments with the *same* label are eligible — the
        # in-class IoU is what F1@k scores.
        same_label = np.array(
            [1.0 if p_labels[i] == g_labels[j] else 0.0 for j in range(len(g_labels))]
        )
        # Guard against zero-length unions (parallel-empty segments
        # cannot happen here, but keep the divide safe).
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = (1.0 * intersection / union) * same_label
            iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)

        if len(iou) == 0:
            fp += 1
            continue

        idx = int(np.argmax(iou))
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = int(len(g_labels) - np.sum(hits))
    return int(tp), int(fp), int(fn)


def frame_accuracy(
    pred_labels: Sequence[int],
    gt_labels: Sequence[int],
    ignore_index: int = -100,
) -> float:
    """Per-frame accuracy with an ignore mask.

    The ``ignore_index`` default matches PyTorch's cross-entropy
    convention (padding frames are marked ``-100``); frames whose GT
    label equals ``ignore_index`` are excluded from both the numerator
    and the denominator. The metric is *not* multiplied by 100 — it is
    returned as a fraction in ``[0, 1]`` so the evaluator can decide on
    display units.

    Args:
        pred_labels: Per-frame predicted labels (length ``T``).
        gt_labels: Per-frame ground-truth labels (length ``T``).
        ignore_index: Label value that marks frames to skip
            (default ``-100``).

    Returns:
        Accuracy as a float in ``[0, 1]``. Returns ``0.0`` when every
        frame is masked out (no valid frames to score).
    """
    pred_arr = np.asarray(pred_labels)
    gt_arr = np.asarray(gt_labels)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape; got {pred_arr.shape} vs {gt_arr.shape}"
        )

    valid = gt_arr != ignore_index
    total = int(valid.sum())
    if total == 0:
        return 0.0
    correct = int(np.sum((pred_arr == gt_arr) & valid))
    return correct / total


def aggregate_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Fold ``(TP, FP, FN)`` totals into ``(precision, recall, F1)`` in %.

    Uses the harmonic mean and degenerates gracefully when either side
    is zero (e.g. a video with no predicted segments at all): all three
    return values are ``0.0`` rather than ``NaN``.

    Args:
        tp: True-positive count.
        fp: False-positive count.
        fn: False-negative count.

    Returns:
        ``(precision, recall, f1)`` triple, each in ``[0, 100]``.
    """
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision * 100.0, recall * 100.0, f1 * 100.0
