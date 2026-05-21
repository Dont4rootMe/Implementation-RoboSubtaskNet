"""Unit tests for ``robosubtasknet.eval.metrics``.

Coverage maps to the §14.1 ``test_metrics.py`` bullets:

* ``fibonacci(0) == []`` and ``fibonacci(1) == [1]`` edge cases (kept
  here in addition to ``test_tcn.py`` so the metrics test file stays
  self-contained, per the explicit instruction in the brief).
* ``edit_score`` returns ``100`` on identical sequences.
* ``f_score`` returns a non-negative ``(TP, FP, FN)`` triple, and the
  total number of predicted segments always equals ``TP + FP`` (every
  predicted segment is either matched or unmatched).
* ``frame_accuracy`` ignores frames labelled ``-100``.
* A hand-worked 3-segment example where F1@50 is computed by hand and
  asserted exactly.

Tensor comparisons use ``torch.testing.assert_close`` with explicit
tolerances. Plain Python numeric assertions use ``math.isclose`` with
explicit ``abs_tol`` / ``rel_tol`` rather than ``==`` on floats.
"""

from __future__ import annotations

import math

from robosubtasknet.eval.metrics import (
    edit_score,
    f_score,
    frame_accuracy,
    get_labels_start_end_time,
)
from robosubtasknet.models import fibonacci


# ---------------------------------------------------------------------------
# fibonacci edge cases (also covered in test_tcn.py — explicit in the brief)
# ---------------------------------------------------------------------------


def test_fibonacci_zero_returns_empty_list() -> None:
    """``fibonacci(0)`` is the empty schedule — no dilated layers."""
    assert fibonacci(0) == []


def test_fibonacci_one_returns_single_element() -> None:
    """``fibonacci(1)`` returns ``[F_2] == [1]`` — one dilation = 1, the
    'dense' starting case."""
    assert fibonacci(1) == [1]


# ---------------------------------------------------------------------------
# edit_score on identical sequences
# ---------------------------------------------------------------------------


def test_edit_score_identical_sequences_is_100() -> None:
    """A perfect prediction yields the maximum edit score, ``100.0``."""
    seq = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    score = edit_score(seq, seq)
    assert math.isclose(score, 100.0, abs_tol=1e-9), (
        f"expected edit_score(seq, seq) == 100.0; got {score}"
    )


def test_edit_score_completely_wrong_sequence_lower_than_100() -> None:
    """Sanity check: a non-identical prediction must score strictly below 100
    (so the identical-sequence test isn't trivially true for any input)."""
    gt = [1, 2, 3]
    pred = [3, 2, 1]
    score = edit_score(pred, gt)
    assert score < 100.0, (
        f"expected edit_score < 100 for a permuted sequence; got {score}"
    )


# ---------------------------------------------------------------------------
# f_score: non-negativity and TP+FP == total predicted segments
# ---------------------------------------------------------------------------


def test_f_score_non_negative_and_total_pred_equals_tp_plus_fp() -> None:
    """For any input, ``f_score`` produces non-negative counts and the
    sum ``TP + FP`` equals the number of predicted segments (MS-TCN's
    matching algorithm classifies every predicted segment as exactly
    one of the two).
    """
    # A non-trivial mix: some predictions match, some don't, and the
    # ground-truth has a label that never appears in predictions
    # (forces an FN).
    gt = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    pred = [1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1]
    tp, fp, fn = f_score(pred, gt, overlap=0.5)

    assert tp >= 0, f"tp must be >= 0; got {tp}"
    assert fp >= 0, f"fp must be >= 0; got {fp}"
    assert fn >= 0, f"fn must be >= 0; got {fn}"

    # MS-TCN's f_score loops over every predicted segment and classifies
    # each as exactly one of TP / FP. So #pred segments == TP + FP.
    pred_segments, _, _ = get_labels_start_end_time(pred)
    assert tp + fp == len(pred_segments), (
        f"TP + FP ({tp + fp}) should equal number of predicted segments "
        f"({len(pred_segments)})"
    )


def test_f_score_perfect_match_is_all_tp() -> None:
    """When pred == gt, every predicted segment is a true positive."""
    seq = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    tp, fp, fn = f_score(seq, seq, overlap=0.5)
    segs, _, _ = get_labels_start_end_time(seq)
    assert tp == len(segs)
    assert fp == 0
    assert fn == 0


# ---------------------------------------------------------------------------
# frame_accuracy ignores -100 frames
# ---------------------------------------------------------------------------


def test_frame_accuracy_ignores_minus_100_frames() -> None:
    """``frame_accuracy`` should exclude both the numerator and the
    denominator any frame whose GT label is ``-100`` (the PyTorch
    cross-entropy "ignore" convention).

    Build a case where the ignored frames are *wrong* — without the
    ignore mask the accuracy would dip; with it, accuracy stays at 1.0.
    """
    gt =   [1, 1, -100, 2, -100, 3, 3]
    pred = [1, 1, 9,    2, 9,    3, 3]  # mismatches only on ignored frames
    acc = frame_accuracy(pred, gt, ignore_index=-100)
    assert math.isclose(acc, 1.0, abs_tol=1e-9), (
        f"expected acc == 1.0 with ignore_index=-100; got {acc}"
    )

    # And confirm the ignore mask actually does something: omitting it
    # (i.e. requiring exact match on every frame) would give 5/7.
    acc_no_ignore = frame_accuracy(pred, gt, ignore_index=-9999)  # any unused value
    assert math.isclose(acc_no_ignore, 5.0 / 7.0, abs_tol=1e-9), (
        f"without ignore mask, expected 5/7 ≈ 0.714; got {acc_no_ignore}"
    )


def test_frame_accuracy_all_masked_returns_zero() -> None:
    """Documented degenerate case: every frame ignored → ``0.0``."""
    gt = [-100, -100, -100]
    pred = [1, 2, 3]
    acc = frame_accuracy(pred, gt, ignore_index=-100)
    assert acc == 0.0


# ---------------------------------------------------------------------------
# Hand-worked 3-segment example — F1@50 by hand
# ---------------------------------------------------------------------------


def test_f1_at_50_hand_worked_three_segments() -> None:
    """Hand-worked F1@50 against a 3-segment ground-truth / prediction pair.

    Setup (20 frames):

        GT    = [1]*10 + [2]*5 + [3]*5    → 3 segments:
                  (label=1, [ 0, 10))
                  (label=2, [10, 15))
                  (label=3, [15, 20))

        Pred  = [1]*4  + [2]*11 + [3]*5    → 3 segments:
                  (label=1, [ 0,  4))
                  (label=2, [ 4, 15))
                  (label=3, [15, 20))

    IoU computation per predicted segment (same-label match only):

      Pred(label=1, [0,4))  vs GT(label=1, [0,10)):
          intersection = min(4,10) - max(0,0)  = 4
          union        = max(4,10) - min(0,0)  = 10
          IoU = 4/10 = 0.4  -> below 0.5 -> FP

      Pred(label=2, [4,15)) vs GT(label=2, [10,15)):
          intersection = min(15,15) - max(4,10) = 5
          union        = max(15,15) - min(4,10) = 11
          IoU = 5/11 ≈ 0.4545  -> below 0.5 -> FP

      Pred(label=3, [15,20)) vs GT(label=3, [15,20)):
          intersection = 5, union = 5
          IoU = 1.0 -> TP

    Final counts:
      TP = 1
      FP = 2
      FN = (# GT segs) - (# hits) = 3 - 1 = 2

    Precision = 1 / (1 + 2) = 1/3
    Recall    = 1 / (1 + 2) = 1/3
    F1@50     = 2 * (1/3) * (1/3) / (1/3 + 1/3) = 1/3 ≈ 33.333...%
    """
    gt = [1] * 10 + [2] * 5 + [3] * 5
    pred = [1] * 4 + [2] * 11 + [3] * 5

    # Sanity: both sides have exactly 3 segments (so the test name's
    # "three segments" claim is structural, not coincidental).
    p_segs, _, _ = get_labels_start_end_time(pred)
    g_segs, _, _ = get_labels_start_end_time(gt)
    assert len(p_segs) == 3, f"expected 3 predicted segments; got {len(p_segs)}"
    assert len(g_segs) == 3, f"expected 3 GT segments; got {len(g_segs)}"

    tp, fp, fn = f_score(pred, gt, overlap=0.5)
    assert (tp, fp, fn) == (1, 2, 2), (
        f"hand-worked F1@50 counts mismatch: expected (TP=1, FP=2, FN=2); "
        f"got (TP={tp}, FP={fp}, FN={fn})"
    )

    # Derive precision / recall / F1 directly from the counts so the
    # arithmetic matches the docstring above.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    assert math.isclose(precision, 1.0 / 3.0, abs_tol=1e-9, rel_tol=1e-9)
    assert math.isclose(recall, 1.0 / 3.0, abs_tol=1e-9, rel_tol=1e-9)
    assert math.isclose(f1, 1.0 / 3.0, abs_tol=1e-9, rel_tol=1e-9)
