"""Corpus-level segmentation evaluator that aggregates the MS-TCN metrics.

Wraps the per-video helpers in :mod:`robosubtasknet.eval.metrics` and
maintains the running statistics needed to report the standard four
numbers from Section 11 of ``IMPLEMENTATION_PLAN.md``:

    * ``Acc``                — micro frame accuracy across the corpus.
    * ``Edit``               — mean per-video edit score.
    * ``F1@10 / F1@25 / F1@50`` — corpus-wide F1, derived from summed
                                  ``(TP, FP, FN)`` over all videos at
                                  IoU thresholds ``0.1 / 0.25 / 0.5``.

Aggregating F1 at the corpus level (rather than averaging per-video F1)
is what the MS-TCN reference does and is the convention the paper's
numbers were reported under; per-video averaging would over-weight
short videos. Acc is also computed micro-style for the same reason.
"""

from __future__ import annotations

from typing import Sequence

from .metrics import (
    aggregate_f1,
    edit_score,
    f_score,
    frame_accuracy,
)


__all__ = ["SegmentationEvaluator"]


# IoU thresholds at which segment-level F1 is reported, in percent units
# (10, 25, 50). Stored as integers so they can be used directly as dict
# keys (``"F1@10"`` etc.) in the returned report.
_DEFAULT_OVERLAPS_PCT: tuple[int, int, int] = (10, 25, 50)


class SegmentationEvaluator:
    """Accumulator for corpus-level segmentation metrics.

    Typical usage::

        evaluator = SegmentationEvaluator(bg_class=[0])
        for pred, gt in zip(predictions, ground_truths):
            evaluator.update(pred, gt)
        report = evaluator.compute()
        # report == {"acc": ..., "edit": ...,
        #            "F1@10": ..., "F1@25": ..., "F1@50": ...,
        #            "precision@10": ..., "recall@10": ...,
        #            "precision@25": ..., "recall@25": ...,
        #            "precision@50": ..., "recall@50": ...,
        #            "num_videos": ...}

    Args:
        bg_class: Optional list of background label ids to skip during
            segment extraction. Pass ``[0]`` (or whatever your mapping
            uses) for datasets like GTEA that exclude a background class
            from segment-level metrics; pass ``None`` to score every
            distinct stretch of labels (the default).
        ignore_index: Label value that marks padded / ignored frames
            when computing frame accuracy. Defaults to ``-100`` to
            match PyTorch's cross-entropy convention.
        overlaps_pct: IoU thresholds (in percent) at which F1 is
            reported. Defaults to ``(10, 25, 50)`` per the paper.
    """

    def __init__(
        self,
        bg_class: list[int] | None = None,
        ignore_index: int = -100,
        overlaps_pct: Sequence[int] = _DEFAULT_OVERLAPS_PCT,
    ) -> None:
        self.bg_class = bg_class
        self.ignore_index = ignore_index
        # Freeze the threshold list as a tuple so it cannot be mutated
        # mid-evaluation (and so the dict-key strings stay stable).
        self.overlaps_pct: tuple[int, ...] = tuple(int(k) for k in overlaps_pct)

        # Running per-threshold confusion-style counters and edit / acc tallies.
        self._tp: dict[int, int] = {k: 0 for k in self.overlaps_pct}
        self._fp: dict[int, int] = {k: 0 for k in self.overlaps_pct}
        self._fn: dict[int, int] = {k: 0 for k in self.overlaps_pct}

        # Edit is per-video and averaged at compute time.
        self._edit_sum: float = 0.0
        self._edit_count: int = 0

        # Frame accuracy is micro: we accumulate correct/total over the
        # whole corpus to avoid biasing toward short videos.
        self._frame_correct: int = 0
        self._frame_total: int = 0

        self._num_videos: int = 0

    def reset(self) -> None:
        """Zero every accumulator. Cheap; safe to call between epochs."""
        for k in self.overlaps_pct:
            self._tp[k] = 0
            self._fp[k] = 0
            self._fn[k] = 0
        self._edit_sum = 0.0
        self._edit_count = 0
        self._frame_correct = 0
        self._frame_total = 0
        self._num_videos = 0

    def update(
        self,
        pred: Sequence[int],
        gt: Sequence[int],
    ) -> None:
        """Accumulate stats from one video's frame-label sequences.

        Both arguments must be per-frame integer label sequences of
        equal length. Padding frames (those marked with ``ignore_index``
        in ``gt``) are excluded from frame accuracy but currently still
        feed into segment extraction — callers that want them stripped
        from segments should slice them out before calling ``update``.

        Args:
            pred: Predicted per-frame labels for this video.
            gt: Ground-truth per-frame labels for this video.
        """
        # Frame accuracy — accumulate micro-counts so we can divide once
        # at compute time and avoid the per-video averaging bias.
        valid_total = 0
        valid_correct = 0
        for p_t, g_t in zip(pred, gt):
            if g_t == self.ignore_index:
                continue
            valid_total += 1
            if p_t == g_t:
                valid_correct += 1
        self._frame_correct += valid_correct
        self._frame_total += valid_total

        # Edit and F1 are segment-level; they don't have a natural notion
        # of an "ignored" frame, so we score the raw sequences. The
        # standard MS-TCN practice (and the paper's evaluation) is to
        # pass the full sequence with ``bg_class`` set instead.
        self._edit_sum += edit_score(pred, gt, bg_class=self.bg_class)
        self._edit_count += 1

        for k in self.overlaps_pct:
            tp, fp, fn = f_score(
                pred,
                gt,
                overlap=k / 100.0,
                bg_class=self.bg_class,
            )
            self._tp[k] += tp
            self._fp[k] += fp
            self._fn[k] += fn

        self._num_videos += 1

    def compute(self) -> dict[str, float]:
        """Roll the accumulated counters into a flat metrics report.

        Returns:
            A dict with the keys

            * ``"acc"``                — micro frame accuracy in %,
            * ``"edit"``               — mean per-video edit score in %,
            * ``"F1@k"``, ``"precision@k"``, ``"recall@k"`` for each
              configured IoU threshold ``k`` (in %),
            * ``"num_videos"`` — count of ``update`` calls so far.

            Returns zeros (rather than raising) when called before any
            ``update`` so it is safe to use inside a training loop that
            may legitimately see an empty validation split.
        """
        report: dict[str, float] = {}

        # Frame accuracy is stored as a fraction; rescale to percent for
        # parity with the paper's reporting convention.
        if self._frame_total > 0:
            report["acc"] = self._frame_correct / self._frame_total * 100.0
        else:
            report["acc"] = 0.0

        if self._edit_count > 0:
            report["edit"] = self._edit_sum / self._edit_count
        else:
            report["edit"] = 0.0

        for k in self.overlaps_pct:
            precision, recall, f1 = aggregate_f1(
                self._tp[k],
                self._fp[k],
                self._fn[k],
            )
            report[f"F1@{k}"] = f1
            report[f"precision@{k}"] = precision
            report[f"recall@{k}"] = recall

        report["num_videos"] = float(self._num_videos)
        return report

    # -- Convenience helpers --------------------------------------------------

    def add_frame_accuracy(
        self,
        pred: Sequence[int],
        gt: Sequence[int],
    ) -> float:
        """Compute (but do not accumulate) per-video frame accuracy.

        Exposed so callers that want to log per-video diagnostics can
        reuse the same masking logic without poking at private state.
        Returns a fraction in ``[0, 1]``, matching ``frame_accuracy``.
        """
        return frame_accuracy(pred, gt, ignore_index=self.ignore_index)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SegmentationEvaluator(bg_class={self.bg_class!r}, "
            f"ignore_index={self.ignore_index}, "
            f"overlaps_pct={self.overlaps_pct}, "
            f"num_videos={self._num_videos})"
        )
