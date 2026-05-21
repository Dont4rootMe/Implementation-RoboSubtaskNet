"""Segmentation evaluation: MS-TCN-compatible metrics and a corpus aggregator.

See ``IMPLEMENTATION_PLAN.md`` Section 11 and :mod:`robosubtasknet.eval.metrics`
for the per-function semantics. Public surface is the metric helpers plus
the :class:`SegmentationEvaluator` accumulator.
"""

from .evaluator import SegmentationEvaluator
from .metrics import (
    aggregate_f1,
    edit_score,
    f_score,
    frame_accuracy,
    get_labels_start_end_time,
    levenstein,
)

__all__ = [
    "SegmentationEvaluator",
    "aggregate_f1",
    "edit_score",
    "f_score",
    "frame_accuracy",
    "get_labels_start_end_time",
    "levenstein",
]
