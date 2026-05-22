"""Class-agnostic boundary segmentation (Stage 1 of the 2-stage pipeline).

Exposes :class:`BoundarySegmenter`, a multi-stage refinement model that emits
per-frame boundary logits (binary: boundary vs non-boundary). Stage 2 of the
pipeline consumes the resulting boundaries to crop and classify clips; that
stage lives elsewhere.
"""

from .model import BoundarySegmenter

__all__ = ["BoundarySegmenter"]
