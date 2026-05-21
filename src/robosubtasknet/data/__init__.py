"""Data ingestion for RoboSubtaskNet.

This package owns dataset loading and collation. Other modules in this
package (augmentations, grammar, annotation) are introduced by separate
agents and are intentionally not re-exported from here to avoid coupling.
"""

from robosubtasknet.data.dataset import (
    RoboSubtaskDataset,
    load_mapping,
    pad_collate,
    parse_split,
)

__all__ = [
    "RoboSubtaskDataset",
    "load_mapping",
    "pad_collate",
    "parse_split",
]
