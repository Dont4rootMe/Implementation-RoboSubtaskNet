"""Per-frame annotation parsers and helpers.

Two annotation conventions are supported:

* **MS-TCN ``.txt``** — one label name per line, one line per frame
  (Section 5.1 of ``IMPLEMENTATION_PLAN.md``).
* **CSV with ``frame_idx,subtask`` header** — the custom RoboSubtask
  format (Section 5.4 of ``IMPLEMENTATION_PLAN.md``).

Both parsers return ``list[str]`` of label names so a single
:func:`labels_to_indices` step maps them to class IDs via a label map.

:func:`segments_from_labels` converts a per-frame integer sequence into
``(start, end_exclusive, label_idx)`` runs, which downstream code uses
for F1@k / Edit metrics and segmentation overlays.
"""

from __future__ import annotations

import csv
from pathlib import Path


def parse_per_frame_labels_txt(path: str | Path) -> list[str]:
    """Parse an MS-TCN-style per-frame label file.

    Each non-empty line is treated as the label for one frame, in order.
    Trailing whitespace is stripped; blank lines are ignored (MS-TCN files
    sometimes end with a trailing newline).

    Args:
        path: Filesystem path to the ``.txt`` annotation file.

    Returns:
        List of label-name strings, one per frame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty (no frames).
    """
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {txt_path}")

    labels: list[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                continue
            labels.append(stripped)

    if not labels:
        raise ValueError(f"Annotation file contained no labels: {txt_path}")
    return labels


def parse_per_frame_labels_csv(path: str | Path) -> list[str]:
    """Parse a per-frame CSV annotation with a ``frame_idx,subtask`` header.

    Frame indices must be a strict 0-based contiguous sequence
    (``0, 1, 2, ...``). Out-of-order or missing indices raise
    :class:`ValueError` with a descriptive message — silent gaps in
    annotations cause hard-to-debug segmentation errors downstream.

    Args:
        path: Filesystem path to the ``.csv`` annotation file.

    Returns:
        List of sub-task name strings, one per frame, ordered by
        ``frame_idx``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the header is malformed, ``frame_idx`` is not an
            integer, or indices are missing / out of order / duplicated.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")

    labels: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Annotation file is empty: {csv_path}") from exc

        normalized = [h.strip().lower() for h in header]
        if normalized[:2] != ["frame_idx", "subtask"]:
            raise ValueError(
                f"Expected CSV header 'frame_idx,subtask' in {csv_path}, got {header!r}"
            )

        expected_idx = 0
        for row_num, row in enumerate(reader, start=2):  # start=2: header is line 1
            if not row or all(cell.strip() == "" for cell in row):
                continue  # tolerate blank rows
            if len(row) < 2:
                raise ValueError(
                    f"{csv_path}:{row_num}: expected at least 2 columns, got {row!r}"
                )
            raw_idx, raw_label = row[0].strip(), row[1].strip()
            try:
                frame_idx = int(raw_idx)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_num}: frame_idx is not an integer: {raw_idx!r}"
                ) from exc

            if frame_idx != expected_idx:
                if frame_idx < expected_idx:
                    raise ValueError(
                        f"{csv_path}:{row_num}: out-of-order or duplicate frame_idx "
                        f"{frame_idx} (expected {expected_idx})"
                    )
                raise ValueError(
                    f"{csv_path}:{row_num}: missing frame_idx {expected_idx} "
                    f"(jumped to {frame_idx})"
                )

            if raw_label == "":
                raise ValueError(
                    f"{csv_path}:{row_num}: empty sub-task label for frame {frame_idx}"
                )
            labels.append(raw_label)
            expected_idx += 1

    if not labels:
        raise ValueError(f"Annotation file contained no frame rows: {csv_path}")
    return labels


def labels_to_indices(labels: list[str], mapping: dict[str, int]) -> list[int]:
    """Strictly map label names to integer class IDs.

    Args:
        labels:  Sequence of label names.
        mapping: ``name -> class_idx`` dictionary (typically the project's
            ``label_map``).

    Returns:
        Parallel list of integer class IDs.

    Raises:
        KeyError: If any label is missing from ``mapping``. The exception
            message lists the offending labels and the known vocabulary,
            so users can spot typos quickly.
    """
    unknown = sorted({label for label in labels if label not in mapping})
    if unknown:
        known = sorted(mapping.keys())
        raise KeyError(
            f"Unknown sub-task label(s) {unknown}. "
            f"Known labels: {known}. Check for typos or update the label map."
        )
    return [mapping[label] for label in labels]


def segments_from_labels(labels: list[int]) -> list[tuple[int, int, int]]:
    """Run-length encode a per-frame integer label sequence into segments.

    Each segment is ``(start, end_exclusive, label_idx)``; i.e. the run
    covers frames ``[start, end_exclusive)``. Concatenating
    ``[labels[s:e] for (s, e, _) in segments]`` reproduces the input.

    Args:
        labels: Per-frame integer class IDs.

    Returns:
        List of ``(start, end_exclusive, label_idx)`` tuples, in order.
        Empty if ``labels`` is empty.
    """
    if not labels:
        return []

    segments: list[tuple[int, int, int]] = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i, current))
            start = i
            current = labels[i]
    segments.append((start, len(labels), current))
    return segments


__all__ = [
    "labels_to_indices",
    "parse_per_frame_labels_csv",
    "parse_per_frame_labels_txt",
    "segments_from_labels",
]
