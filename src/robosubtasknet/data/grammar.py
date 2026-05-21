"""Task → sub-task grammar utilities.

The grammar encodes which ordered sub-task transitions are valid for each
high-level task (see Sections 5.2 and 9.5 of ``IMPLEMENTATION_PLAN.md``).
It is the source of truth for the transition-aware loss term.

A grammar YAML has the shape::

    tasks:
      pick_and_place:
        sequence: [reach, pick, move, place, retract]
      cleaning:
        sequence: [reach, wipe, retract]

This module exposes three helpers:

* :func:`load_grammar`                       — parse + validate the YAML.
* :func:`validate_label_map_against_grammar` — surface missing label names.
* :func:`build_allowed_transitions`          — produce the ``[C, C]`` bool
  mask consumed by the transition loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_grammar(path: str | Path) -> dict[str, dict]:
    """Load and validate a task→sub-task grammar YAML.

    Args:
        path: Filesystem path to the grammar YAML.

    Returns:
        The parsed top-level mapping (``{"tasks": {...}, ...}``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is malformed (missing ``tasks`` key, a task
            entry that is not a mapping, or a ``sequence`` that is missing,
            empty, or contains non-string entries).
    """
    grammar_path = Path(path)
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar file not found: {grammar_path}")

    with grammar_path.open("r", encoding="utf-8") as f:
        grammar: Any = yaml.safe_load(f)

    if not isinstance(grammar, dict):
        raise ValueError(
            f"Grammar root must be a mapping, got {type(grammar).__name__} "
            f"in {grammar_path}"
        )
    if "tasks" not in grammar:
        raise ValueError(f"Grammar must contain a top-level 'tasks' key in {grammar_path}")

    tasks = grammar["tasks"]
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(
            f"'tasks' must be a non-empty mapping of task_name -> spec in {grammar_path}"
        )

    for task_name, task_spec in tasks.items():
        if not isinstance(task_spec, dict):
            raise ValueError(
                f"Task '{task_name}' must be a mapping with a 'sequence' key, "
                f"got {type(task_spec).__name__}"
            )
        if "sequence" not in task_spec:
            raise ValueError(f"Task '{task_name}' is missing required 'sequence' key")
        seq = task_spec["sequence"]
        if not isinstance(seq, list) or len(seq) == 0:
            raise ValueError(
                f"Task '{task_name}' has an empty or non-list 'sequence' "
                f"(got {type(seq).__name__})"
            )
        for item in seq:
            if not isinstance(item, str):
                raise ValueError(
                    f"Task '{task_name}' contains non-string sub-task name: {item!r}"
                )

    return grammar


def validate_label_map_against_grammar(
    label_map: dict[str, int],
    grammar: dict[str, dict],
) -> list[str]:
    """Return sub-task names referenced by the grammar but absent from ``label_map``.

    Useful for emitting a friendly error before constructing the transition
    matrix (which would otherwise raise a bare ``KeyError``).

    Args:
        label_map: Mapping from sub-task name to class index.
        grammar:   Parsed grammar mapping (as returned by :func:`load_grammar`).

    Returns:
        Sorted list of missing sub-task names. Empty if the label map covers
        every name referenced by the grammar. The ``background`` class is
        also reported as missing if not present.
    """
    referenced: set[str] = set()
    for task_spec in grammar.get("tasks", {}).values():
        for name in task_spec.get("sequence", []):
            referenced.add(name)
    referenced.add("background")
    return sorted(referenced - set(label_map.keys()))


def build_allowed_transitions(
    grammar_path: str | Path,
    label_map: dict[str, int],
) -> torch.Tensor:
    """Construct the ``[C, C]`` bool tensor of allowed ordered transitions.

    Rules (Section 9.5 of ``IMPLEMENTATION_PLAN.md``):

      1. All self-transitions are allowed (a sub-task can persist).
      2. For each task in the grammar, allow consecutive transitions along
         its ``sequence`` (i.e. ``seq[i] -> seq[i+1]``).
      3. Background bridges in both directions for every sub-task that
         appears in any task sequence: ``background -> x`` and
         ``x -> background``.

    Args:
        grammar_path: Path to the grammar YAML.
        label_map:    Mapping from sub-task name (must include
            ``"background"``) to class index in ``[0, C)``.

    Returns:
        ``torch.BoolTensor`` of shape ``[C, C]`` where ``allowed[a, b]`` is
        ``True`` iff a frame-to-frame transition from class ``a`` to class
        ``b`` is grammatically valid.

    Raises:
        KeyError: If a sub-task referenced by the grammar (or the implicit
            ``"background"`` class) is missing from ``label_map``. The
            error message lists every missing name.
    """
    grammar = load_grammar(grammar_path)

    missing = validate_label_map_against_grammar(label_map, grammar)
    if missing:
        raise KeyError(
            "label_map is missing sub-task names referenced by the grammar: "
            f"{missing}. Update the label map (and remember the 'background' class)."
        )

    num_classes = len(label_map)
    allowed = torch.eye(num_classes, dtype=torch.bool)

    bg = label_map["background"]

    for task_spec in grammar["tasks"].values():
        seq = [label_map[name] for name in task_spec["sequence"]]
        # Rule 2: consecutive transitions along the task's sequence.
        for a, b in zip(seq[:-1], seq[1:]):
            allowed[a, b] = True
        # Rule 3: background bridges in both directions.
        for c in seq:
            allowed[bg, c] = True
            allowed[c, bg] = True

    return allowed


__all__ = [
    "build_allowed_transitions",
    "load_grammar",
    "validate_label_map_against_grammar",
]
