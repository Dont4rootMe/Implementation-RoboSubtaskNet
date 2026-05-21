"""Tests for the task grammar -> allowed-transitions matrix builder.

Covers Section 14.1 of ``IMPLEMENTATION_PLAN.md``:
    * All self-transitions are allowed.
    * Sequence transitions declared in the grammar are allowed.
    * Forbidden transitions remain disallowed.
    * The ``background`` class bridges to/from every sub-task in any sequence.
"""

from __future__ import annotations

import torch
import yaml

from robosubtasknet.data.grammar import build_allowed_transitions


def _write_tiny_grammar(tmp_path) -> str:
    """Write a tiny grammar YAML matching the schema in Section 9.5."""
    grammar = {
        "tasks": {
            "tiny_pick_and_place": {
                "sequence": ["reach", "pick", "place", "retract"],
            },
        },
    }
    grammar_path = tmp_path / "grammar.yaml"
    with grammar_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(grammar, f)
    return str(grammar_path)


def test_allowed_transitions_matches_grammar(tmp_path):
    label_map = {
        "background": 0,
        "reach": 1,
        "pick": 2,
        "place": 3,
        "retract": 4,
    }
    grammar_path = _write_tiny_grammar(tmp_path)

    allowed = build_allowed_transitions(grammar_path, label_map)

    num_classes = len(label_map)
    assert allowed.shape == (num_classes, num_classes), (
        f"expected [{num_classes}, {num_classes}] matrix, got {tuple(allowed.shape)}"
    )
    assert allowed.dtype is torch.bool, (
        f"expected bool tensor, got dtype={allowed.dtype}"
    )

    # 1. All diagonal entries are True (self-transitions allowed).
    for c in range(num_classes):
        assert bool(allowed[c, c]), f"self-transition at class {c} should be allowed"

    # 2. Sequence transitions declared in the grammar are allowed.
    bg = label_map["background"]
    reach = label_map["reach"]
    pick = label_map["pick"]
    place = label_map["place"]
    retract = label_map["retract"]

    assert bool(allowed[reach, pick]), "(reach -> pick) must be allowed"
    assert bool(allowed[pick, place]), "(pick -> place) must be allowed"
    assert bool(allowed[place, retract]), "(place -> retract) must be allowed"

    # 3. Forbidden transition: backwards along the sequence stays disallowed.
    assert not bool(allowed[pick, reach]), (
        "(pick -> reach) must be forbidden (backwards along the grammar sequence)"
    )

    # 4. Background bridges True both ways for any sub-task that appears
    #    in a task sequence.
    assert bool(allowed[bg, reach]), "(background -> reach) must be allowed"
    assert bool(allowed[reach, bg]), "(reach -> background) must be allowed"
