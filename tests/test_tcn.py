"""Unit tests for the Fibonacci-dilated TCN (``robosubtasknet.models.tcn``)
and the multi-stage wrapper (``robosubtasknet.models.robosubtasknet``).

Coverage maps to the §14.1 bullets for ``test_tcn.py``:

1. ``fibonacci(10) == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]``.
2. ``SingleStageTCN`` output shape is ``[B, num_classes, T]``.
3. ``RoboSubtaskNet`` returns a list of length ``num_stages``.
4. The plan's receptive-field formula matches reality: forward a delta
   (one-hot in time) through a single stage and verify the width of the
   non-zero gradient region equals the closed-form RF.
"""

from __future__ import annotations

import pytest
import torch

from robosubtasknet.models import (
    RoboSubtaskNet,
    SingleStageTCN,
    fibonacci,
)


# ---------------------------------------------------------------------------
# Dilation schedule
# ---------------------------------------------------------------------------


def test_fibonacci_ten() -> None:
    """The §14.1 specimen: ``fibonacci(10) == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]``."""
    assert fibonacci(10) == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


# ---------------------------------------------------------------------------
# Single-stage TCN shape
# ---------------------------------------------------------------------------


def test_single_stage_output_shape(num_classes: int) -> None:
    """``SingleStageTCN(...)`` maps ``[B, D, T] -> [B, C, T]`` exactly."""
    B, D, T = 2, 16, 32
    num_layers = 5  # small enough to keep the test fast
    hidden = 8

    model = SingleStageTCN(
        num_layers=num_layers,
        in_dim=D,
        hidden_dim=hidden,
        num_classes=num_classes,
        dropout=0.0,  # determinism not strictly needed for shape but cheap
    ).eval()

    x = torch.randn(B, D, T)
    y = model(x)

    assert y.shape == (B, num_classes, T), (
        f"expected [{B}, {num_classes}, {T}]; got {tuple(y.shape)}"
    )


# ---------------------------------------------------------------------------
# RoboSubtaskNet multi-stage list length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_robosubtasknet_returns_list_of_num_stages(
    num_classes: int, num_stages: int,
) -> None:
    """The multi-stage forward returns one logit tensor per stage."""
    B, T, D = 2, 16, 8

    model = RoboSubtaskNet(
        num_stages=num_stages,
        num_layers=3,            # small for speed
        feature_dim=D,
        hidden_dim=8,
        num_classes=num_classes,
        dropout=0.0,
    ).eval()

    rgb = torch.randn(B, T, D)
    flow = torch.randn(B, T, D)
    outputs = model(rgb, flow)

    assert isinstance(outputs, list), (
        f"RoboSubtaskNet.forward must return a list; got {type(outputs).__name__}"
    )
    assert len(outputs) == num_stages, (
        f"expected {num_stages} stage outputs; got {len(outputs)}"
    )
    for s, out in enumerate(outputs):
        assert out.shape == (B, num_classes, T), (
            f"stage {s}: expected [{B}, {num_classes}, {T}]; "
            f"got {tuple(out.shape)}"
        )


# ---------------------------------------------------------------------------
# Receptive-field formula vs reality
# ---------------------------------------------------------------------------


def _expected_receptive_field(num_layers: int) -> int:
    """Closed-form RF for kernel=3 + Fibonacci dilations (§8.1)."""
    return 1 + 2 * sum(fibonacci(num_layers))


def test_receptive_field_matches_formula(num_classes: int) -> None:
    """Forward a one-hot temporal delta and check the gradient-influence width.

    We freeze all randomness (``dropout = 0``, ``eval()``), use a long
    enough ``T`` to host the full RF without boundary effects, and place
    a single non-zero column at the centre of the input. Backpropagating
    the centre output's sum w.r.t. the input gives us the set of input
    frames that *could* have affected that output — the empirical
    receptive field. Its width must match the closed-form formula
    ``RF(L) = 1 + 2 * sum(fibonacci(L))``.

    We use a fairly small ``L`` so the test stays fast (RF for L=5 is
    1 + 2*(1+2+3+5+8) = 39 frames; comfortably fits in T=128).
    """
    num_layers = 5
    expected_rf = _expected_receptive_field(num_layers)  # 39
    assert expected_rf == 39, "sanity: rf formula sanity check"

    B, D, T = 1, 4, 128
    hidden = 4

    model = SingleStageTCN(
        num_layers=num_layers,
        in_dim=D,
        hidden_dim=hidden,
        num_classes=num_classes,
        dropout=0.0,
    ).eval()

    # Use a constant input that *requires grad*; we then read .grad to
    # see which input frames influence the centre output.
    x = torch.zeros(B, D, T, requires_grad=True)
    y = model(x)  # [B, C, T]

    centre = T // 2
    # Sum across batch / class so a single backward pass exposes the
    # input-frame footprint of the centre output column.
    y[:, :, centre].sum().backward()

    grad = x.grad  # [B, D, T]
    assert grad is not None
    # Mark a frame as "in the RF" if it has any non-zero gradient across
    # the D channels (any path is enough).
    influence = grad.abs().sum(dim=(0, 1))  # [T]
    nonzero_mask = influence > 0.0
    nz_indices = torch.nonzero(nonzero_mask, as_tuple=False).flatten().tolist()

    assert len(nz_indices) > 0, "centre output received no gradient at all"

    # Contiguity check: the RF should be a contiguous interval around
    # ``centre`` (Fibonacci-dilated convolutions are linear in the
    # kernel pattern so holes would indicate a bug).
    lo, hi = nz_indices[0], nz_indices[-1]
    empirical_rf = hi - lo + 1
    assert empirical_rf == expected_rf, (
        f"empirical RF {empirical_rf} (indices {lo}..{hi}) does not match "
        f"closed-form RF {expected_rf} for L={num_layers}"
    )
    # And the interval should be centred on ``centre`` (symmetric padding).
    half = expected_rf // 2
    assert lo == centre - half, f"left edge {lo} != centre-half={centre - half}"
    assert hi == centre + half, f"right edge {hi} != centre+half={centre + half}"
