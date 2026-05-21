"""Unit tests for ``robosubtasknet.models.fusion.AttentionFusion``.

Coverage maps to the four bullets in §14.1 of ``IMPLEMENTATION_PLAN.md``:

1. Output shape equals input shape.
2. When both streams are identical, the convex combination collapses to
   either input regardless of the learned gate.
3. Gradients flow to both ``rgb`` and ``flow`` inputs.
4. The gate ``alpha`` stays in ``[0, 1]`` for arbitrary inputs (verified
   via ``forward_with_gate``).

All numeric comparisons go through ``torch.testing.assert_close`` with
explicit absolute / relative tolerances so a failure tells you *how far
off* the implementation is rather than just "not exactly equal".
"""

from __future__ import annotations

import pytest
import torch

from robosubtasknet.models import AttentionFusion


# ---------------------------------------------------------------------------
# Shape invariance (§14.1 / §7)
# ---------------------------------------------------------------------------


def test_output_shape_matches_input_shape(
    tiny_batch: dict[str, torch.Tensor],
) -> None:
    """``AttentionFusion`` is a per-frame, per-channel gate — shape preserved."""
    rgb = tiny_batch["rgb"]
    flow = tiny_batch["flow"]
    fusion = AttentionFusion(dim=rgb.shape[-1])

    fused = fusion(rgb, flow)

    assert fused.shape == rgb.shape == flow.shape, (
        f"expected fused.shape == rgb.shape == flow.shape; "
        f"got fused={tuple(fused.shape)}, rgb={tuple(rgb.shape)}, "
        f"flow={tuple(flow.shape)}"
    )


def test_output_shape_with_mlp_hidden(
    tiny_batch: dict[str, torch.Tensor],
) -> None:
    """The MLP variant of the gate must preserve shape too (Section 7.1 ablation)."""
    rgb = tiny_batch["rgb"]
    flow = tiny_batch["flow"]
    fusion = AttentionFusion(dim=rgb.shape[-1], hidden=rgb.shape[-1] // 2)

    fused = fusion(rgb, flow)

    assert fused.shape == rgb.shape


# ---------------------------------------------------------------------------
# Convex-combination collapse when streams are identical (§7.3 invariant 1)
# ---------------------------------------------------------------------------


def test_identical_streams_collapse_to_either_stream(
    tiny_batch: dict[str, torch.Tensor],
) -> None:
    """If rgb == flow, ``alpha * rgb + (1-alpha) * flow == rgb`` exactly.

    The gate value is irrelevant: the convex combination of a vector
    with itself is the vector. Tight tolerance (``atol = 1e-7``) because
    the only error source is float32 round-off.
    """
    rgb = tiny_batch["rgb"]
    # Identical clone — same values, same dtype, same device.
    flow = rgb.clone()
    fusion = AttentionFusion(dim=rgb.shape[-1])

    fused = fusion(rgb, flow)

    torch.testing.assert_close(fused, rgb, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(fused, flow, atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# Gradient flow to both inputs (§14.1 bullet 3)
# ---------------------------------------------------------------------------


def test_gradients_flow_to_both_inputs(
    tiny_batch: dict[str, torch.Tensor],
) -> None:
    """Both ``rgb`` and ``flow`` must receive non-trivial gradient.

    Strategy: tag each input with ``requires_grad_(True)``, sum the
    fused output, backprop, and assert each input got a populated
    ``.grad`` tensor with at least one non-zero entry.
    """
    rgb = tiny_batch["rgb"].clone().detach().requires_grad_(True)
    flow = tiny_batch["flow"].clone().detach().requires_grad_(True)
    fusion = AttentionFusion(dim=rgb.shape[-1])

    fused = fusion(rgb, flow)
    fused.sum().backward()

    assert rgb.grad is not None, "no gradient propagated to rgb input"
    assert flow.grad is not None, "no gradient propagated to flow input"
    assert rgb.grad.shape == rgb.shape
    assert flow.grad.shape == flow.shape
    # Every dimension should see some gradient signal — not just a few
    # lucky entries. A non-trivial fraction must be non-zero.
    assert torch.any(rgb.grad != 0.0), "rgb gradient is identically zero"
    assert torch.any(flow.grad != 0.0), "flow gradient is identically zero"
    assert torch.isfinite(rgb.grad).all(), "rgb gradient contains NaN/inf"
    assert torch.isfinite(flow.grad).all(), "flow gradient contains NaN/inf"


# ---------------------------------------------------------------------------
# Gate range — α ∈ [0, 1] (§14.1 bullet 4, §7.3 invariant 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_gate_stays_in_unit_interval(
    tiny_batch: dict[str, torch.Tensor], scale: float,
) -> None:
    """``alpha`` from ``forward_with_gate`` must be elementwise in ``[0, 1]``.

    The sigmoid keeps it there mathematically; this test guards against
    accidentally swapping in a non-bounded nonlinearity (e.g. tanh) in a
    refactor. ``scale`` exaggerates the inputs so a buggy non-sigmoid
    gate produces obviously out-of-range values.
    """
    rgb = tiny_batch["rgb"] * scale
    flow = tiny_batch["flow"] * scale
    fusion = AttentionFusion(dim=rgb.shape[-1])

    fused, alpha = fusion.forward_with_gate(rgb, flow)

    assert alpha.shape == rgb.shape, (
        f"alpha shape {tuple(alpha.shape)} must equal rgb shape {tuple(rgb.shape)}"
    )
    # Strict bounds: every element in [0, 1]. Use min/max instead of all()
    # so a failing test surfaces the worst offender's value.
    a_min = float(alpha.min().item())
    a_max = float(alpha.max().item())
    assert 0.0 <= a_min, f"alpha min = {a_min} < 0"
    assert a_max <= 1.0, f"alpha max = {a_max} > 1"

    # ``fused`` from ``forward_with_gate`` must match ``forward``'s output
    # for the same inputs — guards against the two paths drifting.
    fused_plain = fusion(rgb, flow)
    torch.testing.assert_close(fused, fused_plain, atol=1e-6, rtol=1e-6)
