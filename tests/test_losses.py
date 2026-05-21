"""Unit tests for the composite loss and its component terms.

Coverage maps to §14.1's ``test_losses.py`` bullets:

* CE branch of ``CompositeLoss`` matches ``F.cross_entropy`` on an
  unmasked input.
* ``TruncatedMSELoss`` is exactly zero when all log-probabilities are
  identical across time (no jumps to penalize).
* ``TransitionLoss`` is zero when predictions are constant across time
  (only self-transitions occur, which are allowed by construction).
* ``TransitionLoss`` is positive when predictions switch between two
  forbidden labels.
* ``CompositeLoss(...)`` returns a tuple ``(loss, components_dict)``
  with the documented keys ``ce``, ``tmse``, ``trans``.

The numerical tests prefer ``torch.testing.assert_close`` with explicit
tolerances (atol=1e-6, rtol=1e-5) — comfortably tighter than any signal
the loss modules can legitimately produce, and loose enough to survive
float32 round-off.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from robosubtasknet.losses import (
    CompositeLoss,
    TransitionLoss,
    TruncatedMSELoss,
)


# ---------------------------------------------------------------------------
# Cross-entropy branch matches F.cross_entropy
# ---------------------------------------------------------------------------


def test_ce_branch_matches_torch_cross_entropy(
    tiny_logits: torch.Tensor,
    tiny_batch: dict[str, torch.Tensor],
    tiny_allowed_transitions: torch.Tensor,
    num_classes: int,
) -> None:
    """A single-stage call to ``CompositeLoss`` with ``lam = gam = 0`` reduces
    to plain cross-entropy on the per-stage logits — and ``components['ce']``
    must equal ``F.cross_entropy`` to within float32 round-off.
    """
    labels = tiny_batch["labels"]
    loss_fn = CompositeLoss(
        num_classes=num_classes,
        allowed_transitions=tiny_allowed_transitions,
        lam=0.0,
        gam=0.0,
    )
    # Pass logits as a single-stage list to isolate the CE contribution.
    total, components = loss_fn([tiny_logits], labels)

    expected = F.cross_entropy(tiny_logits, labels)
    # ``components['ce']`` is a float (.item()'d inside CompositeLoss),
    # so wrap it in a tensor for ``assert_close``.
    torch.testing.assert_close(
        torch.tensor(components["ce"]),
        expected.detach(),
        atol=1e-6,
        rtol=1e-5,
    )
    # With lam = gam = 0 and one stage, the total loss is the CE term itself.
    torch.testing.assert_close(
        total.detach(), expected.detach(), atol=1e-6, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# T-MSE = 0 for time-constant log-probs
# ---------------------------------------------------------------------------


def test_tmse_zero_for_time_constant_logits(num_classes: int) -> None:
    """``TruncatedMSELoss`` operates on log-softmax deltas; if logits are
    identical across time, those deltas vanish identically.

    We construct ``logits`` by drawing a per-class vector and broadcasting
    it across the time axis — the resulting log-softmax is the same at
    every frame, so every ``delta`` in the loss is zero.
    """
    B, T = 2, 16
    base = torch.randn(B, num_classes, 1)         # one slice
    logits = base.expand(B, num_classes, T).contiguous()

    loss = TruncatedMSELoss(tau=4.0)
    val = loss(logits)

    torch.testing.assert_close(
        val.detach(), torch.tensor(0.0), atol=1e-7, rtol=0.0
    )


def test_tmse_positive_when_logits_vary(num_classes: int) -> None:
    """Sanity check: a non-constant logit sequence yields a *positive* T-MSE,
    so the zero-case test above isn't accidentally trivial."""
    torch.manual_seed(0)
    B, T = 2, 16
    logits = torch.randn(B, num_classes, T)

    loss = TruncatedMSELoss(tau=4.0)
    val = loss(logits)

    assert val.item() > 0.0, f"expected T-MSE > 0 on random logits, got {val.item()}"


# ---------------------------------------------------------------------------
# Transition loss zero when predictions are constant
# ---------------------------------------------------------------------------


def test_transition_loss_zero_for_constant_predictions(
    num_classes: int, tiny_allowed_transitions: torch.Tensor,
) -> None:
    """``TransitionLoss`` is built from the *forbidden* mask; the diagonal
    of ``allowed_transitions`` is all-True (every class can self-persist).

    If predictions are constant across time the only "transition" present
    in the joint distribution is ``(c, c)`` — an allowed self-transition.
    Forbidden mass is exactly zero.
    """
    B, T = 2, 16
    # Make the prediction land confidently on class 0 by pushing one logit
    # huge and the rest down. After softmax we get a near-one-hot — the
    # peak is still bilinearly close to (0, 0).
    logits = torch.full((B, num_classes, T), -1e3)
    logits[:, 0, :] = 1e3

    loss_fn = TransitionLoss(tiny_allowed_transitions)
    val = loss_fn(logits)

    torch.testing.assert_close(
        val.detach(), torch.tensor(0.0), atol=1e-6, rtol=0.0
    )


def test_transition_loss_positive_when_switching_between_forbidden_labels(
    num_classes: int,
) -> None:
    """Pick a transition pair that is explicitly *not* allowed and force
    predictions to flip between them — the loss must rise above zero.

    Construction (with ``num_classes = 5``):

    * Allow only self-transitions (identity matrix). Nothing else allowed.
    * So every off-diagonal pair, including the (3, 4) pair we flip
      between, is forbidden.
    * Even frames predict label 3 with near-certainty; odd frames
      predict label 4 with near-certainty. Every consecutive pair is
      either (3, 4) or (4, 3) — both forbidden — so the loss is positive.
    """
    allowed = torch.eye(num_classes, dtype=torch.bool)
    # Confirm the pair we intend to flip between is genuinely forbidden.
    assert not bool(allowed[3, 4]), "test invariant: (3, 4) must be forbidden"
    assert not bool(allowed[4, 3]), "test invariant: (4, 3) must be forbidden"

    B, T = 1, 8  # T must be even for a clean alternating pattern
    assert T % 2 == 0
    logits = torch.full((B, num_classes, T), -1e3)
    # Even frames -> class 3, odd frames -> class 4.
    logits[:, 3, 0::2] = 1e3
    logits[:, 4, 1::2] = 1e3

    loss_fn = TransitionLoss(allowed)
    val = loss_fn(logits)

    assert val.item() > 0.0, (
        f"expected transition loss > 0 when alternating between two "
        f"forbidden labels; got {val.item()}"
    )
    # For near-one-hot predictions, the per-pair forbidden mass should
    # be very close to 1.0; with mean over (T-1) pairs the loss is
    # near 1.0 too. Use a generous floor to keep the test robust to
    # softmax saturation rounding.
    assert val.item() > 0.5, (
        f"forbidden-transition loss unexpectedly small ({val.item()}); "
        f"expected ~1.0 for near-one-hot predictions"
    )
    # Should not exceed the trivial upper bound (1.0 for a per-pair
    # average of probability mass).
    assert val.item() <= 1.0 + 1e-5, (
        f"transition loss exceeded its upper bound: {val.item()}"
    )


# ---------------------------------------------------------------------------
# CompositeLoss return-tuple contract
# ---------------------------------------------------------------------------


def test_composite_loss_returns_tuple_with_expected_keys(
    tiny_logits: torch.Tensor,
    tiny_batch: dict[str, torch.Tensor],
    tiny_allowed_transitions: torch.Tensor,
    num_classes: int,
) -> None:
    """``CompositeLoss.forward`` must return ``(loss, components_dict)``
    with keys ``ce``, ``tmse``, ``trans``.

    The plan (Section 9.4) makes this contract load-bearing: the trainer
    iterates ``components.items()`` for TensorBoard logging.
    """
    labels = tiny_batch["labels"]
    mask = tiny_batch["mask"]
    loss_fn = CompositeLoss(
        num_classes=num_classes,
        allowed_transitions=tiny_allowed_transitions,
        lam=0.15,
        gam=0.15,
        tau=4.0,
    )
    stage_outputs = [tiny_logits, tiny_logits.clone()]

    result = loss_fn(stage_outputs, labels, mask)

    assert isinstance(result, tuple), (
        f"CompositeLoss must return a tuple; got {type(result).__name__}"
    )
    assert len(result) == 2, (
        f"tuple length should be 2 (loss, components); got {len(result)}"
    )
    total, components = result
    assert isinstance(total, torch.Tensor), (
        f"first tuple element must be a torch.Tensor; got {type(total).__name__}"
    )
    assert total.ndim == 0, (
        f"loss must be a scalar (0-d) tensor; got shape {tuple(total.shape)}"
    )
    assert torch.isfinite(total), f"loss is non-finite: {total.item()}"

    assert isinstance(components, dict), (
        f"second tuple element must be a dict; got {type(components).__name__}"
    )
    for key in ("ce", "tmse", "trans"):
        assert key in components, (
            f"components dict missing required key '{key}'; "
            f"present keys: {sorted(components.keys())}"
        )
        # Plan stores per-key floats (``.detach().item()``), but a tensor
        # form is also defensible — accept either as long as it's finite.
        v = components[key]
        if isinstance(v, torch.Tensor):
            v = v.detach().item()
        assert isinstance(v, (int, float)), (
            f"components['{key}'] should be a scalar number; got {type(v).__name__}"
        )
        assert math.isfinite(v), f"components['{key}'] = {v} is not finite"
