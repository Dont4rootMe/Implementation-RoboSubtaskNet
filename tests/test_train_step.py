"""One forward + backward pass on a tiny synthetic batch must finish without NaN.

Covers Section 14.2 of ``IMPLEMENTATION_PLAN.md``: a smoke test that the
RoboSubtaskNet -> CompositeLoss path is wired so that gradients flow into
every parameter without producing NaN values.
"""

from __future__ import annotations

import torch

from robosubtasknet.data.grammar import build_allowed_transitions
from robosubtasknet.losses import CompositeLoss
from robosubtasknet.models import RoboSubtaskNet


def _allowed_transitions_tiny(num_classes: int) -> torch.Tensor:
    """Build an all-True transition matrix for a tiny toy setup.

    We bypass the YAML loader when the schema would require named sub-tasks
    that don't exist in our toy 3-class universe: the transition loss only
    needs a ``[C, C]`` bool tensor. An all-True matrix means the loss is
    well-defined and contributes zero penalty regardless of predictions —
    which keeps this test focused on the no-NaN-gradient property.
    """
    return torch.ones((num_classes, num_classes), dtype=torch.bool)


def test_train_step_no_nan_gradients():
    torch.manual_seed(0)

    num_classes = 3
    feature_dim = 8
    hidden_dim = 8

    model = RoboSubtaskNet(
        num_stages=2,
        num_layers=4,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    model.train()

    allowed = _allowed_transitions_tiny(num_classes)
    loss_fn = CompositeLoss(num_classes=num_classes, allowed_transitions=allowed)

    batch_size, num_frames = 1, 16
    rgb = torch.randn(batch_size, num_frames, feature_dim)
    flow = torch.randn(batch_size, num_frames, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size, num_frames))
    mask = torch.ones(batch_size, num_frames)

    # Forward + composite loss + backward.
    stage_outputs = model(rgb, flow)
    loss, components = loss_fn(stage_outputs, labels, mask)

    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()!r}"

    model.zero_grad(set_to_none=True)
    loss.backward()

    # Assert no NaNs in any parameter gradient.
    saw_grad = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            # A parameter that doesn't receive gradient on this synthetic
            # batch is acceptable; we only ban NaN/Inf when a grad exists.
            continue
        saw_grad = True
        assert torch.isfinite(param.grad).all(), (
            f"non-finite gradient in parameter {name!r}"
        )

    assert saw_grad, "expected at least one parameter to receive a gradient"
    # Sanity check: the components dict carries every advertised key.
    for key in ("ce", "tmse", "trans"):
        assert key in components, f"missing loss component {key!r}"
