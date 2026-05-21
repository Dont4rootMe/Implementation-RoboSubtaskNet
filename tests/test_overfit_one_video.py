"""Overfit a tiny RoboSubtaskNet on a single synthetic video.

Per Section 14.2 of ``IMPLEMENTATION_PLAN.md`` this is the most important
integration test: if the model cannot drive the per-frame cross-entropy
below 0.05 on a single video where the RGB stream is strongly correlated
with the labels, the data path or loss is broken.

Marked ``@pytest.mark.slow`` so CI can opt out cheaply.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from robosubtasknet.losses import CompositeLoss
from robosubtasknet.models import RoboSubtaskNet


@pytest.mark.slow
def test_overfit_one_video():
    torch.manual_seed(0)

    num_classes = 3
    feature_dim = 8
    hidden_dim = 16
    num_layers = 3
    num_stages = 2

    num_frames = 64

    # ------------------------------------------------------------------
    # Synthetic data with a strong RGB <-> label correlation.
    # ------------------------------------------------------------------
    # Build a label sequence that switches classes in contiguous chunks
    # (mirrors how real sub-task labels look) and tie each class to a
    # distinct mean RGB template. Flow is mild noise so the model has to
    # learn to lean on the RGB stream.
    chunk_size = num_frames // num_classes
    labels_list = []
    for c in range(num_classes):
        labels_list.append(torch.full((chunk_size,), c, dtype=torch.long))
    if chunk_size * num_classes < num_frames:
        labels_list.append(
            torch.full((num_frames - chunk_size * num_classes,), num_classes - 1, dtype=torch.long)
        )
    labels = torch.cat(labels_list, dim=0).unsqueeze(0)  # [1, T]

    class_means = torch.randn(num_classes, feature_dim) * 3.0
    rgb_frames = class_means[labels.squeeze(0)]                 # [T, D]
    rgb = (rgb_frames + 0.1 * torch.randn_like(rgb_frames)).unsqueeze(0)  # [1, T, D]
    flow = 0.1 * torch.randn(1, num_frames, feature_dim)
    mask = torch.ones(1, num_frames)

    # ------------------------------------------------------------------
    # Tiny model + composite loss with an all-True transition matrix.
    # Using all-True transitions keeps the test focused on the CE path
    # (the transition penalty is 0 when every transition is allowed).
    # ------------------------------------------------------------------
    model = RoboSubtaskNet(
        num_stages=num_stages,
        num_layers=num_layers,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    model.train()

    allowed = torch.ones((num_classes, num_classes), dtype=torch.bool)
    loss_fn = CompositeLoss(num_classes=num_classes, allowed_transitions=allowed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    final_ce = float("inf")
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        stage_outputs = model(rgb, flow)
        loss, _ = loss_fn(stage_outputs, labels, mask)
        loss.backward()
        optimizer.step()

        # Track the final-stage CE (this is what the assertion gates on).
        with torch.no_grad():
            final_logits = stage_outputs[-1]  # [B, C, T]
            final_ce = F.cross_entropy(final_logits, labels).item()

    assert final_ce < 0.05, (
        f"expected final-stage CE < 0.05 after 100 steps on a tiny synthetic "
        f"video, got {final_ce:.4f} — data path or loss is likely broken"
    )
