"""RoboSubtaskNet: multi-stage TCN with attention-fused two-stream features.

The model first fuses per-frame RGB and optical-flow features via a learned
per-channel gate (``AttentionFusion``), then refines a frame-level sub-task
classification through ``num_stages`` of a Fibonacci-dilated single-stage TCN.

Stage 1 ingests the fused features (``[B, D, T]``). Each subsequent refinement
stage ingests the softmax probabilities of the previous stage's logits — the
standard MS-TCN refinement scheme. The model returns the per-stage logits as a
list so the composite loss can supervise every stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AttentionFusion
from .tcn import SingleStageTCN


class RoboSubtaskNet(nn.Module):
    """Attention-fused, Fibonacci-dilated multi-stage TCN for sub-task segmentation.

    Architecture (per Section 8.3 of the implementation plan):
        1. ``AttentionFusion`` combines RGB and flow streams of shape ``[B, T, D]``
           into a single fused stream of the same shape.
        2. ``stage_1`` is a ``SingleStageTCN`` mapping ``feature_dim`` -> ``num_classes``.
        3. ``refinement`` is a stack of ``num_stages - 1`` ``SingleStageTCN`` modules,
           each mapping ``num_classes`` -> ``num_classes`` on softmax probabilities of
           the previous stage's logits.

    The multi-stage design lets later stages smooth and re-segment over a long
    receptive field (Fibonacci dilations give RF ~463 frames at L=10), reducing
    over-segmentation that a single stage alone tends to produce.
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        feature_dim: int = 1024,
        hidden_dim: int = 64,
        num_classes: int = 9,
        dropout: float = 0.5,
        fusion_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.fusion = AttentionFusion(dim=feature_dim, hidden=fusion_hidden)
        self.stage_1 = SingleStageTCN(
            num_layers, feature_dim, hidden_dim, num_classes, dropout
        )
        self.refinement = nn.ModuleList(
            [
                SingleStageTCN(num_layers, num_classes, hidden_dim, num_classes, dropout)
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> list[torch.Tensor]:
        """Run fusion + multi-stage refinement.

        Args:
            rgb:  Per-frame RGB features of shape ``[B, T, D]``.
            flow: Per-frame optical-flow features of shape ``[B, T, D]``.

        Returns:
            List of ``num_stages`` logit tensors, each of shape ``[B, C, T]``.
            The list is ordered from stage 1 (raw fused features) to the final
            refinement stage. The composite loss supervises every entry.
        """
        # rgb, flow: [B, T, D]
        fused = self.fusion(rgb, flow)   # [B, T, D]
        fused = fused.transpose(1, 2)    # [B, D, T] for Conv1d
        outputs: list[torch.Tensor] = [self.stage_1(fused)]
        for stage in self.refinement:
            outputs.append(stage(F.softmax(outputs[-1], dim=1)))
        return outputs  # list of [B, C, T]

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Return per-frame sub-task labels from the final refinement stage.

        Args:
            rgb:  Per-frame RGB features of shape ``[B, T, D]``.
            flow: Per-frame optical-flow features of shape ``[B, T, D]``.

        Returns:
            Integer label tensor of shape ``[B, T]`` (argmax of the last stage's
            softmax distribution over classes).
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(rgb, flow)[-1]  # [B, C, T]
            probs = F.softmax(logits, dim=1)
            labels = probs.argmax(dim=1)          # [B, T]
        finally:
            if was_training:
                self.train()
        return labels

    def count_parameters(self) -> int:
        """Number of trainable parameters — handy for logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
