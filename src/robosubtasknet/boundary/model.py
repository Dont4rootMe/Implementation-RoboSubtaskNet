"""Class-agnostic temporal boundary detector (Stage 1 of the 2-stage pipeline).

Reuses the existing :class:`AttentionFusion` and Fibonacci-dilated
:class:`SingleStageTCN` building blocks, but replaces the per-class
classification head with a binary per-frame boundary head. The model still
performs MS-TCN-style multi-stage refinement; each refinement stage operates
on the sigmoid of the previous stage's 1-channel logits (rather than the
softmax of multi-class logits, as in the original RoboSubtaskNet).

Outputs are raw logits ``[B, 1, T]`` per stage so a downstream BCE-with-logits
loss can supervise every stage. Use :meth:`predict_probs` at inference time to
obtain a smooth ``[B, T]`` boundary probability map for non-maximum suppression
or peak-picking.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from robosubtasknet.models.fusion import AttentionFusion
from robosubtasknet.models.tcn import SingleStageTCN


class BoundarySegmenter(nn.Module):
    """Class-agnostic temporal boundary detector.

    Reuses ``AttentionFusion`` + Fibonacci-dilated ``SingleStageTCN``. The
    final head of every stage is ``Conv1d(hidden_dim, 1)`` (inside
    ``SingleStageTCN``), producing per-frame boundary logits. Refinement
    stages take ``sigmoid`` of the previous stage's 1-channel logits as input.

    Architecture:
        1. ``AttentionFusion`` combines RGB and flow streams ``[B, T, D]``
           into a single fused stream of the same shape.
        2. Stage 1 is ``SingleStageTCN(num_layers, feature_dim, hidden_dim, 1, dropout)``
           on the fused stream transposed to ``[B, D, T]``.
        3. Each refinement stage is ``SingleStageTCN(num_layers, 1, hidden_dim, 1, dropout)``
           consuming ``sigmoid`` of the previous stage's logits.
        4. ``forward`` returns the list of per-stage logits, each ``[B, 1, T]``.

    Args:
        num_stages: Total number of stages (1 raw + ``num_stages - 1`` refinement).
        num_layers: Number of Fibonacci-dilated layers per stage.
        feature_dim: Per-stream input feature dimensionality ``D`` (e.g. 1024 for I3D).
        hidden_dim: Channel dimension of the residual stream inside each stage
            (MS-TCN convention is 64).
        dropout: Dropout probability inside each ``FibonacciDilatedLayer``.
        fusion_hidden: Hidden width for the fusion gate; ``None`` selects the
            single-Linear gate used by the paper.
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        feature_dim: int = 1024,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        fusion_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fusion = AttentionFusion(dim=feature_dim, hidden=fusion_hidden)
        self.stage_1 = SingleStageTCN(
            num_layers, feature_dim, hidden_dim, num_classes=1, dropout=dropout
        )
        self.refinement = nn.ModuleList(
            [
                SingleStageTCN(
                    num_layers, 1, hidden_dim, num_classes=1, dropout=dropout
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(
        self, rgb: torch.Tensor, flow: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run fusion + multi-stage boundary refinement.

        Args:
            rgb:  Per-frame RGB features of shape ``[B, T, D]``.
            flow: Per-frame optical-flow features of shape ``[B, T, D]``.

        Returns:
            List of ``num_stages`` logit tensors, each of shape ``[B, 1, T]``.
            Ordered from stage 1 (raw fused features) through final refinement.
            No sigmoid is applied; pair with BCE-with-logits during training.
        """
        # rgb, flow: [B, T, D]
        fused = self.fusion(rgb, flow)   # [B, T, D]
        fused = fused.transpose(1, 2)    # [B, D, T] for Conv1d
        outputs: list[torch.Tensor] = [self.stage_1(fused)]
        for stage in self.refinement:
            outputs.append(stage(torch.sigmoid(outputs[-1])))
        return outputs  # list of [B, 1, T]

    @torch.no_grad()
    def predict_probs(
        self, rgb: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Return per-frame boundary probabilities from the final stage.

        Args:
            rgb:  Per-frame RGB features of shape ``[B, T, D]``.
            flow: Per-frame optical-flow features of shape ``[B, T, D]``.

        Returns:
            Probability tensor of shape ``[B, T]``, ``sigmoid`` of the last
            stage's logits with the singleton channel dimension squeezed.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(rgb, flow)[-1]   # [B, 1, T]
            probs = torch.sigmoid(logits).squeeze(1)  # [B, T]
        finally:
            if was_training:
                self.train()
        return probs

    def count_parameters(self) -> int:
        """Number of trainable parameters — handy for logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
