"""Attention-gated fusion of RGB and optical-flow feature streams.

Implements Section 7 of the RoboSubtaskNet implementation plan.

Given per-time-step RGB and flow features ``f_rgb_t, f_flow_t`` in R^D, a
shallow gate network produces a per-frame, per-channel mixing coefficient
``alpha(t) in [0, 1]^D``:

    alpha(t) = sigmoid(W_a [f_rgb_t; f_flow_t] + b_a)
    f_fused_t = alpha(t) * f_rgb_t + (1 - alpha(t)) * f_flow_t

``alpha = 1`` selects the RGB stream and ``alpha = 0`` selects the flow
stream, independently for each feature dimension at each timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """Per-frame, per-channel gated fusion of two equally-sized feature streams.

    The fusion produces ``alpha(t) in [0, 1]^D`` from the concatenation of the
    two input streams and uses it to convex-combine them channelwise:

        alpha = sigmoid(gate(concat(rgb, flow)))
        fused = alpha * rgb + (1 - alpha) * flow

    Args:
        dim: Per-stream feature dimensionality ``D`` (default ``1024``, matching
            the I3D backbone used by the paper).
        hidden: If ``None`` (default), the gate is a single ``nn.Linear(2*D, D)``
            as in the paper's "fully connected shallow layer" description. If an
            integer, the gate is a 2-layer MLP
            ``Linear(2*D, hidden) -> GELU -> Linear(hidden, D)``, a defensible
            variant suggested in Section 7.1 for ablation.

    Diagnostic invariants (Section 7.3):
        1. If both inputs are equal (``rgb == flow``), the convex combination
           collapses regardless of ``alpha``: the fused output equals either
           input exactly, so ``||fused - rgb|| == 0``.
        2. If ``gate.weight`` is zero and ``gate.bias`` is zero (single-Linear
           variant) the sigmoid evaluates to ``0.5`` everywhere and the fused
           feature equals the elementwise mean ``0.5 * (rgb + flow)``.
        3. After training, the per-sub-task mean ``alpha`` should qualitatively
           match Table III of the paper (flow-dominant for ``reach``, ``move``,
           ``wipe``; RGB-dominant for ``pick``, ``place``, ``pour``). Use
           :meth:`forward_with_gate` to log ``alpha`` for this check.
    """

    def __init__(self, dim: int = 1024, hidden: int | None = None) -> None:
        super().__init__()
        self.dim: int = dim
        self.hidden: int | None = hidden
        self.gate: nn.Module
        if hidden is None:
            self.gate = nn.Linear(2 * dim, dim)
        else:
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Fuse RGB and flow features.

        Args:
            rgb: Tensor of shape ``[B, T, D]``.
            flow: Tensor of shape ``[B, T, D]``.

        Returns:
            Fused features of shape ``[B, T, D]``.
        """
        alpha = torch.sigmoid(self.gate(torch.cat([rgb, flow], dim=-1)))
        return alpha * rgb + (1.0 - alpha) * flow

    def forward_with_gate(
        self, rgb: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse RGB and flow features and also return the gate ``alpha``.

        Useful for diagnostics and for matching against Table III of the paper.

        Args:
            rgb: Tensor of shape ``[B, T, D]``.
            flow: Tensor of shape ``[B, T, D]``.

        Returns:
            A tuple ``(fused, alpha)``, both of shape ``[B, T, D]``. ``alpha``
            lies elementwise in ``[0, 1]``.
        """
        alpha = torch.sigmoid(self.gate(torch.cat([rgb, flow], dim=-1)))
        fused = alpha * rgb + (1.0 - alpha) * flow
        return fused, alpha
