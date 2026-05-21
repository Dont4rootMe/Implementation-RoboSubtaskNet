"""Truncated MSE smoothing loss (T-MSE) for action segmentation.

Reference: IMPLEMENTATION_PLAN.md Section 9.2. Penalizes large
log-probability jumps between consecutive frames, clipped at ``tau``
to avoid pathological gradients at class boundaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TruncatedMSELoss(nn.Module):
    r"""Truncated mean-squared error over per-class log-probability deltas.

    Computes

    .. math::
        \mathcal{L}_{\mathrm{T\text{-}MSE}}
            = \frac{1}{T C} \sum_{t, c} \tilde\Delta_{t,c}^{2},
        \quad
        \tilde\Delta_{t,c}
            = \min\!\bigl(|\log p_{t,c} - \log p_{t-1,c}|,\, \tau\bigr).

    Variable-length sequences are supported via an optional ``mask``;
    only frame pairs where *both* endpoints are valid contribute, per
    the masking requirement in Section 16.2.

    Args:
        tau: Truncation threshold in log-probability units (default ``4.0``,
            matching the MS-TCN/RoboSubtaskNet paper default).
    """

    def __init__(self, tau: float = 4.0) -> None:
        super().__init__()
        self.tau = float(tau)

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the truncated MSE smoothness loss.

        Args:
            logits: Per-frame class logits of shape ``[B, C, T]``.
            mask: Optional validity mask of shape ``[B, T]`` with ``1``
                for valid frames and ``0`` for padded ones.

        Returns:
            Scalar loss tensor (zero-dimensional).
        """
        # log_p: [B, C, T]
        log_p = F.log_softmax(logits, dim=1)
        # delta: [B, C, T-1]
        delta = log_p[:, :, 1:] - log_p[:, :, :-1]
        delta = torch.clamp(delta.abs(), max=self.tau)
        sq = delta.pow(2)  # [B, C, T-1]

        if mask is not None:
            # Gate frame pairs where both endpoints are valid: [B, T-1].
            m = mask[:, 1:] * mask[:, :-1]
            sq = sq * m.unsqueeze(1)
            # Preserve the plan's formula: denominator = m.sum() * C,
            # where C == sq.size(1). Clamp to >=1 to guard against
            # degenerate masks (e.g., all-padding batch elements).
            denom = (m.sum() * sq.size(1)).clamp(min=1)
            return sq.sum() / denom
        return sq.mean()
