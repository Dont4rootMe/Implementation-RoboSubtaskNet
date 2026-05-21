"""Composite training loss for RoboSubtaskNet.

Implements the training objective from Section 9 of the implementation plan:

    L = sum_{s=1..S} ( L_CE^(s) + lam * L_TMSE^(s) + gam * L_Trans^(s) )

Each per-stage term combines a cross-entropy classification loss with the
truncated MSE smoothing penalty (Section 9.2) and the transition-aware
grammar penalty (Section 9.3). The composite loss aggregates over all
multi-stage TCN outputs and returns a scalar plus a per-component dict
for logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tmse import TruncatedMSELoss
from .transition import TransitionLoss


class CompositeLoss(nn.Module):
    """Composite multi-stage loss: CE + lam * T-MSE + gam * Transition.

    The total loss is the sum across all stages of the multi-stage TCN of:

        CE(out_s, labels) + lam * TMSE(out_s, mask) + gam * Trans(out_s, mask)

    Args:
        num_classes: Number of sub-task classes (C).
        allowed_transitions: Bool tensor of shape [C, C] where entry (c, c')
            is True iff the ordered transition c -> c' is allowed by the
            task grammar (self-transitions on the diagonal must be True).
        lam: Weight on the truncated MSE smoothing term (paper default 0.15).
        gam: Weight on the transition-aware grammar penalty (grid-searched
            in [0.1, 0.3]).
        tau: Truncation threshold (in log-prob units) for the T-MSE term.
        ce_ignore_index: Label value treated as padding by the
            cross-entropy term (defaults to -100, matching PyTorch).
    """

    def __init__(
        self,
        num_classes: int,
        allowed_transitions: torch.Tensor,
        lam: float = 0.15,
        gam: float = 0.15,
        tau: float = 4.0,
        ce_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)
        self.tmse = TruncatedMSELoss(tau=tau)
        self.trans = TransitionLoss(allowed_transitions)
        self.lam, self.gam = lam, gam

    def forward(
        self,
        stage_outputs: list[torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the multi-stage composite loss.

        Args:
            stage_outputs: List of per-stage logit tensors, each of shape
                [B, C, T]. The list length equals the number of refinement
                stages S in the TCN.
            labels: Ground-truth label tensor of shape [B, T]. Padded
                frames should carry ``ce_ignore_index``.
            mask: Optional binary tensor of shape [B, T] where 1 marks
                valid frames and 0 marks padding. Passed to the T-MSE and
                transition terms for proper averaging.

        Returns:
            A tuple ``(loss, components)`` where ``loss`` is the scalar
            tensor to backpropagate and ``components`` aggregates the
            CE / T-MSE / Trans contributions (as floats) summed across
            stages for logging.
        """
        loss: torch.Tensor = torch.zeros((), device=labels.device)
        components: dict[str, float] = {"ce": 0.0, "tmse": 0.0, "trans": 0.0}
        for out in stage_outputs:
            ce = self.ce(out, labels)
            tm = self.tmse(out, mask)
            tr = self.trans(out, mask)
            loss = loss + ce + self.lam * tm + self.gam * tr
            components["ce"] += ce.detach().item()
            components["tmse"] += tm.detach().item()
            components["trans"] += tr.detach().item()
        return loss, components
