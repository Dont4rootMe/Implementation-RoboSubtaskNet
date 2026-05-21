"""Transition-aware loss for RoboSubtaskNet.

Implements the bilinear forbidden-mass formulation from Section 9.3 of the
implementation plan: penalize predicted probability mass that flows between
labels (c, c') that the task grammar forbids.

For each consecutive frame pair t -> t+1, the per-pair penalty is::

    L_t = sum_{c, c'} p_{t, c} * p_{t+1, c'} * M_{c, c'}

where ``M_{c, c'} = 1`` if the ordered transition (c, c') is *not* in the
allowed set, else 0. The loss is differentiable and is exactly zero when the
predicted distributions only place joint mass on allowed transitions.

Per Section 16.2, variable-length videos must be masked in every loss
component, so this module also supports a per-frame validity mask.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class TransitionLoss(nn.Module):
    """Bilinear transition-aware penalty over forbidden label pairs.

    Construct from a boolean ``[C, C]`` matrix where ``allowed_transitions[c, c']``
    is ``True`` iff the ordered transition (c -> c') is permitted by the task
    grammar (see Section 9.5 of the implementation plan). The diagonal of
    ``allowed_transitions`` MUST be ``True`` for sub-tasks that can persist
    across consecutive frames; otherwise the loss will heavily penalize the
    very common (c, c) self-transition and training will diverge.

    The forbidden mask ``M = ~allowed_transitions`` is registered as a buffer,
    so it is moved alongside the module by ``.to(device)`` / ``.cuda()`` and
    saved/loaded with ``state_dict``.

    Args:
        allowed_transitions: bool tensor of shape ``[C, C]``. ``True`` at
            position ``(c, c')`` iff that ordered transition is allowed.

    Raises:
        ValueError: if ``allowed_transitions`` is not a ``bool`` tensor, not
            square, or has fewer than 2 classes.
    """

    forbidden: torch.Tensor  # type hint for the registered buffer

    def __init__(self, allowed_transitions: torch.Tensor) -> None:
        super().__init__()

        if not isinstance(allowed_transitions, torch.Tensor):
            raise ValueError(
                "allowed_transitions must be a torch.Tensor, got "
                f"{type(allowed_transitions).__name__}"
            )
        if allowed_transitions.dtype != torch.bool:
            raise ValueError(
                "allowed_transitions must have dtype=torch.bool, got "
                f"{allowed_transitions.dtype}"
            )
        if allowed_transitions.ndim != 2 or (
            allowed_transitions.shape[0] != allowed_transitions.shape[1]
        ):
            raise ValueError(
                "allowed_transitions must be a square [C, C] matrix, got "
                f"shape {tuple(allowed_transitions.shape)}"
            )
        if allowed_transitions.shape[0] < 2:
            raise ValueError(
                "allowed_transitions must have C >= 2, got "
                f"C={allowed_transitions.shape[0]}"
            )

        forbidden = (~allowed_transitions).float()
        self.register_buffer("forbidden", forbidden)  # [C, C]

        self._logged_summary: bool = False

    def _log_summary_once(self) -> None:
        """Emit a one-line diagnostic the first time forward is called."""
        if self._logged_summary:
            return
        c = int(self.forbidden.shape[0])
        total = c * c
        n_forbidden = int(self.forbidden.sum().item())
        n_allowed = total - n_forbidden
        density = n_allowed / float(total)
        logger.info(
            "TransitionLoss: C=%d, allowed=%d, forbidden=%d, density=%.4f",
            c,
            n_allowed,
            n_forbidden,
            density,
        )
        self._logged_summary = True

    def forward(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the transition-aware penalty.

        Args:
            logits: Tensor of shape ``[B, C, T]`` (pre-softmax class scores).
            mask: Optional tensor of shape ``[B, T]`` with 1 at valid frames
                and 0 at padding. When provided, only consecutive pairs where
                both endpoints are valid contribute to the loss.

        Returns:
            Scalar tensor with the per-pair-averaged forbidden joint mass.
        """
        self._log_summary_once()

        # Softmax along the class dimension -> p[b, c, t]
        p = F.softmax(logits, dim=1)
        # Consecutive pairs along time
        p_t = p[:, :, :-1]   # [B, C, T-1]
        p_tp1 = p[:, :, 1:]  # [B, C, T-1]

        # joint_forbidden[b, t] = sum_{c, c'} p_t[b,c,t] * p_tp1[b,c',t] * forbidden[c,c']
        joint_forbidden = torch.einsum(
            "bct,bdt,cd->bt", p_t, p_tp1, self.forbidden
        )

        if mask is not None:
            # A pair (t, t+1) is valid only when both endpoints are valid.
            m = mask[:, 1:] * mask[:, :-1]
            # Clamp denominator to >= 1 for numerical safety (no NaN on
            # all-padded batches).
            denom = m.sum().clamp(min=1.0)
            return (joint_forbidden * m).sum() / denom

        return joint_forbidden.mean()
