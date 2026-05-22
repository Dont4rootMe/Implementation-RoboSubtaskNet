"""Losses for the class-agnostic boundary detector (Stage 1).

Stage 1 of the RoboSubtaskNet pipeline emits per-frame boundary logits
of shape ``[B, 1, T]`` from a multi-stage refinement model. Boundary
supervision is *sparse* (most frames carry label 0), so this module
provides:

* :func:`make_soft_boundary_labels` -- Gaussian-soft targets that turn
  isolated unit spikes at boundary frames into smooth peaks of width
  ``sigma`` so that nearby predictions are partially rewarded.
* :class:`BoundaryBCELoss` -- sigmoid binary cross-entropy with optional
  positive-class weighting (``pos_weight``) to counter the class imbalance.
* :class:`BoundarySmoothnessLoss` -- truncated MSE on consecutive sigmoid
  logits, the binary analogue of the T-MSE term in MS-TCN.
* :class:`CompositeBoundaryLoss` -- aggregates BCE + smoothness across
  every refinement stage and returns a logging dict.

All loss components honour an optional ``mask`` of shape ``[B, T]``
(``1`` = valid frame, ``0`` = pad). Denominators are clamped to
``>= 1`` to guard against degenerate masks (e.g. all-padding batch
elements).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_soft_boundary_labels(
    boundaries: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Convert binary boundary spikes to Gaussian-soft labels.

    Each ``1`` in ``boundaries`` is convolved with a 1-D Gaussian of
    standard deviation ``sigma``. The resulting peaks have value
    ``1`` at the true boundary and decay smoothly to ``0`` over a
    window of roughly ``6 * sigma`` frames. The output is clamped to
    ``[0, 1]`` so overlapping peaks from closely-spaced boundaries
    saturate at unity rather than summing past it.

    Args:
        boundaries: Binary boundary tensor of shape ``[B, T]`` with
            entries in ``{0, 1}`` (floating-point or integer dtype).
        sigma: Standard deviation of the Gaussian kernel in frames.
            Must be positive. When ``sigma <= 0`` (or the kernel
            collapses to a single tap) the input is returned
            unchanged (cast to float).

    Returns:
        Soft-label tensor of shape ``[B, T]`` with values in ``[0, 1]``,
        peaked at the original boundary positions.
    """
    if boundaries.dim() != 2:
        raise ValueError(
            f"boundaries must be [B, T]; got shape {tuple(boundaries.shape)}"
        )

    soft = boundaries.float()
    sigma_f = float(sigma)
    if sigma_f <= 0.0:
        return soft.clamp(0.0, 1.0)

    # Kernel half-width = ceil(3 * sigma), giving a ~6-sigma window.
    half = max(1, int(math.ceil(3.0 * sigma_f)))
    radius = torch.arange(-half, half + 1, device=soft.device, dtype=soft.dtype)
    kernel = torch.exp(-(radius**2) / (2.0 * sigma_f * sigma_f))
    # Normalise so the peak (at the spike location) is exactly 1.0;
    # this preserves the interpretation of soft labels as a confidence
    # in [0, 1] regardless of sigma.
    kernel = kernel / kernel.max().clamp(min=1e-12)
    kernel = kernel.view(1, 1, -1)

    # F.conv1d expects [B, C, T]; treat T as the spatial dim.
    x = soft.unsqueeze(1)  # [B, 1, T]
    pad = half
    smoothed = F.conv1d(x, kernel, padding=pad)
    smoothed = smoothed.squeeze(1)  # [B, T]
    return smoothed.clamp(0.0, 1.0)


class BoundaryBCELoss(nn.Module):
    """Sigmoid BCE on per-frame boundary logits with optional ``pos_weight``.

    The loss is computed against Gaussian-soft targets and averaged over
    valid frames only (``mask == 1``).

    Args:
        pos_weight: Multiplicative weight applied to the positive class
            in the BCE term. Larger values up-weight rare boundary
            frames; ``None`` disables re-weighting (equivalent to ``1``).
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight_value: float | None = (
            None if pos_weight is None else float(pos_weight)
        )

    def forward(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked, optionally pos-weighted BCE loss.

        Args:
            logits: Boundary logits of shape ``[B, 1, T]``.
            soft_labels: Gaussian-smoothed targets of shape ``[B, T]``
                with values in ``[0, 1]``.
            mask: Optional validity mask of shape ``[B, T]`` (``1`` =
                valid, ``0`` = pad).

        Returns:
            Scalar loss tensor.
        """
        if logits.dim() != 3 or logits.size(1) != 1:
            raise ValueError(
                f"logits must be [B, 1, T]; got shape {tuple(logits.shape)}"
            )
        # Reduce to [B, T] for elementwise BCE.
        logits2d = logits.squeeze(1)
        target = soft_labels.to(dtype=logits2d.dtype)

        pos_weight_tensor: torch.Tensor | None = None
        if self.pos_weight_value is not None:
            pos_weight_tensor = torch.tensor(
                self.pos_weight_value,
                dtype=logits2d.dtype,
                device=logits2d.device,
            )

        # Per-frame BCE -- we do our own masked averaging below.
        per_frame = F.binary_cross_entropy_with_logits(
            logits2d,
            target,
            pos_weight=pos_weight_tensor,
            reduction="none",
        )

        if mask is None:
            return per_frame.mean()

        m = mask.to(dtype=per_frame.dtype)
        denom = m.sum().clamp(min=1)
        return (per_frame * m).sum() / denom


class BoundarySmoothnessLoss(nn.Module):
    r"""Truncated MSE on consecutive sigmoid logits.

    Penalises sharp frame-to-frame changes in the predicted boundary
    probability, with truncation at ``tau`` (in sigmoid-probability
    units) so genuine boundary transitions are not over-penalised:

    .. math::
        \mathcal{L}_{\mathrm{smooth}}
            = \frac{1}{\sum m_{t,t+1}}
              \sum_{t} m_{t,t+1}
                \min\!\bigl(|\sigma(z_{t+1}) - \sigma(z_{t})|,\,\tau\bigr)^2.

    Args:
        tau: Truncation threshold on the absolute consecutive-frame
            delta of sigmoid logits (default ``4.0``, mirroring the
            MS-TCN T-MSE default).
    """

    def __init__(self, tau: float = 4.0) -> None:
        super().__init__()
        self.tau = float(tau)

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the truncated-MSE smoothness loss.

        Args:
            logits: Boundary logits of shape ``[B, 1, T]``.
            mask: Optional validity mask of shape ``[B, T]``.

        Returns:
            Scalar loss tensor.
        """
        if logits.dim() != 3 or logits.size(1) != 1:
            raise ValueError(
                f"logits must be [B, 1, T]; got shape {tuple(logits.shape)}"
            )
        probs = torch.sigmoid(logits.squeeze(1))  # [B, T]
        delta = probs[:, 1:] - probs[:, :-1]      # [B, T-1]
        delta = torch.clamp(delta.abs(), max=self.tau)
        sq = delta.pow(2)

        if mask is None:
            return sq.mean()

        # Gate frame pairs where both endpoints are valid: [B, T-1].
        m = mask.to(dtype=sq.dtype)
        pair_mask = m[:, 1:] * m[:, :-1]
        denom = pair_mask.sum().clamp(min=1)
        return (sq * pair_mask).sum() / denom


class CompositeBoundaryLoss(nn.Module):
    """Composite Stage-1 loss: BCE + smoothness across all refinement stages.

    For each stage output the total is::

        BCE(out_s, soft_labels) + lam_smooth * Smooth(out_s, mask)

    and the per-stage contributions are summed (matching the MS-TCN
    multi-stage objective). A per-component dict of detached floats is
    returned alongside the differentiable scalar for logging.

    Args:
        pos_weight: Positive-class weight forwarded to
            :class:`BoundaryBCELoss` (default ``10.0`` to counter the
            sparse positive distribution).
        lam_smooth: Weight on the smoothness term (default ``0.15``,
            matching the MS-TCN T-MSE coefficient).
        tau: Truncation threshold forwarded to
            :class:`BoundarySmoothnessLoss`.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        lam_smooth: float = 0.15,
        tau: float = 4.0,
    ) -> None:
        super().__init__()
        self.bce = BoundaryBCELoss(pos_weight=pos_weight)
        self.smooth = BoundarySmoothnessLoss(tau=tau)
        self.lam_smooth = float(lam_smooth)

    def forward(
        self,
        stage_outputs: list[torch.Tensor],
        soft_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Aggregate BCE + smoothness over all multi-stage outputs.

        Args:
            stage_outputs: List of per-stage logit tensors, each of
                shape ``[B, 1, T]``.
            soft_labels: Gaussian-soft boundary targets of shape
                ``[B, T]``.
            mask: Optional validity mask of shape ``[B, T]``.

        Returns:
            Tuple ``(loss, components)`` where ``loss`` is the scalar
            tensor to backpropagate and ``components`` is a dict with
            keys ``"bce"`` and ``"smooth"`` (summed across stages, as
            detached floats) suitable for step-level logging.
        """
        if len(stage_outputs) == 0:
            raise ValueError("stage_outputs must contain at least one tensor")

        device = stage_outputs[0].device
        loss: torch.Tensor = torch.zeros((), device=device)
        components: dict[str, float] = {"bce": 0.0, "smooth": 0.0}
        for out in stage_outputs:
            bce = self.bce(out, soft_labels, mask)
            sm = self.smooth(out, mask)
            loss = loss + bce + self.lam_smooth * sm
            components["bce"] += bce.detach().item()
            components["smooth"] += sm.detach().item()
        return loss, components
