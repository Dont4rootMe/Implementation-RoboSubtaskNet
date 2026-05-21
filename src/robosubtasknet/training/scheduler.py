"""Learning-rate schedulers for the RoboSubtaskNet trainer.

A thin factory over :mod:`torch.optim.lr_scheduler` so the rest of the training
stack can stay scheduler-agnostic. The trainer always calls ``.step()`` once
per epoch and ``.state_dict()`` when checkpointing — even when the user picks
the no-op scheduler — so we provide a stub that implements both methods.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class _NoOpScheduler:
    """Scheduler that does nothing but obeys the ``.step()`` / ``.state_dict()``
    contract the trainer expects.

    Use this when you want a fixed learning rate without sprinkling ``if
    scheduler is not None`` guards through the training loop.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def step(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D401
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _state: dict[str, Any]) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    num_epochs: int,
    **kwargs: Any,
) -> Any:
    """Build a learning-rate scheduler by name.

    Parameters
    ----------
    name:
        One of ``"cosine"``, ``"step"``, or ``"none"`` (case-insensitive).
    optimizer:
        The optimizer whose learning rate the scheduler will modulate.
    num_epochs:
        Total number of training epochs. Used as ``T_max`` for cosine and as a
        fallback when ``step_size`` is not provided for step.
    **kwargs:
        Scheduler-specific options:

        * cosine — ``eta_min`` (default ``0.0``).
        * step — ``step_size`` (default ``max(1, num_epochs // 3)``),
          ``gamma`` (default ``0.1``).

    Returns
    -------
    A scheduler object exposing at least ``.step()`` and ``.state_dict()``.
    """
    key = name.lower().strip()
    if key == "cosine":
        eta_min = float(kwargs.get("eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    if key == "step":
        step_size = int(kwargs.get("step_size", max(1, num_epochs // 3)))
        gamma = float(kwargs.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if key in {"none", "noop", "constant"}:
        return _NoOpScheduler(optimizer)
    raise ValueError(
        f"Unknown scheduler name: {name!r}. Expected 'cosine', 'step', or 'none'."
    )


__all__ = ["build_scheduler"]
