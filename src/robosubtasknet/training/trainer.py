"""Training orchestration for RoboSubtaskNet.

Implements Section 10 (training procedure) and Section 13 (reproducibility,
checkpoint contents, logging) of the implementation plan. The trainer is
deliberately scope-limited: it owns the per-step / per-epoch loop and the
checkpoint payload, but defers metric computation to an evaluator passed by
the caller and defers logging to user-supplied callbacks.

Design notes
------------
* Batch dictionaries are expected to follow the dataset convention:
  ``{"rgb": [B, T, D], "flow": [B, T, D], "labels": [B, T], "mask": [B, T]}``.
* Mixed precision uses :class:`torch.cuda.amp.GradScaler` with
  ``enabled=use_amp and device.type == "cuda"``. On CPU the autocast block
  becomes a no-op and the scaler bypasses scaling so the loop is identical on
  both devices.
* The trainer never instantiates callbacks itself — they are passed in via the
  constructor. The default behavior with no callbacks is to run silently.
"""

from __future__ import annotations

import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .callbacks import Callback


# ---------------------------------------------------------------------------
# Reproducibility helper (Section 13.1)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Also flips cuDNN into deterministic mode and disables its autotuner. This
    can be ~15% slower than the default — recommended for final runs, not for
    development iterations (see Section 13.1 of the implementation plan).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    """Return the current git commit SHA, or ``"unknown"`` on failure.

    Used to stamp every checkpoint so we can always trace a saved artifact
    back to the source tree that produced it.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def _to_device(value: Any, device: torch.device) -> Any:
    """Move a tensor (or pass through anything else) to ``device``.

    The dataset is expected to emit a dict with tensor values, but we keep
    this defensive so a caller that adds e.g. a list of video IDs to the batch
    dict won't crash the loop.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def _detach_float(x: Any) -> float:
    """Convert a tensor or numeric to a plain Python float (zero if NaN/None)."""
    if x is None:
        return 0.0
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)


def _fire(
    callbacks: list[Callback],
    hook: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a hook across every callback, swallowing per-callback errors.

    A failing callback (e.g. a TensorBoard writer running out of disk) should
    not take down training — log to stderr and keep going. We re-raise on
    ``KeyboardInterrupt`` so Ctrl-C still works.
    """
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is None:
            continue
        try:
            fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — see docstring
            import sys

            print(
                f"[Trainer] Callback {type(cb).__name__}.{hook} raised: {exc}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Train / evaluate / fit loop for the RoboSubtaskNet model.

    Parameters
    ----------
    model:
        The network. Expected to take ``(rgb, flow)`` and return a list of
        per-stage logits ``[B, C, T]`` (see ``RoboSubtaskNet.forward``).
    loss_fn:
        A :class:`~robosubtasknet.losses.CompositeLoss`-style callable that
        accepts ``(stage_outputs, labels, mask)`` and returns
        ``(loss_tensor, components_dict)``.
    optimizer:
        Any :class:`torch.optim.Optimizer`.
    scheduler:
        Optional LR scheduler (``.step()`` / ``.state_dict()``); stepped once
        per epoch after the train loop. Pass ``None`` for fixed-LR training.
    device:
        Where to put the model and batches. Accepts a string (``"cuda"``,
        ``"cuda:0"``, ``"cpu"``) or a :class:`torch.device`.
    use_amp:
        Enable :mod:`torch.cuda.amp` mixed precision. Auto-disabled when the
        device is CPU because AMP is CUDA-only.
    grad_clip:
        Max norm for gradient clipping (``5.0`` per Section 10.3). Set to
        ``0`` or ``None`` to disable.
    log_every:
        Print a one-liner every ``log_every`` steps (and fire ``on_batch_end``
        on every step — the throttling only affects stdout, not callbacks).
    callbacks:
        List of :class:`Callback` instances. Hooks fire in list order.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Any = None,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        grad_clip: float = 5.0,
        log_every: int = 50,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        # AMP is only meaningful on CUDA; silently degrade on CPU to avoid
        # forcing every caller to special-case CPU-only environments (e.g. CI).
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.log_every = max(1, int(log_every))
        self.callbacks: list[Callback] = list(callbacks or [])

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self._best_metric: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._global_step: int = 0
        self._stop_training = False

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: Iterable) -> dict[str, float]:
        """Run one epoch of training over ``loader``.

        Returns
        -------
        dict
            Average ``loss`` plus per-component averages (``ce``, ``tmse``,
            ``trans``) over the epoch, suitable for logging.
        """
        self.model.train()

        loss_sum = 0.0
        comp_sums: dict[str, float] = {"ce": 0.0, "tmse": 0.0, "trans": 0.0}
        n_batches = 0
        epoch_start = time.time()

        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True) if self.use_amp else nullcontext()
        )

        for step, batch in enumerate(loader):
            rgb = _to_device(batch["rgb"], self.device)
            flow = _to_device(batch["flow"], self.device)
            labels = _to_device(batch["labels"], self.device)
            mask = _to_device(batch.get("mask"), self.device)

            with amp_ctx:
                outputs = self.model(rgb, flow)
                loss, components = self.loss_fn(outputs, labels, mask)

            # Standard AMP backward sequence: scale -> backward -> unscale ->
            # clip -> step -> update. ``unscale_`` is the only safe place to
            # clip gradients because the scaler's master gradients live in fp32.
            self.scaler.scale(loss).backward()
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(self.grad_clip)
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            loss_val = _detach_float(loss)
            loss_sum += loss_val
            n_batches += 1
            for key in comp_sums:
                if key in components:
                    comp_sums[key] += _detach_float(components[key])

            self._global_step += 1
            current_lr = self.optimizer.param_groups[0].get("lr")
            _fire(
                self.callbacks,
                "on_batch_end",
                step,
                {
                    "step": self._global_step,
                    "loss": loss_val,
                    "components": {k: _detach_float(v) for k, v in components.items()},
                    "lr": current_lr,
                },
            )

            if (step + 1) % self.log_every == 0:
                print(
                    f"  step {step + 1:>5d}  loss={loss_val:.4f}  "
                    f"ce={_detach_float(components.get('ce')):.4f}  "
                    f"tmse={_detach_float(components.get('tmse')):.4f}  "
                    f"trans={_detach_float(components.get('trans')):.4f}"
                )

        denom = max(1, n_batches)
        metrics = {
            "loss": loss_sum / denom,
            "ce": comp_sums["ce"] / denom,
            "tmse": comp_sums["tmse"] / denom,
            "trans": comp_sums["trans"] / denom,
            "epoch_time": time.time() - epoch_start,
            "num_batches": float(n_batches),
        }
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable,
        evaluator: Optional[Callable[..., Any]] = None,
    ) -> dict[str, float]:
        """Run evaluation over ``loader``.

        Parameters
        ----------
        loader:
            Yields the same batch dicts as the train loader.
        evaluator:
            Optional callable. For each batch we call::

                evaluator(predictions, labels, mask)

            where ``predictions`` is ``[B, T]`` (argmax of the final stage's
            softmax). After the loop we call ``evaluator.compute()`` if it
            exposes such a method, and merge the returned dict (if any) into
            the metrics.

        Returns
        -------
        dict
            Averaged losses (``loss``, ``ce``, ``tmse``, ``trans``) plus any
            metrics produced by ``evaluator.compute()``.
        """
        self.model.eval()
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True) if self.use_amp else nullcontext()
        )

        loss_sum = 0.0
        comp_sums = {"ce": 0.0, "tmse": 0.0, "trans": 0.0}
        n_batches = 0

        for batch in loader:
            rgb = _to_device(batch["rgb"], self.device)
            flow = _to_device(batch["flow"], self.device)
            labels = _to_device(batch["labels"], self.device)
            mask = _to_device(batch.get("mask"), self.device)

            with amp_ctx:
                outputs = self.model(rgb, flow)
                loss, components = self.loss_fn(outputs, labels, mask)

            loss_sum += _detach_float(loss)
            n_batches += 1
            for key in comp_sums:
                if key in components:
                    comp_sums[key] += _detach_float(components[key])

            if evaluator is not None:
                # Final-stage prediction. Cast to fp32 for argmax stability in AMP.
                final_logits = outputs[-1].float()
                preds = final_logits.argmax(dim=1)  # [B, T]
                try:
                    evaluator(preds, labels, mask)
                except TypeError:
                    # Fall back to two-arg form for evaluators that don't take a mask.
                    evaluator(preds, labels)

        denom = max(1, n_batches)
        metrics: dict[str, float] = {
            "loss": loss_sum / denom,
            "ce": comp_sums["ce"] / denom,
            "tmse": comp_sums["tmse"] / denom,
            "trans": comp_sums["trans"] / denom,
        }

        if evaluator is not None and hasattr(evaluator, "compute"):
            extra = evaluator.compute()
            if isinstance(extra, dict):
                metrics.update({k: float(v) for k, v in extra.items() if v is not None})
        if evaluator is not None and hasattr(evaluator, "reset"):
            evaluator.reset()

        return metrics

    # ------------------------------------------------------------------
    # Full fit orchestration
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader | Iterable,
        val_loader: DataLoader | Iterable | None,
        num_epochs: int,
        save_dir: str | Path,
        evaluator: Optional[Callable[..., Any]] = None,
        config_snapshot: Any = None,
        monitor: str = "f1_at_50",
        monitor_mode: str = "max",
    ) -> dict[str, Any]:
        """Run the full training schedule.

        Each epoch:
        1. ``on_epoch_begin`` callback hook.
        2. ``train_one_epoch`` on ``train_loader``.
        3. If ``val_loader`` is provided, ``evaluate`` on it.
        4. Step the scheduler (if any).
        5. Build the Section-13.2 checkpoint payload and fire ``on_epoch_end``
           with it inside ``logs["checkpoint_payload"]`` so callbacks like
           :class:`ModelCheckpoint` can persist it.
        6. If any callback sets ``logs["stop_training"] = True`` (e.g.
           :class:`EarlyStopping`), break out of the loop.

        Returns
        -------
        dict
            ``{"best_metric": float | None, "best_epoch": int | None,
              "history": list[dict]}``.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, Any]] = []
        self._stop_training = False
        self._best_metric = None
        self._best_epoch = None

        train_begin_logs: dict[str, Any] = {
            "num_epochs": int(num_epochs),
            "save_dir": str(save_dir),
            "monitor": monitor,
            "monitor_mode": monitor_mode,
        }
        if config_snapshot is not None:
            train_begin_logs["config_text"] = str(config_snapshot)
        _fire(self.callbacks, "on_train_begin", train_begin_logs)

        sha = _git_sha()

        for epoch in range(1, int(num_epochs) + 1):
            _fire(self.callbacks, "on_epoch_begin", epoch, {"epoch": epoch})
            print(f"[Epoch {epoch}/{num_epochs}] training...")

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                print(f"[Epoch {epoch}/{num_epochs}] evaluating...")
                val_metrics = self.evaluate(val_loader, evaluator=evaluator)

            # Step the scheduler at the epoch boundary. We accept the small
            # cost of stepping even on the no-op scheduler so the trainer
            # never has to special-case it.
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    # ReduceLROnPlateau-style schedulers need a metric.
                    plateau_key = monitor if monitor in val_metrics else "loss"
                    self.scheduler.step(val_metrics.get(plateau_key, train_metrics["loss"]))

            # Track the best monitored value so we can include it in the
            # payload for downstream consumers (e.g. ModelCheckpoint).
            current = val_metrics.get(monitor)
            if current is None:
                current = train_metrics.get(monitor)
            if current is not None:
                better = (
                    self._best_metric is None
                    or (monitor_mode == "max" and current > self._best_metric)
                    or (monitor_mode == "min" and current < self._best_metric)
                )
                if better:
                    self._best_metric = float(current)
                    self._best_epoch = epoch

            # ------------------------------------------------------------------
            # Checkpoint payload (Section 13.2). The trainer constructs the
            # payload; callbacks decide whether to persist it (top-k, best, ...).
            # ------------------------------------------------------------------
            scheduler_state = (
                self.scheduler.state_dict() if self.scheduler is not None else None
            )
            payload: dict[str, Any] = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": scheduler_state,
                "epoch": epoch,
                "best_metric": self._best_metric,
                "config_snapshot": (
                    str(config_snapshot) if config_snapshot is not None else None
                ),
                "git_sha": sha,
                "seed": getattr(self, "seed", None),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            epoch_logs: dict[str, Any] = {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "checkpoint_payload": payload,
                "best_metric": self._best_metric,
                "best_epoch": self._best_epoch,
            }
            _fire(self.callbacks, "on_epoch_end", epoch, epoch_logs)

            history.append(
                {
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
            )

            # Print a short summary line for the console reader.
            summary_bits = [f"train_loss={train_metrics['loss']:.4f}"]
            if val_metrics:
                if "loss" in val_metrics:
                    summary_bits.append(f"val_loss={val_metrics['loss']:.4f}")
                if monitor in val_metrics:
                    summary_bits.append(
                        f"val_{monitor}={val_metrics[monitor]:.4f}"
                    )
            print(f"[Epoch {epoch}/{num_epochs}] " + "  ".join(summary_bits))

            if epoch_logs.get("stop_training"):
                self._stop_training = True
                print(f"[Trainer] Early stop requested at epoch {epoch}; halting.")
                break

        _fire(
            self.callbacks,
            "on_train_end",
            {
                "best_metric": self._best_metric,
                "best_epoch": self._best_epoch,
                "history": history,
            },
        )

        return {
            "best_metric": self._best_metric,
            "best_epoch": self._best_epoch,
            "history": history,
        }


__all__ = ["Trainer", "set_seed"]
