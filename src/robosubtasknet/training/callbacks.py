"""Lightweight callback protocol and concrete callbacks for the trainer.

The design follows the Keras / PyTorch-Lightning convention: callbacks are
objects with named hook methods that the trainer invokes at well-defined
points. Implementations subclass :class:`Callback` and override the hooks they
care about; defaults are no-ops.

The trainer passes a ``logs`` dict to every hook. By convention this contains:

* ``on_train_begin`` — ``{"num_epochs": int, "save_dir": str}``.
* ``on_epoch_begin`` — ``{"epoch": int}``.
* ``on_batch_end``  — ``{"epoch": int, "step": int, "components": dict[str, float],
                       "loss": float, "alpha": Optional[Tensor]}``.
* ``on_epoch_end``  — ``{"epoch": int, "train_metrics": dict, "val_metrics": dict,
                       "model_state": ..., "optimizer_state": ...,
                       "scheduler_state": ..., "checkpoint_payload": dict}``.
* ``on_train_end``  — ``{"best_metric": float, "best_epoch": int}``.

Concrete callbacks may inspect or modify the contents, but should fail soft
when keys are missing — different trainers may populate different subsets.
"""

from __future__ import annotations

import heapq
import os
from pathlib import Path
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------
class Callback:
    """Base class for training callbacks. All hooks default to no-ops."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:  # noqa: D401
        return None

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:  # noqa: D401
        return None

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: D401
        return None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: D401
        return None

    def on_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:  # noqa: D401
        return None


# ---------------------------------------------------------------------------
# TensorBoard logger
# ---------------------------------------------------------------------------
class TensorBoardLogger(Callback):
    """Log per-step loss components, epoch metrics, and fusion-gate stats.

    Imports :mod:`torch.utils.tensorboard` lazily so callers that never use it
    do not pay the import cost (and unit tests without tensorboard installed
    can still import this module).

    Parameters
    ----------
    log_dir:
        Directory for the TensorBoard event files.
    flush_secs:
        How often (seconds) the writer flushes events to disk.
    """

    def __init__(self, log_dir: str | os.PathLike, flush_secs: int = 30) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - exercised only without TB
            raise ImportError(
                "TensorBoardLogger requires the 'tensorboard' package. "
                "Install it via `pip install tensorboard`."
            ) from exc
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)
        self._global_step = 0

    # ---- hooks ---------------------------------------------------------
    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        if logs and "config_text" in logs:
            self.writer.add_text("config", str(logs["config_text"]))

    def on_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        self._global_step += 1
        components: dict[str, float] = logs.get("components", {}) or {}
        for name in ("ce", "tmse", "trans"):
            if name in components:
                self.writer.add_scalar(
                    f"train_step/{name}",
                    float(components[name]),
                    self._global_step,
                )
        if "loss" in logs and logs["loss"] is not None:
            self.writer.add_scalar(
                "train_step/loss", float(logs["loss"]), self._global_step
            )
        lr = logs.get("lr")
        if lr is not None:
            self.writer.add_scalar("train_step/lr", float(lr), self._global_step)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        train_metrics: dict[str, float] = logs.get("train_metrics", {}) or {}
        val_metrics: dict[str, float] = logs.get("val_metrics", {}) or {}
        for name, value in train_metrics.items():
            if value is None:
                continue
            self.writer.add_scalar(f"train_epoch/{name}", float(value), epoch)
        for name, value in val_metrics.items():
            if value is None:
                continue
            self.writer.add_scalar(f"val_epoch/{name}", float(value), epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.writer.flush()
        self.writer.close()

    # ---- diagnostics --------------------------------------------------
    def log_gate(self, alpha: torch.Tensor, step: int | None = None) -> None:
        """Log statistics of the fusion gate ``alpha`` for diagnostics.

        ``alpha`` is expected to be a tensor of values in ``[0, 1]`` of shape
        ``[B, T, D]`` or ``[B, D, T]``. We log a scalar mean and a histogram.
        """
        if step is None:
            step = self._global_step
        a = alpha.detach()
        a_float = a.float()
        self.writer.add_scalar("fusion/alpha_mean", a_float.mean().item(), step)
        self.writer.add_scalar("fusion/alpha_std", a_float.std().item(), step)
        # Histograms can be heavy; gate them to avoid blowing up event files.
        try:
            self.writer.add_histogram("fusion/alpha", a_float.flatten(), step)
        except Exception:  # noqa: BLE001 — defensive for old TB versions
            pass


# ---------------------------------------------------------------------------
# Model checkpoint
# ---------------------------------------------------------------------------
class ModelCheckpoint(Callback):
    """Save the top-k checkpoints according to a monitored metric.

    Files are written as ``<save_dir>/epoch=<epoch>-<monitor>=<value>.pt`` plus
    a stable ``<save_dir>/last.pt`` and ``<save_dir>/best.pt``.

    Parameters
    ----------
    save_dir:
        Directory to write checkpoints into. Created on first use.
    save_top_k:
        Keep at most this many checkpoints (excluding ``last``/``best``). Older
        ones are pruned from disk as better epochs arrive.
    monitor:
        Key in ``logs["val_metrics"]`` (preferred) or ``logs["train_metrics"]``
        used to compare checkpoints.
    mode:
        ``"max"`` keeps highest values, ``"min"`` keeps lowest.
    """

    def __init__(
        self,
        save_dir: str | os.PathLike,
        save_top_k: int = 3,
        monitor: str = "f1_at_50",
        mode: str = "max",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = int(save_top_k)
        self.monitor = monitor
        self.mode = mode
        # Min-heap of (signed_score, epoch, path) where score is signed so that
        # the worst entry is always at the top for the configured mode.
        self._heap: list[tuple[float, int, str]] = []
        self._best_score: Optional[float] = None
        self._best_path: Optional[Path] = None

    # ---- helpers ------------------------------------------------------
    def _is_better(self, score: float) -> bool:
        if self._best_score is None:
            return True
        return score > self._best_score if self.mode == "max" else score < self._best_score

    def _signed(self, score: float) -> float:
        # Convert to a value where smaller means "worse" so heapq (a min-heap)
        # puts the worst-kept checkpoint on top, ready for eviction.
        return score if self.mode == "max" else -score

    @staticmethod
    def _extract_score(logs: dict[str, Any], monitor: str) -> Optional[float]:
        for bucket_key in ("val_metrics", "train_metrics", "metrics"):
            bucket = logs.get(bucket_key)
            if isinstance(bucket, dict) and monitor in bucket and bucket[monitor] is not None:
                return float(bucket[monitor])
        if monitor in logs and logs[monitor] is not None:
            return float(logs[monitor])
        return None

    # ---- hooks --------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        payload = logs.get("checkpoint_payload")
        if payload is None:
            return  # Trainer didn't ship a payload; nothing to save.
        score = self._extract_score(logs, self.monitor)
        # Tag the payload so loaders can sanity-check what they're loading.
        payload = dict(payload)
        payload.setdefault("epoch", epoch)
        payload["best_metric"] = score

        fname = self.save_dir / f"epoch={epoch:03d}-{self.monitor}={score!s}.pt"
        torch.save(payload, fname)
        torch.save(payload, self.save_dir / "last.pt")

        # Best-so-far tracking (separate from top-k heap).
        if score is not None and self._is_better(score):
            self._best_score = score
            self._best_path = fname
            torch.save(payload, self.save_dir / "best.pt")

        if score is None:
            return  # Without a score we can't run the top-k policy.

        heapq.heappush(self._heap, (self._signed(score), epoch, str(fname)))
        # Evict worst-kept entries until we're within the budget.
        while len(self._heap) > self.save_top_k:
            _signed, _epoch, evict_path = heapq.heappop(self._heap)
            try:
                Path(evict_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ---- exposed for tests / introspection ----------------------------
    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_path(self) -> Optional[Path]:
        return self._best_path


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping(Callback):
    """Stop training when ``monitor`` plateaus for ``patience`` epochs.

    The callback does not raise; instead it sets ``logs["stop_training"] = True``
    on ``on_epoch_end``. The trainer is expected to honor that flag.

    Parameters
    ----------
    monitor:
        Metric key to track inside ``logs["val_metrics"]`` (or ``train_metrics``).
    patience:
        Number of consecutive epochs without improvement before stopping.
    mode:
        ``"max"`` or ``"min"`` — direction in which the metric should move.
    min_delta:
        Minimum change to be considered an improvement.
    """

    def __init__(
        self,
        monitor: str,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self._best: Optional[float] = None
        self._wait = 0
        self.stopped_epoch: Optional[int] = None

    def _is_improvement(self, score: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "max":
            return score > self._best + self.min_delta
        return score < self._best - self.min_delta

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._best = None
        self._wait = 0
        self.stopped_epoch = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        score = ModelCheckpoint._extract_score(logs, self.monitor)
        if score is None:
            return  # No metric this epoch; don't penalize the patience counter.
        if self._is_improvement(score):
            self._best = score
            self._wait = 0
            return
        self._wait += 1
        if self._wait >= self.patience:
            self.stopped_epoch = epoch
            logs["stop_training"] = True


__all__ = [
    "Callback",
    "TensorBoardLogger",
    "ModelCheckpoint",
    "EarlyStopping",
]
