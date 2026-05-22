#!/usr/bin/env python
"""Train the class-agnostic boundary segmenter (Stage 1 of the V2 pipeline).

Reads pre-extracted ``.npz`` files written by
``scripts/extract_features_boundary.py`` (RGB + Flow I3D features paired with
binary boundary labels) and trains
:class:`robosubtasknet.boundary.BoundarySegmenter` using
:class:`robosubtasknet.boundary.losses.CompositeBoundaryLoss` over Gaussian-soft
boundary targets.

The script reuses the project-wide :class:`robosubtasknet.training.Trainer`,
which expects each batch dict to expose ``labels`` and ``mask``. Our boundary
dataset emits the binary positions as ``boundaries``; we wrap the dataloader so
each batch's ``labels`` key holds Gaussian-smoothed soft targets — that way
``CompositeBoundaryLoss(stage_outputs, soft_labels, mask)`` slots into the
existing ``train_one_epoch`` signature unchanged.

Mirrors the optimizer-group split, scheduler, checkpoint / TensorBoard
callbacks, and ``final.pt`` + ``metadata.json`` outputs used by
``scripts/train.py`` / ``scripts/train_lerobot.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

# Make ``src/`` importable when the package isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Stage-1 class-agnostic boundary segmenter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Features cache directory; must contain *.npz files written by "
             "scripts/extract_features_boundary.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Checkpoint directory (created on demand).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="BCE positive-class weight; balances sparse positive labels.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Std (in feature timesteps) of the Gaussian soft-label kernel.",
    )
    p.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.15,
        help="Weight on the smoothness (truncated-MSE) term inside the "
             "composite boundary loss.",
    )
    p.add_argument(
        "--tmse-tau",
        type=float,
        default=4.0,
        help="Truncation threshold (sigmoid-probability units) of the "
             "smoothness term.",
    )
    p.add_argument("--num-stages", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--feature-dim", type=int, default=1024)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string; auto-detects 'cuda' when available.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP only matters on CUDA).",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_optimizer(
    model: "torch.nn.Module", lr: float, weight_decay: float
) -> "torch.optim.Optimizer":
    """AdamW with weight decay excluded from norm-layer weights and biases.

    Same recipe as ``scripts/train.py`` / ``scripts/train_lerobot.py`` — 1-D
    parameters (biases, LayerNorm/BatchNorm scales) and anything containing
    ``norm`` / ``bn`` in its name lands in the ``weight_decay=0`` group.
    """
    import torch

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = ("norm" in name.lower()) or ("bn" in name.lower())
        if param.ndim <= 1 or is_bias or is_norm:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


class _SoftLabelLoader:
    """Wrap a DataLoader so each batch carries soft boundary labels.

    The base :class:`BoundaryDataset` collates ``boundaries`` as binary
    ``LongTensor [B, T]``; the project-wide :class:`Trainer` reads
    ``batch["labels"]``. This thin wrapper converts the binary positions to
    Gaussian-soft targets via :func:`make_soft_boundary_labels` and exposes
    them under ``labels`` so ``Trainer.train_one_epoch`` works unchanged.

    The original ``boundaries`` key is preserved for any downstream consumer
    that needs the hard targets (e.g. peak-picking or diagnostics).
    """

    def __init__(self, loader: Iterable, sigma: float) -> None:
        from robosubtasknet.boundary.losses import make_soft_boundary_labels

        self._loader = loader
        self._sigma = float(sigma)
        self._make_soft = make_soft_boundary_labels

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self._loader:
            boundaries = batch["boundaries"].float()
            soft = self._make_soft(boundaries, sigma=self._sigma)
            batch_out = dict(batch)
            batch_out["labels"] = soft
            yield batch_out

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._loader)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    # Defer torch + project imports for fast --help.
    import torch
    from torch.utils.data import DataLoader

    from robosubtasknet.boundary import BoundarySegmenter
    from robosubtasknet.boundary.dataset import BoundaryDataset, pad_collate_boundary
    from robosubtasknet.boundary.losses import (
        CompositeBoundaryLoss,
        make_soft_boundary_labels,  # noqa: F401 — re-exported via _SoftLabelLoader
    )
    from robosubtasknet.training import (
        ModelCheckpoint,
        TensorBoardLogger,
        Trainer,
        build_scheduler,
        set_seed,
    )

    # Validation of paths kept early so users get fast feedback.
    if not args.cache.is_dir():
        raise SystemExit(f"--cache directory not found: {args.cache}")
    args.output.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ----- Dataset -------------------------------------------------------- #
    train_ds = BoundaryDataset(
        feature_dir=args.cache,
        feature_dim=args.feature_dim,
    )
    print(f"[train_boundary] dataset: {len(train_ds)} episodes from {args.cache}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate_boundary,
        drop_last=False,
    )
    soft_loader = _SoftLabelLoader(train_loader, sigma=args.sigma)

    # ----- Model ---------------------------------------------------------- #
    model_config = dict(
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = BoundarySegmenter(**model_config)

    # ----- Loss / Optim / Scheduler -------------------------------------- #
    loss_fn = CompositeBoundaryLoss(
        pos_weight=args.pos_weight,
        lam_smooth=args.lambda_smooth,
        tau=args.tmse_tau,
    )
    optimizer = _build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler("cosine", optimizer, num_epochs=args.epochs)

    # ----- Callbacks ----------------------------------------------------- #
    log_dir = args.output / "tb_logs"

    callbacks: list[Any] = []
    if not args.no_tensorboard:
        try:
            callbacks.append(TensorBoardLogger(log_dir=log_dir))
        except ImportError as exc:
            print(
                f"[train_boundary] tensorboard unavailable ({exc}); "
                f"continuing without it."
            )
    # We have no val loader (the cache is one undifferentiated pool), so the
    # monitor falls back to train_metrics. Use loss (min) to keep the
    # top-k / best.pt machinery meaningful.
    callbacks.append(
        ModelCheckpoint(
            save_dir=args.output,
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
    )

    # ----- Config snapshot (also written as metadata.json sidecar) ------- #
    config_snapshot: dict[str, Any] = {
        "cache": str(args.cache),
        "model_config": model_config,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "pos_weight": args.pos_weight,
        "sigma": args.sigma,
        "lambda_smooth": args.lambda_smooth,
        "tmse_tau": args.tmse_tau,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "feature_extraction": {
            "window": 16,
            "stride": 8,
            "flow_method": "tvl1",
        },
    }

    # ----- Trainer ------------------------------------------------------- #
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=(device.type == "cuda" and not args.no_amp),
        grad_clip=args.grad_clip,
        log_every=50,
        callbacks=callbacks,
    )
    trainer.seed = args.seed  # type: ignore[attr-defined]

    print(
        f"[train_boundary] params={model.count_parameters():,} device={device} "
        f"epochs={args.epochs} lr={args.lr} sigma={args.sigma} "
        f"pos_weight={args.pos_weight} lambda_smooth={args.lambda_smooth}"
    )

    # No val_loader: the boundary cache is a single pool (train/val splitting
    # is the caller's responsibility — point at a different --cache). The
    # trainer's internal best-metric tracking falls back to train_metrics
    # when val_metrics is empty; we monitor train loss (min) so best_metric is
    # not perpetually None.
    result = trainer.fit(
        train_loader=soft_loader,
        val_loader=None,
        num_epochs=args.epochs,
        save_dir=args.output,
        evaluator=None,
        config_snapshot=config_snapshot,
        monitor="loss",
        monitor_mode="min",
    )

    # ----- Final checkpoint --------------------------------------------- #
    final_ckpt = args.output / "final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else None
            ),
            "best_metric": result.get("best_metric"),
            "best_epoch": result.get("best_epoch"),
            "model_config": model_config,
            "hyperparameters": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "pos_weight": args.pos_weight,
                "sigma": args.sigma,
                "lambda_smooth": args.lambda_smooth,
                "tmse_tau": args.tmse_tau,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
            },
            "feature_extraction": {
                "window": 16,
                "stride": 8,
                "flow_method": "tvl1",
            },
        },
        final_ckpt,
    )
    print(f"[train_boundary] saved -> {final_ckpt}")

    # Metadata sidecar — pairs any best.pt from ModelCheckpoint with the
    # exact hyperparameters used. No label vocab: Stage 1 is class-agnostic.
    metadata_path = args.output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_config": model_config,
                "hyperparameters": config_snapshot,
                "best_metric": result.get("best_metric"),
                "best_epoch": result.get("best_epoch"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[train_boundary] metadata sidecar -> {metadata_path}")

    print(
        f"[train_boundary] done. best train_loss={result.get('best_metric')} @ "
        f"epoch {result.get('best_epoch')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
