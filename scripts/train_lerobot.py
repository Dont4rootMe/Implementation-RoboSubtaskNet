#!/usr/bin/env python
"""Train RoboSubtaskNet on one or more LeRobot datasets via cached features.

Pipeline:

1. Read ``label_map.json`` (union vocabulary) → ``num_classes``.
2. Build :class:`RoboSubtaskDataset` over the cache (mode=``"npz"``).
3. Build :class:`RoboSubtaskNet`, :class:`CompositeLoss`, AdamW optimizer
   (norms/biases excluded from weight decay), cosine scheduler.
4. ``Trainer.fit(...)`` with TensorBoard + ModelCheckpoint callbacks.
5. Write ``<output>/final.pt`` containing model state plus the metadata
   needed at inference (``label_map``, ``action_text``, ``camera_keys``,
   ``model_config``, ``feature_extraction``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RoboSubtaskNet on LeRobot dataset(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="LeRobot dataset root(s). Used for provenance — features must be in --cache.",
    )
    p.add_argument(
        "--camera",
        nargs="+",
        type=str,
        required=True,
        help="Camera key(s) per --input (1 broadcast or one-to-one).",
    )
    p.add_argument("--cache", type=Path, required=True, help="Features cache dir.")
    p.add_argument("--label-map", type=Path, required=True, help="label_map.json.")
    p.add_argument("--output", type=Path, required=True, help="Checkpoint directory.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grammar",
        type=Path,
        default=None,
        help="Optional grammar YAML; enables transition loss if set.",
    )
    p.add_argument("--lambda-tmse", type=float, default=0.15)
    p.add_argument(
        "--gamma-trans",
        type=float,
        default=0.0,
        help="Transition-loss weight. 0.0 (default) disables it.",
    )
    p.add_argument("--num-stages", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--feature-dim", type=int, default=1024)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def _build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW with weight decay excluded from norm-layer weights and biases."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = ("norm" in name.lower()) or ("bn" in name.lower())
        if p.ndim <= 1 or is_bias or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def main() -> int:
    args = parse_args()

    # Camera/input arity check (consistency w/ extract step)
    if not (len(args.camera) == 1 or len(args.camera) == len(args.input)):
        raise SystemExit(
            f"--camera must have 1 or {len(args.input)} values; got {len(args.camera)}."
        )
    camera_keys = (
        list(args.camera) * len(args.input)
        if len(args.camera) == 1
        else list(args.camera)
    )

    # Label map → num_classes
    if not args.label_map.exists():
        raise SystemExit(f"--label-map not found: {args.label_map}")
    with args.label_map.open("r", encoding="utf-8") as f:
        label_map: dict[str, int] = json.load(f)
    num_classes = len(label_map)
    action_text: dict[int, str] = {int(v): k for k, v in label_map.items()}
    print(f"[train] label_map: {num_classes} classes")

    # Imports deferred for fast --help
    from torch.utils.data import DataLoader

    from robosubtasknet.data import RoboSubtaskDataset, pad_collate
    from robosubtasknet.eval import SegmentationEvaluator
    from robosubtasknet.losses import CompositeLoss
    from robosubtasknet.models import RoboSubtaskNet
    from robosubtasknet.training import (
        ModelCheckpoint,
        TensorBoardLogger,
        Trainer,
        build_scheduler,
        set_seed,
    )

    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Dataset
    if not args.cache.is_dir():
        raise SystemExit(f"--cache directory not found: {args.cache}")
    train_ds = RoboSubtaskDataset(
        feature_dir=args.cache,
        mode="npz",
        feature_dim=args.feature_dim,
    )
    print(f"[train] dataset: {len(train_ds)} episodes from {args.cache}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
        drop_last=False,
    )

    # Model
    model_config = dict(
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    )
    model = RoboSubtaskNet(**model_config)

    # Allowed transitions
    if args.grammar is not None:
        from robosubtasknet.data.grammar import build_allowed_transitions

        if not args.grammar.exists():
            raise SystemExit(f"--grammar file not found: {args.grammar}")
        allowed = build_allowed_transitions(args.grammar, label_map)
        print(f"[train] grammar mask loaded from {args.grammar}")
    else:
        allowed = torch.ones((num_classes, num_classes), dtype=torch.bool)
        if args.gamma_trans != 0.0:
            print(
                "[train] WARNING: --gamma-trans != 0 but no --grammar provided; "
                "transition loss reduces to a constant penalty on confidence."
            )

    loss_fn = CompositeLoss(
        num_classes=num_classes,
        allowed_transitions=allowed,
        lam=args.lambda_tmse,
        gam=args.gamma_trans,
        tau=4.0,
    )

    optimizer = _build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler("cosine", optimizer, num_epochs=args.epochs)

    # Callbacks
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "tb_logs"

    callbacks: list[Any] = []
    if not args.no_tensorboard:
        try:
            callbacks.append(TensorBoardLogger(log_dir=log_dir))
        except ImportError as exc:
            print(f"[train] tensorboard unavailable ({exc}); continuing without it.")
    callbacks.append(
        ModelCheckpoint(
            save_dir=args.output,
            save_top_k=3,
            monitor="F1@50",
            mode="max",
        )
    )

    # Evaluator wrapper
    bg_class = [label_map["background"]] if "background" in label_map else None
    evaluator = SegmentationEvaluator(bg_class=bg_class)

    def evaluator_callable(preds, labels, mask):
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None
        for b in range(preds_np.shape[0]):
            p = preds_np[b]
            g = labels_np[b]
            if mask_np is not None:
                valid = mask_np[b].astype(bool)
                p = p[valid]
                g = g[valid]
            evaluator.update(p.tolist(), g.tolist())

    evaluator_callable.compute = evaluator.compute  # type: ignore[attr-defined]
    evaluator_callable.reset = evaluator.reset  # type: ignore[attr-defined]

    config_snapshot = {
        "label_map": label_map,
        "action_text": action_text,
        "camera_keys": camera_keys,
        "input_paths": [str(p) for p in args.input],
        "model_config": model_config,
        "feature_extraction": {"window": 16, "stride": 8, "flow_method": "tvl1"},
        "lambda_tmse": args.lambda_tmse,
        "gamma_trans": args.gamma_trans,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
    }

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
        f"[train] params={model.count_parameters():,} device={device} "
        f"epochs={args.epochs} lr={args.lr} gamma_trans={args.gamma_trans}"
    )

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=args.epochs,
        save_dir=args.output,
        evaluator=evaluator_callable,
        config_snapshot=config_snapshot,
        monitor="F1@50",
        monitor_mode="max",
    )

    # Save a self-contained final.pt with everything needed for inference.
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
            "label_map": label_map,
            "action_text": action_text,
            "camera_keys": camera_keys,
            "model_config": model_config,
            "feature_extraction": {
                "window": 16,
                "stride": 8,
                "flow_method": "tvl1",
            },
        },
        final_ckpt,
    )
    print(f"[train] saved → {final_ckpt}")

    # Also write a metadata.json sidecar so any best.pt produced by
    # ModelCheckpoint can be paired with the vocabulary at inference time.
    metadata_path = args.output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_map": label_map,
                "action_text": {str(k): v for k, v in action_text.items()},
                "camera_keys": camera_keys,
                "model_config": model_config,
                "feature_extraction": {
                    "window": 16,
                    "stride": 8,
                    "flow_method": "tvl1",
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[train] metadata sidecar → {metadata_path}")

    print(
        f"[train] done. best F1@50={result.get('best_metric')} @ epoch "
        f"{result.get('best_epoch')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
