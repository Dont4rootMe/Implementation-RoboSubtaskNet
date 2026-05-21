"""Training entry-point for RoboSubtaskNet.

Builds the model, composite loss, optimizer, scheduler, and the train / val
dataloaders, then hands everything to ``robosubtasknet.training.Trainer.fit``.

Tries Hydra first (composes ``configs/<dataset>.yaml`` via
``configs/default.yaml``) and falls back to ``argparse`` when Hydra is not
installed. Both paths read the same OmegaConf-style config structure, so
behaviour is identical end-to-end.

Notable choices implementing the spec:

* AdamW with weight decay excluded from norms and biases (Section 16.2 #4).
* Optional ``allowed_transitions`` from the grammar YAML feeds the
  ``CompositeLoss`` transition term (Sections 9.3, 9.5).
* Top-k checkpointing plus a stable ``best.pt`` (Section 13.2).
* TensorBoard logging of per-step losses and per-epoch metrics (Section 13.3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# Make ``src/`` importable when the package isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration loading (Hydra-first, argparse fallback)
# --------------------------------------------------------------------------- #


def _load_config_from_yaml(config_path: Path) -> Any:
    """Compose a single config dict via OmegaConf (Hydra's underlying library).

    This is used as the argparse fallback when Hydra itself isn't available.
    Manually resolves the simple ``defaults: [default]`` chain that the
    in-repo configs use so that per-dataset overrides land on top of the
    shared base.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise RuntimeError(
            "Configuration loading requires either Hydra or OmegaConf. "
            "Install with `pip install omegaconf`."
        ) from exc

    cfg_path = config_path
    cfg = OmegaConf.load(cfg_path)
    defaults = cfg.pop("defaults", None) if hasattr(cfg, "pop") else None
    if defaults is None:
        defaults = cfg.get("defaults") if hasattr(cfg, "get") else None

    merged: Any = OmegaConf.create({})
    if defaults is not None:
        for entry in defaults:
            if isinstance(entry, dict):
                key = next(iter(entry.keys()))
                name = str(entry[key])
            else:
                name = str(entry)
            if name in ("_self_", "self"):
                continue
            child_path = cfg_path.parent / f"{name}.yaml"
            if child_path.exists():
                merged = OmegaConf.merge(merged, _load_config_from_yaml(child_path))
    # ``defaults`` may still be present on cfg; drop before merging.
    if "defaults" in cfg:
        try:
            del cfg["defaults"]
        except Exception:  # noqa: BLE001
            pass
    merged = OmegaConf.merge(merged, cfg)
    return merged


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RoboSubtaskNet for temporal sub-task segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gtea.yaml"),
        help="Path to the dataset YAML (composes default.yaml via 'defaults').",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory (defaults to logging.checkpoint_dir).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Override TensorBoard log directory.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override torch device ('cuda', 'cpu'); auto-detected if omitted.",
    )
    p.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override training.num_epochs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed.",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging even if the config enables it.",
    )
    return p


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """AdamW with weight decay only on weight matrices (Section 16.2 #4).

    Norm-layer parameters and biases get a zero weight-decay parameter group
    so the optimizer doesn't shrink them. This is the standard recipe for
    transformer / TCN models and resolves the common ``Adam with WD on
    biases / norms`` failure mode flagged in the implementation plan.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Heuristics: bias terms and norm-layer weights are 1-D parameters; the
        # convention is to skip weight decay on them.
        is_bias = name.endswith(".bias")
        is_norm = ("norm" in name.lower()) or ("bn" in name.lower()) or (
            "ln" in name.split(".") if "." in name else False
        )
        if p.ndim <= 1 or is_bias or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)

    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # AdamW regardless of the config name -- the spec explicitly calls for it.
    return torch.optim.AdamW(param_groups, lr=lr)


def _resolve_mode(cfg: Any) -> str:
    """Map ``cfg.dataset.feature_format`` to the dataset's ``mode`` argument."""
    fmt = str(cfg.dataset.get("feature_format", "npz")).lower()
    if fmt in ("split2048", "mstcn", "mstcn2048"):
        return "mstcn"
    return "npz"


def _build_dataset(cfg: Any, split: str) -> Any:
    """Construct a :class:`RoboSubtaskDataset` for ``"train"`` or ``"test"``."""
    from robosubtasknet.data import RoboSubtaskDataset

    bundle_key = "train_bundle" if split == "train" else "test_bundle"
    bundle = cfg.dataset.get(bundle_key, None)
    bundle_path = Path(bundle) if bundle else None

    mapping_file = cfg.dataset.get("mapping_file", None)
    mapping_path = Path(mapping_file) if mapping_file else None

    feature_dir = Path(cfg.dataset.feature_dir)
    feature_dim = int(cfg.model.feature_dim)
    mode = _resolve_mode(cfg)

    return RoboSubtaskDataset(
        feature_dir=feature_dir,
        split_file=bundle_path,
        mapping_file=mapping_path,
        mode=mode,
        feature_dim=feature_dim,
    )


def _build_dataloader(
    cfg: Any, dataset: Any, train: bool
) -> DataLoader:
    from robosubtasknet.data import pad_collate

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.dataset.get("num_workers", 0))
    pin_memory = bool(cfg.dataset.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate,
        drop_last=False,
    )


def _build_allowed_transitions(cfg: Any) -> Optional[torch.Tensor]:
    """Return the ``[C, C]`` bool tensor from the grammar, or ``None``.

    A missing grammar (e.g. when training on plain MS-TCN GTEA, which has no
    task-level grammar) collapses the transition loss into an "all allowed"
    matrix -- mathematically equivalent to disabling it.
    """
    grammar_cfg = cfg.get("grammar", None)
    if grammar_cfg is None:
        return None
    grammar_path = grammar_cfg.get("path", None) if hasattr(grammar_cfg, "get") else None
    if not grammar_path:
        return None
    grammar_path = Path(grammar_path)
    if not grammar_path.exists():
        print(f"[train] grammar file not found at {grammar_path}; skipping transition mask.")
        return None

    mapping_file = cfg.dataset.get("mapping_file", None)
    if not mapping_file:
        print("[train] no mapping_file in dataset config; cannot build grammar mask.")
        return None
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        print(f"[train] mapping file {mapping_path} missing; skipping transition mask.")
        return None

    from robosubtasknet.data.dataset import load_mapping
    from robosubtasknet.data.grammar import build_allowed_transitions

    mapping = load_mapping(mapping_path)
    return build_allowed_transitions(grammar_path, mapping)


def _bg_class_from_mapping(cfg: Any) -> list[int] | None:
    """Best-effort lookup of the background class for the evaluator."""
    mapping_file = cfg.dataset.get("mapping_file", None)
    if not mapping_file or not Path(mapping_file).exists():
        return None
    from robosubtasknet.data.dataset import load_mapping

    mapping = load_mapping(Path(mapping_file))
    if "background" in mapping:
        return [int(mapping["background"])]
    return None


# --------------------------------------------------------------------------- #
# Core training routine (Hydra and argparse paths converge here)
# --------------------------------------------------------------------------- #


def run_training(
    cfg: Any,
    *,
    cli_args: Optional[argparse.Namespace] = None,
) -> dict[str, Any]:
    """Build everything and call ``Trainer.fit``.

    Kept side-effect-free except for filesystem writes (checkpoints, logs).
    Returns the dict produced by ``Trainer.fit`` (best metric + history).
    """
    # ---- imports done late so that purely-CLI use cases (``--help``) don't
    # ---- pull torch into module-import time twice.
    from robosubtasknet.losses import CompositeLoss
    from robosubtasknet.models import RoboSubtaskNet
    from robosubtasknet.training import (
        ModelCheckpoint,
        TensorBoardLogger,
        Trainer,
        build_scheduler,
        set_seed,
    )

    # ---- CLI overrides
    if cli_args is not None:
        if cli_args.num_epochs is not None:
            cfg.training.num_epochs = int(cli_args.num_epochs)
        if cli_args.seed is not None:
            cfg.seed = int(cli_args.seed)
        if cli_args.save_dir is not None:
            cfg.logging.checkpoint_dir = str(cli_args.save_dir)
        if cli_args.log_dir is not None:
            cfg.logging.log_dir = str(cli_args.log_dir)
        if cli_args.no_tensorboard:
            cfg.logging.tensorboard = False

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ---- Device
    if cli_args is not None and cli_args.device:
        device_str = str(cli_args.device)
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ---- Data
    print("[train] building datasets...")
    train_ds = _build_dataset(cfg, split="train")
    try:
        val_ds = _build_dataset(cfg, split="test")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[train] no validation set available ({exc}); training without val.")
        val_ds = None

    train_loader = _build_dataloader(cfg, train_ds, train=True)
    val_loader = _build_dataloader(cfg, val_ds, train=False) if val_ds is not None else None

    # ---- Model
    print("[train] building model...")
    model = RoboSubtaskNet(
        num_stages=int(cfg.model.num_stages),
        num_layers=int(cfg.model.num_layers),
        feature_dim=int(cfg.model.feature_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
    )

    # ---- Allowed transitions (grammar mask)
    allowed = _build_allowed_transitions(cfg)
    if allowed is None:
        # Default to all-allowed when no grammar is configured. This makes the
        # transition loss vanish without forcing users to drop the term.
        c = int(cfg.model.num_classes)
        allowed = torch.ones((c, c), dtype=torch.bool)

    # ---- Loss
    loss_fn = CompositeLoss(
        num_classes=int(cfg.model.num_classes),
        allowed_transitions=allowed,
        lam=float(cfg.losses.lambda_tmse),
        gam=float(cfg.losses.gamma_trans),
        tau=float(cfg.losses.tmse_tau),
    )

    # ---- Optimizer (AdamW excluding norms/biases from weight decay)
    optimizer = _build_optimizer(model, cfg)

    # ---- Scheduler
    scheduler = build_scheduler(
        name=str(cfg.training.scheduler),
        optimizer=optimizer,
        num_epochs=int(cfg.training.num_epochs),
    )

    # ---- Callbacks
    log_dir = Path(cfg.logging.log_dir)
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = []
    if bool(cfg.logging.get("tensorboard", True)):
        try:
            callbacks.append(TensorBoardLogger(log_dir=log_dir))
        except ImportError as exc:
            print(f"[train] TensorBoard unavailable ({exc}); continuing without it.")
    callbacks.append(
        ModelCheckpoint(
            save_dir=ckpt_dir,
            save_top_k=int(cfg.logging.get("save_top_k", 3)),
            monitor="F1@50",
            mode="max",
        )
    )

    # ---- Evaluator wrapper for fit()
    bg_class = _bg_class_from_mapping(cfg)
    from robosubtasknet.eval import SegmentationEvaluator

    evaluator = SegmentationEvaluator(bg_class=bg_class)

    def _evaluator_callable(preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        # The Trainer feeds us [B, T] preds and labels. Iterate the batch and
        # update the corpus-level accumulator one video at a time.
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = None
        for b in range(preds_np.shape[0]):
            p = preds_np[b]
            g = labels_np[b]
            if mask_np is not None:
                valid = mask_np[b].astype(bool)
                p = p[valid]
                g = g[valid]
            evaluator.update(p.tolist(), g.tolist())

    # Decorate the callable so Trainer can detect ``.compute``/``.reset``.
    _evaluator_callable.compute = evaluator.compute  # type: ignore[attr-defined]
    _evaluator_callable.reset = evaluator.reset  # type: ignore[attr-defined]

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=bool(cfg.training.get("amp", False)),
        grad_clip=float(cfg.training.get("grad_clip", 0.0) or 0.0),
        log_every=int(cfg.logging.get("log_every", 50)),
        callbacks=callbacks,
    )
    trainer.seed = seed  # type: ignore[attr-defined]

    print(
        f"[train] dataset={cfg.dataset.name}  "
        f"num_classes={cfg.model.num_classes}  "
        f"params={model.count_parameters():,}  device={device}"
    )

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(cfg.training.num_epochs),
        save_dir=ckpt_dir,
        evaluator=_evaluator_callable,
        config_snapshot=cfg,
        monitor="F1@50",
        monitor_mode="max",
    )
    print(
        f"[train] done. best={result.get('best_metric')} @ epoch "
        f"{result.get('best_epoch')}"
    )
    return result


# --------------------------------------------------------------------------- #
# Entry point dispatch
# --------------------------------------------------------------------------- #


def _try_hydra_main() -> Optional[int]:
    """Try Hydra entry-point; ``None`` when Hydra isn't available."""
    try:
        import hydra
        from omegaconf import DictConfig
    except ImportError:
        return None

    config_dir = str((_REPO_ROOT / "configs").resolve())

    @hydra.main(
        version_base=None, config_path=config_dir, config_name="gtea"
    )
    def _entry(cfg: DictConfig) -> None:
        run_training(cfg, cli_args=None)

    _entry()
    return 0


def main() -> int:
    """CLI entrypoint. Prefers Hydra; falls back to argparse."""
    # Detect Hydra-style invocation: any argv element that contains ``=`` (or
    # explicit ``--config-name`` / ``--config-path``) implies the user wants
    # Hydra. Otherwise we go through the argparse path so a plain
    # ``python scripts/train.py --config configs/gtea.yaml`` keeps working.
    hydra_invocation = any(
        ("=" in a) or a in ("--config-name", "--config-path", "--cfg")
        for a in sys.argv[1:]
    )
    if hydra_invocation:
        rc = _try_hydra_main()
        if rc is not None:
            return rc

    parser = _build_argparser()
    args = parser.parse_args()
    cfg = _load_config_from_yaml(args.config)
    run_training(cfg, cli_args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
