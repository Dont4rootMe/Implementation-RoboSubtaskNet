"""Evaluation entry-point: load a checkpoint, score a split, print metrics.

Mirrors Section 11 of ``IMPLEMENTATION_PLAN.md`` -- reports frame accuracy,
edit score, and segment-level F1 at IoU thresholds {10, 25, 50}::

    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --dataset-config configs/gtea.yaml \\
        --split test

Reads the dataset config the same way ``train.py`` does (OmegaConf compose),
rebuilds the model, loads the checkpoint, runs the model over the requested
split, and prints a small metrics table. Metric semantics match the MS-TCN
reference implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

# Make ``src/`` importable when the package isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Argparse-only CLI per Section 11 of the plan."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RoboSubtaskNet checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint .pt produced by scripts/train.py.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Path to the dataset YAML (e.g. configs/gtea.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "test", "val"),
        help="Which split to evaluate (looked up via the config's split files).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override; auto-selects 'cuda' if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the dataloader batch size.",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> Any:
    """Reuse the train.py loader so behaviour stays in lockstep."""
    # Local import to avoid pulling ``train.py``'s side effects when this
    # module is only imported (e.g. for testing).
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from train import _load_config_from_yaml  # noqa: E402

    return _load_config_from_yaml(config_path)


def _build_dataset(cfg: Any, split: str) -> Any:
    """Load the requested split's :class:`RoboSubtaskDataset`.

    Bundle resolution: ``train`` -> ``train_bundle``; everything else ->
    ``test_bundle`` (the config schema treats ``val`` and ``test`` the same).
    """
    from robosubtasknet.data import RoboSubtaskDataset

    bundle_key = "train_bundle" if split == "train" else "test_bundle"
    bundle = cfg.dataset.get(bundle_key, None)
    bundle_path = Path(bundle) if bundle else None

    mapping_file = cfg.dataset.get("mapping_file", None)
    mapping_path = Path(mapping_file) if mapping_file else None

    feature_dir = Path(cfg.dataset.feature_dir)
    feature_dim = int(cfg.model.feature_dim)

    fmt = str(cfg.dataset.get("feature_format", "npz")).lower()
    mode = "mstcn" if fmt in ("split2048", "mstcn") else "npz"

    return RoboSubtaskDataset(
        feature_dir=feature_dir,
        split_file=bundle_path,
        mapping_file=mapping_path,
        mode=mode,
        feature_dim=feature_dim,
    )


def _bg_class_from_mapping(cfg: Any) -> list[int] | None:
    mapping_file = cfg.dataset.get("mapping_file", None)
    if not mapping_file or not Path(mapping_file).exists():
        return None
    from robosubtasknet.data.dataset import load_mapping

    mapping = load_mapping(Path(mapping_file))
    if "background" in mapping:
        return [int(mapping["background"])]
    return None


def _print_metrics_table(metrics: dict[str, float]) -> None:
    """Render the canonical paper-style metrics row.

    Output mirrors Section 11.4 of the implementation plan: F1@10/25/50,
    Edit, Acc -- one number per column. Columns are kept narrow so the
    line fits in a typical 80-column terminal.
    """
    headers = ("F1@10", "F1@25", "F1@50", "Edit", "Acc")
    values = (
        metrics.get("F1@10", 0.0),
        metrics.get("F1@25", 0.0),
        metrics.get("F1@50", 0.0),
        metrics.get("edit", 0.0),
        metrics.get("acc", 0.0),
    )
    col = "  ".join(f"{h:>7s}" for h in headers)
    row = "  ".join(f"{v:7.2f}" for v in values)
    bar = "-" * len(col)
    print(col)
    print(bar)
    print(row)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a state-dict into ``model`` and return the full payload.

    Supports both the rich payloads produced by ``scripts/train.py`` (a dict
    with ``model_state_dict``) and bare ``state_dict`` files.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
        payload = {"model_state_dict": state_dict}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[eval] WARNING: missing keys on load: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[eval] WARNING: unexpected keys on load: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    return payload


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.dataset_config)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data
    dataset = _build_dataset(cfg, split=args.split)
    from robosubtasknet.data import pad_collate

    batch_size = args.batch_size if args.batch_size else int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.dataset.get("num_workers", 0)),
        pin_memory=bool(cfg.dataset.get("pin_memory", False)),
        collate_fn=pad_collate,
        drop_last=False,
    )

    # ---- Model
    from robosubtasknet.models import RoboSubtaskNet

    model = RoboSubtaskNet(
        num_stages=int(cfg.model.num_stages),
        num_layers=int(cfg.model.num_layers),
        feature_dim=int(cfg.model.feature_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
    )
    _load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # ---- Evaluator
    from robosubtasknet.eval import SegmentationEvaluator

    bg_class = _bg_class_from_mapping(cfg)
    evaluator = SegmentationEvaluator(bg_class=bg_class)

    print(
        f"[eval] checkpoint={args.checkpoint}  dataset={cfg.dataset.name}  "
        f"split={args.split}  videos={len(dataset)}  device={device}"
    )

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            flow = batch["flow"].to(device, non_blocking=True)
            labels = batch["labels"]
            mask = batch.get("mask")

            outputs = model(rgb, flow)
            preds = outputs[-1].argmax(dim=1).detach().cpu()
            labels_cpu = labels.detach().cpu()

            for b in range(preds.shape[0]):
                p = preds[b]
                g = labels_cpu[b]
                if mask is not None:
                    valid = mask[b].bool()
                    p = p[valid]
                    g = g[valid]
                evaluator.update(p.tolist(), g.tolist())

    metrics = evaluator.compute()
    print()
    _print_metrics_table(metrics)
    print()
    print(f"num_videos={int(metrics.get('num_videos', 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
