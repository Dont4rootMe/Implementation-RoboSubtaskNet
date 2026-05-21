"""Render predicted vs. ground-truth segment strips for one video.

Reads a trained checkpoint, runs it on a single video, and saves a side-by-side
"strip" PNG suitable for the qualitative-results notebook (Section 13.3 of the
plan -- "log sample segmentations as image strips, predicted vs ground truth").

CLI::

    python scripts/visualize_segmentation.py \\
        --checkpoint checkpoints/best.pt \\
        --video-id S1_Cheese_C1 \\
        --out logs/strips/S1_Cheese_C1.png \\
        --dataset-config configs/gtea.yaml
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

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """CLI per the implementation-plan brief for this script."""
    parser = argparse.ArgumentParser(
        description="Render predicted vs ground-truth segment strips for one video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint .pt produced by scripts/train.py.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="Video stem (without extension) present in the dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/gtea.yaml"),
        help="Dataset YAML; tells us where to find features and labels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override; auto-selects CPU/CUDA otherwise.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title (defaults to the video id).",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> Any:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from train import _load_config_from_yaml

    return _load_config_from_yaml(path)


def _build_dataset(cfg: Any) -> Any:
    """Build a dataset covering *all* feature files in the config's dir.

    We bypass the split bundles here -- the user passed an explicit video id
    and we want to be able to find it regardless of which split it lives in.
    """
    from robosubtasknet.data import RoboSubtaskDataset

    feature_dir = Path(cfg.dataset.feature_dir)
    mapping_file = cfg.dataset.get("mapping_file", None)
    mapping_path = Path(mapping_file) if mapping_file else None

    fmt = str(cfg.dataset.get("feature_format", "npz")).lower()
    mode = "mstcn" if fmt in ("split2048", "mstcn") else "npz"
    feature_dim = int(cfg.model.feature_dim)

    return RoboSubtaskDataset(
        feature_dir=feature_dir,
        split_file=None,
        mapping_file=mapping_path,
        mode=mode,
        feature_dim=feature_dim,
    )


def _find_sample(dataset: Any, video_id: str) -> dict[str, Any]:
    """Pull the sample for ``video_id``; raise if it isn't part of the dataset."""
    try:
        idx = dataset.video_ids.index(video_id)
    except ValueError as exc:  # pragma: no cover - control-flow guard
        raise KeyError(
            f"Video id {video_id!r} not found in dataset at "
            f"{dataset.feature_dir}. Available examples: "
            f"{dataset.video_ids[:5]}..."
        ) from exc
    return dataset[idx]


def _label_palette(num_classes: int) -> np.ndarray:
    """Return a stable ``[C, 3]`` RGB palette suitable for label maps.

    We sample matplotlib's ``tab20`` for the first 20 classes and fall back to
    a ``viridis`` ramp for anything beyond that. The palette is deterministic
    so visualisations are comparable across runs.
    """
    import matplotlib.pyplot as plt

    if num_classes <= 20:
        cmap = plt.get_cmap("tab20", num_classes)
    else:
        cmap = plt.get_cmap("viridis", num_classes)
    return np.asarray([cmap(i)[:3] for i in range(num_classes)], dtype=np.float32)


def _strip_image(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert a ``[T]`` label vector to an ``[H, T, 3]`` strip for matplotlib.

    ``H`` is chosen so the strip is tall enough to read but doesn't dominate
    the figure. Out-of-range labels render as zeros (visually distinct gray).
    """
    labels = np.asarray(labels, dtype=np.int64)
    valid = (labels >= 0) & (labels < palette.shape[0])
    strip = np.zeros((labels.shape[0], 3), dtype=np.float32)
    strip[valid] = palette[labels[valid]]
    # Stack along a height axis so matplotlib displays a tall rectangle.
    return np.broadcast_to(strip[None, :, :], (40, labels.shape[0], 3)).copy()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.dataset_config)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data
    dataset = _build_dataset(cfg)
    sample = _find_sample(dataset, args.video_id)
    rgb = sample["rgb"].unsqueeze(0).to(device)
    flow = sample["flow"].unsqueeze(0).to(device)
    labels_gt = sample["labels"].cpu().numpy().astype(np.int64)

    # ---- Model
    from robosubtasknet.models import RoboSubtaskNet

    model = RoboSubtaskNet(
        num_stages=int(cfg.model.num_stages),
        num_layers=int(cfg.model.num_layers),
        feature_dim=int(cfg.model.feature_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model.predict(rgb, flow).squeeze(0).cpu().numpy().astype(np.int64)

    # Align lengths defensively -- the dataset already truncates to the
    # min(rgb, flow, labels) but the model can still emit a sequence of a
    # slightly different temporal length if upstream changes the I3D stride.
    T = min(preds.shape[0], labels_gt.shape[0])
    preds = preds[:T]
    labels_gt = labels_gt[:T]

    # ---- Render via matplotlib (imported lazily so --help is cheap).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = _label_palette(int(cfg.model.num_classes))
    strip_gt = _strip_image(labels_gt, palette)
    strip_pred = _strip_image(preds, palette)

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(max(8, T / 30.0), 3.2), sharex=True
    )
    title = args.title if args.title else f"{args.video_id}"
    fig.suptitle(f"{title} — segmentation strips (top: GT, bottom: pred)")
    axes[0].imshow(strip_gt, aspect="auto", interpolation="nearest")
    axes[0].set_ylabel("GT")
    axes[0].set_yticks([])
    axes[1].imshow(strip_pred, aspect="auto", interpolation="nearest")
    axes[1].set_ylabel("Pred")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Frame (I3D-feature index)")

    # Best-effort legend: pull class names from the mapping when available.
    legend_handles: list[Patch] = []
    mapping_file = cfg.dataset.get("mapping_file", None)
    name_for: dict[int, str] = {}
    if mapping_file and Path(mapping_file).exists():
        from robosubtasknet.data.dataset import load_mapping

        m = load_mapping(Path(mapping_file))
        name_for = {v: k for k, v in m.items()}
    unique_classes = sorted(
        set(int(x) for x in np.concatenate([labels_gt, preds]))
        - {-100}  # ignore CE padding
    )
    for cls in unique_classes:
        if 0 <= cls < palette.shape[0]:
            name = name_for.get(int(cls), str(int(cls)))
            legend_handles.append(Patch(color=palette[int(cls)], label=name))
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 6),
            bbox_to_anchor=(0.5, -0.02),
            fontsize="small",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
