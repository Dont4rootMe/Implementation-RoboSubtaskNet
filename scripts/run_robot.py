"""Real-time-ish robot execution driver (Section 12).

Online loop::

    while running:
        frame    = capture()
        features = extract(frame)             # buffered window of frames
        subtask  = segment(features)          # final-stage argmax
        execute(subtask)                      # DMP rollout for the new label

Real-time caveats from Section 12.4: full I3D + TV-L1 cannot run real-time on
a single GPU, so we operate at the spec's relaxed target of ~1 fps of sub-task
*updates*. The buffered window is the same one the offline extractor uses
(16 frames, stride 8), so the segmentation model sees the exact feature
distribution it was trained on.

``--dry-run`` prints what each step would do *without* touching the camera,
the segmentation model, or the robot -- useful for wiring smoke tests on
machines that don't have hardware connected.

CLI::

    python scripts/run_robot.py \\
        --checkpoint checkpoints/best.pt \\
        --video-source 0 \\
        --robot-config configs/robot.yaml \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional

# Make ``src/`` importable when the package isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """CLI per the implementation-plan brief for this script."""
    parser = argparse.ArgumentParser(
        description="Online robot execution from a video feed (Section 12).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained segmentation checkpoint (.pt).",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        required=True,
        help=(
            "OpenCV VideoCapture source: integer for a webcam (e.g. '0'), "
            "filesystem path for a video file, or an RTSP/HTTP URL."
        ),
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        required=True,
        help="YAML describing the robot driver and per-sub-task primitives.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/robosubtask.yaml"),
        help="Dataset YAML used to look up num_classes and the label map.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="Number of frames per I3D clip (same as feature extractor).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between consecutive feature windows.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override; auto-detects CUDA/CPU.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Stop after this many loop iterations (0 = run until interrupted).",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=1.0,
        help=(
            "Target sub-task update rate (Hz). 1 Hz follows Section 12.4's "
            "real-time relaxation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the camera, segmentation, and robot. Print what would happen "
            "step-by-step. Useful for smoke testing without hardware."
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #


def _load_yaml(path: Path) -> Any:
    """Minimal YAML loader that doesn't require Hydra/OmegaConf to be present."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise RuntimeError(
            "Loading the robot config requires PyYAML. "
            "Install with `pip install pyyaml`."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_cfg(path: Path) -> Any:
    """Reuse train.py's loader for symmetry with the rest of the scripts."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from train import _load_config_from_yaml

    return _load_config_from_yaml(path)


# --------------------------------------------------------------------------- #
# Pieces of the live loop (kept small + injectable for dry-run / tests)
# --------------------------------------------------------------------------- #


class _VideoSource:
    """Thin wrapper over ``cv2.VideoCapture`` with a string/integer source."""

    def __init__(self, source: str) -> None:
        try:
            import cv2  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "run_robot.py requires opencv-python for video capture. "
                "Install with `pip install opencv-python`."
            ) from exc
        # Integer-looking strings (e.g. "0") refer to a webcam index.
        try:
            opened: Any = int(source)
        except ValueError:
            opened = source
        self._cap = cv2.VideoCapture(opened)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame  # BGR numpy array per cv2 convention.

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:  # pragma: no cover - cv2 lifetime jitter
            pass

    def __enter__(self) -> "_VideoSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _build_segmentation_model(cfg: Any, ckpt_path: Path, device: Any) -> Any:
    """Construct and weight-load the segmentation model."""
    import torch

    from robosubtasknet.models import RoboSubtaskNet

    model = RoboSubtaskNet(
        num_stages=int(cfg.model.num_stages),
        num_layers=int(cfg.model.num_layers),
        feature_dim=int(cfg.model.feature_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def _label_names_from_cfg(cfg: Any) -> list[str]:
    """Return ``[name_for_class_0, name_for_class_1, ...]`` ordered by index."""
    mapping_file = cfg.dataset.get("mapping_file", None)
    if not mapping_file or not Path(mapping_file).exists():
        return [str(i) for i in range(int(cfg.model.num_classes))]
    from robosubtasknet.data.dataset import load_mapping

    mapping = load_mapping(Path(mapping_file))
    inv = {v: k for k, v in mapping.items()}
    return [inv.get(i, str(i)) for i in range(int(cfg.model.num_classes))]


def _segment_features(model: Any, rgb: Any, flow: Any, device: Any) -> int:
    """Run the model on the buffered features and return the latest sub-task label."""
    import torch

    rgb_t = rgb.to(device).unsqueeze(0)   # [1, T_feat, D]
    flow_t = flow.to(device).unsqueeze(0)
    with torch.no_grad():
        preds = model.predict(rgb_t, flow_t)
    return int(preds[0, -1].item())


# --------------------------------------------------------------------------- #
# Main loop -- dry-run and live versions share the same skeleton.
# --------------------------------------------------------------------------- #


def _dry_run(args: argparse.Namespace, dataset_cfg: Any, robot_cfg: Any) -> int:
    """Print the would-be steps without touching hardware or models.

    Walks ``max_iters`` cycles (default 5 in dry-run) and emits a one-liner per
    step. The output is deterministic so it can be diffed in CI.
    """
    label_names = _label_names_from_cfg(dataset_cfg)
    primitives = (robot_cfg.get("primitives") or {}) if isinstance(robot_cfg, dict) else {}
    driver = (robot_cfg.get("driver") or "unspecified") if isinstance(robot_cfg, dict) else "unspecified"

    n_iters = args.max_iters or 5
    print("[dry-run] segmentation checkpoint:", args.checkpoint)
    print("[dry-run] video source:           ", args.video_source)
    print("[dry-run] robot config:           ", args.robot_config)
    print(f"[dry-run] robot driver:            {driver}")
    print(f"[dry-run] sub-task labels:        ", ", ".join(label_names))
    print(f"[dry-run] update target:          {args.update_hz:.2f} Hz "
          f"(period={1.0 / max(args.update_hz, 1e-3):.2f} s)")
    print(f"[dry-run] running {n_iters} fake iterations...")

    for i in range(n_iters):
        # Synthesize a plausible label rotation so the printout looks realistic.
        cls = i % len(label_names)
        name = label_names[cls]
        primitive = primitives.get(name, "(no primitive mapped)")
        print(
            f"[dry-run] step={i:03d}  "
            f"capture-frame -> extract-features (window={args.window}, "
            f"stride={args.stride}) -> segment=({cls}:{name}) -> "
            f"execute={primitive!r}"
        )
    print("[dry-run] complete.")
    return 0


def _live_loop(args: argparse.Namespace, dataset_cfg: Any, robot_cfg: Any) -> int:
    """Real online loop. Capture, segment, execute -- forever (or until --max-iters)."""
    import numpy as np
    import torch

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model
    model = _build_segmentation_model(dataset_cfg, args.checkpoint, device)
    label_names = _label_names_from_cfg(dataset_cfg)

    # ---- Feature extractors (lazy import: heavy deps).
    try:
        from robosubtasknet.features.i3d import I3DFeatureExtractor
    except ImportError as exc:
        raise ImportError(
            "Live execution needs I3DFeatureExtractor (pytorchvideo)."
        ) from exc

    rgb_model = I3DFeatureExtractor(modality="rgb", pretrained=True).to(device).eval()
    flow_model = I3DFeatureExtractor(modality="flow", pretrained=True).to(device).eval()

    # Optical flow estimator -- TV-L1 to match Kinetics pretraining (§6.2).
    try:
        from robosubtasknet.features.flow import compute_flow_from_frames
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise ImportError(
            "Optical flow extraction requires cv2 / opencv-contrib-python."
        ) from exc

    # ---- DMP / executor scaffolding (Phase 2; deliberately permissive).
    from robosubtasknet.execution.dmp import SubtaskDMPLibrary

    library = SubtaskDMPLibrary()  # empty; primitives can be loaded by ext code.
    primitives_map: dict[int, str] = {}
    if isinstance(robot_cfg, dict):
        for name, payload in (robot_cfg.get("primitives") or {}).items():
            if name in label_names:
                primitives_map[label_names.index(name)] = (
                    payload if isinstance(payload, str) else name
                )

    # ---- Capture
    frame_buffer: deque[np.ndarray] = deque(maxlen=int(args.window) + 1)
    period = 1.0 / max(float(args.update_hz), 1e-3)

    print(f"[run-robot] live loop @ {args.update_hz:.2f} Hz "
          f"(target period {period:.2f}s); window={args.window}, stride={args.stride}")
    print(f"[run-robot] device={device}, sub-tasks={label_names}")

    last_label: Optional[int] = None
    iteration = 0
    try:
        with _VideoSource(args.video_source) as cap:
            while True:
                tic = time.time()

                frame = cap.read()
                if frame is None:
                    print("[run-robot] video source returned no frame; exiting.")
                    break
                frame_buffer.append(frame)

                if len(frame_buffer) >= args.window:
                    clip = np.stack(list(frame_buffer)[-args.window:], axis=0)  # [W, H, W_, 3]
                    rgb_features = _extract_rgb_feature(
                        rgb_model, clip, device=device
                    )
                    flow_features = _extract_flow_feature(
                        flow_model, clip, device=device,
                        flow_fn=compute_flow_from_frames,
                    )
                    label_idx = _segment_features(
                        model, rgb_features, flow_features, device
                    )
                    if label_idx != last_label:
                        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
                        primitive = primitives_map.get(label_idx, "(no primitive)")
                        print(
                            f"[run-robot] iter={iteration:04d}  "
                            f"label={label_idx}:{name}  -> execute={primitive}"
                        )
                        last_label = label_idx
                else:
                    if iteration % 5 == 0:
                        print(
                            f"[run-robot] buffering... "
                            f"({len(frame_buffer)}/{args.window} frames)"
                        )

                iteration += 1
                if args.max_iters and iteration >= args.max_iters:
                    print(f"[run-robot] reached --max-iters={args.max_iters}; stopping.")
                    break

                # Throttle to the target update rate (Section 12.4 ~1 fps).
                elapsed = time.time() - tic
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("[run-robot] interrupted by user; shutting down.")
    return 0


def _extract_rgb_feature(rgb_model: Any, clip: Any, device: Any) -> Any:
    """Compute a single-window RGB feature from a ``[W, H, W_, 3]`` BGR clip."""
    import cv2
    import numpy as np
    import torch

    # Resize -> RGB float -> Kinetics normalize. Spatial size 224x224 per I3D.
    rgb_frames = []
    for f in clip:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.resize(f_rgb, (224, 224))
        rgb_frames.append(f_rgb)
    rgb_arr = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0  # [T, H, W, 3]
    tensor = (
        torch.from_numpy(rgb_arr)
        .permute(3, 0, 1, 2)
        .contiguous()
        .to(device)
    )
    tensor = rgb_model.normalize_rgb(tensor)
    feat = rgb_model.extract_clip_features(
        tensor, window=clip.shape[0], stride=clip.shape[0]
    )  # [T_feat, 1024]
    return feat.cpu()


def _extract_flow_feature(flow_model: Any, clip: Any, device: Any, flow_fn: Any) -> Any:
    """Compute a single-window flow feature using TV-L1 + I3D flow stream."""
    import numpy as np
    import torch

    flow_arr = flow_fn(clip, method="tvl1")  # [T-1, H, W, 2] already normalized
    flow_arr = flow_arr.astype(np.float32)
    # Move to torch and resize spatially to match I3D's expected 224x224.
    flow_t = torch.from_numpy(flow_arr).permute(0, 3, 1, 2)  # [T-1, 2, H, W]
    flow_t = torch.nn.functional.interpolate(
        flow_t, size=(224, 224), mode="bilinear", align_corners=False
    )
    # I3D expects [C, T, H, W]; squeeze through extract_clip_features.
    tensor = flow_t.permute(1, 0, 2, 3).contiguous().to(device)
    feat = flow_model.extract_clip_features(
        tensor, window=tensor.shape[1], stride=tensor.shape[1]
    )  # [T_feat, 1024]
    return feat.cpu()


def main() -> int:
    args = parse_args()
    dataset_cfg = _load_dataset_cfg(args.dataset_config)
    robot_cfg = _load_yaml(args.robot_config)

    if args.dry_run:
        return _dry_run(args, dataset_cfg, robot_cfg)
    return _live_loop(args, dataset_cfg, robot_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
