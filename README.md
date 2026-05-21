# RoboSubtaskNet

A clean, open-source reimplementation of **RoboSubtaskNet** — temporal sub-task segmentation for human-to-robot skill transfer in real-world environments (Sharma et al., [arXiv:2602.10015](https://arxiv.org/abs/2602.10015)). The pipeline extracts I3D features from RGB and optical-flow streams of demonstration videos, fuses them with a learned per-frame attention gate, refines per-frame sub-task predictions through a Fibonacci-dilated multi-stage temporal convolutional network, and trains under a composite loss (cross-entropy + truncated MSE + transition-aware penalty). The segmentation output maps to a small vocabulary of manipulation primitives (reach, pick, place, pour, wipe, move, retract, give) suitable for downstream robot execution.

The original authors did not release code. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full design spec, architecture details, datasets, training recipe, evaluation metrics, and phased milestones.

## Quick start

```bash
git clone https://github.com/robosubtasknet/robosubtasknet.git
cd robosubtasknet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Optional extras:

- `pip install -e ".[flow-tvl1]"` — TV-L1 optical flow via `opencv-contrib-python`.
- `pip install -e ".[robot]"` — DMP, detection, and kinematics dependencies for the Phase 2 robot execution stack.

## Repository layout

The repo follows a `src/` layout with the package at `src/robosubtasknet/`. Top-level directories:

- `src/robosubtasknet/` — library code (`data/`, `features/`, `models/`, `losses/`, `training/`, `eval/`, `execution/`).
- `configs/` — Hydra configs (default + per-dataset overrides).
- `scripts/` — entry points for feature extraction, training, evaluation, and robot execution.
- `tests/` — unit and integration tests.
- `data/` — datasets, annotations, precomputed features, and splits (gitignored except for `.gitkeep` markers).
- `checkpoints/`, `logs/` — training artifacts (gitignored).
- `notebooks/` — exploratory and qualitative analysis notebooks.

See Section 3 of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full tree.

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full text.
