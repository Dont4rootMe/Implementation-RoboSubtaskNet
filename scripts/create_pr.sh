#!/usr/bin/env bash
# Open the scaffold PR for RoboSubtaskNet against origin/main.
#
# Prerequisites:
#   - gh CLI installed and authenticated (`gh auth login`) OR the GH_TOKEN
#     environment variable set to a token with `repo` scope.
#   - The current branch is pushed to origin.
#
# Usage:
#   ./scripts/create_pr.sh                                       # uses current branch -> main
#   ./scripts/create_pr.sh scaffold/initial-implementation main  # explicit head + base
#
# The title and body below are tailored to the initial scaffold PR.
# Adjust them when reusing for follow-up PRs, or replace with `gh pr create`
# directly for a different change set.

set -euo pipefail

HEAD_BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
BASE_BRANCH="${2:-main}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install from https://cli.github.com/" >&2
  exit 1
fi

if [[ -z "${GH_TOKEN:-}" ]] && ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' or set GH_TOKEN." >&2
  exit 1
fi

gh pr create \
  --base "${BASE_BRANCH}" \
  --head "${HEAD_BRANCH}" \
  --title "Scaffold full RoboSubtaskNet implementation" \
  --body "$(cat <<'EOF'
## Summary

- Clean reimplementation of **RoboSubtaskNet** ([arXiv:2602.10015](https://arxiv.org/abs/2602.10015), Sharma et al.) — the original authors did not release code, so this PR builds the full pipeline from the spec in `IMPLEMENTATION_PLAN.md` (already on `main`).
- Lands the entire Phase 1 segmentation stack (feature extraction -> attention fusion -> Fibonacci-dilated multi-stage TCN -> composite loss -> evaluator) plus the optional Phase 2 robot-side scaffold (DMPs, detection, IBVS).
- 55 files, ~8.8k LOC, 42 Python modules — all pass `py_compile`; configs validate; the structure mirrors Section 3 of the plan 1:1.

## What's in the box

**Models** (`src/robosubtasknet/models/`)
- `AttentionFusion` — per-frame, per-channel sigmoid gate over RGB and flow features (Section 7)
- `FibonacciDilatedLayer` + `SingleStageTCN` with dilation schedule `d_l = F_{l+1}` and the Section 8.1 receptive-field formula
- `RoboSubtaskNet` — 4-stage refinement model returning per-stage logits

**Losses** (`src/robosubtasknet/losses/`)
- `TruncatedMSELoss` with the Section 9.2 log-prob truncation
- `TransitionLoss` — bilinear forbidden-mass formulation (Section 9.3) driven by a grammar-derived [C, C] bool matrix
- `CompositeLoss` aggregating CE + lambda * T-MSE + gamma * Trans over stages, returning per-component telemetry

**Features** (`src/robosubtasknet/features/`)
- I3D-R50 backbone wrapper (frozen Kinetics weights, 8x temporal downsample)
- TV-L1 and RAFT optical flow with the distributional-mismatch caveat documented
- `extract_features_for_video` orchestrating window/stride sliding + float16 storage per Section 6.3

**Training & eval** (`src/robosubtasknet/{training,eval}/`)
- `Trainer` with AMP, grad clipping, deterministic seeding, callback hooks
- `TensorBoardLogger`, `ModelCheckpoint`, `EarlyStopping`
- MS-TCN-compatible `f_score`, `edit_score`, `frame_accuracy`, `levenstein` (spelling preserved for parity) and a `SegmentationEvaluator` accumulating F1@10/25/50, Edit, Acc

**Phase 2 scaffold** (`src/robosubtasknet/execution/`)
- `DiscreteDMP` / `MultiDOFDMP` / `SubtaskDMPLibrary` (no `pydmps` dep)
- `ObjectDetector` / `PoseEstimator6D` protocols + `YOLOv8Detector` (lazy `ultralytics`)
- `IBVSController` with the Section 12.3 image-Jacobian P-controller

**Around the model**
- Hydra-style configs: `default`, `gtea` (11 classes), `breakfast` (48), `robosubtask` (9) + grammar YAML
- `RoboSubtaskDataset` supporting both MS-TCN concat-2048 `.npy` and our `.npz` format; `pad_collate` with `-100` padding for `ignore_index`
- CLI scripts: `extract_features`, `train` (Hydra+argparse fallback), `evaluate`, `visualize_segmentation`, `run_robot` (with `--dry-run`)
- Tests (8 files): fusion shape/grad/gate-range, fibonacci + receptive-field, loss zero/positive cases, metric edge cases incl. a hand-worked F1@50 example, grammar transitions, single-step no-NaN, single-video overfit (Section 14.2's "most important test")
- CI: GitHub Actions for pytest (Py 3.10 + 3.11 matrix) and ruff + mypy

## Why these specific choices

- **Bilinear transition loss**: Section 9.3 of the paper leaves the exact form underspecified; the bilinear formulation is differentiable, zero on grammar-consistent predictions, and the most defensible reading. Grid-searching gamma in {0.05, 0.1, 0.15, 0.2, 0.3} is listed in Section 16.1 as a follow-up.
- **TV-L1 default for custom data**: matches the Kinetics-pretrained I3D flow stream's input distribution. RAFT is wired up but flagged.
- **Hidden dim 64**: MS-TCN convention. Bumping to 128 is the first knob to try if GTEA F1@50 lands < 75% (Section 16.2 #3).
- **MS-TCN feature split**: GTEA/Breakfast use the published 2048-d features halved into RGB|Flow (Section 5.1) — fastest path to anchor against paper numbers before touching custom data.

## How to verify

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # unit tests (no GPU, no datasets needed)
pytest -q -m slow              # adds the single-video overfit sanity check
ruff check . && ruff format --check . && mypy src/
```

End-to-end runs (after `pip install -e ".[dev]"`):

```bash
python scripts/train.py --config configs/gtea.yaml     # needs MS-TCN GTEA features
python scripts/evaluate.py --checkpoint <ckpt> --dataset-config configs/gtea.yaml
```

## Reviewer focus points

- `losses/transition.py` formulation vs the paper's prose-only spec (Section 16.1 calls this out as an open decision).
- `data/dataset.py` `mode="mstcn"` half-split convention — first 1024 dims = RGB or Flow? The fastest way to verify is the single-stream ablation noted in Section 16.2 #1.
- Variable-length masking: every loss component honors `mask`, but worth double-checking against Section 16.2 #2.

## Phase / milestone

This lands Phase 0-2 of Section 15. Phase 3 (GTEA reproduction, target F1@50 >= 77.5) requires downloading MS-TCN's published features and is intentionally out of scope here.

Generated with Claude Code.
EOF
)"
