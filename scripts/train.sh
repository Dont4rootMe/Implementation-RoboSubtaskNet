#!/usr/bin/env bash
# Train the V2 pipeline: Stage 1 (boundary segmenter) + Stage 2 (VLM LoRA).
#
# Each LeRobot dataset can have a differently-named main camera, so the
# camera key is attached per dataset using "path:camera_key" notation.
#
# Usage:
#   ./scripts/train.sh <boundary_cache_dir> <output_dir> <root1>:<camera1> [<root2>:<camera2> ...]
#
# Examples:
#   # single dataset
#   ./scripts/train.sh ./boundary_cache ./out \
#     /data/lerobot_d1:observation.images.head_left_eye
#
#   # multiple datasets, each with its own camera
#   ./scripts/train.sh ./boundary_cache ./out \
#     /data/lerobot_d1:observation.images.head_left_eye \
#     /data/lerobot_d2:observation.images.front_cam \
#     /data/lerobot_d3:observation.images.wrist_cam
#
# Output structure:
#   <output_dir>/boundary/    # boundary checkpoint, tb logs, final.pt
#   <output_dir>/vlm/         # VLM LoRA adapter, training logs
#
# Pipeline:
#   1) scripts/extract_features_boundary.py  ->  <boundary_cache>/<dataset>__episode_<M>.npz
#   2) scripts/train_boundary.py             ->  <output>/boundary/final.pt
#   3) scripts/train_vlm.py                  ->  <output>/vlm/lora_adapter
#
# This script chains the two stages with sensible defaults. For custom
# hyperparameters, invoke train_boundary.py / train_vlm.py directly.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
usage: $0 <boundary_cache_dir> <output_dir> <root1>:<camera1> [<root2>:<camera2> ...]
example:
  $0 ./boundary_cache ./out \\
    /data/lerobot_d1:observation.images.head_left_eye \\
    /data/lerobot_d2:observation.images.front_cam
EOF
  exit 1
fi

CACHE="$1"; OUT="$2"; shift 2

# Parse positional <path>:<camera> pairs. No extra-args forwarding -- users
# override per-stage hyperparameters by calling the Python scripts directly.
INPUTS=()
CAMERAS=()
for arg in "$@"; do
  if [[ "$arg" == --* ]]; then
    echo "error: unexpected flag '$arg'. train.sh accepts only <path>:<camera> pairs;" >&2
    echo "       for custom hyperparameters, run train_boundary.py / train_vlm.py directly." >&2
    exit 1
  fi
  if [[ "$arg" == *":"* ]]; then
    INPUTS+=("${arg%%:*}")
    CAMERAS+=("${arg#*:}")
  else
    echo "error: missing camera key for '$arg'. Use <path>:<camera_key>." >&2
    exit 1
  fi
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "error: no <path>:<camera> pairs provided." >&2
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

BOUNDARY_OUT="$OUT/boundary"
VLM_OUT="$OUT/vlm"

mkdir -p "$CACHE" "$BOUNDARY_OUT" "$VLM_OUT"

echo "[train.sh] ${#INPUTS[@]} dataset(s):"
for i in "${!INPUTS[@]}"; do
  echo "  - ${INPUTS[$i]}  (camera: ${CAMERAS[$i]})"
done
echo "[train.sh] boundary cache : $CACHE"
echo "[train.sh] boundary output: $BOUNDARY_OUT"
echo "[train.sh] vlm output     : $VLM_OUT"

echo "[train.sh] step 1/3: extracting Stage-1 boundary features..."
python "$REPO_ROOT/scripts/extract_features_boundary.py" \
  --input "${INPUTS[@]}" \
  --camera "${CAMERAS[@]}" \
  --cache "$CACHE"

echo "[train.sh] step 2/3: training Stage-1 boundary segmenter..."
python "$REPO_ROOT/scripts/train_boundary.py" \
  --cache "$CACHE" \
  --output "$BOUNDARY_OUT"

echo "[train.sh] step 3/3: LoRA fine-tuning Stage-2 VLM labeler..."
python "$REPO_ROOT/scripts/train_vlm.py" \
  --input "${INPUTS[@]}" \
  --camera "${CAMERAS[@]}" \
  --output "$VLM_OUT"

echo "[train.sh] done."
echo "  boundary checkpoint: $BOUNDARY_OUT/final.pt"
echo "  vlm lora adapter   : $VLM_OUT"
