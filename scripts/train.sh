#!/usr/bin/env bash
# Train RoboSubtaskNet on one or more LeRobot datasets.
#
# Usage:
#   ./scripts/train.sh <camera_key> <cache_dir> <output_dir> <lerobot_root1> [lerobot_root2 ...]
#
# The same camera key is applied to every input dataset. For per-dataset
# cameras, call the Python scripts directly with paired --input / --camera lists:
#   python scripts/extract_features_lerobot.py --input D1 D2 --camera C1 C2 ...
#   python scripts/train_lerobot.py            --input D1 D2 --camera C1 C2 ...
#
# Pipeline:
#   1) extract_features_lerobot.py  →  <cache>/<dataset>__episode_<M>.npz + label_map.json
#   2) train_lerobot.py             →  <output>/best.pt, final.pt, metadata.json
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat >&2 <<EOF
usage: $0 <camera_key> <cache_dir> <output_dir> <lerobot_root1> [lerobot_root2 ...]
example:
  $0 observation.images.head_left_eye ./features ./checkpoints /data/lerobot_dms1 /data/lerobot_dms2
EOF
  exit 1
fi

CAMERA="$1"; CACHE="$2"; OUT="$3"; shift 3
INPUTS=("$@")
LABEL_MAP="$CACHE/label_map.json"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

mkdir -p "$CACHE" "$OUT"

echo "[train.sh] step 1/2: extracting features from ${#INPUTS[@]} dataset(s)..."
python "$REPO_ROOT/scripts/extract_features_lerobot.py" \
  --input "${INPUTS[@]}" \
  --camera "$CAMERA" \
  --cache "$CACHE" \
  --label-map "$LABEL_MAP"

echo "[train.sh] step 2/2: training..."
python "$REPO_ROOT/scripts/train_lerobot.py" \
  --input "${INPUTS[@]}" \
  --camera "$CAMERA" \
  --cache "$CACHE" \
  --label-map "$LABEL_MAP" \
  --output "$OUT" \
  "${@:5}"  # forward any extra args (e.g. --epochs 100 --lr 1e-3)

echo "[train.sh] done. checkpoint dir: $OUT"
