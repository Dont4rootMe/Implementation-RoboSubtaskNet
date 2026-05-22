#!/usr/bin/env bash
# Train RoboSubtaskNet on one or more LeRobot datasets.
#
# Each LeRobot dataset can have a differently-named main camera, so the
# camera key is attached per dataset using "path:camera_key" notation.
#
# Usage:
#   ./scripts/train.sh <cache_dir> <output_dir> <root1>:<camera1> [<root2>:<camera2> ...] [-- extra_args_for_python]
#
# Examples:
#   # single dataset
#   ./scripts/train.sh ./features ./checkpoints \
#     /data/lerobot_d1:observation.images.head_left_eye
#
#   # multiple datasets, each with its own camera
#   ./scripts/train.sh ./features ./checkpoints \
#     /data/lerobot_d1:observation.images.head_left_eye \
#     /data/lerobot_d2:observation.images.front_cam \
#     /data/lerobot_d3:observation.images.wrist_cam
#
#   # forward extra flags to train_lerobot.py (--epochs, --lr, --grammar, etc.)
#   ./scripts/train.sh ./features ./checkpoints \
#     /data/lerobot_d1:observation.images.head_left_eye \
#     --epochs 100 --lr 1e-3
#
# Pipeline:
#   1) extract_features_lerobot.py  →  <cache>/<dataset>__episode_<M>.npz + label_map.json
#   2) train_lerobot.py             →  <output>/best.pt, final.pt, metadata.json
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
usage: $0 <cache_dir> <output_dir> <root1>:<camera1> [<root2>:<camera2> ...] [--flag value ...]
example:
  $0 ./features ./checkpoints \\
    /data/lerobot_d1:observation.images.head_left_eye \\
    /data/lerobot_d2:observation.images.front_cam
EOF
  exit 1
fi

CACHE="$1"; OUT="$2"; shift 2

# Parse positional <path>:<camera> pairs until we hit a --flag, then
# forward everything else to train_lerobot.py.
INPUTS=()
CAMERAS=()
EXTRA=()
parsing_pairs=true
for arg in "$@"; do
  if $parsing_pairs && [[ "$arg" != --* ]]; then
    if [[ "$arg" == *":"* ]]; then
      INPUTS+=("${arg%%:*}")
      CAMERAS+=("${arg#*:}")
    else
      echo "error: missing camera key for '$arg'. Use <path>:<camera_key>." >&2
      exit 1
    fi
  else
    parsing_pairs=false
    EXTRA+=("$arg")
  fi
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "error: no <path>:<camera> pairs provided." >&2
  exit 1
fi

LABEL_MAP="$CACHE/label_map.json"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

mkdir -p "$CACHE" "$OUT"

echo "[train.sh] ${#INPUTS[@]} dataset(s):"
for i in "${!INPUTS[@]}"; do
  echo "  - ${INPUTS[$i]}  (camera: ${CAMERAS[$i]})"
done

echo "[train.sh] step 1/2: extracting features..."
python "$REPO_ROOT/scripts/extract_features_lerobot.py" \
  --input "${INPUTS[@]}" \
  --camera "${CAMERAS[@]}" \
  --cache "$CACHE" \
  --label-map "$LABEL_MAP"

echo "[train.sh] step 2/2: training..."
python "$REPO_ROOT/scripts/train_lerobot.py" \
  --input "${INPUTS[@]}" \
  --camera "${CAMERAS[@]}" \
  --cache "$CACHE" \
  --label-map "$LABEL_MAP" \
  --output "$OUT" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "[train.sh] done. checkpoint dir: $OUT"
