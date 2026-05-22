#!/usr/bin/env bash
# Auto-label a LeRobot dataset → produce a new LeRobot dataset with
# predicted action_text_id per frame. Video mp4s are hard-linked (os.link),
# not copied, to save disk space.
#
# Usage:
#   ./scripts/inference.sh <input_lerobot> <output_lerobot> <checkpoint> <camera_key>
#
# Example:
#   ./scripts/inference.sh /data/raw_demos /data/auto_labeled ./checkpoints/best.pt \
#       observation.images.head_left_eye
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<EOF
usage: $0 <input_lerobot> <output_lerobot> <checkpoint> <camera_key>
example:
  $0 /data/raw_demos /data/auto_labeled ./ckpts/best.pt observation.images.head_left_eye
EOF
  exit 1
fi

INPUT="$1"; OUTPUT="$2"; CKPT="$3"; CAMERA="$4"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

if [[ ! -d "$INPUT" ]]; then
  echo "error: input dataset not found: $INPUT" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "error: checkpoint not found: $CKPT" >&2
  exit 1
fi

python "$REPO_ROOT/scripts/inference_lerobot.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --checkpoint "$CKPT" \
  --camera "$CAMERA"

echo "[inference.sh] done. output dataset: $OUTPUT"
