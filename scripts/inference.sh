#!/usr/bin/env bash
# Run V2 e2e inference: boundary segmenter + VLM labeler -> new LeRobot dataset.
#
# Auto-labels a LeRobot dataset by (1) detecting subtask boundaries with the
# Stage-1 MS-TCN segmenter, then (2) labeling each segment with the Stage-2
# Qwen2-VL LoRA adapter. Video mp4s in the output dataset are hard-linked
# (os.link), not copied, to save disk space.
#
# Usage:
#   ./scripts/inference.sh <input_lerobot> <output_lerobot> <boundary_ckpt> <vlm_lora_dir> <camera_key>
#
# Example:
#   ./scripts/inference.sh /data/raw_demos ./auto_labeled \
#     ./ckpts/boundary/final.pt ./ckpts/vlm/lora_adapter \
#     observation.images.head_left_eye
set -euo pipefail

if [[ $# -ne 5 ]]; then
  cat >&2 <<EOF
usage: $0 <input_lerobot> <output_lerobot> <boundary_ckpt> <vlm_lora_dir> <camera_key>
example:
  $0 /data/raw_demos ./auto_labeled \\
    ./ckpts/boundary/final.pt ./ckpts/vlm/lora_adapter \\
    observation.images.head_left_eye
EOF
  exit 1
fi

INPUT="$1"; OUTPUT="$2"; BOUNDARY_CKPT="$3"; VLM_DIR="$4"; CAMERA="$5"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

if [[ ! -d "$INPUT" ]]; then
  echo "error: input dataset not found: $INPUT" >&2
  exit 1
fi
if [[ ! -f "$BOUNDARY_CKPT" ]]; then
  echo "error: boundary checkpoint not found: $BOUNDARY_CKPT" >&2
  exit 1
fi
if [[ ! -d "$VLM_DIR" ]]; then
  echo "error: vlm lora directory not found: $VLM_DIR" >&2
  exit 1
fi

python "$REPO_ROOT/scripts/inference_pipeline.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --boundary-checkpoint "$BOUNDARY_CKPT" \
  --vlm-checkpoint "$VLM_DIR" \
  --camera "$CAMERA"

echo "[inference.sh] done. output dataset: $OUTPUT"
