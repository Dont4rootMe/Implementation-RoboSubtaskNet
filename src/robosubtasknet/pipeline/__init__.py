"""End-to-end orchestration for the two-stage RoboSubtaskNet inference pipeline.

The :func:`run_inference_pipeline` function chains the Stage-1 boundary
segmenter (MS-TCN over fused I3D RGB + flow features) with the Stage-2
Qwen2-VL LoRA segment labeler to auto-label a LeRobot dataset. The input
dataset is never modified; a fresh LeRobot dataset is materialized at the
output path with hard-linked videos and a freshly built ``action_text.json``
vocabulary derived from the predicted subtask phrases.
"""

from .inference import run_inference_pipeline

__all__ = ["run_inference_pipeline"]
