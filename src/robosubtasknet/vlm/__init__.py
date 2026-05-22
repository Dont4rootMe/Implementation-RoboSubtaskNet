"""Stage-2 vision-language model wrapper for subtask labeling.

The :class:`SegmentLabelerVLM` class wraps the Qwen2-VL-Instruct family with
optional LoRA adapters (via ``peft``) for fine-tuning. :func:`make_segment_chat`
builds the chat-format payload expected by the Qwen2-VL processor.
"""

from .model import SegmentLabelerVLM, make_segment_chat

__all__ = ["SegmentLabelerVLM", "make_segment_chat"]
