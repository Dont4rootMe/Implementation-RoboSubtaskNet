"""Collator for the Stage-2 Qwen2-VL fine-tuning dataset.

The Stage-1 boundary detector produces a sequence of frame segments. The
Stage-2 vision-language model (Qwen2-VL-Instruct, optionally LoRA-tuned) takes
each segment and emits a short natural-language label describing the subtask
performed in it.

For supervised fine-tuning we feed the model a chat-formatted prompt of the
form::

    <|im_start|>system
    You are a robotics assistant ...<|im_end|>
    <|im_start|>user
    Task description: <task_text>
    <video frames here><|im_end|>
    <|im_start|>assistant
    <subtask_text><|im_end|>

and supervise *only* the assistant turn (everything before it, including the
system / user blocks and the vision tokens, is masked with ``ignore_index``).

This module owns the dataloader-side glue that turns a list of raw
``{frames, task_text, subtask_text, ...}`` samples into a batch dictionary
accepted by ``Qwen2VLForConditionalGeneration.forward``. The heavy lifting -
chat templating, image / video patch extraction, vision token expansion - is
delegated to the HuggingFace processor; this class is responsible for:

1. Building the chat ``messages`` for each sample via
   :func:`robosubtasknet.vlm.model.make_segment_chat`.
2. Running the processor twice per sample - once on the prompt alone and once
   on the prompt + assistant target - so we can locate the assistant span by
   token-length difference rather than by re-tokenising the special tokens
   ourselves (this side-steps the well-known ``<|im_start|>assistant`` /
   ``\n`` tokenisation ambiguity).
3. Padding ``input_ids`` / ``attention_mask`` / ``labels`` to the longest
   sample in the batch and stacking / concatenating the vision tensors that
   the processor returned (``pixel_values_videos``, ``video_grid_thw``, ...)
   in the layout Qwen2-VL's forward expects.

Vision tensor keys are *not* hard-coded: whatever non-text fields the
processor returns are forwarded as-is, which keeps the collator working
across minor Qwen2-VL processor revisions and lets the same code path serve
image-only and video-frame inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch

__all__ = ["VLMSegmentCollator"]


# Text keys we know how to pad ourselves. Everything else returned by the
# processor (pixel_values_videos, video_grid_thw, image_grid_thw, ...) is
# treated as a vision tensor and stacked / concatenated rather than padded.
_TEXT_KEYS = ("input_ids", "attention_mask")


class VLMSegmentCollator:
    """Collate ``{frames, task_text, subtask_text, ...}`` into Qwen2-VL inputs.

    The collator is *processor-driven*: it never tries to know the exact
    tokeniser vocabulary or the exact vision-tensor key names. It builds the
    chat messages, hands them to the processor along with the frames, then
    pads / stacks the resulting tensors into a batch.

    Label masking is done by *length comparison* rather than by string
    matching. For each sample we run the processor twice:

    * once on the prompt only (``add_generation_prompt=True``,
      ``training_target=None``), giving ``L_prompt`` text tokens, and
    * once on the prompt + assistant target (``add_generation_prompt=False``,
      ``training_target=subtask_text``), giving ``L_full`` text tokens.

    We then clone ``input_ids`` into ``labels`` and overwrite the first
    ``L_prompt`` positions with ``ignore_index``. Everything from position
    ``L_prompt`` onwards (the assistant turn, including its closing
    ``<|im_end|>``) is supervised. This is more robust than searching for
    ``<|im_start|>assistant\\n`` in the token stream because the assistant
    header tokens themselves are masked, which matches HuggingFace's own
    ``DataCollatorForCompletionOnlyLM`` semantics and avoids edge cases where
    ``<|im_start|>``, ``assistant``, and ``\\n`` merge or split differently
    depending on the BPE neighbours.

    Parameters
    ----------
    processor
        Qwen2-VL processor obtained from ``AutoProcessor.from_pretrained``.
        Must accept ``text=...``, ``videos=...`` (or ``images=...``) and
        ``return_tensors='pt'``, and must expose ``apply_chat_template``.
    tokenizer
        The underlying tokenizer. Usually ``processor.tokenizer``; passed
        explicitly so callers can substitute a wrapped variant (e.g. one with
        extra pad-token configuration) without monkey-patching the processor.
    max_length
        Maximum sequence length after tokenisation. Sequences longer than
        this are truncated; the assistant turn is *not* re-aligned after
        truncation, so callers should keep ``max_length`` comfortably larger
        than the longest expected prompt+target combination.
    ignore_index
        Fill value used to mask positions in ``labels`` that should not
        contribute to the loss (system + user + vision + assistant header,
        and also any padding positions added during batch collation).
    make_chat
        Optional override for the chat-message builder. Default is
        :func:`robosubtasknet.vlm.model.make_segment_chat`, imported lazily.
        The override is invoked as ``make_chat(task_text, training_target=...)``
        and must return a list of ``{role, content}`` dicts compatible with
        ``processor.apply_chat_template``.
    """

    def __init__(
        self,
        processor: Any,
        tokenizer: Any,
        max_length: int = 1024,
        ignore_index: int = -100,
        make_chat: Callable[..., List[Dict[str, Any]]] | None = None,
    ) -> None:
        if processor is None:
            raise ValueError("processor must not be None.")
        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length: int = int(max_length)
        self.ignore_index: int = int(ignore_index)
        self._make_chat_override = make_chat

        # Resolve a pad id that's safe to use both for input_ids (where the
        # attention mask will be 0) and for labels (which we always mask to
        # ignore_index, so the actual numeric value never reaches the loss).
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            # Last-ditch fallback. The attention mask will hide these
            # positions anyway, but we still need a concrete integer.
            pad_id = 0
        self.pad_token_id: int = int(pad_id)

    # ------------------------------------------------------------------
    # Chat builder
    # ------------------------------------------------------------------
    def _get_make_chat(self) -> Callable[..., List[Dict[str, Any]]]:
        """Resolve the chat-message builder lazily.

        We import :func:`robosubtasknet.vlm.model.make_segment_chat` on first
        use so the collator module can be imported in environments where
        ``transformers`` (a dependency of ``vlm.model``) is missing - for
        instance during static analysis or when only the Stage-1 boundary
        path is exercised in tests.
        """
        if self._make_chat_override is not None:
            return self._make_chat_override
        # Local import to avoid a hard dependency on transformers at import
        # time. Cached on the instance after first lookup.
        from robosubtasknet.vlm.model import make_segment_chat

        self._make_chat_override = make_segment_chat
        return make_segment_chat

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------
    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Qwen2-VL training batch from raw sample dicts.

        Each element of ``batch`` must contain:

        * ``frames`` - sequence of PIL images or ``np.ndarray`` frames for a
          single segment. Forwarded to the processor as a one-element
          ``videos`` list (Qwen2-VL's "list of frames is a video" path).
        * ``task_text`` - top-level task description (string).
        * ``subtask_text`` - ground-truth assistant target (string).

        Any other keys are ignored by this collator but may be consumed by
        upstream wrappers (e.g. for logging segment provenance).

        Returns
        -------
        dict
            Padded batch tensors. ``input_ids``, ``attention_mask`` and
            ``labels`` are 2-D ``LongTensor``\\s of shape ``[B, T_max]``.
            Vision tensors keep whatever leading dimension the processor
            produced and are concatenated along ``dim=0`` so that the batch
            dimension is implicit in the per-sample vision-token layout that
            Qwen2-VL's forward expects. A ``video_grid_thw`` /
            ``image_grid_thw`` tensor, when produced by the processor, is
            similarly concatenated.
        """
        if not batch:
            raise ValueError("VLMSegmentCollator received an empty batch.")

        make_chat = self._get_make_chat()

        per_sample: List[Dict[str, Any]] = []
        prompt_lens: List[int] = []
        for sample in batch:
            self._validate_sample(sample)
            frames = sample["frames"]
            task_text = sample["task_text"]
            subtask_text = sample["subtask_text"]

            # 1) Prompt-only pass: tells us where the assistant turn starts.
            prompt_messages = make_chat(task_text, training_target=None)
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])

            # 2) Full pass: prompt + assistant target. This is what actually
            # gets fed into the model. Re-running the processor (rather than
            # appending tokens) keeps any vision-token expansion consistent
            # with the text and is cheap relative to the model forward.
            full_messages = make_chat(task_text, training_target=subtask_text)
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_inputs = self.processor(
                text=[full_text],
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            per_sample.append(full_inputs)
            # Cap at the truncated length to keep labels well-defined even
            # when the full sequence got clipped: in that case every label
            # position is part of the (truncated) prompt and the loss for
            # this sample is effectively zero - the right behaviour for a
            # broken sample, surfaced via the empty supervised span.
            prompt_lens.append(min(prompt_len, int(full_inputs["input_ids"].shape[-1])))

        # ------------------------------------------------------------------
        # Per-sample label masking (before padding so we don't mask padding
        # positions twice).
        # ------------------------------------------------------------------
        labels_list: List[torch.Tensor] = []
        for inputs, prompt_len in zip(per_sample, prompt_lens):
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            # Mask the prompt span (system + user + vision + assistant header).
            labels[..., :prompt_len] = self.ignore_index
            labels_list.append(labels)

        # ------------------------------------------------------------------
        # Pad text tensors to the longest in the batch.
        # ------------------------------------------------------------------
        seq_lens = [int(inputs["input_ids"].shape[-1]) for inputs in per_sample]
        T_max = max(seq_lens)
        B = len(per_sample)

        input_ids_batch = torch.full(
            (B, T_max), fill_value=self.pad_token_id, dtype=torch.long
        )
        attention_mask_batch = torch.zeros((B, T_max), dtype=torch.long)
        labels_batch = torch.full(
            (B, T_max), fill_value=self.ignore_index, dtype=torch.long
        )

        for i, (inputs, labels) in enumerate(zip(per_sample, labels_list)):
            L = seq_lens[i]
            ids = inputs["input_ids"]
            attn = inputs.get("attention_mask")
            # Squeeze the processor's leading batch-of-one dimension if
            # present; processors are consistent enough to always emit it,
            # but be defensive.
            ids = ids.reshape(-1)[:L].to(torch.long)
            input_ids_batch[i, :L] = ids
            if attn is not None:
                attn = attn.reshape(-1)[:L].to(torch.long)
                attention_mask_batch[i, :L] = attn
            else:
                # No attention mask returned -> every real token attends.
                attention_mask_batch[i, :L] = 1
            labels_batch[i, :L] = labels.reshape(-1)[:L].to(torch.long)

        out: Dict[str, Any] = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

        # ------------------------------------------------------------------
        # Forward all non-text tensors the processor produced. We don't try
        # to rename keys - Qwen2-VL's forward signature owns those names -
        # we just stack them in the layout Qwen2-VL expects:
        #
        # * ``pixel_values_videos`` / ``pixel_values`` carry the visual
        #   patches for *all* samples flattened along ``dim=0`` (the batch
        #   axis is implicit because Qwen2-VL's vision tower processes the
        #   patches grouped by ``*_grid_thw``).
        # * ``video_grid_thw`` / ``image_grid_thw`` carry per-sample
        #   ``(T, H, W)`` triples that tell the model how to regroup the
        #   patches; these also concatenate along ``dim=0``.
        # ------------------------------------------------------------------
        vision_keys = self._collect_vision_keys(per_sample)
        for key in vision_keys:
            values = [inputs[key] for inputs in per_sample if key in inputs]
            if not values:
                continue
            try:
                out[key] = torch.cat(
                    [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values],
                    dim=0,
                )
            except (RuntimeError, ValueError):
                # Shape mismatch on dim 0 - fall back to a python list so
                # the model side can decide what to do. This should not
                # happen for Qwen2-VL's vision tensors but keeps us forward-
                # compatible with processors that emit ragged outputs.
                out[key] = values

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_sample(sample: Dict[str, Any]) -> None:
        """Cheap sanity checks - friendlier errors than a KeyError deep in HF."""
        for key in ("frames", "task_text", "subtask_text"):
            if key not in sample:
                raise KeyError(
                    f"VLMSegmentCollator sample is missing required key {key!r}; "
                    f"got keys {sorted(sample.keys())}."
                )
        if not sample["frames"]:
            raise ValueError(
                "VLMSegmentCollator sample has empty 'frames'; "
                "each segment must contain at least one frame."
            )
        if not isinstance(sample["task_text"], str):
            raise TypeError(
                f"'task_text' must be a string, got {type(sample['task_text'])!r}."
            )
        if not isinstance(sample["subtask_text"], str):
            raise TypeError(
                f"'subtask_text' must be a string, got {type(sample['subtask_text'])!r}."
            )

    @staticmethod
    def _collect_vision_keys(per_sample: Sequence[Dict[str, Any]]) -> List[str]:
        """Collect all non-text keys returned by the processor.

        Preserves insertion order across samples so the resulting batch dict
        has a deterministic key ordering, which makes test assertions and
        logging output stable.
        """
        seen: Dict[str, None] = {}
        for inputs in per_sample:
            for key in inputs.keys():
                if key in _TEXT_KEYS:
                    continue
                if key not in seen:
                    seen[key] = None
        return list(seen.keys())
