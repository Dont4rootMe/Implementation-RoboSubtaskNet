"""Qwen2-VL-Instruct wrapper with optional LoRA adapter for segment labeling.

Stage 2 of the RoboSubtaskNet pipeline assigns a short natural-language label
to each temporal segment proposed by Stage 1. This module wraps the
``Qwen/Qwen2-VL-2B-Instruct`` checkpoint (HuggingFace) with optional
PEFT-LoRA adapters so the same wrapper supports both fine-tuning and
inference paths.

Design notes
------------
- All heavy dependencies (``torch``, ``transformers``, ``peft``) are imported
  lazily inside :meth:`SegmentLabelerVLM.from_pretrained` so this module can be
  imported in environments that do not have the full HuggingFace stack
  available (e.g. unit tests of unrelated components, CI sanity checks).
- The chat template is constructed by :func:`make_segment_chat`. It follows
  the Qwen2-VL multimodal chat format: a system turn, a user turn that
  contains a ``<video>`` placeholder plus the textual prompt, and -- during
  training -- an additional assistant turn carrying the target subtask label.
- The LoRA target modules cover the attention projections and MLP linears of
  Qwen2-VL's language decoder, which is the standard target set for
  Qwen2 / Qwen2.5 LoRA fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You label robot manipulation subtasks. Given a short video segment and the overall task "
    "instruction, respond with a single concise verb phrase describing the subtask "
    "(e.g., 'reach for the cup', 'place on shelf'). No explanations, no punctuation beyond the phrase."
)


# Placeholder string used in the chat template wherever the processor will
# splice in the encoded video frames. The exact value does not matter at the
# template-construction stage -- Qwen2-VL's ``apply_chat_template`` discovers
# the video by inspecting the ``type`` field, not the value -- but we keep an
# explicit sentinel so the structure can be inspected / tested without a live
# processor.
_VIDEO_PLACEHOLDER = "<video>"


# LoRA target modules for Qwen2-VL's language model. These match the standard
# Qwen2 attention + MLP linear projections and are what the official Qwen
# fine-tuning recipes target.
_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def make_segment_chat(
    task_text: str,
    training_target: str | None = None,
) -> list[dict]:
    """Build the Qwen2-VL chat payload for a single segment.

    The returned list is the ``messages`` argument accepted by
    ``Qwen2VLProcessor.apply_chat_template`` (and by HuggingFace's
    ``processor.apply_chat_template`` in general). The ``user`` turn contains
    a ``<video>`` placeholder so the processor knows where to splice in the
    encoded frames; the placeholder string itself is irrelevant -- only the
    ``"type": "video"`` field is consumed -- but we use a visible sentinel so
    the structure is easy to inspect in tests.

    Parameters
    ----------
    task_text:
        The overall episode-level task instruction (e.g. "make a sandwich").
        Inserted into the user prompt so the labeler is conditioned on the
        broader context.
    training_target:
        If provided, an additional ``assistant`` turn is appended carrying the
        ground-truth subtask label. Used to build supervised fine-tuning
        examples; leave as ``None`` for inference / generation.

    Returns
    -------
    A list of role dictionaries shaped as::

        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": "<video>"},
                {"type": "text", "text": "Task: {task_text}. What subtask ..."},
            ]},
            # Optional, only if training_target is given:
            {"role": "assistant", "content": [{"type": "text", "text": training_target}]},
        ]
    """
    user_text = (
        f"Task: {task_text}. What subtask is happening in the video segment? "
        "Answer with a short verb phrase."
    )

    messages: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": _VIDEO_PLACEHOLDER},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    if training_target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": training_target}],
            }
        )

    return messages


class SegmentLabelerVLM:
    """Qwen2-VL-Instruct wrapper with optional LoRA adapter for segment labeling.

    Usage
    -----
    Training (attach a fresh LoRA adapter)::

        vlm = SegmentLabelerVLM.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", lora=True
        )
        vlm.model.train()
        # ... fine-tune via HF Trainer or a custom loop, using
        #     vlm.processor and vlm.model.
        vlm.save_lora("./lora_out")

    Inference (reload a saved adapter)::

        vlm = SegmentLabelerVLM.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", lora_path="./lora_out"
        )
        text = vlm.label_segment(frames_rgb, task_text="make a sandwich")
    """

    DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        model: Any,
        processor: Any,
        tokenizer: Any,
    ) -> None:
        """Store the loaded objects. Prefer :meth:`from_pretrained`.

        Parameters
        ----------
        model:
            The Qwen2-VL ``ConditionalGeneration`` model (possibly already
            wrapped by ``peft.PeftModel``).
        processor:
            The Qwen2-VL multimodal processor that handles video / image
            preprocessing and chat-template rendering.
        tokenizer:
            The underlying text tokenizer (``processor.tokenizer``); kept as a
            separate attribute for convenience.
        """
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        *,
        lora: bool = False,
        lora_path: str | Path | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> "SegmentLabelerVLM":
        """Load Qwen2-VL plus an optional LoRA adapter.

        Parameters
        ----------
        model_name:
            HuggingFace repo id of the base model. Defaults to
            :pyattr:`DEFAULT_MODEL` (``Qwen/Qwen2-VL-2B-Instruct``).
        lora:
            If True (and no ``lora_path`` is given), attach a freshly
            initialized LoRA adapter for training.
        lora_path:
            Directory containing a previously saved PEFT-LoRA adapter (i.e.
            the output of :meth:`save_lora`). When provided, ``lora`` is
            ignored and the adapter is loaded on top of the base model.
        lora_rank, lora_alpha, lora_dropout:
            LoRA hyperparameters used when attaching a fresh adapter. These
            are also the values reported in the implementation plan.
        torch_dtype:
            Dtype string passed to ``from_pretrained`` (e.g. ``"bfloat16"``,
            ``"float16"``, ``"float32"``).
        device:
            Device string (``"cuda"``, ``"cpu"``, or e.g. ``"cuda:0"``). The
            full pipeline targets a single 24 GB GPU; the value is forwarded
            to ``.to(device)`` after loading.

        Returns
        -------
        :class:`SegmentLabelerVLM`

        Raises
        ------
        ImportError
            If ``torch``, ``transformers``, or ``peft`` (when LoRA is
            requested) are not installed.
        """
        # Lazy import of heavy dependencies so the package can be imported
        # without the full HuggingFace stack present.
        try:
            import torch  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised at runtime
            raise ImportError(
                "SegmentLabelerVLM requires 'torch'. Install it with "
                "`pip install torch` (matching your CUDA/CPU setup)."
            ) from e

        try:
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )
        except ImportError as e:  # pragma: no cover - exercised at runtime
            raise ImportError(
                "SegmentLabelerVLM requires 'transformers' >= 4.45 with "
                "Qwen2-VL support. Install it with "
                "`pip install \"transformers>=4.45\"`."
            ) from e

        # Resolve dtype string to a torch.dtype.
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype {torch_dtype!r}; expected one of "
                f"{sorted(dtype_map)}"
            )
        dtype = dtype_map[torch_dtype]

        name = model_name or cls.DEFAULT_MODEL

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=dtype,
        )
        processor = AutoProcessor.from_pretrained(name)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            # Some processor versions expose the text tokenizer under a
            # different attribute; fall back to ``AutoTokenizer`` so the
            # wrapper still works.
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as e:  # pragma: no cover - defensive
                raise ImportError(
                    "Could not obtain a tokenizer from the Qwen2-VL "
                    "processor and 'transformers.AutoTokenizer' is "
                    "unavailable."
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(name)

        # --- LoRA: load existing adapter or attach a fresh one --------- #
        if lora_path is not None:
            try:
                from peft import PeftModel  # type: ignore
            except ImportError as e:  # pragma: no cover - exercised at runtime
                raise ImportError(
                    "Loading a LoRA adapter requires the 'peft' package. "
                    "Install it with `pip install peft`."
                ) from e
            model = PeftModel.from_pretrained(model, str(lora_path))
        elif lora:
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore
            except ImportError as e:  # pragma: no cover - exercised at runtime
                raise ImportError(
                    "Attaching a fresh LoRA adapter requires the 'peft' "
                    "package. Install it with `pip install peft`."
                ) from e
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(_LORA_TARGET_MODULES),
            )
            model = get_peft_model(model, lora_config)

        # Move to the requested device. We tolerate ``device="cpu"`` for
        # debugging on machines without a GPU.
        try:
            model = model.to(device)
        except (RuntimeError, AssertionError) as e:
            # Surface a clearer message if CUDA is unavailable.
            raise RuntimeError(
                f"Failed to move model to device {device!r}: {e}. "
                "If CUDA is unavailable, pass device='cpu'."
            ) from e

        return cls(model=model, processor=processor, tokenizer=tokenizer)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_lora(self, out_dir: str | Path) -> None:
        """Save the LoRA adapter weights (and adapter config) to ``out_dir``.

        Requires that ``self.model`` is a ``peft.PeftModel`` -- i.e. that the
        wrapper was constructed with ``lora=True`` or ``lora_path=...``.

        Parameters
        ----------
        out_dir:
            Directory to write to. Created if it does not exist.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ``PeftModel`` exposes ``save_pretrained`` which writes only the
        # adapter weights + adapter_config.json. We do not gate on the exact
        # class because ``peft`` may not be importable here -- duck-typing on
        # ``save_pretrained`` is sufficient and lets users wrap their own
        # PEFT subclasses.
        if not hasattr(self.model, "save_pretrained"):
            raise RuntimeError(
                "self.model has no 'save_pretrained' method; saving a LoRA "
                "adapter requires a peft.PeftModel-wrapped base model. Did "
                "you forget to pass lora=True to from_pretrained?"
            )

        self.model.save_pretrained(str(out))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def label_segment(
        self,
        frames: list,  # list of np.ndarray HxWx3 RGB uint8
        task_text: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> str:
        """Predict a subtask label for a single video segment.

        Builds the Qwen2-VL chat payload (see :func:`make_segment_chat`),
        runs it through the processor with the supplied frames, generates a
        completion, and returns the decoded assistant turn -- stripped of any
        role markers or special tokens.

        Parameters
        ----------
        frames:
            A list of ``H x W x 3`` ``np.ndarray``s in **RGB** order with
            ``uint8`` dtype. Should contain at least one frame; the Qwen2-VL
            video processor handles temporal subsampling internally.
        task_text:
            The episode-level task instruction used to condition the labeler.
        max_new_tokens:
            Generation cap. Subtask labels are typically very short (one verb
            phrase, ~5-10 tokens), so the default of 64 leaves plenty of
            headroom.
        do_sample:
            If False (default), use greedy decoding. Set to True for sampled
            outputs (e.g. for diversity at inference time).

        Returns
        -------
        The model's predicted subtask label as a plain string.
        """
        # Lazy imports mirror ``from_pretrained``.
        try:
            import torch  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised at runtime
            raise ImportError(
                "SegmentLabelerVLM.label_segment requires 'torch'."
            ) from e

        if not frames:
            raise ValueError(
                "label_segment requires at least one frame; got an empty list."
            )

        messages = make_segment_chat(task_text=task_text, training_target=None)

        # Render the chat template into a prompt string with a trailing
        # generation prompt token (so the model continues with an assistant
        # turn).
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ``Qwen2VLProcessor`` accepts a ``videos`` kwarg as a list of clips,
        # where each clip is itself a list / array of frames. We pass a
        # single clip containing the supplied frames.
        inputs = self.processor(
            text=[prompt],
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )

        # Move tensor inputs to the same device as the model.
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        # Strip the prompt prefix from each row of the generated tensor.
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            prompt_len = input_ids.shape[1]
            generated = generated[:, prompt_len:]

        # Decode and pick the single example.
        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = decoded[0].strip() if decoded else ""
        return text


__all__ = ["SegmentLabelerVLM", "make_segment_chat", "SYSTEM_PROMPT"]
