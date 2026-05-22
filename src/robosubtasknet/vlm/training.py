"""HuggingFace-Trainer wrapper for Stage-2 VLM LoRA fine-tuning.

The Stage-2 pipeline LoRA-fine-tunes Qwen2-VL-2B on
``(segment, task, subtask)`` triplets. We delegate to
:class:`transformers.Trainer` (rather than the project's own
:class:`robosubtasknet.training.trainer.Trainer`) so that the LoRA +
``accelerate`` + gradient-accumulation plumbing is handled by upstream.

This module exposes two helpers:

* :func:`build_hf_training_args` — a thin factory around
  :class:`transformers.TrainingArguments` with sensible defaults for the
  Qwen2-VL LoRA recipe (Section 11 of the implementation plan).
* :func:`train_lora` — runs the training loop and persists the LoRA adapter.

All ``transformers`` / ``peft`` imports are deferred into the function bodies
so importing this module does not force the heavy ML dependencies to load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# TrainingArguments factory
# ---------------------------------------------------------------------------
def build_hf_training_args(
    output_dir: str | Path,
    *,
    epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_steps: int | None = None,
    bf16: bool = True,
    seed: int = 42,
    report_to: str | list[str] = "tensorboard",
    gradient_checkpointing: bool = True,
    max_grad_norm: float = 1.0,
):
    """Build :class:`transformers.TrainingArguments` for the LoRA recipe.

    ``transformers`` is lazy-imported so callers that only need the wrapper
    module (e.g. type-checking, dataset prep) do not pay the import cost.

    Parameters
    ----------
    output_dir
        Directory where the ``Trainer`` will write checkpoints and logs.
    epochs
        Number of full passes over ``train_dataset``.
    per_device_batch_size
        Per-GPU micro-batch size; the effective batch is
        ``per_device_batch_size * gradient_accumulation_steps * world_size``.
    gradient_accumulation_steps
        Number of micro-batches to accumulate before an optimizer step.
    lr, weight_decay, warmup_ratio, max_grad_norm
        Optimizer / scheduler hyperparameters. Defaults follow Section 11 of
        the implementation plan.
    logging_steps, save_steps, eval_steps
        Logging / checkpoint / evaluation cadences in optimizer steps. When
        ``eval_steps`` is ``None`` evaluation is disabled.
    bf16
        Use bfloat16 mixed precision (preferred on Ampere+ / Hopper). Set to
        ``False`` for fp32 or to pair with fp16 elsewhere.
    seed
        Deterministic seed forwarded to ``TrainingArguments.seed``.
    report_to
        Where to stream metrics ("tensorboard", "wandb", "none", or a list).
    gradient_checkpointing
        Toggle activation checkpointing inside ``Trainer``. Note that
        :func:`train_lora` additionally enables the Qwen2-VL-specific
        non-reentrant variant on the underlying model.

    Returns
    -------
    transformers.TrainingArguments
    """
    # Deferred import so that ``import robosubtasknet.vlm.training`` does not
    # force the multi-hundred-MB ``transformers`` stack to load.
    from transformers import TrainingArguments

    output_dir = str(output_dir)

    # Evaluation cadence: enable it only when the caller asks for a non-None
    # ``eval_steps``. Newer versions of ``transformers`` renamed the keyword to
    # ``eval_strategy``; we fall back to ``evaluation_strategy`` for older
    # releases.
    eval_strategy = "steps" if eval_steps is not None else "no"
    common_kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        bf16=bf16,
        seed=seed,
        data_seed=seed,
        report_to=report_to,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        remove_unused_columns=False,  # VLM batches carry non-tensor metadata
        save_total_limit=3,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    if eval_steps is not None:
        common_kwargs["eval_steps"] = eval_steps

    # ``evaluation_strategy`` was renamed to ``eval_strategy`` in
    # ``transformers >= 4.41``. Try the new name first, fall back to the old.
    try:
        return TrainingArguments(eval_strategy=eval_strategy, **common_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy=eval_strategy, **common_kwargs)


# ---------------------------------------------------------------------------
# Training loop wrapper
# ---------------------------------------------------------------------------
def train_lora(
    vlm,  # SegmentLabelerVLM instance with LoRA adapter attached
    train_dataset,  # VLMSegmentDataset
    collator,  # VLMSegmentCollator
    output_dir: str | Path,
    *,
    eval_dataset=None,
    training_args=None,  # transformers.TrainingArguments override
    callbacks: list | None = None,
    **kwargs,
):
    """Run :class:`transformers.Trainer` against the LoRA-wrapped VLM.

    Steps
    -----
    1. Lazy-import :class:`transformers.Trainer`.
    2. Assume ``vlm.model`` is already wrapped with the LoRA adapter (the
       caller does this via ``SegmentLabelerVLM.from_pretrained(lora=True)``
       or ``vlm.attach_lora(...)``).
    3. Build :class:`transformers.TrainingArguments` from ``kwargs`` if the
       caller did not pre-construct one.
    4. If gradient checkpointing is requested, apply the Qwen2-VL-specific
       non-reentrant incantation and enable input gradients (required so the
       LoRA-injected adapter sees gradients flowing into frozen layers).
    5. Instantiate ``Trainer`` and call ``trainer.train()``.
    6. Persist the LoRA adapter under ``output_dir / "lora_adapter"`` via
       :meth:`SegmentLabelerVLM.save_lora`.
    7. Return ``trainer.state.log_history``; the last entry holds the final
       metric snapshot (e.g. ``best_metric`` when ``load_best_model_at_end``
       is set by the caller through ``training_args``).

    Parameters
    ----------
    vlm
        :class:`SegmentLabelerVLM` instance. Must expose ``model``,
        ``tokenizer`` / ``processor``, and ``save_lora``.
    train_dataset
        Map-style dataset yielding chat-formatted samples.
    collator
        Callable turning a list of samples into a model-ready batch.
    output_dir
        Where the trainer writes checkpoints; ``lora_adapter/`` is created
        alongside on completion.
    eval_dataset, training_args, callbacks
        Forwarded to ``Trainer``. ``training_args`` overrides any kwargs.
    **kwargs
        Forwarded to :func:`build_hf_training_args` when ``training_args`` is
        ``None``.

    Returns
    -------
    list[dict]
        ``trainer.state.log_history``. The last record is the final eval /
        training metrics snapshot.
    """
    # Deferred import: keeps the module light when only the factory is used.
    from transformers import Trainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the args object if the caller did not supply one.
    if training_args is None:
        training_args = build_hf_training_args(output_dir, **kwargs)

    # Resolve the underlying nn.Module. ``SegmentLabelerVLM`` may store the
    # PEFT-wrapped model under ``vlm.model``; we work directly on it so that
    # gradient-checkpointing flags propagate to the base transformer.
    model = vlm.model

    # Qwen2-VL + LoRA needs the non-reentrant checkpointing variant and
    # explicit input-grad enablement, otherwise the LoRA adapter sees no
    # gradient (the frozen base layers don't requires_grad their inputs by
    # default). See https://huggingface.co/docs/peft for the canonical recipe.
    if getattr(training_args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older ``transformers`` releases accept a positional/dict kwarg.
            model.gradient_checkpointing_enable({"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ``Trainer`` historically took ``tokenizer=`` to handle saving; modern
    # releases (>= 4.46) renamed it to ``processing_class``. Try the new
    # keyword first, fall back to the old. We pass ``vlm.processor`` when it
    # exists (Qwen2-VL needs the multimodal processor, not the tokenizer
    # alone), otherwise the plain tokenizer.
    processing = getattr(vlm, "processor", None) or getattr(vlm, "tokenizer", None)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    try:
        trainer = Trainer(processing_class=processing, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=processing, **trainer_kwargs)

    trainer.train()

    # Persist the LoRA adapter separately from the HF checkpoint so the caller
    # can ship just the adapter weights for inference.
    adapter_dir = output_dir / "lora_adapter"
    vlm.save_lora(adapter_dir)

    return trainer.state.log_history


__all__ = ["build_hf_training_args", "train_lora"]
