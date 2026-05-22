# VLM LoRA Notes

Design rationale for the segment-labeling stage of the class-agnostic boundary-detection pipeline.

## 1. Problem statement

After the boundary detector slices an episode into class-agnostic segments, each segment still needs a textual subtask label (e.g., "reach for the cup") conditioned on the episode-level task instruction (e.g., "pick and place the cup"). Because we ingest heterogeneous LeRobot datasets, labels must be **free-form text** rather than a fixed vocabulary. LoRA fine-tuning lets the VLM absorb our project's specific phrasing patterns without full-model retraining.

## 2. Base model survey

| Model | Params | Video | License | Notes |
|---|---|---|---|---|
| **Qwen2-VL-2B-Instruct / 7B-Instruct** (Alibaba 2024) | 2B / 7B | Native dynamic frame sampling | Apache 2.0 | SOTA open weights, strong instruction following. Recommended. |
| LLaVA-NeXT-Video-7B (Liu et al. 2024) | 7B | Yes | Apache 2.0 | Strong video, slightly older recipe. |
| InternVL2-2B (Shanghai AI Lab) | 2B | Yes | Apache 2.0 | Competitive but heavier processor setup. |
| MiniCPM-V 2.6 / SmolVLM-Instruct | 2-8B | Limited | Apache 2.0 | Lighter, weaker video understanding. |
| MoLMo (Allen AI) | 7B | Image-centric | Apache 2.0 | Excellent grounding, limited video training. |
| Gemini / GPT-4o-mini-vision (closed) | -- | Yes | Proprietary | Viable as teacher for distillation; not LoRA-able. |

**Recommendation: Qwen2-VL-2B-Instruct** as default. It fits on a single 24 GB GPU with LoRA + gradient checkpointing + bf16. Upgrade to 7B for harder vocabularies or when more compute is available. Closed models stay reserved for offline teacher labeling, not for our trainable path.

## 3. LoRA recipe

- **Library**: `peft` (HuggingFace). Wrap `Qwen2VLForConditionalGeneration` with `LoraConfig`.
- **Target modules**: attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) + MLP (`gate_proj`, `up_proj`, `down_proj`). Standard "all-linear" excluding embeddings and the LM head.
- **Rank**: 16 (balanced quality vs. memory). Bump to 32 for harder vocabularies.
- **Alpha**: 32 (= 2 × rank), standard scaling.
- **Dropout**: 0.05.
- **Vision encoder**: keep **frozen**. Robot footage is close enough to natural video that the pretrained vision tower transfers well; fine-tuning it risks catastrophic forgetting on small datasets.
- **Training data format**: chat template with a `<video>` placeholder for segment frames:

  - System: "You label robot manipulation subtasks. Given a video segment and the overall task, respond with a single concise verb phrase."
  - User: "<video>. Task: pick and place the cup. What subtask?"
  - Assistant: "reach for the cup"

- **Label masking**: supervised loss only on assistant-turn tokens; user/system tokens set to `-100`.
- **Gradient checkpointing**: enable via `model.gradient_checkpointing_enable({"use_reentrant": False})` and `model.enable_input_require_grads()` so LoRA gradients flow.
- **Optimizer**: AdamW, lr `2e-4` (LoRA-friendly), 3% warmup, cosine decay, bf16 mixed precision.

## 4. Data composition

- For each LeRobot dataset, iterate episodes using `action_config` (or derive from `action_text_id` plus `meta/action_text.json`).
- Each segment yields one training sample: 8-16 frames uniformly sampled in `[start, end)`, the episode-level `task_text`, and the segment's `subtask_text`.
- **Skip background segments**: uninformative supervision, hurts more than it helps.
- Cap per-episode segments to avoid long-tail dominance from very long episodes.
- Resize frames to **224 x 224** (Qwen2-VL default vision input).

## 5. Inference details

- Greedy decoding: `do_sample=False`, `max_new_tokens=64`.
- For inputs **without** an episode-level task, optionally first prompt the VLM zero-shot to summarize the full episode ("Describe the overall task..."), then feed that summary as `task_text` for per-segment labeling.
- Free-form outputs are deduped at the orchestrator level into a fresh `meta/action_text.json` per output dataset, so downstream consumers still get integer `action_text_id`s.

## 6. References

- Qwen2-VL technical report, arXiv:2409.12191.
- HuggingFace `peft` documentation.
- LLaVA-NeXT-Video (Liu et al., 2024).
- OpenVLA paper for robotics + VLM context.
