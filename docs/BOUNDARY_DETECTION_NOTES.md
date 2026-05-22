# Boundary Detection: Research and Design Rationale

## 1. Problem Statement

We work with heterogeneous LeRobot datasets whose per-frame annotations use inconsistent
vocabularies (e.g. one dataset says `pick_cube`, another says `grab_object`, a third uses
free-form text). Standard supervised RoboSubtaskNet — cross-entropy over a fixed set of N
subtask classes — does not transfer across such schemas. What is consistent, however, is
the *position* of transitions between behavioral chunks. We therefore reformulate the
problem as **class-agnostic temporal boundary detection**: predict *where* behavior
changes, not *what* it is. Semantic labels are assigned downstream by a LoRA-fine-tuned
VLM applied to each detected segment.

## 2. Published Approaches

- **MS-TCN / MS-TCN++ + binary head (Farha & Gall, CVPR 2019; Li et al., TPAMI 2020).**
  Replace the K-way softmax with a single sigmoid trained on per-frame boundary targets.
  Architecture, dilation schedule, and refinement stages all transfer unchanged. Lowest
  engineering risk: we already have a working MS-TCN+ implementation.

- **ASRF — Action Segment Refinement Framework (Ishikawa et al., WACV 2021).**
  Trains an explicit boundary regression branch in parallel with the classification
  branch and uses predicted boundaries to refine class predictions. Dropping the
  classification head yields exactly the boundary-only formulation we need; ASRF is
  effectively the upper bound for "MS-TCN + boundary head" done right.

- **BSN / BMN (Lin et al., ECCV 2018; Lin et al., AAAI 2019).**
  Boundary-Sensitive / Boundary-Matching Networks predict start- and end-probability per
  frame for action proposal generation. Conceptually closest to our task, but the
  proposal-confidence map and NMS pipeline are oriented toward retrieval-style outputs,
  not contiguous auto-labeling.

- **ActionFormer (Zhang et al., ECCV 2022).**
  Transformer with multi-scale feature pyramid; jointly predicts boundary offsets and
  class scores. A class-agnostic variant (regression head only) is straightforward but
  requires much more data and tuning than MS-TCN for marginal expected gain on our
  modest dataset sizes.

- **Unsupervised: TW-FINCH (Kukleva et al., CVPR 2021).**
  Clusters temporally-weighted features into segments without any labels. Attractive if
  no annotations existed, but we *do* have annotations — only the vocabulary is
  inconsistent. Discarding the class info while keeping the transition positions is
  strictly more informative than full unsupervised clustering.

- **Change-point detection (Truong et al., Signal Processing 2020).**
  `ruptures` + `scipy.signal` offer inference-only, training-free baselines. Useful as
  sanity check and for cold-start on unseen feature distributions, but cannot exploit
  the supervision we actually have.

## 3. Our Choice

We adopt **MS-TCN+ (Fibonacci-dilated) with a single sigmoid boundary head**:

- Reuses the existing, tested `AttentionFusion` + `SingleStageTCN` stack.
- Boundary targets are derived from *any* annotated dataset by detecting transitions in
  `action_text_id` — fully class-agnostic and robust to vocabulary mismatch.
- Targets are smoothed with a Gaussian (σ ≈ 2 feature frames) to convert the sparse
  binary signal into a soft distribution: vanilla 0/1 + BCE collapses because positives
  are < 10% of frames.
- Multi-stage refinement (3–4 stages) sharply reduces over-segmentation.
- Loss = per-frame BCE with `pos_weight ≈ 10` + truncated-MSE temporal smoothness,
  matching the original MS-TCN regularizer.

## 4. Hyperparameter Rationale

- **σ = 2 feature frames** (≈ 16 raw frames at stride 8): wide enough to provide gradient
  to neighbors of the true boundary, narrow enough not to merge adjacent transitions.
- **pos_weight = 10**: matches the empirical 5–10 % positive density on LeRobot dumps.
- **Decoding**: threshold 0.5, then `scipy.signal.find_peaks` with `distance=3` and
  `prominence=0.1`, which suppresses Gaussian-shoulder false positives.

## 5. References

- MS-TCN: https://arxiv.org/abs/1903.01945
- MS-TCN++: https://arxiv.org/abs/2006.09220
- ASRF: https://arxiv.org/abs/2007.06866
- BSN: https://arxiv.org/abs/1806.02964
- BMN: https://arxiv.org/abs/1907.09702
- ActionFormer: https://arxiv.org/abs/2202.07925
- TW-FINCH: https://arxiv.org/abs/2103.11264
- Change-point survey (Truong et al., 2020): https://arxiv.org/abs/1801.00718
