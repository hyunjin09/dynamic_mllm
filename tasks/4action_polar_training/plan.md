# Implementation Plan: Four-Action Image+Question POLAR Training

## Overview

Prepare, but do not launch, two Image+Question POLAR-style training runs over
the completed GQA, ChartQA, and TextVQA exact sequential four-action labels.
Both runs use the same predictor, split, route weights, optimization settings,
validation procedure, and eventual external evaluation; only the supervision
objective differs.

## Frozen contract

- Actions use executor order `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- Predictor output is categorical action logits with shape `[B, 28, 4]`.
- Inputs are frozen Qwen3-Embedding-0.6B question rows plus cached unpooled
  Qwen2.5-VL projected visual rows entering decoder layer 0.
- Start from 6,917 VQA source records and explicitly exclude the 106 records
  with zero replay-valid routes; do not fabricate supervision for them. Train
  on all 248,804 valid routes for the remaining 6,811 records and preserve the
  exact exclusion list in the manifest audit. Do not inherit the binary max-50
  route cap.
- Preserve the existing image-group-disjoint split: 5,945 train and 866
  validation records after those exclusions.
- `duplicated_action_bce` duplicates each valid route, applies one-hot BCE over
  the four mutually exclusive action positions at every layer, and normalizes
  route weights within each input.
- `exact_set_nll` uses the product of per-layer categorical probabilities and
  maximizes weighted probability mass over the complete valid-route set.
- Retain POLAR's all-FULL weight `0.3` when a cheaper valid route exists;
  otherwise route weight is `1.0`, normalized within input.
- Optimization matches the binary full10 comparison: 10 epochs, AdamW,
  `5e-4`, weight decay `0.01`, cosine schedule, 10 warmup steps, effective
  batch 128, gradient clip `1.0`, BF16, seed `20260809`, deterministic
  algorithms, and no early stopping.
- Save a checksum-bound checkpoint and run complete internal validation after
  every epoch.
- Select one best checkpoint per objective prospectively by maximum validation
  valid-set Hit@1, then Hit@5, lower nearest-valid Hamming, lower objective
  loss, and earlier epoch.
- After training, evaluate only ChartQA, MMMU-Pro Standard, MMMU-Pro Vision,
  and all three POPE splits (14,960 rows). External outcomes are not inspected
  before checkpoint selection.

## Task list

1. Define and test the four-action predictor, objective, decoder, and metrics
   contracts.
2. Build and audit the complete predictor manifest and deterministic split.
3. Add Image+Question collators and a fresh visual-feature extraction/audit
   path.
4. Add the ten-epoch trainer, epoch checkpointing, validation, and selection.
5. Add the narrowed resumable external evaluation and result merger.
6. Freeze run configs, execute non-training preflights, and perform final code
   review.

## Stop boundary

This readiness task may build deterministic manifests, configs, tests, and
CPU-only audit artifacts. It must not launch visual-cache GPU extraction,
training, or external model generation.
