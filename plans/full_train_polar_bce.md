# Full10 POLAR-Style Duplicated-BCE Comparator Amendment

This amendment authorizes the missing full-scale comparator to
`plans/full_train.md`. It does not alter the completed exact-set-NLL runs or
their artifacts.

## Fixed scientific comparison

Run the same two direct factorized 28-bit predictors for ten epochs:

- Question-only;
- Image+Question using the frozen unpooled projected Qwen2.5-VL visual rows.

Change only the training supervision from grouped exact valid-set NLL to the
already validated POLAR-style duplicated valid-route BCE. For every input and
selected valid mask, compute ordinary mean-over-28-bits BCE. Preserve the
existing within-input route weights: when ALL-ON coexists with a cheaper valid
route, its relative weight is `0.3` and every other selected route has relative
weight `1.0`; normalize weights within the input so every unique input has
equal total training weight.

## Frozen matched settings

- Data: 6,043 positive training and 874 positive validation inputs from the
  unchanged image-group-disjoint GQA/TextVQA/ChartQA manifest.
- Routes: unchanged deterministic diverse maximum of 50 valid masks per input.
- Architecture: frozen Qwen3-Embedding-0.6B question encoder, unchanged POLAR
  cross-attention/cross-layer backbone, and direct factorized 28-bit head.
- Optimization: seed `20260809`, 10 epochs, effective batch 128, AdamW,
  learning rate `5e-4`, weight decay `0.01`, cosine schedule, 10 warmup steps,
  BF16, deterministic algorithms, and no early stopping.
- Initialization: same seed and same shared parameter initialization as the
  completed exact-set-NLL runs; only the Image+Question visual projection is
  modality-specific.
- Validation: retain grouped complete valid sets and the common exact set-NLL,
  Valid-Set Hit@1/Hit@5, nearest-valid Hamming, and decoded-mask diversity
  metrics. Select the external-evaluation checkpoint by the unchanged
  best-Hit@1 tie hierarchy, not by duplicated-BCE training loss.
- Save all ten epoch checkpoints and complete training histories.

## Execution and evaluation

Run Question-only on one node02 GPU and Image+Question on one node07 GPU,
concurrently. After both checkpoint selections are frozen, run the joint
external parity/determinism preflight. If it passes, evaluate the two
modalities concurrently on the unchanged 22,307-record external suite:

- ChartQA: 2,500;
- TextVQA: 5,000;
- MMStar validation: 1,500;
- MMMU validation: 847;
- MMMU-Pro Standard test: 1,730;
- MMMU-Pro Vision test: 1,730;
- POPE adversarial/popular/random: 3,000 each.

DocVQA remains excluded. Use current live ALL-ON as the paired scientific
baseline; retain the historical cache only as audit data. Report each benchmark
and the three scientific suites separately, including the frozen 8,982-record
image-disjoint POPE sensitivity.

## Gates and stops

- Before training, the full-batch BF16 preflight must show finite duplicated
  BCE, finite predictor gradients, no frozen-encoder gradients, deterministic
  repeated logits, and shared initialization equality for both modalities.
- Stop a run only for a technical integrity failure such as NaN/Inf, data or
  checksum mismatch, gradient leakage, corrupt output, failed parity, or
  unrecoverable runtime failure.
- Do not modify the exact-set-NLL artifacts, data split, route cap, weighting,
  optimizer, architecture, decoding, evaluator, or benchmark population in
  response to intermediate BCE results.
- This action compares the two supervision formulations. It does not authorize
  architecture redesign, new labels, additional training objectives, or a
  routing/latency claim.

