# Proposed Next Direction: Query-Conditioned Visual Refinement

## Status

Proposed strategic pivot only. The v2 harmful-READ path, v3 structured-null
confirmation, and v4 local routing direction remain closed. This document does
not authorize an experiment, training, manifest freeze, or implementation.

## Scientific hypothesis

The prefix-causal model performs useful but query-blind visual consolidation.
Its four local actions can retain or remove that computation but cannot let the
question refine the visual state. A bounded post-question replay of relevant,
already encoded visual tokens should therefore provide more semantically
specific answer support than compute- and geometry-matched query-blind replay.

This is a capacity hypothesis. It is not a routing, acceleration, harmfulness,
or deployability claim.

## Why this direction, not another local policy

- Early WRITE is strongly answer-aligned, so indiscriminately skipping visual
  update is the wrong primitive.
- READ and WRITE have different functional and compute roles, so a single
  depth/skip decision obscures the relevant interface.
- Four-action rankings are heterogeneous, but the v4 outcome-aware query
  oracle has little practical pooled frontier advantage.
- Under common padding, current visual state/WRITE is exactly query-invariant.
  The missing capability is a causal `question -> visual refinement` path.
- V2 Outcome B and prompt/heavy-tail sensitivity require matched content
  controls and behavioral coherence rather than reference likelihood alone.

Multi-layer keep/drop combinations remain technically untested, but they do
not introduce this missing capability and are too close to the closed local
skipping direction.

## Minimum falsification experiment

### Frozen scope

- Model: the same pinned Qwen2.5-VL-7B-Instruct revision, base weights frozen.
- Dataset: 100 new GQA images, exactly two questions per image.
- Eligibility: the questions target two distinct, unambiguous scene-graph
  objects with resolvable boxes; no image/record overlap with inspected v2–v4
  or calibration data.
- No router, learned selector, adapter, probe, or fine-tuning.
- The box coordinates select tokens only. No label, object name, program,
  answer, or box metadata is passed to the model.

### New capability

Use the existing vision encoding once. Deterministically map each target box to
the native merged-token grid and pool it to a fixed `4x4` window of 16 visual
tokens. Replay those 16 hidden-width visual tokens after the literal question
and before the answer prefix. Because replay rows occur after the question,
their decoder updates can causally integrate question context before the
answer is scored.

This adds post-question visual contextualization. It does not remove or search
another decoder READ/WRITE route.

### Frozen conditions

Every replay condition uses the same 16-token count, delimiters, positions,
attention rules, precision, and suffix:

1. own-question target window;
2. paired other-question target window;
3. geometry-matched non-target window;
4. deterministic random window;
5. uniform whole-image 16-token summary;
6. no-replay baseline, secondary because it is not compute-matched.

The paired target windows are crossed within each image, so both questions see
both visual regions. This balances region content without choosing cases from
outcomes.

### Primary statistic

For image `I` with questions `q1,q2` and their target replay windows `r1,r2`,
use per-token accepted-reference log-likelihood to compute

\[
D_I=\frac{1}{2}
\left[
S(q_1,r_1)-S(q_1,r_2)
+S(q_2,r_2)-S(q_2,r_1)
\right].
\]

Use the image as the bootstrap unit. Freeze one practical threshold before
outcomes; the inherited `0.05` nats/token is the default unless an outcome-
blind identity/no-op analysis requires a larger floor.

Reference likelihood is not sufficient by itself. Also require:

- positive median and 20%-trimmed `D_I`;
- target replay outperforming matched non-target, random, and whole-image
  replay under the same token budget;
- no domination by the largest 5% of images;
- coherent official GQA greedy correctness: more target-specific corrections
  than regressions and no equally large improvement from controls;
- deterministic scoring, valid answer spans, and no metadata/answer leakage.

### Entry gate

Before scientific outcomes are inspected, verify:

- deterministic box-to-token mapping and sufficient eligible matched regions;
- identical replay tensor shapes and token budgets across conditions;
- unchanged literal question and accepted answer;
- replay padding masked from scoring;
- FULL/no-replay reproduction of the original frozen model;
- finite, deterministic target and control executions;
- no target labels or answers in model inputs or serialized replay states.

A new technical rule or insufficient eligible pool stops for approval rather
than changing the design.

## Success and kill rules

The capacity hypothesis survives only if the image-clustered 95% CI for mean
`D_I` lies above zero, the median and 20%-trimmed effects are positive, the
predeclared practical threshold is met by a robust summary, and correctness
behavior is coherent. Target replay must beat every compute-matched replay
control, not only the no-replay baseline.

Kill the direction if any of the following occurs:

- the primary CI crosses zero;
- the result disappears under median or 20% trimming;
- non-target, random, or whole-image replay performs similarly;
- gains are prompt-format-sensitive or concentrated in a small extreme tail;
- likelihood changes lack correctness coherence;
- the replay interface cannot be validated without changing the causal
  estimand.

After a kill, do not train a selector/adapter, search replay locations or token
budgets, fall back automatically to high-resolution crops, or resume
multi-layer skipping.

## Interpretation boundary if successful

A positive result would show that the frozen model has latent headroom for
query-conditioned visual contextualization when given an oracle-selected
existing-token window. It would not show that:

- target regions are learnably selectable before answer generation;
- the method accelerates inference;
- an adapter or memory architecture will succeed;
- local visual computation was harmful;
- the result generalizes beyond the pinned GQA/model setting.

Only then may a separate plan propose a small learned selector or low-rank
memory adapter with parameter- and compute-matched controls.

## Competing directions not selected

- High-resolution target-crop revisitation adds new pixels and a second vision
  pass, making a positive result harder to attribute to query conditioning.
- Explicit query-writable memory is scientifically interesting but requires
  adapter training and capacity-matched controls before the cheaper capacity
  hypothesis has been established.
- Multi-layer suppression remains a variant of the closed action family and
  does not repair query-blind visual WRITE.

## Expected cost

Approximately 100 images x two questions x six conditions = 1,200 frozen
teacher-forced and deterministic-greedy branches, with one cached vision
encoding per image/question layout and only 16 replay rows. This is a low-cost
falsification relative to the completed 6,720-branch v4 discovery.

## Next bounded action

If explicitly approved, perform an outcome-blind feasibility and protocol
freeze for this one experiment, then stop for execution approval. Do not begin
inference automatically.

TEST_QUERY_CONDITIONED_VISUAL_REFINEMENT
