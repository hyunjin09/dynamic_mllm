# Amendment 01: Exact Valid-Set NLL and Held-Out Binary-Policy Gate

Date: 2026-08-09

Status: approved; this amendment supersedes the BP-0 representation rejection
and question-only predictor-input clauses of
`dynamic_mllm_binary_visual_polar_plan_v1.md`. All other binary action,
executor, data, model-freezing, and evaluation constraints remain active.

## 1. Corrected interpretation of BP-0

The completed BP-0 audit remains valid evidence about label integrity and
structure:

- all 4,000 records parsed with zero invalid records;
- all eight benchmark/difficulty cells matched the source audit;
- image-group splitting was feasible with zero cross-split image groups;
- empirical per-bit marginal decoding had substantially lower cached-valid-set
  top-k coverage than canonical run decoding.

The empirical-marginal comparison is diagnostic only. It approximates
duplicated-route BCE/marginal fitting and does not evaluate the optimum induced
by exact valid-set likelihood. It must not reject the direct binary head.

## 2. Frozen primary representation and objective

Retain one factorized 28-logit binary head. For a complete mask
`m in {0,1}^28`, define

\[
P_\theta(m\mid x)=\prod_{l=1}^{28}
\sigma(z_l(x))^{m_l}[1-\sigma(z_l(x))]^{1-m_l}.
\]

For the record-local, deduplicated, retained valid set `V_x`, optimize exactly

\[
\mathcal L(x)=-\log\sum_{m\in V_x}w_mP_\theta(m\mid x).
\]

The loss must compute the probability of every complete mask before the
log-sum-exp. It must not average mask bits into a hard or soft target.

Route weights are positive and normalized to sum to one within each record.
Before normalization, every retained route has weight `1.0`, except a valid
all-ON route receives the frozen relative multiplier `0.25` when at least one
shorter valid route exists. Route capping remains deterministic at 50 and must
retain the sparsest route and valid all-ON anchor.

The canonical maximal-run predictor remains an untrained structured fallback.
It may not replace the direct head unless held-out failure evidence implicates
route structure rather than optimization, predictor generalization, or cache
coverage.

## 3. Loss-aligned BP-0 amendment gate

Replace the old representation rejection with two checks:

1. Label-integrity gate: the already completed schema, source-count, route,
   image-identity, and group-split checks must pass.
2. Objective-consistency gate: on contradictory synthetic valid masks, the
   implemented weighted loss must match an independent complete-mask formula,
   remain finite, and permit deterministic optimization to concentrate top-1
   probability on one coherent member of the valid set rather than its
   marginal hybrid.

The objective-consistency check is an implementation sanity check only. It is
not predictor-training, held-out generalization, or evidence that the learned
policy works.

## 4. Predictor input amendment

The output remains the same direct binary head. Its shared pre-action input is
now image plus question:

- question tokens: frozen Qwen3-Embedding-0.6B hidden states under the pinned
  revision;
- image token: the mean of the valid initial visual rows emitted by the frozen
  pinned Qwen2.5-VL vision path after its native merger, using the record's
  frozen image preprocessing and resolution;
- fusion: separate trainable projections place question and image features in
  the same `d_model=256` space; the single image token is appended to question
  tokens before the existing layer-query cross-attention and cross-layer
  encoder;
- no answer, MCTS route, correctness, or execution outcome enters the predictor
  input.

The visual feature is cached once per record, checksummed, and frozen before
training. The Qwen2.5-VL base, vision encoder, and Qwen3 encoder receive no
gradients. Mean pooling is frozen prospectively; no visual pooling search is
authorized.

## 5. Revised stages and gates

### BP-0A — exact-objective consistency

Verify the implemented formula, route weights, masking of padded routes,
finite gradients, and coherent-mode concentration. Save a versioned report and
checksum. Failure blocks all later stages.

### BP-1 — executor and cached-label reproduction

Retain the original 16-fixture model-scale gate unchanged: native all-ON
parity, exact OFF compact-text oracle, deterministic arbitrary masks, cache
geometry, and exact cached generated-token reproduction must pass.

### BP-2 — image-group data and feature freeze

Freeze the deterministic 75%/12.5%/12.5% manifest with zero image overlap and
at least 40 records per benchmark/difficulty/split cell. Retain records without
a cached successful route as evaluation-only. Extract and checksum the frozen
image feature for every record; extraction must reproduce deterministically on
technical fixtures.

### BP-3 — one bounded direct-head training run

Train only the shared image/question predictor with the exact valid-set NLL.
Use the frozen configuration: seed `20260809`, batch size 32, learning rate
`3e-4`, AdamW weight decay `0.01`, gradient clipping `1.0`, and 10 epochs.
Select the checkpoint using validation set-NLL, breaking ties by validation
top-1 cached-valid-set membership. Do not use test outcomes for checkpoint or
hyperparameter selection.

### BP-4 — image-group-disjoint held-out online gate

Evaluate the selected checkpoint once on the complete frozen test split.
Freshly execute every predicted top-1 mask, including masks absent from the
MCTS cache, and freshly execute all-ON under the same pinned runtime.

Report:

- top-1 predicted-mask membership in the cached valid set, both over all test
  records and over records with at least one cached valid mask;
- fresh official task correctness for predicted masks and all-ON;
- image-grouped and benchmark/difficulty-stratified accuracy differences;
- exact analytic decoder-layer FLOPs for each route, route ON counts, and
  reduction relative to all-ON; no wall-clock acceleration claim;
- cached-search oracle correctness (any successful searched mask), sparsest
  cached-valid-route compute, and gaps to the learned predictor;
- uncached predicted-mask rate and correctness.

The original online success gate remains: the image-grouped 95% CI for macro
accuracy difference versus all-ON must lie above `-0.01`, mean analytic local
decoder FLOP reduction must be at least 20%, and no easy cell may regress by
more than 0.02. The cached-search oracle is explicitly limited to the masks
searched by MCTS and is not a global oracle.

## 6. Failure diagnosis

Any failure report must separate:

- objective/optimization failure: loss, gradients, or training-set membership
  do not improve under the exact objective;
- predictor generalization failure: training/validation behavior is adequate
  but held-out cached membership or online correctness fails;
- incomplete valid-mask cache: predicted masks are uncached yet succeed online,
  or cache membership understates fresh correctness;
- binary-factorization limitation: after optimization and cache-coverage
  alternatives are ruled out, coherent held-out route structure remains
  unrecoverable by the direct head.

No failure automatically authorizes canonical segmentation, another routing
head, new MCTS, or base-model fine-tuning.
