# Operational Research Plan: Binary Route Label Regeneration

## Active WeMath2.0-Pro greedy-recovery detour (2026-08-18)

The user reopened label extraction for the 2,278 WeMath2.0-Pro records where
current ALL-ON is wrong and the completed hard-cap-400 MCTS cache contains no
valid route. The active plan is
`plans/dynamic_mllm_wemath2pro_greedy_recovery_plan.md`.

The supplied frozen Phase-1/Phase-2 package defines the search geometry but is
not a compatible runtime wrapper. Preserve its ten greedy orders, acceptance
rule, Phase-2 proposal families, and seeds while using the current verified
Transformers 5.3.0 binary executor, native image processing, 96-token greedy
generation, and MathRuler contract. Preserve the MCTS cache and write all new
positive and negative routes to a separate recovery output root.

Current stage: G3 Phase 1 running. G0 froze exactly 2,278 records / 1,104 image
groups with all linked MCTS checksums passing; G1 deterministic search tests
pass; G2 passed every check on 5/5 records after adding the required
deterministic-CuBLAS environment setting. Slurm jobs 101708 (node06, shards
0--1) and 101709 (node07, shards 2--3) run one process per GPU. Preserve all
raw routes and cap only the derived valid-route supervision view at 50 masks
per sample. Phase 2 remains gated on complete global Phase-1 aggregation.

## Active Pareto-filtered predictor amendment (2026-08-17)

The user authorized `plans/filter_train.md`: filter the exact frozen max-50
GQA/TextVQA/ChartQA training supervision per sample by stored-score/visual-ON
Pareto dominance, freeze one common manifest, then run matched ten-epoch
Image+Question duplicated-BCE and exact-valid-set-NLL predictors and the
unchanged 22,307-record no-DocVQA external evaluation.

The manifest/readiness gate is complete and PASS. All 8,000 population records,
the 7,000/1,000 image-group split, and 6,043/874 positive train/validation
inputs are unchanged. The parent 237,802 route occurrences reduce to 9,905
Pareto routes. Both objectives use identical features, direct 28-bit head,
initialization, optimizer, schedule, threshold decoding, and checkpoint rule;
only the objective/data grouping differs. Training/evaluation jobs must use one
generic A6000 each, without a node pin, and may begin only through the project
scheduler. Previous artifacts remain immutable.

Evidence: `outputs/binary_pareto_v1/` and
`workspace/phase_memory/phase_24_binary_pareto_filter_train.md`.

## Active WeMath2.0-Pro label-analysis amendment (2026-08-17)

Status: **completed**. All 4,544 prospectively valid records passed strict
source/contract/budget/trace/validity integrity; the other eight source rows
remain prospectively technical-invalid. Of 4,544 records, 2,266 have at least
one valid route. The diagnostic max-50 exact weighted duplicated-BCE oracle has
13.72% selected-valid Hit@1 and mean nearest-valid Hamming 5.10, while 94.93%
of selected route occurrences are Pareto-dominated. The labels are
conditionally suitable for exact valid-set NLL and route ranking, but not for
unfiltered duplicated-BCE complete-mask supervision. No training was run.
Evidence: `outputs/wemath2pro_mcts_label_analysis_v1/` and
`reports/wemath2pro_mcts_training_suitability.md`.

## Active label-geometry analysis amendment (2026-08-16)

The user authorized `plans/mcts_bce_analysis.md` as one bounded, label-only
research action. It audits the frozen 8,000-record raw MCTS cache, the exact
7,000/1,000 image-group split, the deterministic selected max-50 valid routes,
and the complete-mask oracle induced by the actual duplicated-BCE weighting.
No predictor training, MCTS regeneration, new Qwen inference, route-selection
change, or new method is authorized. The result must distinguish raw search
geometry, selected-supervision geometry, bitwise BCE hybridization, dominated
FULL-route pressure, and learned-predictor failure before recommending any
later action.

Status: **completed** as Outcome C + Outcome E. The exact weighted BCE label
oracle is selected-valid for 5.93% of 6,917 positive inputs; raw and selected
route diversity remain high; 95.83% of selected route occurrences are
Pareto-dominated. No follow-on training is authorized. Evidence:
`reports/binary_mcts_label_geometry_and_bce_oracle_report.md` and
`outputs/binary_mcts_label_geometry_v1/`.

## Active full10 duplicated-BCE comparator amendment (2026-08-15)

The user authorized `plans/full_train_polar_bce.md`: repeat the completed
Question-only and Image+Question ten-epoch direct-predictor runs with the
validated POLAR-style duplicated-route BCE while preserving all data,
max-50 route selection, 0.3 ALL-ON weighting, architecture, optimization,
initialization, validation, decoding, and external-evaluation settings. Run the
two modalities concurrently on one node02 and one node07 GPU, then evaluate
their frozen best-Hit@1 checkpoints concurrently on the unchanged 22,307-record
suite. Existing exact-set-NLL outputs are immutable comparators.

Active source plan:
`plans/dynamic_mllm_label_regeneration_plan.md`

Source-plan SHA-256:
`634f2736d287c647cda7b21755b2ace753db29316ecc9c51523218b498380918`

The source plan was explicitly amended on 2026-08-10 to use a minimal 15-record
smoke, immediate full extraction after it passes, and post-extraction predictor
splits. It supersedes use of the old MCTS v2 cache as ground-truth supervision.
The regenerated cache passed P9 on 2026-08-12. The bounded P10-P13 predictor
diagnostics and the subsequently approved `plans/full_train.md` ten-epoch
Question-only/Image+Question comparison are complete. These actions did not
change the frozen label-generation contract. External predictor evaluation is
not admitted by the full10 internal result.

On 2026-08-12, the user explicitly amended the primary derived supervision cap
from 32 to 50 valid routes per image-query to match POLAR's maximum. The same
deterministic diverse 50-route set must be used by duplicated BCE and exact
set-NLL; raw routes remain untruncated.

## Active objective

Regenerate complete 28-bit layer-wise visual ON/OFF route labels under one
frozen, reproducible Qwen2.5-VL-7B execution contract. Every mask is evaluated
by actual route-conditioned greedy generation and the frozen benchmark metric.

The regenerated raw cache must later support single-route supervision, direct
binary exact valid-set NLL, multi-route/top-K prediction, candidate reranking,
and a derived POLAR segment representation without changing the raw label
source.

## Fixed scope and non-goals

- Data: 8,000 records—GQA 4,000, TextVQA 2,000, ChartQA 2,000; balanced by
  historical all-ON status. Do not add DocVQA.
- Generate all 8K raw labels before predictor splitting. After extraction,
  freeze an exact image-group-disjoint 7,000 train / 1,000 validation split
  using identifiers/group metadata rather than route outcomes. After internal
  validation freezes the predictor checkpoint and inference rule, evaluate on
  the 7,500-record ChartQA/TextVQA core-VQA subset, 5,807-record MMStar/MMMU
  suite, and 9,000-record POPE suite. DocVQA is excluded. Do not pool the three
  suites into one accuracy.
- Route space: unrestricted 28-bit layer-wise visual ON/OFF. Do not constrain
  MCTS to POLAR segments or an early-to-late decision order.
- Execution: frozen base MLLM; native Qwen processor defaults; no project
  `max_image_tokens` override; deterministic greedy benchmark evaluation.
- Historical correct/wrong membership and old mask validity are metadata or
  optional proposals only. Fresh executor outputs and scores are authoritative.
- Preserve all evaluated positive and negative masks plus raw MCTS traces.
- No router, controller, probe, predictor, or base-model training was part of
  P0-P9. P10 remains a separate authorization boundary.

## Frozen execution order and gates

| Step | Required action | Gate / stop rule |
|---|---|---|
| P0 | Inspect existing MCTS v2 code/documentation and freeze the minimal model/processor/executor/generation/evaluator contract. | Deterministic source hashes substitute for a Git commit when unavailable; native processing has no custom visual-token cap. |
| P1 | Deterministically select and freeze five GQA, five TextVQA, and five ChartQA smoke records. | Exactly 15 records selected without intervention outcomes. |
| P2 | Run minimal smoke. | Binary ALL-ON equals native generated tokens on 15/15; frozen mixed routes reproduce exact tokens and scores. Any failure stops extraction. |
| P3 | Run full 8K unrestricted graph MCTS. | Recompute authoritative all-ON per sample; 200 simulations when current-correct, 400 default and at most 600 adaptively when current-wrong. |
| P4 | Verify raw-cache completeness. | Exactly 8,000 terminal records; rerun only failed/incomplete records under the unchanged contract. |
| P5 | Build per-sample and current all-ON/correction summaries. | Counts reconcile with raw cache; historical status remains metadata only. |
| P6 | Analyze route diversity and transitions. | Compute ON counts, transitions, Hamming distances, and practical pairwise diversity summaries. |
| P7 | Freeze image-group-disjoint predictor splits. | Exactly 7,000/1,000, no image overlap, deterministic seed 20260809, dataset/historical-source-cell balance, and no route-outcome optimization. Expanded external evaluation remains separate. |
| P8 | Build derived supervision views. | Raw cache unchanged; single-best, diverse max-50 valid set, positive/negative ranking, and derived POLAR segments. |
| P9 | Freeze cache, checksums, and report. | All raw/derived artifacts, failures, commands, versions, quality/diversity/contract-drift results complete. |
| P10 | Predictor experiments. | Closed until explicit later authorization after P9. |

## Label requirements

- Aim for roughly 20 diverse valid routes when naturally found; do not force a
  count or discard samples with fewer routes.
- Save samples with zero valid routes; exclude them from positive set-NLL
  training views but retain them for negative/reranker analysis.
- When more than 50 routes are valid, retain all in raw storage and derive a
  diverse 50-route subset using minimum-budget/anchor inclusion, ON-count
  stratification, Hamming diversity, and transition diversity.
- For every route store identifiers, split/image group, 28-bit mask, token IDs,
  prediction, raw score/threshold/validity, mask geometry, token counts, and
  generation metadata.

## Hard stops

Stop rather than adapt silently if all-ON/native parity fails, identical masks
are nondeterministic, native processing causes an unhandled resource failure,
benchmark scoring drifts, split leakage appears, reproducibility metadata is
missing, or route semantics change after label generation begins.

## Current state

Completed: **P0 — execution-contract freeze** and **P1 — smoke-manifest
freeze**.

- Frozen contract SHA-256:
  `64f525f5d0a4333e1aeae27f41b9055c8da19a9a0fc566ab3c7db270ea37fc7d`.
- Frozen source manifest: exactly 8,000 records with the approved six source
  cells.
- Frozen smoke manifest: exactly five GQA, five TextVQA, and five ChartQA
  records; one representative mixed mask per dataset.
- All artifact sidecar checksums pass. Because the project root is not a Git
  checkout, deterministic hashes of the active executor, MCTS, evaluator,
  runner, documentation, and plan files are authoritative.

Completed: **P2 — minimal scheduled GPU smoke**. Job `99740` achieved exact
binary ALL-ON/native generated-token parity on 15/15 records and exact repeated
tokens/scores for all three frozen mixed masks. The saved JSON report and
checksum pass.

Completed: **P3 — full 8K unrestricted MCTS extraction**, **P4 — strict cache
reconciliation**, **P5 — per-sample/outcome summaries**, and **P6 — route
diversity/transition analysis**. P4 verified all
8,000 records against their source rows and frozen execution contract with no
missing, duplicate, malformed, error, temporary, or zero-byte records. P5
reverified every raw-record checksum and produced 8,000 per-sample summaries.
The current executor has 4,045 ALL-ON-correct and 3,955 ALL-ON-wrong records;
2,872/3,955 current-wrong records have at least one correcting evaluated mask.
P6 analyzed all 528,047 valid masks and 36,163,535 exact within-sample mask
pairs without applying the later 50-route cap. P7 then froze all 8,000 records
into exactly 7,000 train and 1,000 validation records using seed `20260809`.
There are zero cross-split image groups and the validation set has the exact
historical balance GQA 250/250, TextVQA 125/125, and ChartQA 125/125.
P8 derived all five training-ready views without changing the raw cache:
single-best, max-50 diverse valid-set, matched binary-predictor, complete
positive/negative route-ranking, and canonical POLAR-segment manifests. It
selected 237,802 of 528,047 raw valid routes across 6,917 positive samples and
retained all 2,642,998 evaluated routes in the ranking view. Independent
streaming verification passed. P9 then froze a self-contained 50-file
artifact/code inventory, command provenance, final report, and 53-entry
checksum ledger. Independent `sha256sum -c` verification passed for all 53
entries. P0-P9 are complete.

## Next bounded action

The active evaluation contains 7,500 ChartQA/TextVQA core-VQA records, 5,807
multiple-choice records, and 9,000 POPE records, for 22,307 total; DocVQA is
excluded. Core VQA and multiple choice have zero MCTS image overlap. POPE has
one shared image across 18 records; report the official 9,000 and a frozen
8,982-record image-disjoint sensitivity result. P7 identities, P8 derived
supervision, and the P9 integrity chain remain frozen. The bounded P10 smoke is
now complete. The literal BCE-versus-set-NLL admission comparison passes, but
the exact-set checkpoint decoded constant ALL-ON on all 18 executions and gave
zero compute reduction. A deterministic 150-record audit reproduced its
Hit@1 with constant ALL-ON. P11 resolved this boundary as Outcome C. The
separately approved P12 canonical structured-head comparison then completed as
Outcome B. P13 subsequently added native pre-routing image information under a
matched direct-head smoke. It improved valid-set probability calibration but
decoded constant ALL-ON on all 150 selected validation inputs, so its
prospective execution gate failed. The user then explicitly authorized the
fixed full10 comparison in `plans/full_train.md`; both ten-epoch runs are now
complete. Their best-Hit@1 checkpoints equal constant ALL-ON (`58.12%`), and
later diversity reduces full-validation route quality. Actual frozen-60
best-checkpoint executions are both `50%` with no corrections or regressions.
Therefore do not run the external suite and do not start another objective.
Any different formulation requires a new approved research action.

## Full10 POLAR-matched direct-predictor result (2026-08-13)

The fixed full10 comparison used all 6,043 positive train and 874 positive
validation inputs, deterministic diverse max-50 routes, ALL-ON weight `0.3`,
exact one-of-valid-set NLL, batch 128, AdamW `5e-4`, cosine schedule, ten
epochs, and shared seed/initialization.

- Question-only best-Hit@1 is epoch 2: Hit@1 `58.12%`, nearest Hamming `3.779`,
  `99.66%` ALL-ON, mean ON `27.997`.
- Image+Question best-Hit@1/best-NLL is epoch 4: Hit@1 `58.12%`, nearest
  Hamming `3.780`, `100%` ALL-ON, mean ON `28.000`.
- At epoch 10, unique masks rise to 122 and 64 while Hit@1 falls to `55.03%`
  and `55.84%`. Diversity is not useful under the frozen metrics.
- Group-disjoint input shuffles substantially worsen set-NLL, confirming
  probability-level question/image conditioning, but selected complete masks
  remain constant.
- On the frozen balanced 60, both best checkpoints score `50%` with W→C=0 and
  C→W=0. Question-only epoch 10 has two uncached ChartQA corrections and no
  regressions, but it is not validation-selected.

Decision: **full10 complete; direct factorized predictor not admitted to
external evaluation**. Evidence:
`reports/binary_polar_full10_polar_matched_results.md` and
`outputs/binary_polar/full10/`.

## P11 POLAR-weighted input-dependence result (2026-08-13)

P11 kept the frozen question encoder, POLAR backbone, direct factorized 28-bit
head, 300/150 smoke identities, max-50 route sets, optimizer, seed, and
checkpoint rule. It changed only the matched route weights: ALL-ON received
relative weight `0.3` when a cheaper valid route coexisted.

- Label geometry confirms a dominant shortcut: ALL-ON covers 58.12% of
  positive validation inputs and coexists with a cheaper valid mask in 57.89%.
  The best non-ALL-ON constant, ALL-OFF, covers 17.39%.
- Exact set-NLL contains input signal: aligned validation set-NLL is `14.8699`
  versus `15.5089` under a deterministic within-dataset question shuffle.
- That signal does not yield useful decoded routing: the selected exact
  checkpoint is 98% ALL-ON on 150 route-validation records and 95% ALL-ON on
  the frozen 60-record execution set.
- Bounded execution gives exact set-NLL 50% accuracy, identical to FULL, with
  zero corrections and zero regressions. Its three non-FULL masks are uncached
  and remain wrong. The cached MCTS oracle is 100% on the deliberately balanced
  subset.

Decision: **Outcome C — input signal exists but factorized top-1 remains poor.**
Do not launch full direct-head training. The subsequent P12 structured-head
comparison was separately authorized and is reported below.

Evidence: `reports/binary_polar_p11_results.md` and
`outputs/binary_polar/p11/`.

## P12 canonical structured-head result (2026-08-13)

P12 reused the exact P11 data, max-50 route sets, 0.3 ALL-ON relative weight,
question encoder, POLAR layer encoder, optimizer, seed, 300/150 identities,
two-epoch budget, and 60-record execution subset. It replaced only the direct
28-bit head with a lossless maximal-run boundary/operation head under exact
valid-set NLL.

- Canonical round trip passes for all 237,802 selected route occurrences with
  zero ambiguity, but masks have mean/median 14.11/14 segments and only 3.65%
  have at most eight segments.
- The selected checkpoint is 100% ALL-ON on 150 validation records. Hit@1 is
  57.33%, nearest-valid Hamming is 3.693, and decoded-mask entropy is zero.
- Probability conditioning survives: aligned structured set-NLL is `17.4918`
  versus `18.1939` shuffled. Both decode the same 150 ALL-ON masks.
- All 60 actual execution predictions are ALL-ON. Accuracy is 50%, identical
  to FULL; W→C=0, C→W=0, and compute reduction is zero.

Decision: **Outcome B — Structured representation also collapses.** Do not
full-train P12 and do not stack another route-structure module. Raw P11/P12
set-NLL values are not compared across representations because their output
probability spaces differ.

Evidence: `reports/binary_polar_p12_results.md` and
`outputs/binary_polar/p12/`.

## P13 multimodal input-isolation result (2026-08-13)

P13 restored the P11 direct factorized head and held the exact valid-set NLL,
max-50 route sets, 0.3 ALL-ON weight, split, 300/150 smoke identities, two
epochs, optimizer, seed, checkpoint rule, and decoder fixed. It changed only
the predictor-visible input among Question, Image, and Image+Question.

- The visual feature is the full BF16 sequence of projected Qwen2.5-VL visual
  rows entering decoder layer 0. A 502-record/500-image-group cache passed
  leakage, finite-value, repeat, shape, and checksum checks.
- The selected Image+Question aligned set-NLL is `14.4944`, better than
  Question-only `14.8699`. Its question-, image-, and both-shuffled values are
  `14.5019`, `14.8748`, and `14.9335`, respectively.
- The probability improvement does not survive decoding: Image+Question is
  ALL-ON for 150/150 records, with one unique mask, 57.33% cached Hit@1, 3.693
  nearest-valid Hamming, and 28 mean VISUAL_ON layers. These are the constant
  ALL-ON metrics.
- The prospective gate fails nonconstant behavior, mask diversity, and decoded
  improvement. The frozen 60-record Qwen execution was not run.

Decision: **Outcome B — More input information changes probabilities but not
routes.** Do not full-train P13 and do not automatically pivot to candidate-
route utility/ranking. Such an objective change requires a new approved plan.

Evidence: `reports/binary_polar_p13_results.md` and
`outputs/binary_polar/p13/`.

## Frozen downstream predictor comparison amendment (2026-08-12)

The predictor experiment after P9 is a loss-only comparison using the same
POLAR-style question encoder, direct 28-bit binary predictor, diverse maximum
of 50 regenerated valid routes, image-group-disjoint split, initialization,
optimizer, schedule, training budget, decoding, and execution evaluator.

- Baseline: duplicated `(input, valid mask)` rows with equal within-input route
  weights and ordinary per-bit BCE.
- Proposed: grouped exact set-NLL over complete Bernoulli masks with equal
  within-input route weights and masked `logsumexp`.
- Primary later behavioral evidence: actual frozen-MLLM execution of every
  predicted top-1 mask, including uncached masks.
- Cached valid-set Hit@1 is diagnostic; uncached does not mean invalid.
- The factorized binary head is unchanged. Structured segmentation remains a
  fallback only after a held-out binary-head failure attributable to route
  structure.

Implementation, synthetic deterministic sanity testing, and the bounded
matched smoke are complete. The audited matched comparison configuration
is `configs/binary_polar_loss_comparison_v2.yaml`. Its P9/data checksums,
deterministic 300/150 smoke identities, real Qwen3 BF16 encoder preflight, and
smoke gate pass after a BF16 validation-autocast repair. Exact set-NLL improves
over duplicated BCE in route metrics and smoke execution, but only by selecting
constant ALL-ON in the executed subset. The predictor is question-only, matching
released POLAR; adding image features would no longer be the approved loss-only
comparison. Full training and external evaluation remain unexecuted pending an
explicit user decision.

## Concurrent approved We-Math2.0-Pro extraction

The user approved a separate all-sample We-Math2.0-Pro extraction without
cancelling or modifying active P3. Its source plan is
`plans/dynamic_mllm_wemath2pro_label_extraction_plan.md`.

- Population: inventory all 4,552 Pro records; MCTS the 4,544 records with a
  nonempty question and answer; preserve the other eight as technical-invalid.
- Scoring: official MathRuler mathematical-equivalence accuracy under a frozen
  direct `<answer>` prompt.
- Search: the same unrestricted 28-bit graph MCTS and 200/400/600 budget.
- Gate: five-record exact ALL-ON/native parity plus deterministic mixed masks.
- Hardware after the gate: node06, 8 GPUs, 96 CPUs, 240 GB RAM, with node06
  NCCL safeguards.
- Isolation: separate manifests, contract hash, logs, and output cache; do not
  overwrite or merge the original 8K artifacts.
