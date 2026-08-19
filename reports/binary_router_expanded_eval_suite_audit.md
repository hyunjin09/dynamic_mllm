# Binary Router Expanded Evaluation-Suite Audit

Date: 2026-08-12

Protocol amendment: DocVQA is excluded from the active binary-predictor
evaluation by explicit user direction. The bundled DocVQA manifest, baseline,
and historical SW31 results remain unchanged as preserved reference artifacts.

## Decision

The expanded reference bundle now supplies sufficient held-out evaluation to
remove the separate 500-record internal test. Freeze the regenerated MCTS
population as:

| Role | Total | GQA | TextVQA | ChartQA |
|---|---:|---:|---:|---:|
| Train | 7,000 | 3,500 | 1,750 | 1,750 |
| Internal validation | 1,000 | 500 | 250 | 250 |

The validation set is used for checkpoint selection and the matched smoke/full
training gate. It is not reported as final test evidence. After checkpoints
and inference are frozen, run the complete external evaluation suite described
below.

This supersedes the provisional 6,500/1,000/500 recommendation in
`reports/binary_router_p7_split_and_external_eval_audit.md`. That recommendation
was made before the bundle contained verified core-VQA and POPE evaluation
suites. The new evidence directly resolves its main objection: final evaluation
is no longer limited to the already-inspected, shifted multiple-choice suite.

Exact train/validation identities are still unfrozen. Freezing them is the next
bounded P7 action; no predictor training or router evaluation was run here.

## Expanded frozen evaluation population

### Core VQA

| Benchmark | Split | Records | Metric |
|---|---|---:|---|
| ChartQA | test | 2,500 | relaxed accuracy |
| TextVQA | validation | 5,000 | EvalAI consensus |
| Total | | 7,500 | report separately and micro-average |

Core VQA uses each manifest `prompt`, the project single-word-or-phrase suffix,
native ChartQA/TextVQA images, and benchmark-specific correctness thresholds.
Fractional score and thresholded correctness must both be reported for
TextVQA.

### Previous multiple-choice benchmarks

| Benchmark | Split/configuration | Records |
|---|---|---:|
| MMStar | validation | 1,500 |
| MMMU | validation, multiple-choice only | 847 |
| MMMU-Pro | standard test | 1,730 |
| MMMU-Pro | vision test | 1,730 |
| Total | | 5,807 |

These use exact first-standalone A--J option-letter correctness. They remain an
external task-transfer suite whose earlier SW31/admission outcomes have already
been inspected.

### POPE

| Split | Records | Metric |
|---|---:|---|
| adversarial | 3,000 | yes/no accuracy |
| popular | 3,000 | yes/no accuracy |
| random | 3,000 | yes/no accuracy |
| Total | 9,000 | report separately and micro-average |

POPE uses 500 unique images repeatedly across its three question categories.
It is a hallucination-specific suite and must not be pooled with core-VQA or
multiple-choice accuracy.

The active evaluation population contains 22,307 records. Results must be
reported by benchmark and by the bundle's three scientific suites; there is no
single 22,307-record overall accuracy.

## Integrity and overlap audit

The bundle verifier passed with full image checking:

- external multiple choice: 5,807 UIDs and 6,173 image references;
- bundled core VQA: 12,849 UIDs and 12,849 image references, of which the
  7,500 ChartQA/TextVQA records are selected for the active evaluation;
- POPE: 9,000 UIDs and 9,000 image references;
- all frozen manifests, baseline caches, reference rows, metrics, thresholds,
  source alignment, and image hashes passed the bundle contract.

The outcome-blind overlap audit loaded no router prediction, likelihood,
correctness, or intervention outcome. It compared the evaluation manifests
against all 8,000 MCTS records using identifiers, normalized text, and exact
image SHA-256.

| Benchmark | Eval records | Shared MCTS images | Eval records on shared images | Exact image-question overlap |
|---|---:|---:|---:|---:|
| ChartQA test | 2,500 | 0 | 0 | 0 |
| TextVQA validation | 5,000 | 0 | 0 | 0 |
| DocVQA validation (audited; excluded) | 5,349 | 0 | 0 | 0 |
| MMStar validation | 1,500 | 0 | 0 | 0 |
| MMMU validation MC | 847 | 0 | 0 | 0 |
| MMMU-Pro standard | 1,730 | 0 | 0 | 0 |
| MMMU-Pro vision | 1,730 | 0 | 0 | 0 |
| POPE adversarial | 3,000 | 1 | 6 | 0 |
| POPE popular | 3,000 | 1 | 6 | 0 |
| POPE random | 3,000 | 1 | 6 | 0 |

Some generic question strings recur across datasets—44 ChartQA and 136 TextVQA
normalized strings in the active suite—but none shares its image, identifier,
or exact image-question pair with an MCTS sample. These are not record leakage.

POPE has one image-content overlap with the MCTS population, appearing in six
records per POPE split (18 total). Preserve the official full 9,000-record POPE
result for standard comparability, and pre-specify a secondary strict
image-disjoint result over 8,982 records. Do not delete or replace the 18 cases
after predictor outcomes are visible.

Evidence:
`outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json`.

## Evaluation semantics for the new predictor

The bundle's original SW31 policy is not the new direct binary predictor. The
new evaluator must:

1. construct the bundle-defined Qwen input and benchmark prompt;
2. construct one predictor text input before base-model execution;
3. predict one complete static 28-bit mask;
4. execute that mask with the verified binary executor from layer 0;
5. apply the unchanged dense text computation and frozen benchmark scorer.

Do not reuse:

- the SW31 checkpoint or its online hidden-state features;
- the forced first-eight visual-ON prefix used only by the prior external
  admission experiment;
- the admission classifier, threshold, or all-ON fallback;
- any route selected using benchmark answers or cached correctness.

Use these prospective predictor inputs:

- Core VQA and POPE: the manifest `question` field;
- MMStar/MMMU/MMMU-Pro: ordered `instruction_text_chunks`, which contain the
  question and options but exclude the fixed answer suffix.

This preserves the predictor's pre-action question-text contract. Both
duplicated-BCE and exact-set-NLL checkpoints must receive exactly the same
texts.

## Required direct-predictor evaluation preflight

Before the 22,307-record run:

1. verify the bundle and frozen manifests/checksums;
2. freeze each objective's checkpoint using only internal validation;
3. freeze predictor text construction and top-1 threshold decoding;
4. verify direct-executor all-ON generation/logit/scorer parity on a small
   stratified subset from every suite;
5. use one identical decoding policy for all-ON and predicted masks within a
   suite—do not inherit the prior K=8 sparse EOS difference;
6. verify repeated predicted-mask execution is deterministic;
7. preserve multi-image order for MMMU/MMMU-Pro;
8. verify every output UID occurs exactly once and all outputs serialize;
9. stop rather than use a cached baseline if parity fails.

The bundle's cached all-ON rows may be used only after this parity preflight.
The original SW31 reference results remain historical comparators rather than
the execution path for the new predictor.

## Final evaluation outputs

For both matched objectives, report:

- benchmark-native score and thresholded correctness;
- paired score/correctness delta against all-ON;
- all-ON wrong to predicted correct and all-ON correct to predicted wrong;
- unchanged correct and unchanged wrong;
- mean, median, and distribution of visual-ON layer count;
- unique predicted masks and mask-transition summaries;
- benchmark-level uncertainty with image clustering where images repeat;
- the official POPE 9,000 result plus the frozen 8,982 image-disjoint
  sensitivity result.

Cached valid-set Hit@1 and MCTS-oracle recovery apply only to the internal MCTS
validation data. They are undefined for the external suites and must not be
invented.

## Why the internal test is removed

The earlier 500-record internal test protected against relying solely on an
already-inspected and distribution-shifted multiple-choice suite. The expanded
bundle now adds 7,500 selected core-VQA records, including image-disjoint
ChartQA and TextVQA evaluations under the same task-native metrics, plus POPE.
Keeping a separate internal test would reduce training from 7,000 to
6,500 without resolving a remaining validity gap large enough to justify the
cost.

The strongest objection is that GQA lacks a same-benchmark external test.
Accordingly, do not claim GQA in-domain generalization. Report the available
cross-task and core-VQA evidence precisely.

## Current blocker and next action

No trained direct binary predictor checkpoint exists in the project artifacts.
The reference bundle contains only the old SW31/admission checkpoints. The
expanded evaluation therefore cannot run yet.

Next freeze the exact 7,000/1,000 image-group-disjoint P7 identities. Then
complete P8/P9, run the separately authorized matched predictor smoke/full
training, freeze checkpoints from validation, implement and preflight the
static-mask bundle adapter, and finally run all three evaluation suites. Do not
run the old SW31 scripts as a substitute for evaluating the new predictor.
