# Binary Router P7 Split and External-Evaluation Audit

Date: 2026-08-12

> Superseded later on 2026-08-12 by
> `reports/binary_router_expanded_eval_suite_audit.md`. The bundle subsequently
> added verified ChartQA/TextVQA and POPE suites, supporting a revised
> 7,000-train/1,000-validation design with external testing.

## Decision

Use the regenerated 8,000-record GQA/TextVQA/ChartQA population as:

| Role | Total | GQA | TextVQA | ChartQA |
|---|---:|---:|---:|---:|
| Train | 6,500 | 3,250 | 1,625 | 1,625 |
| Internal validation | 1,000 | 500 | 250 | 250 |
| Internal test | 500 | 250 | 125 | 125 |

Use all 5,807 records in
`eval/reference/shared_prefix_eval_20260812/data/heldout_mmstar_mmmu_final_v2`
as a separate **external transfer test** after the checkpoint and inference
rule have been frozen from internal validation. Do not use this external bundle
for training, checkpoint selection, threshold selection, or hyperparameter
tuning.

This supersedes the earlier approximate 6,000/1,000/1,000 operational target,
but it does not eliminate an in-domain test. Training receives 81.25% of the
regenerated population while a 500-record outcome-blind internal test remains
available for the primary matched duplicated-BCE versus exact-set-NLL
comparison.

The exact split identities have not been frozen in this audit. Freezing them is
the next bounded P7 action.

## What the reference evaluation bundle does

The Markdown protocol defines a reproducible external multiple-choice
evaluation for Qwen2.5-VL-7B-Instruct at revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, SDPA, under Transformers
5.3.0. Its population is:

| Benchmark | Split/configuration | Records |
|---|---|---:|
| MMStar | validation | 1,500 |
| MMMU | validation, multiple-choice only | 847 |
| MMMU-Pro | standard test | 1,730 |
| MMMU-Pro | vision test | 1,730 |
| Total | | 5,807 |

The bundle contains 6,173 image references and 5,306 unique image hashes. It
uses the manifest prompt verbatim, a single user message, the Qwen chat
template, greedy generation with `max_new_tokens=16`, repetition penalty 1.05,
and first-standalone-option-letter scoring over A--J.

The bundled policy is not the predictor being developed in this repository.
It uses a live SW31 per-layer router, forces layers 0--7 visual-ON, and then
uses a separate prefix admission gate to select all-ON or the sparse
continuation. Its locked external result was negative: all-ON accuracy was
44.62%, learned admission accuracy was 44.05% (difference -0.57 percentage
points; paired bootstrap 95% CI [-1.12, 0.00]), and learned admission used a
mean 25.01 visual-ON layers. These inspected outcomes are historical context,
not evidence about the new direct-mask predictor.

## What can and cannot be reused

Reusable components:

- the authoritative 5,807-record manifest and bundled images;
- image/query construction, placeholder ordering, processor policies, and
  multiple-choice scoring;
- the pinned Qwen model snapshot and software contract;
- manifest/checksum verification, GPU sharding, merge/completeness checks,
  paired outcome categories, bootstrap/report structure, and the all-ON
  comparison framework;
- the saved all-ON baseline only after a direct parity preflight confirms that
  the new runner reproduces its prompt, processing, decoding, and scores.

Components that must not be carried into the new evaluation:

- the SW31 checkpoint and online hidden-state router;
- the forced eight-layer dense prefix;
- the prefix-admission checkpoint, threshold, and fallback rule;
- the assumption that the bundled sparse EOS policy or sparse execution path
  is automatically equivalent to the current static-mask executor.

The new adapter must obtain one complete 28-bit mask from the trained predictor
before Qwen execution and pass it to the verified static binary executor. It
must not force the first eight bits ON and must not add an admission gate.

The current predictor consumes question text only. For these multiple-choice
records, its external input must be frozen prospectively as the ordered
`instruction_text_chunks` (including the question and answer options but
excluding the answer suffix). This choice must be identical for both loss
objectives. It is an adapter definition, not an opportunity to retune the
predictor.

## Overlap audit

The outcome-blind audit compared the 8,000 regenerated-label records against
all 5,807 external records using every reliable common key available.

| Check | Overlap |
|---|---:|
| UID | 0 |
| Sample ID | 0 |
| Benchmark name | none |
| Exact image SHA-256 | 0 |
| Normalized question/instruction text | 0 |
| Normalized full prompt | 0 |
| Exact single-image/question pair | 0 |

All 6,173 external image references matched their manifest-declared SHA-256
values. The audit found 7,477 unique MCTS image hashes and 5,306 unique
external image hashes, with no intersection. Evidence is saved in
`outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json`.

Therefore the external population is image- and record-disjoint from the MCTS
population. It is nevertheless not an untouched publication test: its prior
SW31/admission outcomes have already been inspected, and it differs from the
training benchmarks in task format and distribution.

## Split feasibility and frozen selection rule

The MCTS population contains 7,497 image groups:

| Dataset | Records | Image groups | Multi-question groups | Maximum group size |
|---|---:|---:|---:|---:|
| GQA | 4,000 | 3,834 | 163 | 3 |
| TextVQA | 2,000 | 1,925 | 75 | 2 |
| ChartQA | 2,000 | 1,738 | 258 | 4 |

An exact 6,500/1,000/500 image-group-disjoint partition is feasible. As a
constructive sufficiency check, even singleton groups alone provide more than
the required validation-plus-test counts in every dataset and historical
all-ON stratum. The final split should use a stable seed-hash order and a
deterministic grouped constrained assignment, rather than selecting only
singleton groups.

The split contract is:

- group by `image_group_id`, with no group crossing train, validation, or
  internal test;
- seed `20260809`;
- exact dataset totals shown in the decision table;
- internal validation historically balanced within every dataset;
- internal test historically balanced overall: GQA 125/125, TextVQA 62/63,
  and ChartQA 63/62 historical correct/wrong;
- use only dataset, image-group identifier, and historical source-cell status
  for selection;
- do not use current all-ON result, valid-route count, correction discovery,
  route diversity, predictor outcomes, or external outcomes.

Historical status is used only to retain the source population's designed
balance. It is not an authoritative current correctness label.

Feasibility evidence and its checksum are saved in
`outputs/label_regeneration/v1/post_generation/predictor_split_design_audit_v1.json`.

## Role of each split

### Training

Train the matched duplicated-route BCE and exact valid-set NLL predictors on
the same 6,500 input records and same capped valid-mask views. Records with no
valid cached route remain in the manifest but cannot contribute a positive
valid-set loss.

### Internal validation

Use the 1,000 records for checkpoint/epoch selection, optimization sanity, and
the already approved bounded smoke-to-full decision. Actual execution should
cover every predicted top-1 mask, including uncached masks. Cached valid-set
Hit@1 is defined only where a cached positive set exists.

### Internal test

Use the 500 records once for the primary unbiased, in-domain comparison of the
two loss formulations. Report actual executed benchmark performance,
FULL-wrong to predicted-correct, FULL-correct to predicted-wrong, unchanged
outcomes, average and distribution of visual-ON layers, cached Hit@1 where
defined, and oracle-gap recovery where the MCTS cache supports it.

### External transfer test

After freezing the selected checkpoints and mask-decoding rule, execute both
predictors on all 5,807 MMStar/MMMU/MMMU-Pro records. Report benchmark-wise and
joint actual accuracy, paired change relative to the bundle-matched all-ON
baseline, correction/regression counts, visual-ON counts, and paired clustered
uncertainty. Cached MCTS Hit@1 and MCTS oracle-gap recovery are not available
for this external population and must not be fabricated.

## Required adapter preflight before external execution

Before the full external run:

1. verify the bundle inventory and image/model checksums;
2. freeze the selected predictor checkpoints, direct-mask decoding, and exact
   predictor input text;
3. verify the direct binary executor's all-ON generation and score parity
   against the bundle baseline on a small stratified subset;
4. verify deterministic repeated execution for predicted mixed masks;
5. verify that all 5,807 UIDs load and that multi-image ordering is preserved;
6. ensure both loss objectives use the identical input, decoding, Qwen
   execution, scorer, and generation settings;
7. stop on parity failure rather than reusing the cached baseline silently.

The bundle's differing cached all-ON and sparse EOS sets are a concrete parity
risk. The external adapter must freeze one prospectively justified policy and
demonstrate baseline equivalence before scientific comparison.

## Alternatives considered

The first proposal was 7,000 train / 1,000 validation with the external bundle
as the only test. Its advantage is 500 additional training records. Its
strongest objection is decisive: it removes the only unbiased in-domain test,
while the external population is already inspected, multiple-choice, and
distribution-shifted. A 6,000/1,000/1,000 split would preserve more in-domain
test power but uses fewer training records than necessary for the user's
train-most preference. The selected 6,500/1,000/500 design preserves both
requirements with exact grouped feasibility.

## Current blocker and next bounded action

No trained direct binary predictor checkpoint was found in the current project
artifacts; the repository contains the implemented and sanity-tested matched
training pipeline, not a completed predictor. Therefore evaluation cannot run
yet.

The next bounded action is to freeze the exact P7 split identities under the
contract above and create checksum-bound train/validation/internal-test
manifests. P8 derived supervision and predictor smoke training remain separate
later actions. The external evaluation should run only after internal
validation has frozen the trained checkpoints.
