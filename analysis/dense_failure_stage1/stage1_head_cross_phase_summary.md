# Stage-1 Failure Head: Problem and Cross-Phase Evidence Summary

Date: 2026-09-02

Status: retrospective synthesis; no new training or model execution

Scope: Stage-1 dense-failure prediction, the Phase-49 cross-dataset OOD
diagnostic, the original corrective search, and the Phase-59 canonical-source
search

## Executive summary

The deployed Stage-1 failure head works well on the historically selected
population but does not generalize to the new canonical-source population.
The immediate supported failure is the **old Stage-1 head together with its
frozen normalization and calibration**. It assigns high failure scores to many
new samples that the dense Qwen model answers correctly.

This is not just a threshold problem. On the new population, failure ranking
itself degrades or reverses, especially for ChartQA:

| Population | ChartQA AUROC | GQA AUROC | TextVQA AUROC | Overall AUROC |
|---|---:|---:|---:|---:|
| Historical train | 0.9900 | 0.8906 | 0.9873 | 0.9529 |
| New canonical-source pool | 0.3469 | 0.7453 | 0.6846 | 0.4056 |

The earlier Phase-49 OOD experiment had already shown that Stage-1 failure
prediction and conservative calibration did not transfer uniformly across
held-out benchmarks. Phase 59 exposed a different and stronger failure axis:
**within-benchmark source shift**. The dataset names remained GQA, ChartQA,
and TextVQA, but new source identities produced substantially different hidden
state distributions.

The two corrective searches remain technically valid under their frozen
contracts. Dense answers use current-runtime LMMS correctness, and all retained
corrective routes replay exactly. However, the new search is not a clean
sample-count-only extension of the old search because the frozen Stage-1 gate
selects a radically different population.

## 1. Stage-1 head and trigger rule in scope

The head used for the two full corrective-label searches was the shared
`state_layer_random4` predictor, referred to as **Shared Random-4**. It was
trained on dense hidden-state features from the historical 7,999-sample
population.

The frozen trigger rule was:

```text
trigger if score(layer) > 0.9253839280601031
at any layer from 0 through 27
```

The trigger is therefore an any-layer rule. A sample has 28 opportunities to
cross the threshold. This amplifies a score-distribution shift, but it does not
explain the shift itself.

The Stage-1 target was current dense all-on LMMS correctness:

```text
y = 1: current dense answer is wrong
y = 0: current dense answer is correct
```

The head does not use W-to-C membership or four-action labels as its target.

### Gate-selection history

The gate history contains two distinct decisions that should not be conflated:

1. Phase 52 selected **Shared Fixed-L27** as the best aggregate utility gate on
   the old validation split. Its threshold was `0.8497647428417646` at layer
   27. Shared Random-4 was the validation runner-up.
2. Phase 53 compared gates as substrates for bounded corrective treatment.
   Shared Random-4 was selected on validation as the better **dynamic treatment
   substrate**, with slightly higher population rescue and conditional
   correctability than the alternatives.
3. Phase 54 consequently froze Shared Random-4 for the train corrective-search
   population. Both the original Phase-56 search and the new Phase-59 search
   reused this exact gate, checkpoint, normalization, and threshold.

Thus, the search pipeline did not use the Phase-52 fixed-L27 aggregate winner;
it used the Phase-53/54 dynamic treatment gate.

## 2. Phase 49: cross-dataset OOD diagnostic

### Question tested

Phase 49 trained independent layerwise linear probes on two datasets and
evaluated on the third held-out dataset. All data came from the original
7,999-sample population.

Examples:

```text
train: GQA + ChartQA
OOD target: TextVQA
```

Target data was excluded from training, layer selection, and threshold
calibration.

### Results

| Source datasets | OOD target | Source-selected layer | OOD AUROC | Input-control AUROC | Correct preservation at source 99% threshold | Wrong recall |
|---|---|---:|---:|---:|---:|---:|
| GQA + TextVQA | ChartQA | 26 | 0.4502 | 0.5472 | 0.3053 | 0.5840 |
| ChartQA + TextVQA | GQA | 20 | 0.6160 | 0.5432 | 0.1380 | 0.9385 |
| GQA + ChartQA | TextVQA | 22 | 0.7236 | 0.5219 | 0.9950 | 0.0920 |

Target-descriptive layer-21 AUROCs were higher for ChartQA and TextVQA
(`0.8018` and `0.8046`) but only `0.5975` for GQA. This showed that some
failure signal remained accessible, while source-only layer selection and
calibration were unstable.

### Conclusion and limit

Phase 49 did **not** clear the head for general OOD deployment. It concluded
that benchmark-general transfer was mixed and that conservative thresholds did
not transfer uniformly.

It also did not test the shift later encountered in Phase 59. Phase 49 tested:

```text
old GQA/ChartQA population -> old held-out TextVQA population
```

Phase 59 instead tested:

```text
old selected ChartQA/TextVQA identities
    -> new canonical-source ChartQA/TextVQA identities
```

Same benchmark name does not imply the same image, question, or hidden-state
distribution.

## 3. Historical population and original corrective search

### Population and gate behavior

The original train population contained 6,399 of the 7,999 completed current-
runtime samples:

| Quantity | Count |
|---|---:|
| Dense correct | 3,199 |
| Dense wrong | 3,200 |
| Triggered correct | 39 |
| Triggered wrong | 1,881 |
| Correct preservation | 98.78% |
| Wrong trigger recall | 58.78% |
| Trigger precision | 97.97% |

Behavior was already dataset-dependent. Train wrong-trigger recall was:

| Dataset | Wrong-trigger recall |
|---|---:|
| GQA | 0.2675 |
| ChartQA | 0.9013 |
| TextVQA | 0.9150 |

The gate therefore missed most GQA failures even in the historical
population, despite maintaining high aggregate precision.

### Corrective-search results

Every one of the 1,881 triggered-wrong samples received exhaustive single-
intervention search. The 1,183 samples without a successful single route then
received cap-200 MCTS.

| Outcome | Count | Fraction of 1,881 triggered wrong |
|---|---:|---:|
| Single-fixable | 698 | 0.3711 |
| Additional MCTS-only fixable | 209 | 0.1111 |
| Total bounded fixable | 907 | 0.4822 |
| Unresolved under bounded search | 974 | 0.5178 |

The search retained 7,628 exact-replay single routes and 442 exact-replay MCTS
routes. The 39 triggered-correct samples supplied separate FULL-preservation
supervision.

### Interpretation

This search established substantial bounded corrective support **conditional
on the historical Shared Random-4 trigger population**. It did not establish
that the Stage-1 head generalized to new sources.

## 4. New canonical-source population and current search

### Candidate construction

Phase 59 froze 4,000 new candidates before observing dense correctness or
fixability:

| Dataset | Candidates |
|---|---:|
| GQA | 2,000 |
| ChartQA | 1,000 |
| TextVQA | 1,000 |

The pool used pinned canonical training sources, outcome-blind metadata
stratification, unique SHA-256 image groups, and zero UID or image-content
overlap with all 8,000 historical candidates.

### Dense outcomes and frozen-gate behavior

| Dataset | Dense correct | Dense wrong | Triggered correct | Triggered wrong |
|---|---:|---:|---:|---:|
| GQA | 1,264 | 736 | 90 | 156 |
| ChartQA | 884 | 116 | 765 | 83 |
| TextVQA | 981 | 19 | 836 | 18 |
| Overall | 3,129 | 871 | 1,691 | 257 |

Compared with the original population:

| Metric | Historical train | New canonical pool |
|---|---:|---:|
| P(trigger \| wrong) | 0.5878 | 0.2951 |
| P(trigger \| correct) | 0.0122 | 0.5404 |
| Trigger precision | 0.9797 | 0.1320 |

The high false-trigger rate is concentrated in ChartQA and TextVQA:

| Correct samples | Historical trigger rate | New trigger rate | New median maximum score | New median layer-0 score |
|---|---:|---:|---:|---:|
| ChartQA | 0.0088 | 0.8654 | 1.0000 | 0.9801 |
| TextVQA | 0.0138 | 0.8522 | 0.9984 | 0.9527 |
| GQA | 0.0131 | 0.0712 | 0.6167 | 0.4505 |

Most new ChartQA and TextVQA false triggers begin at layer 0. The head is not
gradually becoming uncertain late in computation; the new inputs are already
outside its learned risk geometry at the first decoded layer.

### Current corrective-search results

Every one of the 257 new triggered-wrong samples received exhaustive single
search. The 182 without a successful single route then received cap-200 MCTS.

| Outcome | Count | Fraction of 257 triggered wrong |
|---|---:|---:|
| Single-fixable | 75 | 0.2918 |
| Additional MCTS-only fixable | 33 | 0.1284 |
| Total bounded fixable | 108 | 0.4202 |
| Unresolved under bounded search | 149 | 0.5798 |

The search retained 950 exact-replay single routes and 66 exact-replay MCTS
routes. All 1,691 triggered-correct samples received exact FULL-preservation
replay.

The expanded corpora contain:

| Corpus | Route records | Unique bases |
|---|---:|---:|
| A: FULL preservation | 1,730 | 1,730 |
| B: single corrective | 8,578 | 773 |
| C: MCTS corrective | 508 | 242 |

### Interpretation

Bounded correctability among triggered-wrong samples is moderately similar to
the old search (`0.4202` versus `0.4822`). The main discontinuity is not the
four-action executor or route search. It is which samples the Stage-1 head
admits:

- far fewer wrong samples are triggered;
- vastly more correct samples are triggered;
- the resulting preservation population dominates the new triggered cohort.

Therefore Phase 59 is a canonical-source expansion, not a pure scale-only
replication of the original search population.

## 5. What the post-hoc head diagnostic establishes

### Validity checks

The new run bound the same:

- Shared Random-4 checkpoint;
- normalization artifact;
- global threshold and strict comparison;
- Qwen2.5-VL snapshot and dense generation contract;
- hidden-state feature definitions;
- LMMS evaluator and binary correctness rules.

All 4,000 dense rows and score trajectories completed. Representative
triggered-correct ChartQA and TextVQA rows had exact generated-answer/GT
agreement and the expected LMMS correctness label.

This rules out a simple population-count error, model-version change, or
obvious evaluator substitution.

### Why changing only the threshold is insufficient

The new failure AUROC is below random overall and inverted on ChartQA. A
different threshold can trade recall against false triggers, but it cannot
repair reversed ranking.

As a post-hoc sensitivity, applying the Phase-52 fixed-L27 threshold to the new
Shared-head scores would still trigger 48.4% of correct samples overall and
only 31.2% of wrong samples. Switching from the dynamic threshold to that old
fixed control therefore does not address the new-source ranking failure.

### Observable source correlates

New correct populations move toward visual-token regimes associated with old
wrong rows:

| Dataset | Historical correct median visual tokens | Historical wrong | New correct |
|---|---:|---:|---:|
| ChartQA | 630 | 580 | 580 |
| TextVQA | 888 | 962 | 999 |
| GQA | 234 | 234 | 234 |

GQA has the most stable token regime and the smallest false-trigger shift.
Token count is not proven causal, but this is evidence consistent with the head
using source/image-format information encoded in the hidden states.

## 6. Diagnosis

### Supported

The frozen Stage-1 head, normalization, and calibration have a severe
canonical-source/OOD generalization failure. This is a real score-trajectory
shift concentrated in ChartQA and TextVQA, not merely a bookkeeping artifact.

### Suspected

The head learned source-, image-format-, or representation-level shortcuts that
were correlated with correctness in the historically selected population.
The visual-token shift is indirect evidence, not proof of the exact causal
feature.

### Unknown

It is not yet known whether the head architecture itself lacks capacity. The
failure may instead be caused by old weights, old normalization statistics, or
the old population's selection bias. It is also unknown whether benchmark-
train familiarity or another source property explains why many new samples are
easy for dense Qwen but look high-risk to Stage 1.

Nothing in these results implicates the Stage-2 four-action head as the cause of
the new trigger shift. Stage 2 receives the population selected by Stage 1.

## 7. What remains valid and what does not

### Still valid

- Current-runtime dense answers and LMMS correctness labels for both
  populations.
- Historical conditional search result: 907/1,881 bounded-fixable among old
  triggered wrong.
- New conditional search result: 108/257 bounded-fixable among new triggered
  wrong.
- Exact replay and provenance of every retained single, MCTS, and preservation
  route.
- Separation of preservation, single-fixable, MCTS-only, and unresolved
  records.

### Not supported

- Treating the new trigger rate as calibrated failure probability.
- Claiming the 4,000-sample search is a clean count-only scale-up of the old
  search.
- Interpreting 1,691 triggered-correct rows as evidence that dense Qwen became
  unreliable.
- Assuming that the prior leave-one-dataset-out experiment cleared the head for
  new within-dataset sources.
- Concluding that a threshold-only adjustment will repair the head.
- Concluding, without another diagnostic, that the head architecture itself is
  the root cause.

## 8. Current decision boundary

The immediate problem is the deployed Stage-1 head plus its frozen
normalization/calibration on the new source. Before treating expanded A+B as a
clean Stage-2 scale experiment, a separate authorized action should decide how
to handle this population shift.

The smallest discriminating experiment would keep the Stage-1 architecture and
features fixed and fit it on the new current-runtime labels under an image-
group-disjoint split:

- if held-out AUROC recovers, the architecture is adequate and the old fit,
  normalization, or calibration was the problem;
- if held-out AUROC remains poor, the hidden-state representation or head
  capacity is the deeper limitation.

That experiment has **not** been run or authorized by this summary.

## 9. Authoritative evidence

- Phase-49 OOD diagnostic:
  `analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md`
- Shared-head evaluation:
  `analysis/dense_failure_stage1/shared_global_gate/decision_summary.md`
- Gate winner selection:
  `analysis/dense_failure_stage1/gate_winner_selection/decision_summary.md`
- Treatment-substrate selection:
  `analysis/dense_failure_stage1/treatment_correctability/decision_summary.md`
- Historical trigger map:
  `analysis/dense_failure_stage1/trigger_map/decision_summary.md`
- Historical corrective search:
  `analysis/dense_failure_stage2/full_corrective_labels/decision_summary.md`
- New data-scale search:
  `analysis/dense_failure_stage2/data_scale_search/summaries/data_scale_search_summary.md`
- Post-hoc new-source head diagnostic:
  `analysis/dense_failure_stage2/data_scale_search/diagnostics/posthoc_gate_shift_diagnostic.md`
