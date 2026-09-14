# Full-Benchmark End-to-End Evaluation Plan — Revised External Scope

## 1. Why This Experiment Matters

The current best method-level candidate is:

- Robust ALL-source Stage-1
- P90 operating point
- Stage-2 Experiment A (preservation + single-intervention supervision)

On the 800-sample historical validation set, this candidate produced:

- W→C = 4
- C→W = 1
- Net = +3

The router does not need to reproduce oracle trajectories perfectly. The method-level criterion is:

> **Does routing correct more originally wrong answers than it breaks originally correct answers?**

Primary success criterion:

`W→C > C→W`

This phase directly tests the paper-relevant end-to-end claim at full benchmark scale.

## 2. What This Experiment Can Establish

This phase can establish:

- whether the frozen method is net-positive at full benchmark scale;
- the effect size across ChartQA, TextVQA, MMMU-Pro, and POPE;
- whether the effect is broad or concentrated in specific tasks;
- whether C→C preservation remains acceptable;
- whether the small validation gain survives larger evaluation.

## 3. What This Experiment Cannot Establish

A weak or negative result should not be overinterpreted as evidence that:

- dynamic visual routing is impossible;
- Stage-2 should be abandoned;
- READ/WRITE decomposition is invalid;
- single-intervention supervision can never work.

A negative result establishes only that the **current frozen implementation** is not yet sufficiently net-positive at scale.

## 4. Frozen Primary Candidate

Freeze exactly one primary candidate before evaluation:

- Stage-1: robust ALL-source Shared Random-4 head
- Operating point: P90
- Stage-2: Experiment A, single-supervision router
- Stage-2 architecture: shared structured READ/WRITE router
- Actions: FULL / READ_ONLY / WRITE_ONLY / IGNORE

Do not change:

- Stage-1 weights or normalization;
- P90 threshold;
- Stage-2 checkpoint;
- Stage-2 architecture;
- executor semantics;
- generation settings;
- LMMS-Eval correctness contract.

Record exact hashes and code commit in `protocol.md`.

## 5. Why P90 Is Primary

P90 is the only retained operating point where the current single-supervision Stage-2 produced a positive end-to-end validation signal.

P98/P95 produced no answer changes in the same validation experiment.

P90 is therefore the prospectively frozen full-evaluation candidate. This does not make it the final universal threshold.

## 6. Full Evaluation Scope

Run the established project evaluation pipeline on the following four benchmark families:

```text
1. ChartQA
2. TextVQA
3. MMMU-Pro
4. POPE
```

The agent should reuse the **existing project-specific evaluation contracts, task names, variants, splits, and lmms-eval conventions already used previously for these four families**.

Do not introduce new benchmark families in this phase.

Do not add:

```text
GQA
DocVQA
MMStar
base MMMU
```

unless separately requested later.

The evaluation population must be the normal full benchmark population defined by the existing project pipeline.

Do not:

```text
subsample based on correctness
subsample based on Stage-1 score
subsample based on known fixability
subsample based on prior search membership
```

The method must be evaluated end-to-end on the full established evaluation scope.

## 7. Evaluation-Freeze Rule

Before launching the four-family evaluation, freeze exactly:

```text
Robust ALL-source Stage-1
P90
Stage-2 Experiment A
```

Do not use the external results to choose a different Stage-1 threshold or Stage-2 checkpoint afterward.

If the existing project evaluation setup already distinguishes development versus held-out reporting internally, reuse that contract exactly rather than redefining it here.

## 8. Paired Dense vs Routed Evaluation

Evaluate every UID twice with identical inputs and generation settings.

Dense control:

`all layers FULL`

Routed method:

`robust Stage-1 → P90 trigger → frozen Stage-2 A`

The only intended difference is the routing intervention.

## 9. Routed Runtime Contract

Before Stage-1 trigger:

`action = FULL`

If Stage-1 never triggers:

`remain FULL through L27`

If Stage-1 triggers at `l*`:

- activate Stage-2;
- at each layer `j >= l*`, read the actual current routed state;
- predict one of four actions;
- execute it;
- use the resulting routed state at the next layer.

Do not change the frozen handoff semantics.

## 10. Primary Per-Sample Outcomes

For each UID classify the paired result as exactly one of:

- C→C
- C→W
- W→C
- W→W

Save UID, dataset, dense prediction/correctness, routed prediction/correctness, and transition type.

## 11. Primary Metrics

For each benchmark and pooled overall report:

- N
- Dense accuracy
- Routed accuracy
- Absolute accuracy change
- W→C count
- C→W count
- Net correction = W→C − C→W
- W→C rate among Dense-W
- C→C preservation among Dense-C

Primary method metric:

`Net correction = W→C − C→W`

Equivalent:

`ΔAccuracy = Net / N`

## 12. Primary Results Table

| Benchmark family | N | Dense Acc | Routed Acc | ΔAcc | W→C | C→W | Net | C→C preservation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | | | | | | | | |
| TextVQA | | | | | | | | |
| MMMU-Pro | | | | | | | | |
| POPE | | | | | | | | |
| Overall | | | | | | | | |

## 13. Paired Uncertainty

Use UID-level paired bootstrap for:

- ΔAccuracy
- Net correction rate
- W→C rate
- C→C preservation

Report 95% confidence intervals by benchmark and pooled.

Do not make statistical significance a mandatory condition for calling a small but consistent result scientifically interesting.

## 14. Stage-1 Admission Metrics

Secondary diagnostics per benchmark:

- trigger rate;
- P(trigger | Dense-C);
- P(trigger | Dense-W);
- trigger precision;
- median trigger layer;
- early/middle/late trigger distribution.

These explain the full-eval behavior but are not the primary method metric.

## 15. Stage-2 Behavior Metrics

Among triggered samples report:

- fraction using at least one non-FULL action;
- post-trigger FULL fraction;
- number of non-FULL actions/sample;
- first non-FULL layer;
- trigger-to-first-non-FULL delay;
- READ_ONLY count;
- WRITE_ONLY count;
- IGNORE count.

These are mechanism diagnostics, secondary to final correctness.

## 16. Selectivity Metrics

Report:

- triggered W;
- triggered W with any non-FULL;
- W→C;
- triggered C;
- triggered C with any non-FULL;
- C→W.

Also compute when defined:

`Rescue precision = W→C / (W→C + C→W)`

## 17. Dataset-Specific Interpretation

Do not require identical gains across all three tasks.

A result such as:

- GQA ≈ neutral
- ChartQA positive
- TextVQA positive

can still support a selective/task-dependent method claim.

However, any benchmark with substantial negative regression must be reported explicitly.

## 18. No Oracle/Fixability Filtering

Primary evaluation must include the full benchmark population.

Do not evaluate only:

- triggered W;
- known single-fixable samples;
- known MCTS-fixable samples;
- search-labeled samples.

Known fixability may only be joined later as secondary explanatory analysis.

## 19. Primary Interpretation Cases

### Strong positive

W→C > C→W on multiple benchmarks and pooled Net > 0.

Interpretation: the learned selective router produces a real benchmark-scale correction signal.

### Narrow positive

Pooled Net > 0 but concentrated in one or two tasks.

Interpretation: the method works selectively but remains task-dependent.

### Neutral

W→C ≈ C→W, or almost no answers change.

Interpretation: the current candidate is too conservative or under-generalizes at scale.

Do not abandon the direction. Use trigger/action logs to distinguish Stage-1 admission from Stage-2 action bottlenecks.

### Negative

C→W > W→C.

Interpretation: the current policy is not conservative enough at full scale.

The next improvement should prioritize preservation/confidence rather than simply increasing intervention frequency.

## 20. Improvement Decision Tree After Full Evaluation

Do not execute any improvement during this phase.

### Positive but very few W→C

Ask whether coverage can be increased while keeping low C→W.

### Many triggers, almost no non-FULL

Focus next on Stage-2 abstention/generalization.

### Non-FULL happens, but W→C remains low

Focus next on action/timing quality.

### C→W dominates

Focus next on conservative action selection / preservation.

### One dataset uniquely weak

Decompose whether the bottleneck is Stage-1 admission, Stage-2 action choice, or task-specific fixability.

## 21. Do Not Improve Before Reading the Full Result

During this phase do not:

- retrain Stage-2;
- add MCTS supervision;
- switch to Experiment C;
- add valid-set loss;
- add on-policy states;
- change P90;
- change Stage-1;
- add layer embeddings;
- add Stage-1 latent features;
- run new search.

## 22. Execution Integrity

Before full launch run a small parity smoke.

Verify:

- dense baseline reproduces expected LMMS behavior;
- no-trigger routed path equals dense path;
- P90 threshold matches the frozen value;
- Stage-2 checkpoint hash is correct;
- four-action semantics match the training/search executor;
- all output rows have unique UIDs;
- no partial samples are silently counted.

Record GPU count, wall time, GPU-hours, failures, and retries.

## 23. Required Outputs

Use:

`analysis/dense_failure_stage2/full_benchmark_eval/`

Create:

- `protocol.md`
- `manifests/<benchmark_family>_full_manifest.jsonl`
- `dense/<benchmark_family>_results.jsonl`
- `routed/<benchmark_family>_results.jsonl`
- `paired/<benchmark_family>_paired.jsonl`
- `paired/all_paired.jsonl`
- `metrics/benchmark_summary.csv`
- `metrics/transition_counts.csv`
- `metrics/paired_bootstrap.csv`
- `metrics/stage1_admission.csv`
- `metrics/stage2_action_behavior.csv`
- `metrics/dataset_breakdown.csv`
- `metrics/compute_summary.csv`
- `figures/dense_vs_routed_accuracy.png`
- `figures/net_correction_by_benchmark.png`
- `figures/rescue_vs_regression.png`
- `figures/stage1_trigger_behavior.png`
- `figures/stage2_nonfull_behavior.png`
- `summaries/full_benchmark_eval_summary.md`
- `summaries/method_level_decision.md`
- `summaries/next_improvement_recommendation.md`
- `artifact_manifest.json`

## 24. `full_benchmark_eval_summary.md` Must Answer

1. Were all four established benchmark families evaluated under their existing project contracts?
2. What is Dense accuracy for ChartQA, TextVQA, MMMU-Pro, and POPE?
3. What is Routed accuracy for each family?
4. What are W→C and C→W counts for each family?
5. Is Net correction positive for each family?
6. Is pooled Net correction positive?
7. What are the paired uncertainty intervals?
8. How much C→C preservation is retained?
9. How often does Stage-1 trigger?
10. How often does Stage-2 actually use non-FULL?
11. Is the effect broad or concentrated in particular benchmark families?
12. Which benchmark family is the dominant source of rescue or regression?

## 25. `method_level_decision.md`

Choose one:

- A. Full-scale net-positive method signal
- B. Narrow/task-specific positive signal
- C. Neutral/under-active current method
- D. Regression-dominated current method

For the decision, explicitly state:

- what the evaluation supports;
- what it does not support;
- why the result matters;
- why one negative benchmark or a small gain should not be overinterpreted.

## 26. `next_improvement_recommendation.md`

After reading the full result, recommend **exactly one next improvement direction**.

The recommendation must answer:

- Why this improvement matters
- Which full-eval failure mode motivates it
- What experiment would discriminate whether it helps
- What a positive result would mean
- What a negative result would mean
- What should not be concluded from one negative result

Do not execute the improvement in this phase.

## 27. Stop Rule

STOP after:

- full ChartQA evaluation;
- full TextVQA evaluation;
- full MMMU-Pro evaluation;
- full POPE evaluation;
- paired dense-vs-routed analysis;
- method-level decision;
- one next-improvement recommendation.

Do not automatically retrain or modify the model after reading the result.

## 28. Core Principle

The router does not need to be perfect.

The method only needs to satisfy:

`W→C > C→W`

often enough to improve final benchmark accuracy.

Therefore this phase prioritizes **end-to-end correctness** over oracle action recall or trajectory imitation.

The full-scale question is:

> **Does the current selective visual-computation router improve more answers than it breaks across ChartQA, TextVQA, MMMU-Pro, and POPE?**
