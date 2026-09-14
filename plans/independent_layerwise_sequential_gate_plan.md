# Independent Layer-Wise Failure Predictors + Layer-Specific Thresholds: Sequential Gate Plan

## 1. Goal

Build and evaluate a conservative sequential failure gate using the **already-trained 28 independent layer-wise failure predictors**.

Do not train a new shared predictor in this phase.

The gate should:

```text
continue FULL by default

at layer l:
    compute failure score p_l

if p_l exceeds the layer-specific threshold tau_l:
    trigger intervention
    stop evaluating later gate layers for that sample
```

The main question is:

> **Can the existing independent predictors be turned into a useful sequential gate by calibrating a separate threshold for each layer?**

This phase evaluates only Stage 1:

```text
CONTINUE FULL
vs
TRIGGER INTERVENTION
```

Do not connect the gate to learned four-action treatment yet.

---

## 2. Inputs

Use the existing:

```text
28 independent linear failure predictors
```

trained on the dense all-on hidden states.

For each sample and layer:

```text
p_{i,l} = predictor_l(h_{i,l})
```

where:

```text
p_{i,l}
=
predicted probability / score that the final dense all-on answer will be wrong
```

Use the frozen train / validation / test split from the previous layer-wise failure-predictability experiment.

Do not retrain the predictors.

---

## 3. Why Layer-Specific Thresholds

The 28 predictors were trained independently.

Therefore:

```text
p_5 = 0.8
```

and:

```text
p_21 = 0.8
```

do not necessarily have the same calibration meaning.

Use:

```text
tau_0, tau_1, ..., tau_27
```

rather than forcing one raw-score threshold across all layers.

The gate is:

```text
for l = 0 ... 27:
    if p_l > tau_l:
        trigger at layer l
        break

if no layer triggers:
    keep FULL through the entire model
```

---

## 4. Important Metric: Sample-Level Preservation

Do not calibrate each layer independently to 99% correct preservation and assume the full sequential policy also preserves 99%.

The gate is sequential:

```text
a correct sample is admitted if ANY layer triggers
```

Therefore the relevant preservation metric is:

```text
sample-level correct preservation
=
fraction of dense-correct samples that never trigger at any layer
```

All threshold selection must be evaluated at the full trajectory level.

---

## 5. Validation Score Collection

For every validation sample, save:

```text
UID
dataset
dense correctness
p_0
p_1
...
p_27
```

Create:

```text
validation_scores.jsonl
```

Also save the same score matrix for test, but do not use test for threshold selection.

---

## 6. Layer-Specific Threshold Calibration

### Preferred V1: shared correct-tail quantile

Use the validation dense-correct samples to define a separate threshold for every layer.

For one global tail parameter `alpha`:

```text
tau_l(alpha)
=
(1 - alpha) quantile of p_l
among validation dense-correct samples
```

Example:

```text
alpha = 0.001

tau_0  = 99.9th percentile of correct scores at layer 0
tau_1  = 99.9th percentile of correct scores at layer 1
...
tau_27 = 99.9th percentile of correct scores at layer 27
```

Thus:

- numeric threshold is layer-specific;
- calibration rule is shared;
- only one scalar `alpha` is swept.

Do not freely optimize 28 unrelated thresholds on validation in V1.

---

## 7. Sweep the Global Tail Parameter

Sweep a fixed reasonable grid of `alpha`.

For each `alpha`:

1. compute `tau_0 ... tau_27`;
2. run the full sequential first-trigger policy on validation;
3. report:
   - sample-level correct preservation;
   - wrong detection rate;
   - failure precision;
   - total trigger rate;
   - median first-trigger layer;
   - mean first-trigger layer;
   - no-trigger fraction.

The main curve is:

```text
x-axis:
sample-level correct preservation

y-axis:
wrong detection rate
```

---

## 8. Select Conservative Operating Points

Using validation only, choose operating points that satisfy as closely as possible:

```text
>= 99% sample-level correct preservation
>= 98% sample-level correct preservation
>= 95% sample-level correct preservation
```

For each selected point freeze:

```text
alpha
tau_0 ... tau_27
```

Do not retune on test.

If no `alpha` satisfies a requested preservation level, report that directly.

---

## 9. Validation Results

For each operating point report:

| Target preservation | Actual preservation | Wrong detection | Failure precision | Trigger rate | Median trigger layer |
|---:|---:|---:|---:|---:|---:|
| >=99% | | | | | |
| >=98% | | | | | |
| >=95% | | | | | |

Also save:

```text
per-layer tau_l
number of first triggers at each layer
```

---

## 10. Dataset-Wise Breakdown

At each selected operating point report separately for:

```text
GQA
ChartQA
TextVQA
```

Metrics:

```text
correct preservation
wrong detection recall
failure precision
trigger rate
median trigger layer
```

Use the same global `alpha`-derived threshold vector for all datasets.

Do not introduce dataset-specific thresholds in V1.

---

## 11. Test Evaluation

After validation freezes:

```text
alpha
tau_0 ... tau_27
```

apply the sequential gate once to the untouched test split.

For each validation-selected operating point report:

```text
actual correct preservation
wrong detection recall
failure precision
trigger rate
median trigger layer
mean trigger layer
no-trigger fraction
```

Do not modify thresholds after seeing test results.

---

## 12. Trigger-Layer Analysis

For dense-wrong samples that trigger, report the first-trigger-layer distribution:

```text
layer 0 ... 27
```

Also summarize:

```text
early: 0-8
middle: 9-18
late: 19-27
never triggered
```

Do the same for dense-correct false triggers.

Main question:

> **Does the gate detect failures early enough to leave useful remaining depth for later intervention?**

---

## 13. Score-Trajectory Examples

For a small deterministic set of examples visualize:

```text
p_0 ... p_27
```

together with:

```text
tau_0 ... tau_27
```

Include representative:

```text
correct + never trigger
correct + false trigger
wrong + early trigger
wrong + late trigger
wrong + never trigger
```

This is for interpretation only.

---

## 14. Single-Layer Baseline

Compare the sequential gate with simple fixed-layer gates at:

```text
layer 14
layer 21
layer 27
```

For each fixed layer, select its threshold on validation at the same:

```text
99%
98%
95%
```

sample-level correct-preservation targets.

Report:

| Gate | Correct preservation | Wrong detection |
|---|---:|---:|
| L14 only | | |
| L21 only | | |
| L27 only | | |
| Sequential layer-specific | | |

Question:

> Does observing failure scores over multiple layers provide more coverage than simply waiting for one strong layer?

---

## 15. No Treatment Yet

When the gate triggers, do **not** execute four-action intervention in this phase.

A trigger means only:

```text
"admit this sample for possible intervention"
```

The goal is to isolate:

```text
failure detection
+
correct-sample preservation
```

before treatment quality is introduced.

---

## 16. Decision Criteria

### Case A — Sequential gate is useful

If:

```text
correct preservation >= 99%
wrong detection is meaningfully above zero
first triggers leave useful remaining depth
```

then Stage 1 is promising.

Next phase can examine:

```text
shared predictor
+
global risk-budget formulation
```

before connecting treatment.

### Case B — Sequential ≈ fixed single layer

If the sequential policy provides little improvement over L14/L21/L27:

> the sequential machinery may not be necessary.

A fixed-layer failure gate may be preferable.

### Case C — Repeated testing destroys preservation

If sample-level preservation falls sharply despite conservative per-layer thresholds:

> accumulated false-trigger risk is the main problem.

This motivates a later global risk-budget formulation.

### Case D — Strong dataset calibration mismatch

If one common calibration rule behaves very differently across GQA, ChartQA, and TextVQA:

> document the mismatch and do not silently add benchmark-specific thresholds.

This becomes a motivation for the next calibration/shared-predictor experiment.

---

## 17. Required Figures

Create:

```text
preservation_vs_wrong_detection.png
trigger_layer_distribution.png
dataset_preservation_recall.png
sequential_vs_single_layer.png
score_trajectory_examples.png
```

---

## 18. Required Outputs

Use:

```text
analysis/dense_failure_stage1/independent_sequential_gate/
```

Create:

```text
protocol.md

validation_scores.jsonl
test_scores.jsonl

threshold_sweep.csv
selected_operating_points.json
layer_specific_thresholds.csv

validation_results.csv
test_results.csv
dataset_breakdown.csv
trigger_layer_distribution.csv
single_layer_comparison.csv

figures/
    preservation_vs_wrong_detection.png
    trigger_layer_distribution.png
    dataset_preservation_recall.png
    sequential_vs_single_layer.png
    score_trajectory_examples.png

decision_summary.md
```

---

## 19. Final Questions

The report must answer:

1. Can the existing 28 independent predictors form a conservative sequential failure gate?
2. At 99% / 98% / 95% sample-level correct preservation, how many dense-wrong samples are detected?
3. At what layers do detected failures first trigger?
4. Does sequential gating outperform using one fixed strong layer?
5. Does one common quantile-calibration rule behave consistently across GQA, ChartQA, and TextVQA?
6. Is the independent-probe gate already sufficient, or is a shared predictor / global risk-budget formulation justified?

---

## 20. Stop Rule

STOP after the independent-predictor sequential-gate analysis.

Do not yet:

```text
train a shared predictor
add learnable layer embeddings
optimize a global risk-budget model
resume W→C repair
run four-action treatment
train Stage 2
```

Use this result to decide whether the next step should be:

```text
shared predictor + global risk budget
```

or whether the existing independent layer-wise gate is already sufficient.
