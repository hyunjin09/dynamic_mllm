# Stage-1 Gate Winner and Threshold Selection Plan

## 1. Goal

Choose one **Stage-1 failure gate** and one **operating threshold** before connecting any four-action treatment.

Current candidates:

1. `Independent Sequential`
2. `Shared All-28 Sequential`
3. `Shared Random-4 Sequential`
4. `Shared Fixed-L27` as a simple strong control

The goal is not to force 99% preservation.

Instead, explicitly measure the trade-off:

```text
more conservative threshold
-> fewer correct samples triggered
-> fewer wrong samples detected

more aggressive threshold
-> more wrong samples detected
-> more correct samples unnecessarily triggered
```

Then choose the Stage-1 gate/threshold that gives the best validation trade-off and confirm it once on test.

No four-action treatment is run in this phase.

---

## 2. Frozen Data

Use the existing frozen split:

```text
Train: 6,399
Validation: 800
Test: 800
```

Population:

```text
7,999 total
3,999 dense-correct
4,000 dense-wrong
```

Use current lmms-eval dense correctness only.

Do not change:
- hidden-state features;
- predictor checkpoints;
- train/val/test split;
- image grouping;
- dense labels.

---

## 3. Frozen Gate Candidates

Do not retrain models.

Evaluate the already completed predictors:

### A. Independent Sequential

```text
28 independent linear predictors
+
existing layer-specific calibration
+
first-trigger sequential policy
```

### B. Shared All-28 Sequential

```text
shared predictor trained with all 28 layers
+
one global score threshold
+
first-trigger policy
```

### C. Shared Random-4 Sequential

```text
shared predictor trained with random 4 layers/sample/epoch
+
one global score threshold
+
first-trigger policy
```

### D. Shared Fixed-L27

Use the already validation-selected strong fixed layer:

```text
layer 27 only
```

This checks whether sequential gating is actually necessary.

---

# 4. Threshold Sweep

For every gate, sweep the threshold over the full validation score range.

For each threshold compute at the **sample level**:

```text
correct preservation
wrong detection recall
failure precision
total trigger rate
number of correct samples triggered
number of wrong samples triggered
median trigger layer
mean trigger layer
```

For sequential gates:

```text
trigger layer
=
first layer crossing the threshold
```

For the fixed-L27 gate:

```text
trigger layer = 27
```

---

# 5. Main Pareto Curve

For every candidate plot:

```text
x-axis: correct preservation
y-axis: wrong detection recall
```

Do not select the winner from one pre-chosen preservation target.

Keep the full trade-off curve.

Also report precision along the curve.

---

# 6. Treatment-Agnostic Admission Utility

Because the dataset is approximately 50:50 correct/wrong, compute a simple Stage-1 utility:

```text
U_gate
=
# wrong samples triggered
-
# correct samples triggered
```

Normalized version:

```text
U_rate
=
(wrong_triggered - correct_triggered) / N
```

Interpretation:

- triggering a dense-wrong sample is potentially useful;
- triggering a dense-correct sample is unnecessary risk.

This is **not final routed accuracy**.

It is only a treatment-agnostic Stage-1 selection score.

Do not call it rescue or regression because treatment has not been executed yet.

---

# 7. Validation Winner Selection

Select one candidate gate + threshold using validation only.

Primary criterion:

```text
maximize U_rate
```

Subject to:

```text
failure precision >= 90%
```

The 90% precision requirement reflects the intended conservative policy:

```text
when the gate fires,
most admitted samples should actually be dense-wrong.
```

Tie-breakers, in order:

1. higher correct preservation;
2. higher wrong detection recall;
3. earlier median trigger;
4. simpler gate.

Do not use test to change this decision.

---

# 8. Preserve Reference Operating Points

Even though the winner is selected by validation utility, also save the thresholds closest to:

```text
99%
98%
95%
90%
```

validation correct preservation.

These are reference points only.

They will later be useful when treatment is connected.

For each gate save:

| Val preservation | Threshold | Wrong recall | Precision | Median trigger |
|---:|---:|---:|---:|---:|
| ~99% | | | | |
| ~98% | | | | |
| ~95% | | | | |
| ~90% | | | | |

---

# 9. Dataset-Wise Check

At the validation-selected winner threshold, report separately:

```text
GQA
ChartQA
TextVQA
```

For each:

```text
correct preservation
wrong detection recall
failure precision
trigger rate
median trigger layer
```

Do not introduce dataset-specific thresholds.

The goal is only to see whether the winner is catastrophically concentrated on one dataset.

---

# 10. Untouched Test Confirmation

After gate and threshold are frozen from validation, evaluate the selected winner once on test.

Report:

```text
test correct preservation
test wrong detection recall
test failure precision
test trigger rate
test U_rate
median trigger layer
dataset-wise breakdown
```

Also evaluate the other frozen candidates on test for comparison, but do not change the winner post hoc.

---

# 11. Pairwise Comparison

Compare the validation-selected winner against:

```text
Independent Sequential
Shared All-28
Shared Random-4
Shared Fixed-L27
```

Report:

| Gate | Test preservation | Wrong recall | Precision | U_rate | Median trigger |
|---|---:|---:|---:|---:|---:|
| Independent Sequential | | | | | |
| Shared All-28 Sequential | | | | | |
| Shared Random-4 Sequential | | | | | |
| Shared Fixed-L27 | | | | | |

Use simple UID-level bootstrap confidence intervals for:

```text
winner minus best alternative U_rate
winner minus best alternative wrong detection
```

Do not require statistical significance to proceed, but report uncertainty.

---

# 12. Winner Interpretation

## Case A — Shared sequential clearly wins

Use the shared sequential predictor as the Stage-1 substrate.

Keep independent probes as a baseline.

## Case B — Independent sequential clearly wins

Keep the independent gate.

There is no need to force a shared architecture.

## Case C — Fixed L27 matches or beats sequential gates

Prefer the fixed-layer gate because it is simpler.

Sequential gating is not justified if it provides no meaningful trade-off gain.

## Case D — Several gates are effectively tied

Prefer, in order:

```text
simpler gate
higher preservation
later calibration stability
```

Keep the others as baselines.

---

# 13. What Happens After the Winner Is Chosen

The next phase will connect Stage 1 to treatment.

For the selected Stage-1 winner, retain several thresholds:

```text
winner threshold
~99% preservation point
~98% point
~95% point
```

Then evaluate:

```text
triggered subset
->
four-action treatment feasibility
```

The eventual final threshold may change once actual:

```text
W→C rescue
C→W regression
final routed accuracy
```

are available.

Therefore the threshold selected here is the **Stage-1 admission operating point**, not necessarily the final paper threshold.

---

# 14. Required Figures

Create only:

```text
gate_pareto_curves.png
gate_precision_vs_recall.png
gate_utility_vs_preservation.png
winner_trigger_layer_distribution.png
dataset_winner_breakdown.png
```

---

# 15. Required Outputs

Use:

```text
analysis/dense_failure_stage1/gate_winner_selection/
```

Create:

```text
protocol.md

validation_threshold_sweeps/
    independent.csv
    shared_all28.csv
    shared_random4.csv
    shared_fixed_l27.csv

validation_summary.csv
reference_operating_points.csv
selected_winner.json

test_comparison.csv
dataset_breakdown.csv
bootstrap_comparison.json

figures/
    gate_pareto_curves.png
    gate_precision_vs_recall.png
    gate_utility_vs_preservation.png
    winner_trigger_layer_distribution.png
    dataset_winner_breakdown.png

decision_summary.md
```

---

# 16. Final Questions

The report must answer:

1. Which Stage-1 gate has the best validation preservation-vs-detection trade-off?
2. What threshold maximizes the conservative Stage-1 admission utility?
3. Does the winner retain high failure precision?
4. Does the winner generalize from validation to test without large preservation drift?
5. Does sequential gating actually beat a fixed strong layer?
6. Is performance dominated by one benchmark?
7. Which gate and threshold should be carried forward into the four-action treatment phase?

---

# 17. Stop Rule

STOP after selecting and test-confirming the Stage-1 winner.

Do not yet:

```text
resume W→C repair
train READ_OFF / WRITE_OFF / BOTH_OFF
run learned four-action treatment
run MCTS
perform external evaluation
```

The next separately authorized phase will ask:

```text
Given the selected Stage-1 gate,
does intervention on admitted samples actually improve final accuracy?
```
