# Shared Stage-1 Failure Predictor + Global Risk Budget Plan

## 1. Goal

The independent 28-probe sequential gate is informative but not robust enough for final treatment admission.

Observed issues:

- validation-selected preservation did not transfer reliably to test;
- repeated layer-wise triggering accumulated false-positive risk;
- calibration differed substantially across GQA, ChartQA, and TextVQA;
- sequential gating gave only a small gain over a fixed strong layer at 99% preservation and underperformed at 98% / 95%.

The next step is therefore to replace:

```text
28 independent predictors
+
28 separately calibrated score spaces
```

with:

```text
one shared failure predictor
+
learnable layer identity
+
one trajectory-level global risk decision
```

The experiment should answer:

> Can a shared layer-conditioned failure predictor produce a more coherent cross-layer score space and a more stable conservative sequential gate?

Do not connect the gate to four-action treatment yet.

---

## 2. Fixed Data Contract

Use the existing current-runtime dense all-on dataset and frozen split:

```text
7,999 samples
3,999 dense-correct
4,000 dense-wrong

Train: 6,399
Validation: 800
Test: 800
```

Keep:

- zero UID overlap;
- zero image-group overlap;
- current lmms-eval correctness labels;
- existing compact hidden-state feature definition.

Input feature per layer:

```text
h_l =
concat(
    text_final,
    text_mean,
    visual_mean
)
```

Dimension:

```text
10,752
```

Do not add:

- W→C labels;
- four-action route labels;
- benchmark ID;
- sparse-router outcomes;
- answer text;
- GT answer.

---

## 3. Shared Predictor Architecture

Use one predictor for all 28 decoder layers.

For layer `l`:

```text
h_l
+
learnable layer embedding e_l
->
shared predictor
->
p_l = P(final dense answer is wrong)
```

### 3.1 Layer embedding

Learn:

```text
e_0, e_1, ..., e_27
```

Recommended initial size:

```text
d_layer = 32 or 64
```

The layer embedding provides depth-specific context while the predictor weights are shared.

### 3.2 Feature projection

Project the 10,752-dimensional hidden feature to a compact router dimension.

Recommended:

```text
10,752
-> 512
```

or:

```text
10,752
-> 256
```

Use a simple learned linear projection.

### 3.3 Predictor

V1:

```text
Projected hidden feature
+
layer embedding
->
2-layer MLP
->
1 scalar logit
```

Example:

```text
Linear(input_dim -> 512)
GELU
Dropout if needed
Linear(512 -> 1)
```

Keep the model small.

Do not use a Transformer, recurrent model, or attention across layers in V1.

---

## 4. Required Architecture Baselines

Train three small variants under the same protocol.

### A. State only

```text
h_l -> shared MLP
```

No layer embedding.

### B. Layer only

```text
e_l -> failure probability
```

No sample state.

This measures pure depth prior.

### C. State + layer embedding — main

```text
[h_l ; e_l] -> shared MLP
```

Desired result:

```text
State + Layer
>= State only
>> Layer only
```

Do not add more architecture variants before these are complete.

---

# 5. Training Scheme Comparison

Compare exactly two supervision schemes.

## Arm A — All-28

For every sample, supervise all 28 layers.

For sample `i`:

```text
L_i
=
mean over l=0...27 of BCE(p_i,l, y_i)
```

where:

```text
y_i = final dense wrong/correct
```

Every sample contributes equal total weight.

---

## Arm B — Random-4

For every sample and epoch:

```text
sample 4 layers uniformly without replacement from 0...27
```

Compute:

```text
L_i
=
mean BCE over the 4 sampled layers
```

Use a deterministic epoch-dependent seed.

Across epochs, every layer should receive approximately balanced exposure.

Do not bias sampling toward late layers in this experiment.

---

## 6. Training Hyperparameters

Use the same optimizer and training budget for All-28 and Random-4.

Recommended starting point:

```text
epochs: 10
optimizer: AdamW
learning rate: 5e-4
scheduler: cosine
precision: bf16 or fp32 for the small head
```

Because backbone features are already extracted:

```text
no Qwen forward
no backbone gradients
```

Training should be cheap.

Use validation for checkpoint selection only.

Do not touch test until all model and calibration choices are frozen.

---

# 7. Predictor Evaluation Before Gating

For every checkpoint / selected model, report:

```text
overall validation AUROC
overall validation AUPRC
layer-wise AUROC
layer-wise AUPRC
```

Also report dataset-wise:

```text
GQA
ChartQA
TextVQA
```

Compare against the existing 28 independent linear probes.

Primary question:

> Does sharing across layers preserve or improve the strong failure predictability while producing a single coherent predictor?

---

# 8. Sequential Gate

For each sample produce:

```text
p_0, p_1, ..., p_27
```

The simplest sequential policy is:

```text
for l = 0 ... 27:
    if p_l > tau:
        trigger at l
        break

if no trigger:
    continue FULL
```

Unlike the previous independent-probe gate, use **one global raw-score threshold** because the shared predictor defines one common score space.

---

# 9. Global Trajectory-Level Risk Calibration

Do not calibrate each layer independently.

Choose `tau` directly from the full validation trajectories.

For each candidate threshold:

1. apply the sequential policy to all validation samples;
2. compute sample-level correct preservation;
3. compute wrong detection;
4. compute precision;
5. record first-trigger distribution.

Define:

```text
Correct preservation(tau)
=
P(no layer triggers | dense-correct)
```

and:

```text
Wrong detection(tau)
=
P(at least one layer triggers | dense-wrong)
```

This directly accounts for repeated testing across 28 layers.

---

## 10. Global Risk Operating Points

Using validation only, select thresholds satisfying:

```text
>= 99% sample-level correct preservation
>= 98% sample-level correct preservation
>= 95% sample-level correct preservation
```

For each target freeze one global `tau`.

Do not use test to adjust the threshold.

This is the primary global risk-budget formulation in V1.

---

# 11. Optional Depth-Window Sensitivity

Evaluate the same trained predictor and same trajectory-level calibration under two gate windows.

### Main

```text
layers 0-27
```

### Sensitivity

```text
layers 16-27
```

Reason:

- all layers carry failure signal;
- layers 16-27 showed a stronger independent-probe plateau;
- very early triggering may leave the model before useful visual reasoning has developed.

Treat 16-27 only as a sensitivity / efficiency arm.

Do not claim early layers are uninformative.

---

# 12. Test Evaluation

Once predictor, training scheme, gate window, and validation thresholds are frozen:

Evaluate test once.

For each preservation target report:

```text
actual test correct preservation
wrong detection recall
failure precision
false-trigger count
trigger rate
median first-trigger layer
mean first-trigger layer
never-trigger fraction
```

Also report:

```text
GQA
ChartQA
TextVQA
```

separately.

Do not recalibrate on test.

---

# 13. Main Baselines

Compare against:

### Baseline 1 — Best fixed-layer gate

Use the existing strong fixed layers:

```text
L14
L21
L27
```

with validation-selected thresholds.

### Baseline 2 — Independent sequential gate

Use the completed Phase-50 result.

### Main — Shared sequential gate

```text
shared predictor
+
layer embedding
+
global trajectory threshold
```

Required comparison:

| Gate | Test Correct Preservation | Wrong Recall | Precision | Median Trigger |
|---|---:|---:|---:|---:|
| Fixed best layer | | | | |
| Independent sequential | | | | |
| Shared All-28 | | | | |
| Shared Random-4 | | | | |

---

# 14. Dataset Calibration Check

At each global threshold, report per-dataset:

```text
correct preservation
wrong recall
precision
trigger rate
median trigger layer
```

Measure:

```text
preservation spread
=
max(dataset preservation)
-
min(dataset preservation)
```

and:

```text
wrong-recall spread
```

The shared predictor should ideally reduce the large dataset calibration mismatch seen with independent probes.

Do not add dataset-specific thresholds in this phase.

---

# 15. Score-Space Analysis

To verify that the shared predictor creates a more coherent cross-layer score space, report:

```text
correct-score distribution by layer
wrong-score distribution by layer
score mean/std by layer
```

Also compute, for the same sample:

```text
correlation of p_l across neighboring layers
```

This analysis is descriptive.

The main question is whether one global threshold has a stable semantic meaning across depth.

---

# 16. Decision Criteria

## Case A — Shared gate improves robustness

Desired pattern:

```text
test preservation close to validation target
+
wrong recall comparable to or better than independent gate
+
smaller dataset preservation spread
```

Then use the shared predictor as the Stage-1 gate.

Next step:

```text
connect high-confidence triggered samples to treatment analysis
```

---

## Case B — Shared predictor works but sequential gate adds little

If:

```text
shared fixed-layer gate
≈
shared sequential gate
```

then prefer the simpler fixed-layer deployment.

---

## Case C — Shared predictor preserves AUROC but global threshold still fails

Then the main remaining problem is calibration / global risk control rather than representation.

The next experiment should study a stronger global risk-budget formulation.

Do not immediately redesign hidden-state features.

---

## Case D — Shared predictor substantially underperforms independent probes

Then failure readout is strongly layer-specific.

The independent probes remain the better substrate, and the next work should focus on better trajectory-level risk aggregation rather than parameter sharing.

---

# 17. OOD Follow-Up

Do not run OOD until the best shared in-domain model is selected.

If the shared predictor is viable in-domain, repeat the already established leave-one-dataset-out protocol:

```text
GQA + ChartQA -> TextVQA
GQA + TextVQA -> ChartQA
ChartQA + TextVQA -> GQA
```

Use:

```text
source-only training
source-only checkpoint selection
source-only global threshold selection
target evaluated once
```

This OOD step is a follow-up after the in-domain decision, not part of model tuning.

---

# 18. Required Figures

Create only:

```text
shared_vs_independent_layerwise_auroc.png
global_preservation_vs_wrong_detection.png
shared_trigger_layer_distribution.png
dataset_calibration_comparison.png
all28_vs_random4.png
```

---

# 19. Required Outputs

Use:

```text
analysis/dense_failure_stage1/shared_global_gate/
```

Create:

```text
protocol.md

configs/
    state_only.json
    layer_only.json
    state_layer_all28.json
    state_layer_random4.json

training/
    histories/
    checkpoints/

evaluation/
    layerwise_metrics.csv
    global_threshold_sweep.csv
    selected_operating_points.json
    test_results.csv
    dataset_breakdown.csv
    trigger_layer_distribution.csv
    calibration_analysis.csv

baselines/
    fixed_layer_comparison.csv
    independent_gate_comparison.csv

figures/
    shared_vs_independent_layerwise_auroc.png
    global_preservation_vs_wrong_detection.png
    shared_trigger_layer_distribution.png
    dataset_calibration_comparison.png
    all28_vs_random4.png

decision_summary.md
```

---

# 20. Final Questions

The final report must answer:

1. Can one shared predictor with learnable layer embeddings match the 28 independent probes?
2. Does All-28 or Random-4 supervision work better?
3. Can one global trajectory-level threshold achieve stable 99% / 98% / 95% correct preservation?
4. Does the shared gate reduce validation-to-test preservation drift?
5. Does it reduce GQA / ChartQA / TextVQA calibration spread?
6. Does sequential gating outperform a fixed strong layer?
7. Are layers 0-27 or 16-27 preferable for actual gating?
8. Is the shared predictor ready to become the Stage-1 admission gate for later four-action treatment?

---

# 21. Stop Rule

STOP after the shared-predictor + global-risk-gate in-domain comparison.

Do not yet:

```text
resume W→C repair
train READ_OFF / WRITE_OFF / BOTH_OFF
run four-action treatment
run MCTS
perform external evaluation
```

Only if the shared gate is robust enough should the next phase connect:

```text
high-confidence predicted dense failure
->
visual intervention
```
