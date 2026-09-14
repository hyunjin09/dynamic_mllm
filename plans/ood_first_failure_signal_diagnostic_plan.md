# OOD-First Dense Failure-Signal Analysis Plan

## 1. Goal

Before training the final shared Stage-1 failure predictor, first test whether the already observed dense-failure signal:

```text
hidden state at layer l
→ final dense answer correct / wrong
```

actually transfers across unseen benchmarks.

At the same time, test whether the signal is more than simple **static input difficulty**.

The immediate questions are:

1. Does layer-wise dense-failure prediction generalize to an unseen dataset?
2. Does intermediate hidden-state information outperform an input-only baseline?
3. Does failure predictability become more transferable as decoder computation proceeds?
4. Is the evidence strong enough to justify building the shared layer-conditioned Stage-1 predictor?

This plan stops before training the final shared predictor.

---

## 2. Data

Use the current-runtime dense all-on dataset:

```text
7,999 samples
3,999 dense-correct
4,000 dense-wrong
```

Datasets:

```text
GQA
ChartQA
TextVQA
```

Use only:

```text
current dense hidden-state features
current lmms-eval correct/wrong labels
```

Do not use:

```text
W→C labels
four-action routes
MCTS labels
sparse-router outputs
dataset ID as an input feature
```

---

## 3. Main OOD Setup: Leave-One-Dataset-Out

Run three OOD experiments.

### OOD-A

```text
Train / source validation:
GQA + ChartQA

OOD test:
TextVQA
```

### OOD-B

```text
Train / source validation:
GQA + TextVQA

OOD test:
ChartQA
```

### OOD-C

```text
Train / source validation:
ChartQA + TextVQA

OOD test:
GQA
```

For each run:

```text
target dataset must not be used for:
- training
- checkpoint selection
- threshold selection
- probability calibration
- layer selection
```

The held-out target benchmark is evaluated only after the source protocol is fixed.

---

## 4. Reuse the Simple Layer-Wise Linear Probe

Do not train the shared MLP + layer embedding yet.

Use the same simple diagnostic form as the previous in-domain analysis:

```text
Probe_l(h_l)
→ P(final dense answer is wrong)
```

Train one independent linear probe for each layer:

```text
l = 0 ... 27
```

This keeps the experiment focused on **whether the failure signal itself transfers**, rather than on architecture optimization.

---

## 5. OOD Metrics

For every layer and every leave-one-dataset-out run report:

```text
AUROC
AUPRC
balanced accuracy
```

Also report the conservative operating points.

On the source validation set, choose thresholds satisfying:

```text
correct preservation >= 99%
correct preservation >= 98%
correct preservation >= 95%
```

Transfer each threshold unchanged to the OOD target dataset.

On the OOD target report:

```text
actual correct preservation
wrong detection recall
failure precision
false-intervention rate
```

Do not recalibrate the threshold on the OOD target.

---

## 6. Main OOD Table

Produce:

| Source train datasets | OOD target | Best / representative AUROC | AUPRC | Target correct preservation | Target wrong recall |
|---|---|---:|---:|---:|---:|
| GQA + ChartQA | TextVQA | | | | |
| GQA + TextVQA | ChartQA | | | | |
| ChartQA + TextVQA | GQA | | | | |

Also keep the full layer-wise curves.

---

# 7. Input-Only vs Hidden-State Control

This control should be run in the same OOD setting.

The question is:

> **Is the model exposing a computation-dependent failure signal, or are we mostly learning which inputs are intrinsically difficult for Qwen?**

Use a simple linear probe only.

Compare:

```text
Input-only / pre-decoder representation
Layer 0 hidden representation
Layer 14 hidden representation
Layer 21 hidden representation
Layer 27 hidden representation
```

Use the same source train / source validation / OOD target splits.

Do not introduce a new large model.

---

## 8. Input-Only Feature

Use the cleanest available representation before decoder computation.

Prefer a fixed representation that contains the image/question input but no decoder-layer processing.

Possible examples, depending on what is already saved / directly available:

```text
pooled input text embedding
pooled visual encoder output
or their concatenation
```

Do not add benchmark ID.

Do not engineer benchmark-specific metadata features.

Document exactly what is used.

---

## 9. Key Comparison

For each OOD target, report:

| Representation | AUROC | AUPRC |
|---|---:|---:|
| Input-only | | |
| Layer 0 | | |
| Layer 14 | | |
| Layer 21 | | |
| Layer 27 | | |

Also report:

```text
Delta AUROC(layer l over input-only)
```

The most important question is:

> Does intermediate decoder computation add OOD-transferable failure information beyond static input difficulty?

---

# 10. Compare With In-Domain Results

Reuse the already completed in-domain layer-wise probe results as the reference.

For each dataset compute:

```text
OOD AUROC drop
=
in-domain AUROC
-
leave-one-dataset-out AUROC
```

Do the same for:

```text
wrong recall at source-calibrated 99% preservation
```

This gives a direct measure of how benchmark-specific the failure signal is.

---

# 11. Layer-Wise OOD Curves

For each held-out dataset produce:

```text
layer 0 ... 27 OOD AUROC
layer 0 ... 27 OOD AUPRC
```

Questions:

1. Is the signal already transferable at layer 0?
2. Does OOD predictability improve with depth?
3. Does the best OOD layer occur before explicit answer commitment around layers 25-27?
4. Is GQA substantially harder than ChartQA/TextVQA even under OOD?

---

# 12. Interpretation Cases

## Case A — Hidden-state OOD is strong and clearly above input-only

Example pattern:

```text
input-only OOD: ~0.60
layer 14/21 OOD: ~0.75-0.85
```

Interpretation:

> Intermediate decoder computation exposes transferable failure information beyond static input difficulty.

This strongly supports proceeding to the shared Stage-1 predictor.

---

## Case B — Input-only is already strong, but hidden states improve further

Example:

```text
input-only: 0.75
layer 21:   0.84
```

Interpretation:

> Static failure susceptibility exists, but decoder computation progressively refines a more informative failure signal.

This is also a strong and defensible result.

---

## Case C — Hidden-state OOD ≈ input-only OOD

Interpretation:

> Most of the apparent failure predictability may be input-difficulty prediction rather than computation-emergent failure awareness.

Proceed cautiously with the failure-awareness claim.

---

## Case D — OOD collapses near chance

Interpretation:

> The signal is strongly benchmark-dependent.

Do not claim a benchmark-general failure-awareness signal.

The later method would need to be framed as a predictor trained on the deployment task mixture.

---

# 13. Decision on the Shared Stage-1 Predictor

Only after this OOD diagnostic is complete, decide whether to train:

```text
shared predictor
+
learnable layer embedding
```

If OOD transfer is meaningful, proceed to compare:

```text
All-28 supervision
vs
Random-4 layer supervision
```

Then evaluate both:

```text
in-domain
+
leave-one-dataset-out OOD
```

Do not run this shared-predictor training in the current plan.

---

# 14. Required Figures

Create only:

```text
ood_layerwise_auroc.png
ood_layerwise_auprc.png
input_vs_hidden_ood.png
in_domain_vs_ood_drop.png
ood_wrong_recall_at_fixed_preservation.png
```

---

# 15. Required Outputs

Use:

```text
analysis/dense_failure_stage1/ood_signal_diagnostic/
```

Create:

```text
protocol.md

leave_textvqa_out/
    layerwise_metrics.csv
    selective_metrics.csv
    summary.md

leave_chartqa_out/
    layerwise_metrics.csv
    selective_metrics.csv
    summary.md

leave_gqa_out/
    layerwise_metrics.csv
    selective_metrics.csv
    summary.md

input_only_vs_hidden.csv
in_domain_vs_ood.csv

figures/
    ood_layerwise_auroc.png
    ood_layerwise_auprc.png
    input_vs_hidden_ood.png
    in_domain_vs_ood_drop.png
    ood_wrong_recall_at_fixed_preservation.png

decision_summary.md
```

---

# 16. Final Questions

The final report must answer:

### Q1
> Does dense-failure prediction transfer when the entire target benchmark is unseen during training?

### Q2
> How much does performance drop from in-domain to OOD for GQA, ChartQA, and TextVQA?

### Q3
> Does intermediate hidden-state information outperform a static input-only difficulty predictor under OOD transfer?

### Q4
> Does OOD failure predictability improve with decoder depth?

### Q5
> At conservative source-selected thresholds, is there still useful wrong detection on the unseen target dataset?

### Q6
> Is the evidence strong enough to justify the claim that failure-related information is benchmark-general rather than purely benchmark-specific?

### Q7
> Should we proceed to the shared Stage-1 predictor with learnable layer embeddings?

---

# 17. Stop Rule

STOP after:

```text
three leave-one-dataset-out linear-probe experiments
+
input-only vs hidden-state control
+
in-domain vs OOD comparison
```

Do not yet:

```text
train the shared Stage-1 predictor
add learnable layer embeddings
compare All-28 vs Random-4
resume W→C repair
run four-action intervention
train Stage 2
add external OOD benchmarks
```

The next step should be chosen only after we know whether the failure signal itself transfers beyond the training benchmarks.
