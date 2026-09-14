# Layer-Wise Dense-Failure Predictability Analysis Plan

## 1. Goal

Before training the shared Stage-1 failure predictor, measure **when final dense failure becomes predictable from intermediate hidden states**.

We already observed that explicit answer identity emerges very late, mainly around layers 25-27.

Now test:

> **Does information about eventual dense failure become linearly predictable earlier than explicit answer commitment?**

This analysis should determine:
- whether early layers contain useful failure information;
- when failure predictability begins to rise;
- whether failure awareness precedes answer emergence;
- which layer range should later receive failure-prediction supervision.

Do not train the final shared failure predictor yet.

---

## 2. Data

Use the current-runtime dense all-on population:

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

Use the saved 28-layer dense hidden-state features and current lmms-eval correctness labels.

Target:

```text
y = 0  dense final answer correct
y = 1  dense final answer wrong
```

Do not use W→C labels, four-action routes, sparse-router outputs, or MCTS labels.

---

## 3. Train / Validation / Test Split

Create one deterministic image-group-disjoint split.

Target approximately:

```text
Train: ~6,400
Validation: ~800
Test: ~800
```

Requirements:
- zero UID overlap;
- zero image-group overlap;
- approximately preserve GQA / ChartQA / TextVQA proportions;
- approximately preserve current correct/wrong balance.

Do not force exact counts if image-group boundaries make that impossible.

Freeze the split before fitting any probe.

The test split must remain untouched during probe selection and analysis design.

---

## 4. Main Experiment: 28 Independent Linear Probes

Train one independent binary linear probe for each decoder layer:

```text
Probe_0(h_0)   -> P(final dense wrong)
Probe_1(h_1)   -> P(final dense wrong)
...
Probe_27(h_27) -> P(final dense wrong)
```

Use the same hidden-state feature definition for every layer.

The purpose is **diagnostic**, not maximum classification performance.

Use a simple linear classifier first.

Do not add:
- learnable layer embeddings;
- shared multi-layer predictor;
- MLP;
- sequence model;
- four-action supervision.

Those come later.

---

## 5. Probe Training

For each layer independently:

```text
input: hidden feature h_l
target: current dense correct/wrong
loss: binary cross entropy
```

Use identical optimization settings across layers.

Use validation only for normal checkpoint/optimization selection if needed.

Do not use test during training.

Record:

```text
train loss
validation loss
validation AUROC
validation AUPRC
```

---

## 6. Primary Layer-Wise Metrics

For each layer 0-27 report:

```text
AUROC
AUPRC
balanced accuracy
```

Main table:

| Layer | AUROC | AUPRC | Balanced Acc |
|---:|---:|---:|---:|
| 0 | | | |
| 1 | | | |
| ... | | | |
| 27 | | | |

Main figure:

```text
x-axis: decoder layer
y-axis: failure-prediction AUROC
```

---

## 7. Conservative Selective Metrics

Because the future policy will be:

```text
uncertain -> stay FULL
high-confidence failure -> intervene
```

AUROC alone is not sufficient.

For each layer, choose thresholds on validation that satisfy:

```text
dense-correct preservation >= 99%
dense-correct preservation >= 98%
dense-correct preservation >= 95%
```

At each operating point report:

```text
wrong detection recall
failure precision
false-deviation rate
```

Required table:

| Layer | Wrong Recall @99% Preserve | @98% | @95% |
|---:|---:|---:|---:|
| 0 | | | |
| ... | | | |
| 27 | | | |

This is the most relevant metric for future selective intervention.

---

## 8. Dataset-Wise Breakdown

For every layer, evaluate separately on:

```text
GQA
ChartQA
TextVQA
```

Report layer-wise AUROC curves for all three datasets.

Questions:
- Does failure predictability emerge at similar depths?
- Is one dataset systematically earlier/later?
- Does the overall curve hide dataset-specific behavior?

Do not tune probes separately by dataset unless explicitly reported as a diagnostic.

---

## 9. Compare Against Answer Emergence

Use the already completed corrected answer-position analysis as a reference.

Known observation:

```text
explicit correct-answer commitment emerges very late,
mainly around layers 25-27
```

Compare:

```text
failure-predictability curve
vs
answer-emergence curve
```

Central question:

> **Does failure awareness emerge before explicit answer commitment?**

Possible interpretation:

### Case A

```text
failure AUROC rises strongly before layer 25
```

Then:

> failure information is accessible before final answer identity is explicitly formed.

This supports earlier Stage-1 gating.

### Case B

```text
failure AUROC rises only around layers 25-27
```

Then:

> failure awareness and answer commitment emerge at similar depths.

Early failure supervision/gating would be less justified.

---

## 10. Define a Candidate Failure-Awareness Region

After observing validation curves, identify a descriptive candidate region where failure prediction becomes meaningfully useful.

Do not choose the region from one arbitrary AUROC threshold alone.

Consider jointly:

```text
AUROC
AUPRC
wrong recall at 99% correct preservation
stability across neighboring layers
dataset consistency
```

Example only:

```text
layers 0-10: near chance
layers 11-17: emerging signal
layers 18-27: strong signal
```

Do not assume these boundaries in advance.

---

## 11. Test Evaluation

After the analysis protocol and any candidate region are fixed from validation:

Evaluate all 28 probes once on the untouched test split.

Report:

```text
test AUROC
test AUPRC
wrong recall @99/98/95% preservation
dataset-wise results
```

Do not retune thresholds using test labels.

---

## 12. Required Figures

Create only:

1. `layerwise_failure_auroc.png`
2. `layerwise_failure_auprc.png`
3. `wrong_recall_at_fixed_preservation.png`
4. `dataset_layerwise_auroc.png`
5. `failure_vs_answer_emergence.png`

The fifth figure is the most scientifically important.

---

## 13. Required Outputs

Use:

```text
analysis/dense_failure_stage1/layerwise_failure_probe/
```

Create:

```text
split_manifest.jsonl
split_audit.md

probe_config.json
layerwise_probe_metrics.csv
layerwise_selective_metrics.csv
dataset_layerwise_metrics.csv
test_metrics.csv

checkpoints/
    layer_00.pt
    ...
    layer_27.pt

figures/
    layerwise_failure_auroc.png
    layerwise_failure_auprc.png
    wrong_recall_at_fixed_preservation.png
    dataset_layerwise_auroc.png
    failure_vs_answer_emergence.png

analysis_summary.md
```

---

## 14. Final Questions

The final report must answer:

### Q1
> At what depth does final dense failure first become meaningfully predictable from hidden states?

### Q2
> Is useful failure predictability present before explicit answer emergence at layers 25-27?

### Q3
> At 99% / 98% / 95% dense-correct preservation, how much wrong-sample detection is possible at each layer?

### Q4
> Is the failure-awareness depth consistent across GQA, ChartQA, and TextVQA?

### Q5
> Is there a defensible depth range for later Stage-1 failure-predictor supervision and gating?

### Q6
> Based on this analysis, should the next training experiment compare:
> - all-layer supervision,
> - informative-depth-only supervision,
> - random-k sampling from the informative depth range?

---

## 15. Stop Rule

STOP after the 28-layer linear-probe analysis and test evaluation.

Do not yet:
- train the shared failure predictor;
- add learnable layer embeddings;
- train MLP heads;
- select random-k training;
- resume W→C repair;
- run four-action intervention.

Use this analysis to decide the Stage-1 training design next.
