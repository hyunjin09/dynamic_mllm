# Stage-1 ALL-Source Mixed Training + Dataset-OOD Evaluation Plan

## 1. Objective

Train one shared Stage-1 dense-failure predictor on:

```text
ALL = Historical + Canonical
```

using the exact Shared Random-4 architecture, the same Stage-1 feature definition, and the same frozen old normalization.

Then evaluate two robustness axes:

1. **Source robustness**
   - Can one head maintain useful failure ranking on both Historical and Canonical populations?

2. **Dataset OOD robustness**
   - When one benchmark is completely excluded from head training, does the failure signal transfer to that unseen benchmark?

Do not modify Stage-2, regenerate corrective labels, or calibrate thresholds in this phase.

---

## 2. Motivation

Previous specialist results:

```text
Historical specialist:
    Historical held-out AUROC ≈ 0.8885
    Canonical AUROC          ≈ 0.4056

Canonical specialist:
    Canonical OOF AUROC      = 0.8178
    Historical cross-eval    = 0.5511
```

The same architecture can therefore learn either source regime individually, but fitted boundaries transfer poorly across regimes.

Main question:

> Can source-diverse training recover a shared failure boundary?

Secondary question:

> Does source diversity also improve leave-one-dataset-out transfer?

---

## 3. Terminology

### Historical

Original quota-selected population from the historical previous-Qwen all-on source regime.

### Canonical

New outcome-blind canonical-source population selected before observing dense correctness.

### ALL

```text
ALL = Historical ∪ Canonical
```

`ALL` means all source regimes currently available in this project, not all possible real-world sources.

---

## 4. Frozen Components

Keep fixed:

```text
Qwen2.5-VL backbone
Stage-1 feature definition
feature block concatenation
Shared Random-4 architecture
shared-across-layer parameterization
Random-4 layer sampling mechanism
optimizer family
scheduler family
regularization recipe
old normalization artifact
```

Train only new Stage-1 head weights.

Do not add:

```text
source ID
dataset ID
visual-token count
new metadata features
new layer embeddings
larger MLP
new normalization
```

The main changed variable is the training population.

---

## 5. ALL-Source Training Sampler

Do not simply concatenate all rows and sample naturally.

Use four top-level cells:

```text
Historical-C
Historical-W
Canonical-C
Canonical-W
```

with approximately equal expected draw probability:

```text
25%
25%
25%
25%
```

This suppresses trivial source/outcome priors.

Do not dataset-balance the main run because Canonical TextVQA has only 19 wrong samples and would be severely oversampled.

Within each source×correctness cell, preserve the natural dataset mixture.

---

## 6. Canonical Evaluation Protocol

Use the same 5-fold group-disjoint OOF protocol as the canonical-refit diagnostic.

For fold `k`:

```text
TRAIN:
    Historical train
    +
    Canonical folds except k

EVAL:
    Canonical fold k
```

Requirements:

```text
zero UID overlap
zero image-content-group overlap
zero SHA-256 image overlap where applicable
```

Concatenate all held-out predictions to obtain canonical OOF scores.

---

## 7. Historical Evaluation Protocol

Historical validation/test remain untouched.

For each of the five ALL fold models, evaluate on the frozen Historical held-out population.

Preferred primary historical score:

```text
average the five fold-model scores per historical sample
then compute AUROC/AUPRC
```

Also report per-fold Historical AUROC and mean ± std.

Do not select the best fold using Historical held-out performance.

---

## 8. Main Source-Robustness Comparison

Produce:

| Head | Historical held-out AUROC | Canonical held-out AUROC |
|---|---:|---:|
| Historical specialist | 0.8885 | 0.4056 |
| Canonical specialist | 0.5511 | 0.8178 |
| ALL-source head | ? | ? |

Also report AUPRC.

The ALL head does not need to beat both specialists on their own source. The target is robustness.

---

## 9. Robustness Metrics

Report:

```text
Average-source AUROC
= (AUROC_Historical + AUROC_Canonical) / 2
```

```text
Worst-source AUROC
= min(AUROC_Historical, AUROC_Canonical)
```

Worst-source AUROC is a primary robustness statistic.

---

## 10. Dataset × Source Breakdown

For the ALL head report:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

For each:

```text
N_C
N_W
AUROC
AUPRC
bootstrap CI
```

Canonical TextVQA must be qualified because it has only 19 wrong samples.

---

## 11. Layerwise Source Robustness

For layers 0...27 compute:

```text
Historical AUROC/AUPRC
Canonical AUROC/AUPRC
worst-source AUROC
```

Questions:

1. Does source mixing repair early-layer canonical inversion?
2. Which layers are consistently robust across source regimes?
3. Is robustness broad or concentrated?

Diagnostic only; do not post-hoc change the deployed gate.

---

## 12. Source-Conditioned Score Shift

Within Dense-C and Dense-W separately, compare ALL-head scores between Historical and Canonical.

Check whether the extreme canonical-Dense-C high-risk behavior disappears.

Do not require perfect source invariance.

---

## 13. Dataset-OOD Experiments

Run three leave-one-dataset-out experiments.

### OOD-ChartQA

Train:

```text
Historical GQA
Canonical GQA
Historical TextVQA
Canonical TextVQA
```

Evaluate:

```text
Historical ChartQA held-out
Canonical ChartQA
```

Historical Phase-49 reference:

```text
GQA + TextVQA → ChartQA ≈ 0.4502
```

### OOD-TextVQA

Train:

```text
Historical GQA
Canonical GQA
Historical ChartQA
Canonical ChartQA
```

Evaluate:

```text
Historical TextVQA held-out
Canonical TextVQA
```

Historical reference:

```text
GQA + ChartQA → TextVQA ≈ 0.7236
```

### OOD-GQA

Train:

```text
Historical ChartQA
Canonical ChartQA
Historical TextVQA
Canonical TextVQA
```

Evaluate:

```text
Historical GQA held-out
Canonical GQA
```

Historical reference:

```text
ChartQA + TextVQA → GQA ≈ 0.6160
```

---

## 14. LODO Sampling

Within each leave-one-dataset-out run, again use source×correctness balanced sampling:

```text
Historical-C
Historical-W
Canonical-C
Canonical-W
```

Only samples from the two non-target datasets may enter these cells.

Do not use target-dataset samples for training, early stopping, threshold calibration, or layer selection.

---

## 15. OOD Metrics

For each target report:

```text
Historical-target AUROC
Canonical-target AUROC
Historical-target AUPRC
Canonical-target AUPRC
average target-source AUROC
worst target-source AUROC
```

Primary matrix:

| Train datasets | OOD target | Historical target AUROC | Canonical target AUROC |
|---|---|---:|---:|
| GQA + TextVQA | ChartQA | ? | ? |
| GQA + ChartQA | TextVQA | ? | ? |
| ChartQA + TextVQA | GQA | ? | ? |

---

## 16. Phase-49 Comparison

Where protocols are reasonably comparable:

| Target | Historical-only Phase-49 AUROC | ALL-source LODO Historical target | ALL-source LODO Canonical target |
|---|---:|---:|---:|
| ChartQA | 0.4502 | ? | ? |
| GQA | 0.6160 | ? | ? |
| TextVQA | 0.7236 | ? | ? |

Interpret cautiously because source composition and training counts differ.

---

## 17. Layerwise OOD Analysis

For each LODO target compute layerwise AUROC over L0...L27 on both target source regimes.

Report average target-source AUROC by layer and worst target-source AUROC by layer.

Questions:

1. Are some layers consistently benchmark-transferable?
2. Does source diversity improve OOD mostly at early or late layers?
3. Is OOD failure concentrated in one source regime or both?

---

## 18. Thresholds Are Out of Scope

Do not calibrate a deployment threshold yet.

Required sequence:

```text
ranking robustness
→ select final Stage-1 candidate
→ threshold calibration later
```

Do not generate a new trigger map in this phase.

---

## 19. Interpretation Cases

### Case A — ALL is source-robust and OOD improves

Interpretation:

> Source-diverse training yields a shared failure boundary robust across observed source regimes and more transferable across datasets.

Next: robust threshold calibration.

### Case B — ALL is source-robust but OOD remains mixed/weak

Interpretation:

> Stage-1 is robust across Historical/Canonical for the in-scope benchmark mixture, but is not a universal benchmark-agnostic failure detector.

Still usable for the current method.

### Case C — ALL compromises both sources

Interpretation:

> A single shared boundary may face genuine regime conflict.

Before changing architecture, inspect sampling, dataset support, and score geometry.

### Case D — ALL favors one source and collapses on the other

Interpretation:

> Source balancing alone is insufficient.

Stop before threshold calibration.

---

## 20. Stage-2 Is Frozen

Do not:

```text
regenerate trigger maps
regenerate Stage-2 labels
rerun single search
rerun MCTS
train Stage-2
```

First select a robust Stage-1 candidate.

---

## 21. Required Outputs

Use:

```text
analysis/dense_failure_stage1/all_source_robustness/
```

Create:

```text
protocol.md

main_all/
    folds/
        fold_manifest.jsonl
        fold_summary.csv
    training/
        fold_0/
        fold_1/
        fold_2/
        fold_3/
        fold_4/
        checkpoint_manifest.json
    predictions/
        canonical_oof_scores.jsonl
        historical_ensemble_scores.jsonl
    metrics/
        source_summary.csv
        robustness_metrics.csv
        dataset_source_breakdown.csv
        layerwise_auroc.csv
        layerwise_auprc.csv
        source_conditioned_score_stats.csv
    figures/
        historical_vs_canonical_roc.png
        source_robustness_comparison.png
        layerwise_source_auroc.png
        worst_source_auroc_by_layer.png
        source_conditioned_score_distributions.png

lodo/
    chartqa/
        protocol.md
        training/
        predictions/
        metrics.csv
        layerwise_auroc.csv
    textvqa/
        protocol.md
        training/
        predictions/
        metrics.csv
        layerwise_auroc.csv
    gqa/
        protocol.md
        training/
        predictions/
        metrics.csv
        layerwise_auroc.csv
    lodo_summary.csv
    phase49_comparison.csv
    figures/
        lodo_auroc_matrix.png
        phase49_vs_all_source_lodo.png
        lodo_layerwise_auroc.png
        lodo_worst_source_auroc.png

summaries/
    all_source_training_summary.md
    dataset_ood_summary.md
    stage1_robustness_decision.md

artifact_manifest.json
```

---

## 22. `all_source_training_summary.md` Must Answer

1. What is ALL-head Historical held-out AUROC?
2. What is ALL-head Canonical OOF AUROC?
3. What are average-source and worst-source AUROC?
4. How do these compare with Historical and Canonical specialists?
5. Which dataset×source cells remain weak?
6. Does canonical ChartQA remain non-inverted?
7. Does canonical Dense-C risk inflation disappear?
8. Which layers are most source-robust?
9. Does source mixing produce a useful shared boundary?
10. Is one shared Stage-1 viable across both observed source regimes?

---

## 23. `dataset_ood_summary.md` Must Answer

1. What is GQA+TextVQA → ChartQA on Historical ChartQA?
2. What is the same OOD model on Canonical ChartQA?
3. What is GQA+ChartQA → TextVQA on both source regimes?
4. What is ChartQA+TextVQA → GQA on both source regimes?
5. Does ALL-source LODO improve Phase-49 Historical-only OOD results?
6. Does source diversity improve worst-source OOD performance?
7. Which OOD target remains least transferable?
8. How uncertain is Canonical TextVQA?
9. Is Stage-1 source-robust but benchmark-specific, or broadly benchmark-transferable?

---

## 24. `stage1_robustness_decision.md`

Choose one:

### Decision A — Robust shared Stage-1 candidate found

Historical and Canonical rankings are both healthy and no source catastrophically collapses.

Next phase:

```text
robust threshold calibration
```

### Decision B — Source-robust but benchmark-OOD limited

ALL works across the two source regimes but leave-one-dataset-out remains weak/mixed.

Conclusion:

> Stage-1 is valid for the current benchmark mixture, but should not be claimed as a universal failure detector.

Next:

```text
threshold calibration on the in-scope ALL mixture
```

### Decision C — Shared-head conflict remains

ALL training cannot maintain useful ranking on both Historical and Canonical.

Stop before threshold calibration and diagnose the conflict.

---

## 25. Stop Rule

STOP after:

```text
ALL-source mixed training/evaluation
+
three leave-one-dataset-out evaluations
+
robustness decision
```

Do not automatically run:

```text
threshold calibration
new trigger map
Stage-2 label compatibility audit
corrective search
Stage-2 retraining
test deployment
```

---

## 26. Final Experimental Structure

```text
                        Robust Stage-1?
                              │
              ┌───────────────┴────────────────┐
              │                                │
       SOURCE ROBUSTNESS                 DATASET OOD
              │                                │
      ALL = Historical+Canonical        Leave one benchmark out
              │                                │
      Shared Random-4 head          ┌──────────┼──────────┐
              │                     │          │          │
       ┌──────┴──────┐          →ChartQA   →TextVQA    →GQA
       │             │
 Historical       Canonical
 held-out           OOF
```

The new question is:

> **Can one shared Stage-1 failure boundary survive both source shift and dataset shift without collapsing back onto regime-specific shortcuts?**
