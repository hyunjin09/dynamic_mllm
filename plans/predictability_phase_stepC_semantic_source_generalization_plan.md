# Predictability & Generalization Phase — Step C
## Semantic-Similarity, Question-Family, Source-Regime, and Dataset Generalization

## 0. Executive purpose

Step B produced the in-domain evidence pattern:

```text
Evidence category:
D — Stage-1 strong / Stage-2 weak
```

Primary Step-B results:

```text
Stage-1 current head:
    AUROC = 0.7869
    AUPRC = 0.6860
    W recall @95% C-preserve = 0.3029
    W recall @98% C-preserve = 0.1609
    peak layer AUROC = 0.826 at layer 20

Stage-2 READ utility:
    target = q_F - q_WO
    joint-router Spearman = 0.0416
    harmful AUROC = 0.5193

Stage-2 WRITE utility:
    target = q_F - q_RO
    joint-router Spearman = 0.0347
    harmful AUROC = 0.5115
```

Step C does **not** redesign these targets.

It asks:

> **Is the Stage-1 signal genuinely transferable beyond semantically similar questions and the same source mixture?**

and, for Stage-2:

> **Is the weak in-domain utility signal concentrated in semantically/source-similar regimes, or is it uniformly weak?**

This step uses the full internal corpus and frozen Step-A targets.

It does **not** run the external four-benchmark evaluation. External transfer belongs to Step D.

---

## 1. Scientific questions

Step C must answer five questions.

### Q1 — Similar-question dependence

Does prediction performance depend strongly on how similar a held-out question is to questions present in the training fold?

### Q2 — Unseen question-family generalization

If entire semantic question families are held out during training, does Stage-1 failure predictability remain strong? Does any Stage-2 utility signal survive?

### Q3 — Historical ↔ Canonical source-regime transfer

Does a predictor learned under one source/population regime transfer to the other?

### Q4 — Leave-one-dataset-out generalization

Does the learned signal transfer across GQA, ChartQA, and TextVQA when one dataset family is absent from training?

### Q5 — What likely explains Stage-1 predictability?

Compare:

```text
question-neighbor baseline
nuisance baseline
linear hidden-state predictor
current hidden-state predictor
```

under increasingly difficult semantic/source shifts.

---

## 2. Frozen targets

Do not change the Step-B targets.

Stage-1:

```text
eventual all-FULL final failure
```

Stage-2 READ:

```text
U_READ_PRIMARY = q_F - q_WO
```

Stage-2 WRITE:

```text
U_WRITE_PRIMARY = q_F - q_RO
```

Do not switch to:

```text
factorial targets
best-action labels
correctness-flip labels
MCTS-valid labels
```

because of Step-B performance.

---

## 3. Frozen reference models

Step C is a generalization study, not another capacity search.

### Stage-1 primary

```text
M3 current Stage-1 head
```

Controls:

```text
M0 nuisance-only
M1 linear hidden-state probe
question-kNN baseline
```

### Stage-2 primary

```text
M3 router-style joint [z_R ; z_W]
```

for READ and WRITE separately.

Controls:

```text
M0 nuisance-only
M1 simple multimodal linear
question-kNN utility baseline
```

Do not rerun the entire Step-B capacity ladder.

---

## 4. Frozen optimization

Reuse Step-B:

```text
feature preprocessing
optimizer
learning rate
batching
epoch/early-stop rule
model width
loss
seed policy
```

without OOD-specific retuning.

For every train/test regime:

```text
fit only on allowed training data
early-stop only on a group-disjoint calibration subset from training data
evaluate held-out regime once
```

Do not tune on held-out semantic clusters, source regimes, or datasets.

---

## 5. Full internal population

Use all Step-A eligible internal data:

```text
10,399 UIDs
9,982 image groups

Stage-1:
291,172 states

Stage-2 dense:
1,413 P90-triggered UIDs
1,385 image groups
15,185 dense post-trigger states
```

No small scientific subset.

---

## 6. Frozen semantic question representation

Before any Step-C result is inspected, freeze exactly one label-blind question encoder contract.

Preferred order:

```text
1. an already-installed/cached sentence embedding model with a stable text-only contract;
2. if unavailable, a frozen text-only Qwen2.5 representation using a fixed layer and pooling rule.
```

Whichever is used, record:

```text
model/hash
tokenizer
text normalization
pooling
L2 normalization
```

Do not compare multiple encoders and choose the one with the strongest result.

Question embeddings may use only question text.

Never use:

```text
image
answer
Dense C/W label
utility label
dataset label
```

---

## 7. Semantic embedding unit

Compute one question embedding per UID.

For image/content groups with multiple questions, also compute:

```text
group embedding
=
mean of L2-normalized UID embeddings
then renormalize
```

Use group embeddings for semantic clustering so image-group disjointness is preserved.

---

# Part I — Similar-question dependence

## 8. Reuse Step-B OOF folds

Reuse the exact Step-B 5-fold image-group-disjoint registry.

For held-out UID `i` in fold `f`, compute:

```text
sim_i
=
max cosine(question_i, question_j)
for j in outer-training groups only
```

Never search nearest questions in the held-out fold.

---

## 9. Similarity bins

Create five equal-frequency bins over held-out UID nearest-training similarity:

```text
Q1 = least similar 20%
Q2
Q3
Q4
Q5 = most similar 20%
```

Use bin edges from concatenated label-blind OOF similarity values.

Each state inherits its UID's bin.

---

## 10. Stage-1 similarity dependence

For each similarity bin report:

```text
AUROC
AUPRC
```

for:

```text
M0 nuisance
M1 linear
M3 current head
question-kNN
```

Also report M3 by layer within bins where support permits.

Primary comparison:

```text
Q1 vs Q5
```

Report:

```text
ΔAUROC = AUROC_Q5 - AUROC_Q1
```

with group-bootstrap uncertainty.

---

## 11. Stage-2 similarity dependence

For READ and WRITE separately, by similarity bin report:

```text
Spearman
harmful AUROC
Precision@top10%
```

for:

```text
M0 nuisance
M1 linear
M3 joint router
question-kNN
```

Key question:

> Is the weak pooled Stage-2 signal concentrated only in the most semantically familiar questions?

---

## 12. Continuous similarity trend

Also measure whether prediction error changes monotonically with nearest-train similarity.

Stage-1:

```text
prediction residual / confidence
vs similarity
```

Stage-2:

```text
absolute utility prediction error
vs similarity
```

Use a descriptive rank correlation.

---

## 13. Question-kNN baseline — Stage-1

Use:

```text
k = 5
```

nearest training UIDs by cosine similarity.

Predict:

```text
mean eventual-failure label of the 5 nearest training questions
```

No hidden-state features.

---

## 14. Question-kNN baseline — Stage-2

For a held-out Stage-2 state at layer `l`, restrict candidate training states to the same absolute layer.

Use:

```text
k = 5 nearest training questions with an eligible state at layer l
```

Predict:

```text
mean READ utility
mean WRITE utility
```

If exact-layer support is insufficient, report unsupported rather than changing layers post hoc.

---

# Part II — Question-family OOD

## 15. Label-blind semantic clustering

Construct semantic question-family clusters at the image/content-group level.

Primary clustering:

```text
K = 100
```

Use deterministic spherical/cosine k-means with frozen seed/settings.

No labels, source IDs, outcomes, or utilities enter clustering.

---

## 16. Cluster audit

Before model training report:

```text
groups per cluster
UIDs per cluster
dataset composition
P90-triggered support
representative questions nearest each centroid
```

Do not manually edit clusters after reading targets.

---

## 17. Five-fold cluster-disjoint OOD

Assign whole semantic clusters to five OOD folds.

Each cluster appears in exactly one test fold.

Balance using label-blind metadata only:

```text
group count
dataset composition
P90-triggered support
```

Do not use failure/utility labels.

---

## 18. Stage-1 cluster OOD

For each fold:

```text
train on 80% semantic clusters
test on unseen 20%
```

Train/evaluate:

```text
M0 nuisance
M1 linear
M3 current head
question-kNN
```

Report concatenated OOD:

```text
AUROC
AUPRC
W recall @95% train-calibrated C-preserve
W recall @98% train-calibrated C-preserve
```

---

## 19. Stage-2 cluster OOD

For READ and WRITE separately:

```text
train M0 / M1 / M3 on train clusters
test on unseen clusters
```

Report:

```text
Spearman
harmful AUROC
harmful AUPRC
Precision@top5%
Precision@top10%
```

Also report Dense-C-only and Dense-W-only where supported.

---

## 20. Cluster-OOD degradation

Compare against Step-B ID:

Stage-1:

```text
ΔAUROC_cluster = AUROC_clusterOOD - AUROC_ID
```

Stage-2:

```text
Δrho_cluster = Spearman_clusterOOD - Spearman_ID
```

Use group-bootstrap CIs.

---

# Part III — Historical ↔ Canonical source-regime transfer

## 21. Why this test is mandatory

Earlier Stage-1 work showed severe population/source sensitivity.

Therefore explicitly test source-regime transfer using the new full measurement corpus.

---

## 22. Historical → Canonical

Train on all Historical internal data.

Test on all Canonical data.

Run:

```text
Stage-1 M0/M1/M3
Stage-2 READ M0/M1/M3
Stage-2 WRITE M0/M1/M3
```

No Canonical labels or statistics may be used for training, normalization, early stopping, or threshold calibration.

---

## 23. Canonical → Historical

Run the reverse direction with identical frozen settings.

Report it even if support is smaller.

---

## 24. Within-dataset source transfer

Where support exists, report separately:

```text
Historical GQA → Canonical GQA
Historical ChartQA → Canonical ChartQA
Historical TextVQA → Canonical TextVQA
```

and reverse where feasible.

This prevents pooled source results from being explained only by dataset-mixture changes.

---

## 25. Source-transfer metrics

Stage-1:

```text
AUROC
AUPRC
test C preservation under train-calibrated 95/98% thresholds
test W recall
```

Stage-2:

```text
Spearman
harmful AUROC
harmful AUPRC
Precision@top10%
```

Compare ID vs Hist→Canon vs Canon→Hist.

---

# Part IV — Dataset leave-one-out

## 26. Three primary LODO experiments

Run:

```text
Train ChartQA + TextVQA → Test GQA
Train GQA + TextVQA     → Test ChartQA
Train GQA + ChartQA     → Test TextVQA
```

Use all eligible Historical/Canonical rows from the training datasets.

The held-out dataset contributes no labels or preprocessing statistics.

---

## 27. Stage-1 LODO

For each held-out dataset train:

```text
M0 nuisance
M1 linear
M3 current head
question-kNN
```

Report:

```text
AUROC
AUPRC
train-calibrated 95% C-preserve:
    test C preservation
    test W recall
train-calibrated 98% C-preserve:
    test C preservation
    test W recall
```

Calibration drift is part of the result.

---

## 28. Stage-2 LODO

For READ and WRITE separately train:

```text
M0 nuisance
M1 linear
M3 joint router
question-kNN
```

Evaluate:

```text
Spearman
harmful AUROC
harmful AUPRC
Precision@top5%
Precision@top10%
```

Also report:

```text
held-out triggered UIDs
held-out states
harmful prevalence
```

---

## 29. Pairwise dataset transfer matrix

Secondary analysis:

```text
Train GQA     → Test ChartQA / TextVQA
Train ChartQA → Test GQA / TextVQA
Train TextVQA → Test GQA / ChartQA
```

Run for Stage-1 and, where support permits, Stage-2 READ/WRITE.

Purpose:

> Is there any shared transferable direction between specific dataset pairs?

Do not use pairwise transfer to select a preferred source.

---

## 30. No full-data refit in Step C evaluation

Every Step-C claim must come from an explicit semantic/source/dataset holdout.

Full internal refit is reserved for Step D external evaluation.

---

## 31. Layerwise generalization

Stage-1:

```text
AUROC by layer
```

for cluster/source/LODO where support permits.

Question:

> Does the layer-20-ish failure signal remain transferable?

Stage-2:

```text
Spearman by Early / Middle / Late
```

Avoid over-fragmenting sparse OOD sets.

---

## 32. Trigger-relative Stage-2 generalization

Report:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers
```

for READ/WRITE utility.

Question:

> Is any weak utility signal concentrated near the Stage-1 handoff?

---

## 33. Nuisance-residual evidence

For each OOD regime report:

```text
M3 performance
M0 nuisance performance
M3 - M0 gap
```

A strong Stage-1 transfer is more compelling if the hidden-state advantage persists while nuisance-only degrades.

---

## 34. Similarity-conditioned OOD

For cluster/source/LODO test samples compute nearest training-question similarity.

If support permits, report performance for:

```text
lower-half semantic similarity
upper-half semantic similarity
```

This distinguishes:

```text
dataset/source OOD but semantically familiar
```

from:

```text
dataset/source OOD + semantically unfamiliar
```

---

# Primary result tables

## 35. Stage-1 table

| Regime | M0 Nuisance AUROC | M1 Linear AUROC | M3 Current AUROC | M3 AUPRC | W Recall @95% C-Preserve |
|---|---:|---:|---:|---:|---:|
| Step-B ID | | | | | |
| Least-similar Q1 | | | | | |
| Question-cluster OOD | | | | | |
| Hist→Canon | | | | | |
| Canon→Hist | | | | | |
| LODO GQA | | | | | |
| LODO ChartQA | | | | | |
| LODO TextVQA | | | | | |

---

## 36. Stage-2 READ table

| Regime | M0 rho | M1 rho | M3 rho | Harmful AUROC | Precision@Top10% |
|---|---:|---:|---:|---:|---:|
| Step-B ID | | | | | |
| Least-similar Q1 | | | | | |
| Question-cluster OOD | | | | | |
| Hist→Canon | | | | | |
| Canon→Hist | | | | | |
| LODO GQA | | | | | |
| LODO ChartQA | | | | | |
| LODO TextVQA | | | | | |

Do the same for WRITE.

---

# Figures

## 37. Similarity-dependence figures

Create one figure each for:

```text
Stage-1 AUROC vs nearest-question similarity quintile
Stage-2 READ Spearman vs similarity quintile
Stage-2 WRITE Spearman vs similarity quintile
```

Overlay:

```text
question-kNN
nuisance
hidden-state predictor
```

---

## 38. Generalization ladder figures

Stage-1:

```text
ID
→ least-similar questions
→ question-cluster OOD
→ source-regime transfer
→ dataset LODO
```

Stage-2:

same ladder using Spearman.

This is the main internal generalization summary.

---

## 39. Statistical uncertainty

Use image/content-group bootstrap.

Report 95% CIs for:

Stage-1:

```text
AUROC
AUPRC
ID→OOD AUROC drop
```

Stage-2:

```text
Spearman
harmful AUROC
ID→OOD Spearman drop
```

Never bootstrap individual layer-state rows independently.

---

## 40. Seed policy

Use the Step-B seed policy.

Primary neural models:

```text
3 seeds
```

if used in Step B.

Report seed mean/SD.

Question-kNN and deterministic linear probes need one fit.

---

## 41. No OOD hyperparameter tuning

Forbidden:

```text
changing model width for a held-out dataset
changing question encoder after seeing results
using OOD labels to set harmful thresholds
changing K=100 after seeing cluster performance
using held-out normalization statistics
```

All OOD contracts are training-side only.

---

# Interpretation

## 42. Stage-1 categories

### S1-A — Strong semantic/source generalization

```text
least-similar questions remain strong
cluster OOD modest drop
Hist→Canon and LODO retain substantial performance
hidden-state model stays above nuisance/kNN
```

Interpretation:

> Eventual failure is encoded in a reasonably transferable hidden-state signal.

### S1-B — Similar-question dependence

```text
Q5 high
Q1 drops strongly
question-kNN tracks hidden-state model
cluster OOD drops sharply
```

Interpretation:

> Much of Stage-1 learnability depends on semantic/template familiarity.

### S1-C — Source-specific signal

```text
ID strong
semantic cluster OOD acceptable
source transfer or dataset LODO collapses
```

Interpretation:

> Failure signal exists but is tied to source/task regime.

---

## 43. Stage-2 categories

Step-B Stage-2 ID performance was already near chance.

### S2-A — Weak everywhere

```text
ID weak
all similarity bins weak
cluster/source/LODO weak
```

Interpretation:

> Current-state local READ/WRITE utility is not readily learnable even before meaningful OOD shift.

### S2-B — High-similarity niche

```text
Q5 materially stronger
Q1 near chance
cluster OOD collapses
```

Interpretation:

> Local utility is predictable only for familiar question families.

### S2-C — Dataset-specific niche

```text
one dataset/source has materially stronger predictability
others remain weak
```

Interpretation:

> Stage-2 utility predictability may exist only in restricted task regimes.

Do not promote the best dataset to a global result.

---

## 44. Combined interpretation

### Combined A

```text
Stage-1 generalizes
Stage-2 remains weak
```

Interpretation:

> Detecting impending failure is substantially easier and more transferable than identifying which local visual computation should be suppressed.

### Combined B

```text
Stage-1 ID strong but OOD collapses
Stage-2 weak
```

Interpretation:

> Stage-1 contains learnable but source/template-sensitive failure information, while Stage-2 additionally lacks strong ID utility signal.

### Combined C

```text
Stage-1 generalizes
specific Stage-2 niche generalizes
```

Interpretation:

> Utility predictability exists only under restricted identifiable regimes.

---

## 45. What Step C can establish

Step C can establish:

```text
how much Stage-1 relies on semantically similar training questions
whether Stage-1 survives unseen question families
whether Stage-1 survives Historical/Canonical source shift
whether Stage-1 transfers across dataset families
whether weak Stage-2 signal is similarity/source specific
whether question-kNN or nuisance baselines explain transfer
```

---

## 46. What Step C cannot establish

Step C does not establish:

```text
external four-benchmark transfer
deployment accuracy improvement
causal hidden-state mechanism
whether richer Stage-2 history/counterfactual inputs can work
whether on-policy training can rescue Stage-2
```

---

# Step-D readiness

## 47. Readiness conditions

Step D may begin only after:

```text
nearest-question similarity analysis complete
question-kNN baselines complete
question-cluster OOD complete
Historical↔Canonical transfer complete
dataset LODO complete
no split/group leakage
all OOD preprocessing fit on training data only
all models use frozen Step-B targets/hyperparameters
```

State:

```text
READY_FOR_STEP_D = true / false
```

---

# Required outputs

## 48. Output directory

Use:

```text
analysis/predictability_generalization/stepC_generalization/
```

Create:

```text
protocol.md
frozen_contract.json

question_semantics/
    encoder_contract.json
    uid_question_embeddings.npy
    group_question_embeddings.npy
    nearest_train_similarity.jsonl
    similarity_bins.csv
    question_knn_predictions_stage1.jsonl
    question_knn_predictions_stage2_read.jsonl
    question_knn_predictions_stage2_write.jsonl

similarity_analysis/
    stage1_similarity_metrics.csv
    stage2_read_similarity_metrics.csv
    stage2_write_similarity_metrics.csv
    continuous_similarity_trends.csv

question_cluster_ood/
    cluster_assignments.jsonl
    cluster_summary.csv
    cluster_fold_registry.jsonl
    representative_questions.csv
    stage1_metrics.csv
    stage2_read_metrics.csv
    stage2_write_metrics.csv

source_transfer/
    historical_to_canonical_stage1.csv
    historical_to_canonical_stage2_read.csv
    historical_to_canonical_stage2_write.csv
    canonical_to_historical_stage1.csv
    canonical_to_historical_stage2_read.csv
    canonical_to_historical_stage2_write.csv
    within_dataset_source_transfer.csv

dataset_lodo/
    lodo_stage1.csv
    lodo_stage2_read.csv
    lodo_stage2_write.csv
    pairwise_stage1_transfer.csv
    pairwise_stage2_read_transfer.csv
    pairwise_stage2_write_transfer.csv

controls/
    nuisance_generalization.csv
    question_knn_generalization.csv
    similarity_conditioned_ood.csv

statistics/
    group_bootstrap_ci.csv
    id_to_ood_degradation.csv
    seed_metrics.csv

figures/
    stage1_similarity_dependence.png
    stage2_read_similarity_dependence.png
    stage2_write_similarity_dependence.png
    stage1_generalization_ladder.png
    stage2_read_generalization_ladder.png
    stage2_write_generalization_ladder.png
    stage1_source_transfer_matrix.png
    stage2_read_source_transfer_matrix.png
    stage2_write_source_transfer_matrix.png
    hidden_state_vs_question_knn.png

summaries/
    stepC_generalization_summary.md
    stepD_readiness.md

artifact_manifest.json
```

---

## 49. `stepC_generalization_summary.md` must answer

1. Which frozen question encoder was used?
2. What is the nearest-training-question similarity distribution?
3. How does Stage-1 AUROC change from least-similar to most-similar questions?
4. How strong is the question-kNN Stage-1 baseline?
5. Does hidden-state Stage-1 retain an advantage over question-kNN and nuisance in the least-similar bin?
6. How does READ utility predictability change with question similarity?
7. How does WRITE utility predictability change with question similarity?
8. Is any Stage-2 signal concentrated only in highly similar questions?
9. How does Stage-1 perform under K=100 semantic-cluster OOD?
10. How do READ/WRITE perform under semantic-cluster OOD?
11. What is Historical→Canonical Stage-1 performance?
12. What is Canonical→Historical Stage-1 performance?
13. Does source transfer hold within each dataset?
14. What are Stage-1 LODO results for GQA, ChartQA, TextVQA?
15. What are READ/WRITE LODO results?
16. Does Stage-1 retain train-calibrated high-C-preservation behavior under OOD?
17. Does the full hidden-state model retain an advantage over nuisance under OOD?
18. Does question-kNN explain much of Stage-1 transfer?
19. Are there any datasets/regimes where Stage-2 utility becomes materially predictable?
20. What is the Stage-1 generalization ladder?
21. What is the Stage-2 generalization ladder?
22. Is the evidence more consistent with a shared failure signal, semantic familiarity, source specificity, or a mixture?
23. Does the Stage-1/Stage-2 asymmetry become stronger or weaker under shift?
24. What does Step C not justify concluding?

---

## 50. `stepD_readiness.md`

State:

```text
READY_FOR_STEP_D = true / false
```

Also record one Stage-1 category:

```text
S1-A strong semantic/source generalization
S1-B similarity/template dependence
S1-C source-specific signal
mixed
```

and one Stage-2 category:

```text
S2-A weak everywhere
S2-B high-similarity niche
S2-C dataset-specific niche
mixed
```

Do not redesign Step-D targets based on these categories.

---

## 51. Stop rule

STOP after:

```text
1. frozen question semantic embeddings
2. full nearest-training similarity analysis
3. full question-kNN baselines
4. full 5-fold semantic-cluster OOD
5. full Historical→Canonical transfer
6. full Canonical→Historical transfer
7. within-dataset source transfer
8. three full dataset LODO experiments
9. pairwise transfer matrices
10. uncertainty + control analyses
11. Step-C summary
12. Step-D readiness decision
```

Do not run:

```text
external ChartQA/TextVQA/MMMU-Pro/POPE transfer
new Stage-2 objective
new router architecture
new counterfactual labels
```

during Step C.

---

## 52. Core principle

Step B established:

```text
Stage-1:
failure is learnable in-domain

Stage-2:
current local READ/WRITE utility is barely learnable in-domain
```

Step C now asks whether the Stage-1 signal is broader than semantic familiarity and source-specific shortcuts, while characterizing whether the weak Stage-2 signal has any restricted regime where it becomes meaningful.

The internal generalization ladder is:

```text
ID
→ semantically unfamiliar questions
→ unseen question families
→ source-regime shift
→ unseen dataset family
```

Only after this ladder is complete should Step D test external benchmark transfer.
