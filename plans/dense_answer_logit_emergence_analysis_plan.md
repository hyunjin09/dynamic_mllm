# Dense Answer-Logit Emergence Analysis Plan

## 1. Goal

Before training the Stage-1 failure predictor, first analyze how answer confidence evolves across decoder depth.

Main questions:

1. For dense-correct samples, when does the correct answer become clearly preferred?
2. For dense-wrong samples, when does the eventual wrong answer become preferred over the ground-truth answer?
3. Are early layers mostly ambiguous, with failure/correctness becoming distinguishable only in middle or late layers?
4. Is there a meaningful depth after which failure supervision should begin?

Do not train the failure predictor yet.

---

## 2. Data

Use the newly generated current-runtime dense all-on dataset:

- GQA
- ChartQA
- TextVQA
- approximately 7,999 successfully completed samples
- each sample has:
  - dense final answer
  - lmms-eval correctness
  - layer-wise dense hidden states for layers 0-27

Use the current-runtime dense correctness labels only.

No W→C labels, four-action routes, or sparse-router outputs are needed for this analysis.

---

## 3. Main Analysis: Layer-Wise Answer Logits

Start with a simple first-answer-token logit-lens analysis.

For each layer `l = 0...27`, use the saved text-side hidden state at the answer-start/query position and apply the model's final normalization + LM head.

For each sample obtain:

```text
GT score at layer l
final predicted-answer score at layer l
```

Use the first answer token for this initial analysis.

Do not add a new learned probe here.

---

## 4. Correct Samples

For samples whose dense final answer is correct:

Track how the ground-truth answer score changes with depth.

Primary curve:

```text
layer
→ mean / median GT answer score
```

Also compute a margin against the strongest competing token:

```text
correct_margin_l
=
GT score - strongest non-GT score
```

Question:

> At what depth does the correct answer become clearly preferred?

---

## 5. Wrong Samples

For samples whose dense final answer is wrong:

Track both:

```text
GT answer score
eventual wrong-answer score
```

Primary margin:

```text
wrong_margin_l
=
GT score - eventual wrong-answer score
```

Interpretation:

```text
margin > 0
→ GT preferred

margin ≈ 0
→ ambiguous

margin < 0
→ eventual wrong answer preferred
```

Question:

> At what depth does the eventual wrong answer begin to dominate the correct answer?

---

## 6. Layer-Wise Population Curves

Produce the following curves over layers 0-27.

### Correct samples

```text
mean GT score
median GT score
mean correct_margin
median correct_margin
```

### Wrong samples

```text
mean GT score
mean eventual-wrong score
median GT score
median eventual-wrong score
mean GT-minus-wrong margin
median GT-minus-wrong margin
```

Also show confidence intervals or bootstrap bands if easy to compute.

---

## 7. Sample-Wise Emergence Layer

Define a simple emergence rule.

For correct samples:

```text
correct emergence layer
=
first layer where correct_margin > delta
for 3 consecutive layers
```

For wrong samples:

```text
wrong emergence layer
=
first layer where wrong_margin < -delta
for 3 consecutive layers
```

Choose a small fixed `delta` prospectively from the score scale.

If choosing `delta` is awkward, also report the simpler zero-crossing version:

```text
first persistent layer where margin changes sign
```

Do not tune `delta` to maximize a later training result.

---

## 8. Wrong-Sample Taxonomy

Classify wrong samples into a small number of trajectory types.

### A. Early-wrong

The wrong answer is preferred from early layers onward.

### B. Progressive-wrong

Early layers are ambiguous, then the wrong answer gradually becomes dominant.

### C. Answer erosion

The GT answer is preferred at some middle layer, but later the trajectory flips toward the eventual wrong answer.

### D. Ambiguous / no clear separation

The margin remains small or unstable through most of the network.

Report the fraction of wrong samples in each category.

Keep the taxonomy simple and deterministic.

---

## 9. Dataset Breakdown

Repeat the main curves for:

```text
GQA
ChartQA
TextVQA
```

Questions:

- Does failure confidence emerge at similar depths across datasets?
- Is one dataset systematically earlier or later?
- Is answer erosion concentrated in a particular dataset?

Do not over-interpret small dataset differences.

---

## 10. Main Outputs

Create:

```text
analysis/dense_failure_stage1/logit_emergence/
```

with:

```text
layerwise_correct_summary.csv
layerwise_wrong_summary.csv
sample_emergence_layers.csv
wrong_trajectory_taxonomy.csv
dataset_breakdown.csv

figures/
    correct_margin_by_layer.png
    wrong_gt_vs_pred_by_layer.png
    wrong_margin_by_layer.png
    emergence_layer_histogram.png

analysis_summary.md
```

---

## 11. What to Decide From This Analysis

The final report should answer:

### Q1

> Are early layers actually uninformative about eventual correctness?

### Q2

> Around which layer does correct/wrong answer preference begin to separate reliably?

### Q3

> Do wrong samples usually become progressively wrong, or do many show answer erosion?

### Q4

> Is there a defensible layer range after which failure-prediction supervision should begin?

### Q5

> Should the next Stage-1 training compare:
>
> - all-layer supervision
> - post-emergence supervision
> - random-k layers sampled only from the informative depth range?

---

## 12. Optional Follow-Up Only If the Pattern Is Clear

If the first-token analysis shows a strong and interpretable depth pattern, validate it on a smaller subset using teacher-forced sequence-level answer scores.

Do not run the sequence-level analysis by default.

The first-token analysis is the primary next step.

---

## 13. Stop Rule

Stop after the analysis report.

Do not yet:

- train the failure predictor;
- choose the final supervision range;
- resume W→C repair;
- train four-action routing.

Use the observed layer-wise confidence dynamics to decide the Stage-1 training strategy next.
