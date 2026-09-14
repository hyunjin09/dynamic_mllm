# Stage-1 Trigger Map Audit Plan

## 1. Objective

Freeze the current Stage-1 dynamic failure gate and construct a complete **trigger map** over the existing dense FULL dataset.

This phase is intentionally narrow.

The goal is to answer:

> **For each sample, does Stage-1 trigger, and if so, at which layer does the first trigger occur?**

The output of this phase will define the exact population that is handed from Stage-1 to the future Stage-2 visual-computation policy.

Do **not** run new four-action corrective search in this phase.

---

## 2. Frozen Stage-1 Configuration

Use the already selected dynamic Stage-1 model:

```text
Shared Random-4
```

with its existing frozen checkpoint and frozen operating threshold/control.

Do not:

```text
retrain the predictor
change the threshold
perform dataset-specific calibration
select a new gate
compare architectures for winner selection
```

The purpose is to characterize the currently frozen dynamic gate, not improve it.

If the exact frozen checkpoint / threshold is stored in prior Phase-52/53 artifacts, load it directly and record its identifier in `protocol.md`.

---

## 3. Reuse Existing Outputs First

Before launching any model forward pass, audit whether the trigger map can be reconstructed from existing saved outputs.

Check, in order:

```text
previous shared_random4 trigger manifests
saved per-layer failure scores
saved dense feature shards
Phase-52 gate outputs
Phase-53 treatment-correctability manifests
```

Preferred order:

```text
existing trigger manifest
    ↓
existing per-layer scores
    ↓
recompute gate scores from stored dense features
    ↓
only if necessary: rerun model forward
```

Do not rerun the full MLLM if the required information already exists.

Record exactly which source was used.

---

## 4. Population

Use the current-runtime dense FULL dataset and its existing split assignments.

Primary splits:

```text
train
validation
test
```

Preserve the previously frozen group-disjoint split.

For every executable sample, use the already established dense FULL correctness label:

```text
dense_correct = 1
dense_wrong   = 0
```

Do not change the evaluation metric or binary correctness threshold.

The trigger audit must use the same current-runtime LMMS-Eval correctness labels already used in Stage-1 development.

---

## 5. Per-Sample Trigger Definition

For sample `i`, let the Stage-1 failure predictor produce:

```text
p_i,0
p_i,1
...
p_i,27
```

where:

```text
p_i,l = predicted probability / score of eventual dense failure at layer l
```

Apply the existing frozen sequential trigger rule exactly as previously defined.

The first trigger layer is:

```text
l_i* = first layer at which the frozen gate fires
```

If the gate never fires:

```text
triggered = false
first_trigger_layer = null
```

If it fires:

```text
triggered = true
first_trigger_layer = l_i*
```

Do not retrospectively choose a later or "better" trigger.

---

## 6. Required Per-Sample Manifest

Create one row per sample with at least:

```text
uid
dataset
split
group_id               # if available from frozen split logic

dense_correct
dense_wrong

triggered
first_trigger_layer

score_at_trigger
threshold_or_control_used
```

Also preserve the complete layer-wise score vector if available:

```text
score_l0
score_l1
...
score_l27
```

Prefer storing the full 28-layer score vector because it enables later threshold / persistence analyses without rerunning the model.

Optional but useful:

```text
dense_prediction
reference_answer
question_id
image_group_id
```

Do not make these optional fields a blocking requirement.

---

## 7. Core Partition

For each split, partition samples into exactly four groups:

```text
A. Dense-C, no trigger
B. Dense-C, trigger
C. Dense-W, no trigger
D. Dense-W, trigger
```

These groups have different future meanings:

```text
Dense-C, no trigger
→ safe default path
→ Stage-2 never invoked

Dense-C, trigger
→ false-positive Stage-1 admission
→ future C→C preservation population

Dense-W, no trigger
→ Stage-1 miss
→ no Stage-2 opportunity under current architecture

Dense-W, trigger
→ future Stage-2 corrective-search population
```

Freeze the exact UID lists.

---

## 8. Primary Metrics

Compute separately for train / validation / test.

### 8.1 Correct preservation before Stage-2

```text
P(no trigger | Dense-C)
```

Equivalent:

```text
# Dense-C with no trigger
/
# all Dense-C
```

This is the current Stage-1-only preservation rate.

Also report:

```text
P(trigger | Dense-C)
```

as the Stage-1 false-admission rate.

### 8.2 Wrong trigger recall

```text
P(trigger | Dense-W)
```

Equivalent:

```text
# Dense-W with trigger
/
# all Dense-W
```

This is the fraction of dense failures that become eligible for future Stage-2 treatment.

Also report:

```text
P(no trigger | Dense-W)
```

as the Stage-1 miss rate.

### 8.3 Trigger precision

Among all triggered samples:

```text
P(Dense-W | trigger)
```

This quantifies how enriched the Stage-2 admission population is for true dense failures.

---

## 9. First-Trigger-Layer Distribution

For triggered Dense-C and triggered Dense-W separately, report:

```text
count by first_trigger_layer 0...27
fraction by first_trigger_layer 0...27
```

Produce both:

```text
exact-layer histogram
```

and grouped bins:

```text
Early:  0-8
Middle: 9-18
Late:   19-27
```

Do not infer that earlier is automatically better.

The purpose here is descriptive:

> **When does the current Stage-1 gate become confident enough to hand off the sample?**

---

## 10. Cumulative Trigger Curves

For every layer `l`, compute:

```text
Wrong cumulative trigger:
P(first_trigger_layer <= l | Dense-W)

Correct cumulative trigger:
P(first_trigger_layer <= l | Dense-C)
```

Plot both curves on the same axes.

This answers:

> By layer `l`, how much of the dense-wrong population has been admitted, and how much of the dense-correct population has also been admitted?

This curve will later be useful when reasoning about the trade-off between:

```text
earlier intervention opportunity
vs
C→C preservation risk
```

Do not optimize a new threshold from this curve in the current phase.

---

## 11. Trigger Score Analysis

For triggered samples, inspect:

```text
score_at_trigger
```

separately for Dense-C and Dense-W.

Report:

```text
mean
median
IQR
5th / 95th percentile
```

Optionally report:

```text
score margin above threshold
```

This can reveal whether false-trigger Dense-C samples barely cross the threshold while Dense-W samples cross with larger margin.

This is descriptive only.

Do not change the gate rule yet.

---

## 12. Dataset Breakdown

Report separately for:

```text
GQA
ChartQA
TextVQA
```

Metrics:

```text
Dense-C preservation
Dense-W trigger recall
trigger precision
median trigger layer for Dense-W
median trigger layer for Dense-C
```

Also report early/middle/late trigger proportions for Dense-W.

Do not tune per-dataset thresholds.

---

## 13. Train Population for Future Stage-2

For the **train split only**, create the exact future Stage-2 candidate manifests.

### 13.1 Triggered Dense-C manifest

These samples will later provide conservative preservation supervision.

For now only record:

```text
uid
dataset
first_trigger_layer
dense_correct = true
```

Future default label source:

```text
FULL suffix
```

No four-action search is performed in this phase.

### 13.2 Triggered Dense-W manifest

These samples will later enter corrective suffix search.

Record:

```text
uid
dataset
first_trigger_layer
dense_correct = false
```

This list defines the future search workload.

Do not yet label samples as FIXABLE / UNRESOLVED.

That belongs to the next phase.

---

## 14. Validation/Test Usage Contract

Validation and test trigger maps may be constructed and analyzed.

However:

```text
train triggered-W
→ future Stage-2 label generation / corrective search

validation triggered-W
→ future model selection / treatment evaluation only

test triggered-W
→ future final held-out treatment evaluation only
```

Do not use validation or test corrective trajectories as Stage-2 training labels.

Freeze this separation now.

---

## 15. Sanity Checks

Before accepting the trigger map, verify:

### Check 1 — sample counts

For each split:

```text
Dense-C + Dense-W = total samples
```

and:

```text
C/no-trigger
+ C/trigger
+ W/no-trigger
+ W/trigger
= total samples
```

### Check 2 — trigger consistency

For every triggered sample:

```text
no earlier layer satisfies the frozen trigger rule
```

### Check 3 — no-trigger consistency

For every non-triggered sample:

```text
no layer 0...27 satisfies the frozen trigger rule
```

### Check 4 — score reproducibility

If per-layer scores already existed, randomly verify a small subset against freshly computed gate scores from stored features.

No full rerun is needed unless mismatch is found.

### Check 5 — prior aggregate consistency

The validation/test preservation, wrong recall, and precision reconstructed from the trigger map should match the previously reported Shared Random-4 results up to exact deterministic equality or explained numerical tolerance.

Any mismatch must be investigated before continuing.

---

## 16. Required Figures

Create:

```text
trigger_hist_wrong.png
trigger_hist_correct.png
trigger_cumulative_c_vs_w.png
trigger_score_distribution.png
dataset_trigger_breakdown.png
```

Keep figures diagnostic and simple.

---

## 17. Required Outputs

Use:

```text
analysis/dense_failure_stage1/trigger_map/
```

Create:

```text
protocol.md

manifests/
    trigger_map_train.jsonl
    trigger_map_val.jsonl
    trigger_map_test.jsonl

    train_triggered_correct.jsonl
    train_triggered_wrong.jsonl

    val_triggered_correct.jsonl
    val_triggered_wrong.jsonl

    test_triggered_correct.jsonl
    test_triggered_wrong.jsonl

metrics/
    split_summary.csv
    trigger_by_layer.csv
    trigger_by_depth_bin.csv
    cumulative_trigger.csv
    dataset_breakdown.csv
    trigger_score_stats.csv

figures/
    trigger_hist_wrong.png
    trigger_hist_correct.png
    trigger_cumulative_c_vs_w.png
    trigger_score_distribution.png
    dataset_trigger_breakdown.png

decision_summary.md
```

---

## 18. `decision_summary.md` Must Answer

1. What fraction of Dense-C samples are retained without triggering?
2. What fraction of Dense-W samples are detected by Stage-1?
3. Among triggered samples, what fraction are actually Dense-W?
4. At which layers do Dense-W samples most commonly first trigger?
5. At which layers do false-trigger Dense-C samples first trigger?
6. How different are the cumulative trigger curves for Dense-C and Dense-W?
7. How strongly does trigger behavior differ by dataset?
8. Exactly how many train Dense-W samples will require future corrective suffix search?
9. Exactly how many train Dense-C samples will provide future preservation supervision?
10. Is the trigger map consistent with the previously reported Shared Random-4 aggregate metrics?

---

## 19. Interpretation Boundaries

Allowed conclusions:

```text
Stage-1 detects X% of dense failures.
Stage-1 leaves Y% of dense-correct samples untouched.
Triggered wrong samples tend to trigger at these depths.
False-trigger correct samples tend to trigger at these depths.
Dataset behavior differs by this amount.
```

Do not conclude yet:

```text
early trigger is better
early trigger causes higher correctability
a triggered wrong sample is fixable
a particular four-action treatment should be used
Stage-2 improves final accuracy
```

Those require the next treatment-label phase.

---

## 20. Stop Rule

STOP after the trigger map and diagnostic report are complete.

Do not:

```text
run new four-action search
generate Stage-2 action labels
train Stage-2
change Stage-1 thresholds
add persistence / EMA trigger logic
resume broad W→C route repair
```

The next separately authorized phase will be:

```text
Triggered-W corrective suffix search
→ determine FIXABLE vs UNRESOLVED
→ extract trajectory-conditioned Stage-2 supervision
```

The current phase must only establish:

```text
WHO triggers
WHEN they trigger
WHETHER they were Dense-C or Dense-W
```
