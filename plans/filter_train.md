We have completed the MCTS label-geometry and duplicated-BCE oracle analysis for the frozen GQA/TextVQA/ChartQA binary-routing cache.

The next task is authorized:

> **Pareto-filter the existing GQA, TextVQA, and ChartQA valid-route supervision, freeze the resulting Pareto-efficient training manifests, then train matched BCE and exact valid-set NLL predictors for 10 epochs and run the full evaluation pipeline.**

Do not regenerate MCTS.

Do not modify the base MLLM or binary executor.

The main scientific question is:

> **Were the previous BCE/NLL failures primarily caused by dominated supervision, or does an objective-level failure remain even after training only on Pareto-efficient routes?**

---

# 1. Starting evidence

The frozen label audit established:

```text
selected valid route occurrences = 237,802
Pareto-dominated                = 227,897
dominated fraction              = 95.83%

mean routes/sample:
original = 34.38
Pareto   = 1.43

BCE label oracle:
original Hit@1 = 5.93%
Pareto Hit@1   = 73.41%

mean BCE-oracle ON:
original = 17.21
Pareto   = 9.78
```

The existing MCTS cache itself is not geometrically deficient:

```text
raw mean pairwise Hamming = 13.36 / 28
raw valid routes/positive = 76.34
raw mean ON               = 15.26
```

The max-50 selector also preserves the measured route geometry.

Therefore:

```text
DO NOT regenerate MCTS.
DO NOT search for more routes.
DO NOT redesign the predictor.
```

The controlled intervention is:

```text
existing valid routes
       ↓
Pareto filtering
       ↓
frozen Pareto-efficient supervision
       ↓
BCE vs exact set-NLL
```

---

# 2. Datasets

Use exactly the existing frozen datasets:

```text
GQA
TextVQA
ChartQA
```

Use the existing image-group-disjoint train/validation split.

Expected frozen population:

```text
8,000 total records
7,000 train
1,000 validation

6,917 positive inputs
6,043 positive train
874 positive validation
```

Verify these identities against the existing manifests before doing anything.

Do not change sample membership.

Do not silently remove zero-positive samples from population accounting.

---

# 3. Existing route semantics

Preserve the current 28-bit route definition:

```text
1 = VISUAL_ON
0 = TEXT_ONLY
```

ALL-ON:

```text
1111111111111111111111111111
```

means normal/full visual execution.

Validity remains exactly the frozen benchmark criterion already used by the regenerated labels.

Do not redefine correctness.

---

# 4. Pareto dominance definition

For each sample independently, consider its valid routes.

Route `b` dominates route `a` if:

```text
stored_score(b) >= stored_score(a)
AND
ON_count(b) < ON_count(a)
```

where:

```text
ON_count(route)
=
number of VISUAL_ON layers
```

This means `b` is at least as good in benchmark utility and strictly cheaper in visual-layer compute.

Remove route `a` from the training supervision.

Do not use route frequency as part of dominance.

Do not invent another continuous utility.

Use the stored benchmark score already frozen in the cache.

---

# 5. Pareto filtering must be performed separately per sample

For every positive record in:

```text
GQA
TextVQA
ChartQA
```

perform:

```text
raw/selected valid route set
        ↓
compute score + ON count
        ↓
remove Pareto-dominated routes
        ↓
retain all Pareto-efficient routes
```

Do not force exactly one route per sample.

If multiple routes are genuinely non-dominated, preserve all of them.

Example:

```text
A: score 1.0, 28 ON
B: score 1.0, 17 ON
C: score 1.0,  8 ON
```

becomes:

```text
{C}
```

But:

```text
A: score 1.0, 10 ON
B: score 0.8,  6 ON
C: score 0.6,  3 ON
```

may retain multiple routes if none dominates another.

---

# 6. Important: filter the exact supervision source used by training

Do not accidentally Pareto-filter one route view and train on another.

Determine the exact route manifest/source currently consumed by the full10 BCE/NLL dataset code.

Use that as the parent supervision source.

The preferred workflow is:

```text
frozen current selected supervision
        ↓
Pareto filtering
        ↓
new frozen Pareto manifest
```

Do not independently rerun the old max-50 selector unless required to recreate the exact frozen source deterministically.

---

# 7. Create a frozen Pareto supervision manifest

Create one canonical manifest containing the filtered routes for all three datasets.

Each record should preserve at least:

```text
UID
dataset
image-group identity
FULL correctness
original selected-valid route count
Pareto-efficient route count
each retained 28-bit mask
stored benchmark score
ON count
original route metadata needed for provenance
```

Also preserve source-manifest references and hashes.

The BCE and NLL runs must consume **the exact same Pareto manifest**.

This is critical.

Do not create separately filtered BCE and NLL labels.

---

# 8. Pareto filtering integrity audit

Before training, verify:

```text
every retained route was present in original valid supervision
every retained route satisfies validity
every retained route is 28 bits
ON counts are exact
no exact duplicates
no dominated route remains
every positive sample retains >= 1 Pareto route
train/validation identity unchanged
image-group disjointness unchanged
```

For every removed route, verify that at least one retained or valid route provides a valid dominance witness.

Generate deterministic hashes for the new manifest.

Do not start training until this gate passes.

---

# 9. Required Pareto geometry report before training

Report overall and separately for:

```text
GQA
TextVQA
ChartQA
train
validation
```

the following before vs after:

```text
positive sample count
route occurrences
mean routes/sample
median routes/sample
p90 routes/sample
fraction with exactly 1 Pareto route
fraction with 2
fraction with >=3
mean ON
minimum ON
median ON
ALL-ON sample presence
ALL-OFF sample presence
```

Also report:

```text
fraction of ALL-ON occurrences removed
fraction of samples whose ALL-ON route is removed
```

Expected prior-analysis sanity values are approximately:

```text
mean original selected routes ≈ 34.38
mean Pareto routes            ≈ 1.43
median Pareto routes          = 1
```

Do not force exact agreement if the canonical training source differs slightly; explain any discrepancy.

---

# 10. Recompute label-only oracles on the frozen training manifest

Before model training, recompute the following using the actual final Pareto manifest.

## 10.1 BCE label oracle

Using the exact route weighting that the BCE run will consume:

```text
q_i,l
=
Σ_m alpha_i,m m_l
/
Σ_m alpha_i,m
```

then decode with the deployed rule:

```text
q_i,l >= 0.5 → ON
```

Report:

```text
Hit@1
nearest-valid Hamming
mean ON
ALL-ON %
ALL-OFF %
unique masks
```

## 10.2 NLL shortcut geometry

Report:

```text
fraction of Pareto sets containing ALL-ON
most common complete route coverage
top-5 route coverage
top-50 route coverage
```

The purpose is to verify that the previous ALL-ON NLL shortcut has actually been removed from most supervision sets.

This is diagnostic only.

Do not stop the approved training based on these results unless there is a technical integrity failure.

---

# 11. Primary predictor architecture

Use the existing **P13/full10 Image+Question direct factorized predictor**.

Use:

```text
Frozen Qwen3 question-token embeddings
+
frozen native Qwen2.5-VL projected visual rows
entering decoder layer 0
```

with the already established minimal fusion:

```text
question projection
visual projection
concatenate visible tokens
28 learned layer queries
cross-attention
cross-layer encoder
direct 28-bit binary head
```

Do not use the P12 segment head.

Do not add a new encoder.

Do not pool the P13 visual features.

Do not change the visual feature cache definition.

---

# 12. Two matched training runs

Run exactly two primary experiments.

## Run A — Pareto + duplicated BCE

```text
Input:
Image + Question

Labels:
Pareto-efficient complete masks only

Loss:
existing POLAR-style duplicated-route BCE
```

For a sample with multiple Pareto routes, retain the existing duplicated-route supervision behavior.

Use the route weighting defined by the current BCE implementation unless filtering makes the ALL-ON special case irrelevant.

Document exact weights.

---

## Run B — Pareto + exact valid-set NLL

```text
Input:
Image + Question

Labels:
the exact same Pareto-efficient complete masks

Loss:
existing exact one-of-valid-set NLL
```

Use:

```text
L_i
=
-log Σ_{m ∈ P_i} w_im P_theta(m | x_i)
```

where `P_i` is the frozen Pareto-efficient set.

Use the already validated stable logsumexp implementation.

Do not alter factorization or decoding.

---

# 13. Important matched-control rule

Everything except the training objective must be matched.

Keep the same:

```text
Pareto manifest
train/validation identities
Image+Question features
predictor architecture
shared initialization
optimizer
learning rate
batch size
scheduler
warmup
epochs
seed
decoder threshold
evaluation code
```

The main variable is:

```text
duplicated BCE
vs
exact valid-set NLL
```

---

# 14. Optimization configuration

Use the same POLAR-style full10 setup already established:

```text
epochs          = 10
effective batch = 128
learning rate   = 5e-4
optimizer       = AdamW
scheduler       = cosine
warmup steps    = 10
seed            = frozen matched seed
```

If physical batch size must be lower for Image+Question, use gradient accumulation.

Record:

```text
physical batch
gradient accumulation
effective batch
optimizer steps
scheduler steps
```

for both runs.

Do not silently change the effective batch size.

---

# 15. Parallel training

Run the two experiments concurrently if resources permit.

Example:

```text
GPU 0 → Pareto BCE
GPU 1 → Pareto NLL
```

or an equivalent isolated allocation.

Use independent:

```text
output directories
logs
checkpoint directories
```

Do not allow one run's outcome to change the other's configuration.

---

# 16. Save every epoch

Mandatory:

```text
pareto_bce/
    epoch_01/
    ...
    epoch_10/

pareto_nll/
    epoch_01/
    ...
    epoch_10/
```

Each checkpoint must preserve enough information to reproduce validation:

```text
model state
optimizer state
scheduler state
epoch
global step
config
seed
hash
```

Do not overwrite epochs.

---

# 17. Validate every epoch

Evaluate on the full frozen positive validation set after each epoch.

For both BCE and NLL report:

```text
train loss
validation loss

valid-set Hit@1
Hit@5 if already available and semantically valid
nearest-valid Hamming

unique predicted masks
mask entropy
ALL-ON %
ALL-OFF %
mean VISUAL_ON layers
```

Also report separately for:

```text
GQA
TextVQA
ChartQA
```

at minimum:

```text
Hit@1
nearest Hamming
ALL-ON %
mean ON
unique masks
```

---

# 18. Important: validation target is the Pareto set

The primary internal valid-set metrics must use the **new Pareto-efficient supervision set**, not the old unfiltered valid set.

However, also optionally report compatibility with the original valid set:

```text
Pareto Hit@1
Original-valid Hit@1
```

This helps determine whether predictions leave the Pareto frontier while remaining behaviorally valid.

Clearly label the two.

Do not mix them.

---

# 19. Checkpoint selection

After all 10 epochs have completed, select the best checkpoint independently for BCE and NLL using the same prospective hierarchy.

Recommended:

```text
1. maximize Pareto-valid Hit@1
2. minimize nearest-Pareto Hamming
3. minimize validation loss
4. earlier epoch
```

If the repository already has a frozen full10 checkpoint-selection rule that should remain comparable, preserve it and additionally report the Pareto-specific best diagnostic checkpoint.

Do not select a checkpoint using external execution results.

Also always report:

```text
best validation checkpoint
best validation-loss checkpoint
epoch 10
```

separately.

---

# 20. Training-trajectory interpretation

For both runs produce a 10-epoch table:

```text
Epoch
Val loss
Pareto Hit@1
Original-valid Hit@1
Nearest Pareto Hamming
Unique masks
ALL-ON
ALL-OFF
Mean ON
```

We specifically want to know whether:

```text
BCE:
hybridization disappears after Pareto filtering?

NLL:
ALL-ON collapse disappears after Pareto filtering?
```

Do not judge success from sparsity alone.

---

# 21. Compare to the old full10 models

Use the frozen previous full10 results as historical baselines.

Compare:

```text
Old unfiltered BCE
Old unfiltered NLL
New Pareto BCE
New Pareto NLL
ALL-ON
```

Report changes in:

```text
ALL-ON fraction
mean ON
unique masks
validation Hit@1
nearest Hamming
```

This is necessary to isolate the effect of Pareto filtering.

---

# 22. Primary scientific hypotheses

## Hypothesis H1 — dominated labels caused most BCE failure

Prediction:

```text
Pareto BCE
>>
unfiltered BCE
```

in complete-mask coherence and possibly execution.

## Hypothesis H2 — dominated FULL routes caused NLL ALL-ON collapse

Prediction:

```text
Pareto NLL
```

no longer converges to ALL-ON.

## Hypothesis H3 — BCE still suffers residual multimodal hybridization

Because Pareto filtering leaves mean ~1.43 routes/sample rather than exactly one, some samples remain multimodal.

If:

```text
Pareto NLL > Pareto BCE
```

especially on multi-Pareto-route samples, that supports residual objective mismatch.

---

# 23. Split validation by Pareto-set multiplicity

This analysis is important.

Partition validation samples into:

```text
|P_i| = 1
|P_i| = 2
|P_i| >= 3
```

For BCE and NLL separately report:

```text
Hit@1
nearest Hamming
mean ON
ALL-ON %
```

Why:

For:

```text
|P_i| = 1
```

BCE and NLL supervision are nearly equivalent.

For:

```text
|P_i| > 1
```

BCE can still average multiple complete modes whereas NLL can place mass on one complete mode.

This is one of the most scientifically informative comparisons in the experiment.

---

# 24. Analyze FULL-correct vs FULL-wrong separately

Retain the existing taxonomy.

## Group A

```text
FULL wrong
correction route exists
```

## Group B

```text
FULL correct
cheaper valid route exists
```

## Group C

```text
FULL correct
no cheaper route found
```

Report validation results per group:

```text
Pareto Hit@1
mean ON
ALL-ON %
```

This tells us whether:

```text
Group A:
router learns correction programs

Group B:
router learns safe compute reduction

Group C:
router correctly stays FULL
```

---

# 25. External evaluation

After checkpoint selection, run the existing frozen external evaluation pipeline for both selected models.

Use the same external frozen suite used for previous full10 evaluation.

Do not change benchmark membership.

At minimum include the currently available:

```text
ChartQA
TextVQA
MMStar
MMMU
MMMU-Pro standard
MMMU-Pro vision
POPE adversarial
POPE popular
POPE random
```

and the existing pooled reporting groups where applicable.

Use the current live ALL-ON execution as the scientific baseline, exactly as in the previous external evaluation.

Do not use stale historical cache correctness as the main baseline.

---

# 26. External evaluation must actually execute predicted masks

For each external record:

```text
input
↓
Pareto-trained predictor
↓
28-bit top-1 mask
↓
unchanged Qwen2.5-VL binary executor
↓
benchmark evaluation
```

Do not infer correctness merely from whether a mask appeared in training/MCTS cache.

Uncached predicted masks must still be executed.

---

# 27. Required external metrics

For every benchmark and each of:

```text
Pareto BCE
Pareto NLL
ALL-ON baseline
```

report:

```text
ALL-ON/base accuracy
router accuracy
delta
ratio vs base
W→C / Rescue
C→W / Harm
mean ON layers
ON reduction
unique masks
ALL-ON fraction
```

Also report:

```text
unchanged correct
unchanged wrong
number of behavior-changing executions
```

Use the same definitions as the prior external reports.

---

# 28. Accuracy–compute tradeoff

The primary downstream interpretation must jointly consider:

```text
accuracy
and
mean ON layers
```

Examples:

Potential success:

```text
base accuracy = 0.86
router        = 0.86
mean ON       = 10
```

Not sufficient:

```text
base accuracy = 0.86
router        = 0.78
mean ON       = 7
```

Do not label a predictor successful merely because it is sparse.

---

# 29. Compare BCE vs NLL directly

The final report must include a direct table such as:

```text
                 Pareto BCE    Pareto NLL    ALL-ON
Val Pareto Hit@1
Val nearest Hamming
ALL-ON %
Unique masks
Mean ON
External accuracy
W→C
C→W
```

overall and per benchmark where meaningful.

---

# 30. Required decision outcomes

Choose the closest supported interpretation.

## Outcome A — Both BCE and NLL substantially improve

Evidence:

```text
ALL-ON collapse reduced
Pareto Hit@1 improves
execution retains/improves accuracy
mean ON drops
```

Interpretation:

> Dominated supervision was the primary bottleneck.

---

## Outcome B — NLL succeeds, BCE remains weaker

Evidence:

```text
Pareto NLL produces coherent useful routes
Pareto BCE still shows hybridization,
especially for |P_i| > 1.
```

Interpretation:

> Dominated routes were one problem, but independent duplicated BCE retains a residual multimodal complete-route mismatch.

This would strongly favor complete-route supervision.

---

## Outcome C — BCE succeeds, NLL still collapses

Evidence:

```text
Pareto BCE predicts useful sparse routes
Pareto NLL converges to another common route/mode.
```

Interpretation:

> Removing dominated ALL-ON was insufficient to eliminate set-likelihood mode collapse.

---

## Outcome D — Internal route metrics improve, external execution does not

Interpretation:

> Pareto labels improve supervision coherence but do not yet yield robust held-out behavioral routing.

Do not hide this behind internal Hit@1.

---

## Outcome E — Both remain poor

Interpretation:

> Label dominance was a real problem but not the main deployable-router bottleneck.

The next step should reconsider predictor generalization or candidate-route scoring/ranking.

Do not implement that pivot automatically.

---

# 31. Important ablation: singleton vs multimodal Pareto sets

This must be emphasized in the final interpretation.

If BCE and NLL are approximately equal on:

```text
|P_i| = 1
```

but NLL is substantially better on:

```text
|P_i| >= 2
```

that is direct evidence supporting:

> complete-route supervision matters specifically when multiple non-dominated route modes remain.

Conversely, if they remain equivalent everywhere, Pareto filtering may have removed most of the meaningful distinction between the objectives.

---

# 32. Do not change anything else

Do not:

```text
regenerate MCTS
increase MCTS simulations
change max route search
change the base model
change binary executor semantics
change visual features
change Image+Question fusion
use P12 segment head
add beam search
add RL
add compute penalty
add extra sparsity regularization
change correct/wrong ratio
oversample Group A
change threshold
tune on external evaluation
```

This experiment must isolate:

```text
unfiltered supervision
        ↓
Pareto-efficient supervision
```

and then compare:

```text
BCE vs NLL
```

---

# 33. Required artifact structure

Create clean new directories such as:

```text
outputs/binary_pareto_v1/
    manifests/
    audits/
    oracle_analysis/
    training/
        bce/
        nll/
    validation/
    external_eval/
```

Do not overwrite any existing P10–P13/full10/MCTS artifacts.

---

# 34. Required final report

Create:

```text
binary_pareto_bce_nll_full10_results.md
```

The report must include:

1. source supervision manifests;
2. exact Pareto-dominance definition;
3. before/after filtering statistics;
4. dataset-wise filtering statistics;
5. Pareto manifest integrity checks;
6. Pareto BCE oracle;
7. Pareto NLL shortcut/common-route geometry;
8. BCE 10-epoch trajectory;
9. NLL 10-epoch trajectory;
10. singleton vs multi-Pareto validation analysis;
11. Groups A/B/C analysis;
12. old unfiltered vs new Pareto comparison;
13. selected checkpoint identities and hashes;
14. external Qwen execution results;
15. accuracy–compute comparison;
16. BCE vs NLL direct comparison;
17. exact Outcome A/B/C/D/E decision;
18. recommendation for the next research step.

---

# 35. Execution principle

> **Without regenerating MCTS, remove compute-dominated supervision from the frozen GQA/TextVQA/ChartQA valid-route sets, freeze one common Pareto-efficient label manifest, then train matched Image+Question BCE and exact-set-NLL predictors for 10 epochs and execute their selected masks on the existing external evaluation suite to determine whether dominated labels caused the previous routing failures and whether residual multimodal supervision still favors a complete-route objective over independent BCE.**
