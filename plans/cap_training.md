We have completed the previous unfiltered and strict-Pareto experiments.

The next authorized experiment tests an intermediate supervision policy:

> **Instead of keeping every valid route or aggressively Pareto-filtering to nearly one route per sample, retain all valid routes whose VISUAL_ON count is below a fixed absolute compute budget.**

Run a four-way duplicated-BCE sweep with:

```text
CAP = 24
CAP = 22
CAP = 20
CAP = 18
```

Train each condition for the full 10 epochs on a separate GPU, in parallel, then run the same frozen external execution evaluation.

Do not run new MCTS.

Do not use strict Pareto filtering in this experiment.

---

# 1. Scientific motivation

Previous supervision extremes behaved poorly:

```text
Unfiltered supervision
≈ 34 valid routes / positive sample
→ many dense/dominated routes
→ trained predictors remained too ON-heavy

Strict Pareto supervision
≈ 1.43 routes / positive sample
→ removes almost all expensive alternatives
→ trained predictors became extremely sparse / ALL-OFF-heavy
→ poor behavioral route quality
```

This experiment tests an intermediate policy:

```text
keep several valid alternatives
BUT
remove routes above an absolute visual-compute budget
```

For cap `C`, define the supervision set:

```text
V_i(C) = {
    m ∈ V_i :
    VISUAL_ON_COUNT(m) <= C
}
```

Example:

```text
Original valid routes:

28 ON
26 ON
24 ON
21 ON
18 ON
12 ON
 8 ON

CAP=24:
24, 21, 18, 12, 8

CAP=22:
21, 18, 12, 8

CAP=20:
18, 12, 8

CAP=18:
18, 12, 8
```

The hypothesis is that an intermediate cap may preserve enough alternative valid routes for robust training while avoiding both:

```text
dense ALL-ON-like supervision
```

and

```text
overly aggressive minimum-compute supervision.
```

---

# 2. Frozen source supervision

Use the original frozen GQA/TextVQA/ChartQA valid-route supervision that existed **before strict Pareto filtering**.

Datasets:

```text
GQA
TextVQA
ChartQA
```

Use the same:

```text
regenerated MCTS cache
selected max-50 valid-route view
image-group-disjoint split
UID identities
route validity criteria
```

as the previous full10 predictor experiments.

Do not:

```text
regenerate MCTS
rerun route search
change route validity
apply Pareto filtering
select minimum-ON routes only
```

The only label transformation is:

```text
ON-count <= CAP
```

---

# 3. Create four frozen cap manifests

Construct exactly four supervision manifests:

```text
cap24
cap22
cap20
cap18
```

For each sample and cap:

```text
retain route m iff:

route is valid
AND
VISUAL_ON_COUNT(m) <= CAP
```

Do not rank the surviving routes.

Do not preferentially select minimum-ON routes.

Do not perform additional diversity sampling.

Use all surviving routes from the existing frozen selected supervision, subject only to the existing maximum route representation if already imposed upstream.

---

# 4. Samples with no valid route under a cap

This must be handled explicitly.

For every cap, compute:

```text
number of positive samples with >=1 surviving route
number with 0 surviving routes
coverage fraction
```

Do not assign an empty BCE target.

Do not convert a zero-route sample to ALL-OFF or ALL-ON.

Do not invent a fallback route.

---

# 5. Primary matched-sample training population

Because lower caps may remove all valid routes for some samples, directly training each cap on a different input population would confound:

```text
route-cap effect
with
training-data composition
```

Therefore use a **common matched training/validation population** for the primary four-way comparison.

Define:

```text
COMMON_ELIGIBLE
=
samples having at least one valid route under CAP=18
```

Because CAP=18 is the strictest condition, every COMMON_ELIGIBLE sample also has at least one valid route under caps 20, 22, and 24.

Use the same COMMON_ELIGIBLE train UIDs and validation UIDs for all four models.

For each model, only the surviving route set differs.

Thus the controlled comparison is:

```text
same image-query samples
same predictor
same optimization
same initialization
different maximum allowed VISUAL_ON count in supervision
```

Also report cap-native coverage on the full original positive population, but do not use different sample populations for the primary training comparison.

If existing repository constraints make this impossible, stop and report the issue before substituting another design.

---

# 6. Pre-training geometry audit

Before training, produce one table:

```text
Metric                 cap24   cap22   cap20   cap18
---------------------------------------------------
full positive coverage
common-eligible count
routes/sample mean
routes/sample median
singleton %
>=2 routes %
>=5 routes %
mean route ON
median route ON
mean pairwise Hamming
bit entropy
ALL-ON presence
ALL-OFF presence
```

Also report separately for:

```text
GQA
TextVQA
ChartQA
```

This audit is required for interpretation but does not stop the authorized sweep unless there is an integrity failure.

---

# 7. BCE label-oracle audit

For each cap, before model training, compute the exact duplicated-BCE label oracle using the actual route weights consumed by training.

For sample `i`, layer `l`:

```text
q_i,l =
Σ_m alpha_i,m * m_l
/
Σ_m alpha_i,m
```

Decode with the exact deployed rule:

```text
q_i,l >= 0.5 → ON
```

Report:

```text
cap
BCE-oracle mean ON
ALL-ON %
ALL-OFF %
unique oracle masks
cap-valid Hit@1
original-valid Hit@1
nearest cap-valid Hamming
```

This is label-only evidence.

Do not use oracle outcomes to change the four authorized caps.

---

# 8. Predictor architecture

Use the existing full10 **Image+Question direct duplicated-BCE predictor** unchanged.

Use exactly the same:

```text
frozen Qwen3 question embeddings
frozen native Qwen2.5-VL projected visual rows
question/visual projections
28 learned layer queries
cross-attention
cross-layer encoder
direct 28-bit binary head
threshold decoder
```

Do not use:

```text
Question-only
P12 segment head
new vision encoder
new fusion module
route scorer
```

The purpose is to isolate the supervision cap.

---

# 9. BCE objective

Use exactly the existing POLAR-style duplicated-route BCE implementation.

For each surviving valid route, preserve the current route-example supervision semantics.

Since all four caps are below 28, ALL-ON cannot survive the cap filter, so the previous ALL-ON `0.3` weighting should normally be irrelevant.

Verify this rather than assuming it.

Do not add:

```text
compute loss
ON penalty
entropy regularization
sparsity penalty
RL
ranking loss
```

The cap itself is the only compute-control intervention.

---

# 10. Optimization

Use the same established full10 POLAR-style configuration:

```text
epochs          = 10
effective batch = 128
learning rate   = 5e-4
optimizer       = AdamW
scheduler       = cosine
warmup steps    = 10
seed            = same matched seed
```

Preserve the same weight decay and other optimizer details used by the previous full10 duplicated-BCE run.

If physical batch size must be smaller:

```text
use gradient accumulation
```

so that:

```text
effective batch size = 128
```

for all four runs.

---

# 11. Matched initialization

The four experiments must share identical initialization for every common parameter.

Before training, verify and record:

```text
shared initialization hash
seed
layer-query hash
cross-attention hash
cross-layer encoder hash
binary-head hash
visual projection hash
question projection hash
```

The only difference between experiments must be the cap-filtered supervision.

---

# 12. Parallel GPU execution

Launch all four trainings simultaneously on separate GPUs.

Preferred mapping:

```text
GPU 0 → CAP=24
GPU 1 → CAP=22
GPU 2 → CAP=20
GPU 3 → CAP=18
```

Use explicit isolated device assignment, e.g. equivalent to:

```text
CUDA_VISIBLE_DEVICES=0 → cap24
CUDA_VISIBLE_DEVICES=1 → cap22
CUDA_VISIBLE_DEVICES=2 → cap20
CUDA_VISIBLE_DEVICES=3 → cap18
```

Each process must have independent:

```text
output directory
checkpoint directory
stdout/stderr log
training history
evaluation outputs
```

Do not allow concurrent jobs to overwrite shared mutable artifacts.

Shared feature caches may be read-only.

---

# 13. No early stopping

Run all four models through all 10 epochs.

Do not stop because:

```text
validation loss worsens
ALL-OFF rises
mean ON becomes low
Hit@1 plateaus
another cap appears better
```

Only stop for a technical failure such as:

```text
NaN / Inf
corrupted data
incorrect supervision
gradient leak
checkpoint corruption
runtime failure
```

Do not change one cap's settings based on another cap's intermediate results.

---

# 14. Save every epoch

Mandatory.

Directory structure should resemble:

```text
outputs/binary_cap_sweep_v1/

    cap24/
        epoch_01/
        ...
        epoch_10/

    cap22/
        epoch_01/
        ...
        epoch_10/

    cap20/
        epoch_01/
        ...
        epoch_10/

    cap18/
        epoch_01/
        ...
        epoch_10/
```

Each checkpoint must include:

```text
model state
optimizer state
scheduler state
epoch
global step
config
seed
manifest hash
checkpoint hash
```

Do not overwrite earlier checkpoints.

---

# 15. Validate every epoch

Evaluate every epoch on the frozen matched validation population.

Report for every cap/epoch:

```text
train BCE loss
validation BCE loss

cap-valid Hit@1
original-valid Hit@1
nearest cap-valid Hamming
nearest original-valid Hamming

mean VISUAL_ON
ALL-ON %
ALL-OFF %

unique masks
mask entropy
```

Also report dataset-wise:

```text
GQA
TextVQA
ChartQA
```

for:

```text
original-valid Hit@1
cap-valid Hit@1
mean ON
ALL-OFF %
unique masks
```

---

# 16. Full trajectory table

Produce a 10-epoch trajectory for each cap.

Example:

```text
CAP=24

Epoch | Val loss | Cap Hit | Orig Hit | Hamming | Mean ON | ALL-OFF | Unique
1
2
...
10
```

Repeat for:

```text
24
22
20
18
```

Also produce one compact cross-cap trajectory figure/table.

We specifically want to observe whether lower caps cause a continuum such as:

```text
high cap
→ denser but safer routes

lower cap
→ more aggressive computation reduction

too-low cap
→ ALL-OFF / performance collapse
```

Do not assume that this pattern must occur.

---

# 17. Checkpoint selection

Use the same frozen validation checkpoint-selection policy as the prior full10 duplicated-BCE experiment.

Do not invent a different selection rule after seeing the sweep.

Document exactly which metric/tie-breaking rule selects the checkpoint.

External execution results must never be used for checkpoint selection.

Always preserve and report:

```text
selected checkpoint
lowest-val-loss checkpoint
epoch 10
```

even if they differ.

---

# 18. External evaluation

After all four trainings are complete and checkpoints are selected, run the same frozen external Qwen execution evaluation used for the previous full10 BCE evaluation.

Evaluate the selected checkpoint for:

```text
CAP=24
CAP=22
CAP=20
CAP=18
```

Use the same current live ALL-ON baseline.

Do not substitute historical cached correctness for current live execution.

---

# 19. Parallel external evaluation

If resources permit, execute the four selected routers in parallel using the same GPU mapping:

```text
GPU 0 → cap24 external eval
GPU 1 → cap22 external eval
GPU 2 → cap20 external eval
GPU 3 → cap18 external eval
```

Ensure that benchmark outputs are isolated per cap.

Do not reuse one model's execution result for another model even when predicted masks match.

---

# 20. External benchmark suite

Use exactly the same frozen external suite as the previous full10 evaluation.

At minimum retain all currently active benchmarks such as:

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

plus the same existing pooled reporting groups.

Do not modify benchmark membership for this sweep.

---

# 21. Required external metrics

For every cap and benchmark report:

```text
N
ALL-ON baseline accuracy
router accuracy
accuracy delta
ratio vs ALL-ON

Harm / C→W
Rescue / W→C

unchanged correct
unchanged wrong

mean ON layers
ON reduction
ALL-ON %
ALL-OFF %
unique predicted masks
behavior-changing executions
```

Actual execution is the primary behavioral evaluation.

Cached cap-valid Hit@1 is only a diagnostic.

---

# 22. Cross-cap accuracy–compute frontier

Produce a primary comparison table:

```text
Cap | External Accuracy | Delta vs FULL | Mean ON | ON Reduction | Harm | Rescue
```

overall where pooling is valid and separately per benchmark.

The main scientific question is:

> **Is there an intermediate supervision cap that gives a substantially better accuracy–visual-compute tradeoff than both unfiltered supervision and strict Pareto supervision?**

---

# 23. Compare against previous endpoints

Include the existing historical endpoints:

```text
Unfiltered duplicated BCE
Strict Pareto duplicated BCE
ALL-ON / FULL
```

alongside:

```text
Cap24
Cap22
Cap20
Cap18
```

for at least:

```text
router accuracy
mean ON
ALL-OFF fraction
unique masks
Harm
Rescue
```

The conceptual continuum should therefore be:

```text
Unfiltered
    ↓
Cap24
    ↓
Cap22
    ↓
Cap20
    ↓
Cap18
    ↓
Strict Pareto
```

Do not force monotonic interpretation if the data disagree.

---

# 24. Important interpretation rule

A lower mean ON count is not automatically better.

For example:

```text
Mean ON = 8
Accuracy collapses
```

is not a successful router.

The target is:

```text
high retained task performance
+
meaningful visual-layer reduction
```

Pay particular attention to:

```text
Harm = FULL-correct → router-wrong
Rescue = FULL-wrong → router-correct
```

A lower-cap model that mainly increases Harm is not useful.

---

# 25. Key hypotheses

## H1 — Cap24 remains too permissive

Possible behavior:

```text
many valid routes remain
BCE still learns dense masks
small compute reduction
```

## H2 — Intermediate caps 22/20 may provide the best tradeoff

Possible behavior:

```text
remove expensive supervision
retain multiple alternatives
avoid strict-Pareto ALL-OFF behavior
```

## H3 — Cap18 may become overly aggressive

Possible behavior:

```text
less supervision coverage
more sparse predictions
higher Harm / accuracy degradation
```

These are hypotheses only.

Do not bias checkpoint selection or reporting toward them.

---

# 26. Analyze cap-native supervision coverage

Although training uses the matched common-eligible population, report how useful each cap would be on the complete original positive training corpus.

For each cap report:

```text
number / fraction of original positive samples
with at least one route <= cap
```

This is important because a cap with good routing performance but very low supervision coverage may be less practical for later large-scale training.

---

# 27. Do not change during this experiment

Do not:

```text
apply Pareto filtering
use min-ON-only routes
change MCTS
generate more MCTS routes
change correct/wrong ratio
change predictor architecture
change input modality
use Question-only
change BCE to NLL
add compute regularization
add route ranking
add beam search
use RL
change decoder threshold
tune individual caps separately
use external evaluation for model selection
```

This experiment must isolate:

```text
maximum VISUAL_ON count allowed in BCE supervision.
```

---

# 28. Required final report

Create:

```text
binary_bce_layer_cap_24_22_20_18_full10_results.md
```

The report must contain:

1. exact source supervision;
2. exact cap filtering definition;
3. common-eligible matched train/val counts;
4. full-population cap coverage;
5. cap24/22/20/18 label geometry;
6. BCE label-oracle geometry per cap;
7. initialization/config hashes;
8. 10-epoch training trajectory for every cap;
9. validation trajectory for every cap;
10. selected checkpoint identities;
11. external execution results;
12. per-benchmark Harm/Rescue;
13. accuracy–ON frontier;
14. comparison with unfiltered BCE;
15. comparison with strict Pareto BCE;
16. explicit recommendation of which cap, if any, deserves further study.

---

# 29. Final decision

End with one of these interpretations:

### Outcome A — clear intermediate sweet spot

One or more caps retain near-FULL performance while materially reducing mean ON and outperform both unfiltered and strict-Pareto supervision.

### Outcome B — monotonic accuracy/compute tradeoff only

Lower caps simply reduce compute while progressively damaging performance, with no especially useful operating point.

### Outcome C — all caps remain too dense

Even cap18 remains close to the unfiltered BCE behavior.

### Outcome D — all caps become too aggressive

Even cap24 causes severe sparsity/Harm similar to strict Pareto.

### Outcome E — cap supervision does not resolve route selection

Masks change substantially with cap, but no cap produces useful executed accuracy–compute behavior.

Do not launch another experiment automatically after this decision.

---

# 30. Execution principle

> **Using the same frozen GQA/TextVQA/ChartQA MCTS supervision and the same Image+Question duplicated-BCE predictor, train four matched 10-epoch models whose only difference is whether valid training routes are restricted to at most 24, 22, 20, or 18 VISUAL_ON layers; run all four in parallel on separate GPUs, save every epoch, and execute the selected checkpoints on the same frozen external benchmark suite to identify whether an intermediate supervision budget produces a useful accuracy–compute frontier between unfiltered and strict-Pareto training.**
