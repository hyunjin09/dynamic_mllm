We have completed the bounded P11–P13 diagnostics.

The next task is intentionally different from the previous smoke-test sequence:

> **Run the full predictor training for 10 epochs using the released POLAR-style optimization setting, for both Question-only and Image+Question predictors in parallel, and preserve every epoch checkpoint so that we can inspect the complete learning trajectory.**

This run is explicitly authorized to proceed through all 10 epochs unless there is a technical failure.

Do not stop early because the predictor temporarily collapses to ALL-ON.

The purpose is to determine whether the collapse observed in the previous two-epoch smokes persists after a POLAR-scale optimization schedule.

---

# 1. Scientific question

Previous bounded results showed:

```text
P11 Question-only:
probability-level question signal exists
but selected top-1 ≈ ALL-ON

P13 Image+Question:
visual information further improves valid-set probability
but selected top-1 = ALL-ON
```

However, those runs used only two epochs.

POLAR's released representative training configuration uses approximately:

```text
epochs       = 10
batch size   = 128
learning rate = 5e-4
optimizer    = AdamW
scheduler    = cosine
warmup steps = 10
max valid routes per sample = 50
```

Therefore, before abandoning the direct exact-valid-set predictor, run a full 10-epoch optimization trajectory under the closest matched configuration.

The primary question is:

> **Does longer POLAR-style optimization eventually convert the probability-level input signal into useful nonconstant complete-mask predictions?**

---

# 2. Two training runs

Run exactly two primary predictor trainings.

## Run A — Question-only

Use the established P11 direct predictor:

```text
Question
↓
Frozen Qwen3-Embedding-0.6B
↓
question token projection
↓
28 learned layer queries
↓
cross-attention
↓
cross-layer encoder
↓
direct factorized 28-bit binary head
↓
VISUAL_ON / TEXT_ONLY logits
```

No image representation is exposed.

---

## Run B — Image+Question

Use the established P13 multimodal predictor:

```text
Question tokens
+
native projected Qwen2.5-VL visual rows
↓
projection to common predictor dimension
↓
concatenate visible token streams
↓
same 28 layer queries
↓
same cross-attention
↓
same cross-layer encoder
↓
same direct factorized 28-bit binary head
```

Use the exact frozen visual feature definition from P13:

```text
projected Qwen2.5-VL visual-token rows
entering decoder layer 0
```

Do not pool them.

Do not introduce a new vision model.

---

# 3. Run the two trainings in parallel

Launch the Question-only and Image+Question experiments concurrently if two GPUs are available.

Preferred layout:

```text
GPU 0 → Question-only
GPU 1 → Image+Question
```

or equivalent isolated devices.

Each run must have:

```text
independent output directory
independent logs
independent checkpoint directory
same data identities
same optimizer settings
same random seed
same training budget
```

Do not allow the two processes to overwrite shared outputs.

Do not change the training configuration of one run based on intermediate results from the other.

---

# 4. Full training data

Use the **full existing predictor training split**, not the previous 300-input smoke subset.

Use the same frozen regenerated labels and image-group-disjoint split established in P11/P13.

Expected positive-route population from the existing frozen manifest is approximately:

```text
Train positive inputs:      6,043
Validation positive inputs:   874
```

Verify these counts against the current frozen manifest before training.

Use:

```text
GQA
TextVQA
ChartQA
```

only.

Do not:

```text
regenerate MCTS labels
modify train/validation membership
move zero-positive rows into the positive objective
inspect test outcomes
```

Preserve the existing image-group split exactly.

---

# 5. Valid-route supervision

Use exactly the existing selected MCTS valid-route sets.

For every positive input:

```text
maximum valid masks = 50
```

Use the same deterministic diverse max-50 route-selection policy already frozen in P11–P13.

Do not replace it with:

```text
shortest-50
highest-score-50
new random subset
new MCTS sampling
```

unless the frozen current implementation already defines otherwise.

---

# 6. Training objective

Use **our exact one-of-valid-set NLL**, not POLAR duplicated-path BCE.

For input `x` with selected valid-mask set `V_x`:

```text
P_theta(m | x)
=
product over 28 layers of
Bernoulli(m_l ; p_l)
```

and:

```text
L(x)
=
-log sum_{m in V_x} w_m P_theta(m | x)
```

Implement via the already validated stable `logsumexp` code.

Do not change the loss implementation from the validated P11/P13 exact-set version.

---

# 7. POLAR-compatible route weighting

Preserve the P11/P13 weighting rule.

Default:

```text
w_m = 1.0
```

If:

```text
ALL-ON is valid
AND
a selected valid route with fewer than 28 VISUAL_ON layers exists
```

then:

```text
w_ALL-ON = 0.3
w_other  = 1.0
```

Use the same within-input normalization as P11/P13.

Do not add:

```text
extra sparsity loss
compute penalty
entropy regularization
route-length penalty
ON-count penalty
RL reward
```

The purpose of this experiment is not to invent a new objective.

---

# 8. POLAR-matched optimization configuration

Use the released POLAR-style representative setting:

```text
epochs          = 10
batch size      = 128
learning rate   = 5e-4
optimizer       = AdamW
scheduler       = cosine
warmup steps    = 10
max valid masks = 50
```

Preserve the existing predictor-compatible AdamW weight decay unless there is a directly frozen POLAR-equivalent value already used in the repository.

If the effective batch size differs because of memory constraints, use gradient accumulation so that:

```text
effective batch size = 128
```

for both experiments.

Do not silently lower the effective batch.

Record:

```text
physical batch size
gradient accumulation
effective batch size
number of optimizer steps
warmup steps
total scheduler steps
```

in the final report.

---

# 9. Initialization

Use matched initialization between the two runs wherever architectures overlap.

Requirements:

```text
same seed
same layer-query initialization
same cross-attention initialization
same cross-layer encoder initialization
same direct binary-head initialization
```

The Image+Question-specific visual projection is the only additional component.

Record:

```text
full initialization hashes
shared-state hashes
seed
```

before training.

---

# 10. No early stopping

This is critical.

Previous experiments selected checkpoints after two epochs using a frozen validation rule.

This experiment instead asks whether **longer optimization changes the trajectory**.

Therefore:

> **Run all 10 epochs.**

Do not terminate because:

```text
ALL-ON fraction is high
Hit@1 plateaus
validation NLL rises temporarily
route diversity temporarily decreases
```

Only stop for:

```text
NaN / Inf
corrupt checkpoint
data leakage
gradient leak into frozen components
unrecoverable technical/runtime failure
```

If one run technically fails while the other continues, repair the failure without changing scientific settings and resume/restart it as needed.

---

# 11. Save every epoch checkpoint

This is mandatory.

After every epoch, save a complete predictor checkpoint.

Required layout should resemble:

```text
question_only/
    epoch_01/
    epoch_02/
    ...
    epoch_10/

image_question/
    epoch_01/
    epoch_02/
    ...
    epoch_10/
```

Each checkpoint must contain enough information to reproduce validation decoding:

```text
model state
optimizer state
scheduler state
epoch
global step
config
seed
checkpoint hash
```

Do not overwrite earlier epochs.

---

# 12. Validate every epoch

After each epoch, evaluate on the same frozen full validation split.

For every checkpoint report:

```text
train exact set-NLL
validation exact set-NLL
cached valid-set Hit@1
cached valid-set Hit@5 if already supported
nearest-valid Hamming
number of unique top-1 masks
ALL-ON fraction
ALL-OFF fraction
mean VISUAL_ON layers
top-1 mask entropy
```

Also report dataset-wise:

```text
GQA
TextVQA
ChartQA
```

for at least:

```text
Hit@1
Hamming
ALL-ON fraction
mean ON
```

---

# 13. Preserve the entire trajectory

The primary output is not only one best checkpoint.

Produce a 10-epoch trajectory table for both runs.

Example:

```text
Epoch | Val NLL | Hit@1 | Hamming | Unique | ALL-ON | Mean ON
1
2
3
...
10
```

We specifically want to know whether the model moves through phases such as:

```text
early:
constant ALL-ON

middle:
diversity emerges

later:
route quality improves or deteriorates
```

Do not hide non-selected epochs.

---

# 14. Checkpoint selection after all 10 epochs

Only after all ten checkpoints exist, identify several checkpoints separately.

## A. Best validation Hit@1 checkpoint

Use the established route-quality hierarchy:

```text
maximize Hit@1
then Hit@5 where available
then minimize nearest-valid Hamming
then minimize validation set-NLL
then earlier epoch
```

## B. Best validation set-NLL checkpoint

Identify independently.

## C. Lowest ALL-ON / highest-diversity checkpoint

Diagnostic only.

Do not call this the "best model" unless route-quality metrics support it.

## D. Final epoch 10 checkpoint

Always preserve and report it.

This separation matters because previous results showed that diversity could increase while route quality became worse.

---

# 15. Conditioning diagnostics

At least for:

```text
best Hit@1 checkpoint
best set-NLL checkpoint
epoch 10
```

run the existing aligned/shuffled diagnostics.

## Question-only

Compare:

```text
aligned question
vs
within-dataset shuffled question
```

## Image+Question

Reuse P13's four conditions:

```text
aligned image + aligned question
aligned image + shuffled question
shuffled image + aligned question
shuffled image + shuffled question
```

Report:

```text
set-NLL
Hit@1
nearest Hamming
unique masks
ALL-ON fraction
mean ON
```

Use the existing frozen deterministic shuffle permutations where possible.

Do not create outcome-dependent shuffles.

---

# 16. Primary comparisons

At the end compare:

```text
constant ALL-ON
Question-only full training
Image+Question full training
cached MCTS oracle
```

Also retain the P11/P13 two-epoch results for historical reference.

The critical questions are:

```text
1. Does 10-epoch training break ALL-ON collapse?

2. If route diversity emerges, does Hit@1 / Hamming improve?

3. Does Image+Question outperform Question-only after sufficient optimization?

4. Does aligned input still outperform shuffled input?

5. Do later checkpoints move toward genuinely sample-specific masks,
   or merely diverse but poor masks?
```

---

# 17. Actual Qwen execution

Do not execute every epoch through Qwen unless runtime is negligible.

After the full 10-epoch training and validation trajectory is available, run actual binary Qwen execution on the existing frozen 60-record evaluation manifest for the following checkpoints:

```text
Question-only:
- best Hit@1
- epoch 10

Image+Question:
- best Hit@1
- epoch 10
```

If best-Hit@1 and epoch-10 are identical, execute only once.

Optionally execute the best-set-NLL checkpoint if it differs materially and remains prospectively interpretable.

Use exactly the same frozen 60 records as P11/P12:

```text
per dataset:
10 FULL-correct
10 FULL-wrong / MCTS-fixable
```

Report:

```text
accuracy
W→C
C→W
unchanged correct
unchanged wrong
mean VISUAL_ON
ALL-ON fraction
cached-mask count
uncached-mask count
uncached-mask accuracy
```

Do not choose additional checkpoints based on their 60-record execution outcomes.

---

# 18. Important interpretation

A successful long run is **not** merely:

```text
ALL-ON decreases
```

We already saw in P13 epoch 2 that diversity can increase while route quality worsens.

Useful evidence requires:

```text
nonconstant predictions
+
improved route quality
+
preferably useful execution
```

The strongest desired outcome is:

```text
Hit@1 > constant ALL-ON baseline
nearest-valid Hamming < ALL-ON baseline
ALL-ON fraction materially < 100%
W→C > 0
C→W controlled
mean VISUAL_ON < 28
```

---

# 19. Do not change during this experiment

Do not:

```text
change to P12 segment head
add route scorer/ranker
add beam search
add autoregressive route generation
add GRPO/PPO/RL
regenerate MCTS routes
change ALL-ON weight
add compute regularization
change visual features
pool image features
add a new encoder
tune threshold using validation outcomes
modify train/validation split
stop training early
```

If results are poor, finish the authorized 10 epochs and report them.

Do not repair scientific failure by changing the method mid-run.

---

# 20. Required final report

Create:

```text
reports/binary_polar_full10_polar_matched_results.md
```

The report must contain:

1. exact training configuration;
2. data counts and frozen split;
3. initialization hashes;
4. Question-only 10-epoch trajectory;
5. Image+Question 10-epoch trajectory;
6. dataset-wise validation trajectories;
7. checkpoint hashes for every epoch;
8. best-Hit@1 / best-NLL / final checkpoints;
9. aligned-vs-shuffled diagnostics;
10. constant ALL-ON comparison;
11. actual 60-record Qwen execution;
12. Question-only vs Image+Question comparison;
13. whether longer POLAR-style optimization breaks constant collapse;
14. whether any resulting diversity is actually useful;
15. recommendation for the next research step.

Preserve all P10–P13 artifacts.

---

# 21. Execution principle

> **Run the Question-only and Image+Question direct exact-valid-set predictors for the full 10-epoch POLAR-style optimization schedule, save every epoch, and use the complete trajectory to determine whether the two-epoch ALL-ON collapse was an optimization-budget artifact or a persistent limitation of direct route generation.**
