# Stage-2 V1 Training Plan — Revised
## Shared READ/WRITE Router, 698-Sample Pilot First

## 1. Objective

Train the first Stage-2 router using the **cleanest current supervision first**, while explicitly treating the 698 SINGLE_FIXABLE W samples as a **V1 generalization pilot**, not as the final sufficient dataset by assumption.

The central question is:

> **Given a frozen Stage-1 handoff, can a small shared Stage-2 router learn post-trigger layer-wise visual actions from current routed visual/text states, and does that learned policy transfer under free rollout?**

The first experiment should use:

```text
Corpus A: 39 triggered Dense-C preservation samples
Corpus B: 698 SINGLE_FIXABLE Dense-W samples
```

Do not add MCTS-only samples until the V1 result tells us whether additional data/trajectory diversity is actually needed.

---

# 2. Why Start with 698 W Samples

The 698 W samples are not the entire dense-wrong pool.

They are the subset satisfying:

```text
Dense FULL = wrong
Stage-1 triggers
a replay-valid single-intervention corrective route exists
```

They are therefore the cleanest current Stage-2 supervision.

The 698 samples contain:

```text
7,628 replay-valid successful single routes
mean 10.93 routes/sample
median 7 routes/sample
```

so they provide substantial route/state diversity, although they are still only 698 independent base samples.

Important interpretation:

```text
698 is sufficient for a V1 learning/generalization pilot
but not assumed sufficient for the final router
```

Do not treat 7,628 routes as 7,628 independent samples.

---

# 3. Staged Data Strategy

Use a strictly staged progression.

## V1 — clean single-intervention supervision

Train with:

```text
Corpus A
+
Corpus B
```

where:

```text
Corpus A = 39 triggered-C preservation samples
Corpus B = 698 SINGLE_FIXABLE W samples
```

Purpose:

> Determine whether the simple router formulation itself learns and transfers.

## V1.5 — only if V1 indicates data/diversity limitation

Add:

```text
209 MCTS_ONLY_FIXABLE W samples
```

so the corrective-positive W population becomes:

```text
698 + 209 = 907
```

Purpose:

> Determine whether additional corrective states and multi-layer trajectory diversity improve generalization.

Do not automatically proceed to V1.5 just because 698 sounds small.

First diagnose V1.

---

# 4. Frozen Components

Freeze:

```text
Qwen2.5-VL backbone
Stage-1 failure predictor
Stage-1 threshold
Stage-1 trigger rule
Stage-1 trigger map
```

Train only:

```text
Stage-2 router
```

Do not jointly optimize Stage-1 and Stage-2.

Stage-1 defines:

```text
WHEN corrective mode begins
```

Stage-2 defines:

```text
WHAT action to take at each post-trigger layer
```

---

# 5. Runtime Contract

For sample `i`, frozen Stage-1 first triggers at:

```text
l_i*
```

Then:

```text
j < l_i*:
    FULL
    Stage-2 inactive
```

and:

```text
j >= l_i*:
    Stage-2 active
```

Deployment:

```text
FULL prefix
    ↓
Stage-1 trigger at l*
    ↓
Stage-2 handoff

L_l*     → predict action → execute
L_l*+1   → predict action from new routed state → execute
...
L27
```

Every decision after the first non-FULL action must use the **actual routed state**, not the original dense trajectory state.

---

# 6. Stage-2 Action Space

Predict:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

with the existing executor semantics:

| Action | READ | WRITE |
|---|---:|---:|
| FULL | 1 | 1 |
| READ_ONLY | 1 | 0 |
| WRITE_ONLY | 0 | 1 |
| IGNORE | 0 | 0 |

---

# 7. Router Input Principle

Do not use Stage-1 latent features in V1.

Do not use learnable layer embeddings in V1.

The router should rely on the **current routed computation state**.

At post-trigger layer `j`, define:

```text
V_j = current routed visual-token states
T_j = current routed text/control-token states
q_j = last current text/control-token state
```

The router contains separate READ and WRITE branches because the two decisions have different semantics.

---

# 8. READ Branch

READ asks:

> **Should the current text/reasoning state directly read visual evidence at this layer?**

Use:

```text
current text/query state
+
current visual-token states
```

Minimal formulation:

```text
z_R = CrossAttention(
    query = q_j,
    key   = V_j,
    value = V_j
)
```

Use a lightweight shared cross-attention module.

Do not use layer-specific READ parameters.

---

# 9. WRITE Branch

WRITE asks:

> **Should the current visual representation continue to be updated/contextualized at this layer?**

Use one shared learned visual-pooling query:

```text
q_W
```

for all layers:

```text
z_W = CrossAttention(
    query = q_W,
    key   = V_j,
    value = V_j
)
```

Do not use:

```text
layer-specific WRITE queries
explicit layer embeddings
```

The same WRITE pooling query must operate across all post-trigger layers.

---

# 10. Action Head

Fuse:

```text
u_j = [z_R ; z_W]
```

Then:

```text
small MLP
→ 4 logits
```

Recommended:

```text
Linear
→ GELU
→ optional LayerNorm/Dropout
→ Linear(4)
```

Keep the router small.

The same Stage-2 router is shared across all post-trigger layers.

---

# 11. No Explicit Layer Embedding

Exclude:

```text
learnable layer embeddings
one-hot layer identity
sinusoidal depth embeddings
layer-specific READ query
layer-specific WRITE query
```

V1 should test:

> Is the current routed hidden state itself enough to determine the action?

If V1 clearly underfits, explicit depth information may be tested later as one bounded ablation.

---

# 12. V1 Training Corpus

## Corpus A — Preservation

```text
39 triggered Dense-C train samples
```

For each one:

```text
trigger layer l*
L_l* ... L27 = FULL
```

These are the correct samples Stage-2 actually encounters because Stage-1 falsely admitted them.

Purpose:

```text
C→C preservation
```

Dense-C samples that never trigger are not Stage-2 training examples.

---

## Corpus B — Simple correction

```text
698 SINGLE_FIXABLE Dense-W train samples
```

Each has at least one replay-valid single-intervention W→C trajectory.

There are:

```text
7,628 successful single routes
```

but training must remain sample-balanced.

---

# 13. Sample-Balanced Route Sampling

For each W draw:

```text
1. choose W UID uniformly from the 698 samples
2. choose one successful single route uniformly for that UID
```

This prevents a sample with 60 discovered successful routes from receiving 60× the weight of a sample with one successful route.

Resample the route across epochs.

---

# 14. Only Post-Trigger States Are Eligible

Stage-2 is inactive before the trigger.

Therefore:

```text
j < l*
```

must never be used as Stage-2 supervision.

For a route:

```text
trigger = L5
corrective intervention = L18
```

eligible states are:

```text
L5 ... L27
```

and not:

```text
L0 ... L4
```

---

# 15. Positive-Anchored Random-4 Sampling for W

Do not train on the entire suffix.

For each selected successful W route, sample approximately four states:

```text
1. mandatory corrective non-FULL state

2. one FULL state after trigger but before intervention
   if available

3. one FULL state after intervention
   if available

4. one additional random FULL state
   from the remaining post-trigger suffix
```

Boundary cases may backfill from any available post-trigger FULL state.

Expected per-W draw:

```text
1 non-FULL
3 FULL
```

This avoids the naive ~96% FULL state distribution while preserving the important behavior:

```text
trigger
→ often wait with FULL
→ intervene later
```

---

# 16. Sampling for Triggered-C

For each triggered-C draw:

```text
choose one of the 39 C samples uniformly
sample up to 4 FULL states from its post-trigger suffix
```

Do not use pre-trigger C states.

---

# 17. Initial C:W Mixture

Start with:

```text
C:W sample-draw ratio = 1:2
```

so approximately:

```text
1/3 of sample draws = preservation C
2/3 = corrective W
```

The 39 C samples will be repeated heavily.

This is intentional because natural sample frequency provides too little preservation supervision.

Do not use focal loss yet.

Do not use class-weighted CE yet.

---

# 18. Initial Loss

Use plain:

```text
4-way cross entropy
```

First address imbalance through the sampler.

Only if the V1 smoke/full run still collapses toward FULL should later experiments consider:

```text
class-weighted CE
focal loss
```

Do not introduce them preemptively.

---

# 19. Multi-Valid Correction Handling

Do not collapse all 7,628 routes into one deterministic route per sample.

For V1:

```text
sample one successful route per W sample per draw/epoch
```

This exposes alternative valid corrections stochastically.

Do not implement a set-valued loss in the first V1.

Preserve all route metadata so it can be added later if label inconsistency becomes a diagnosed problem.

---

# 20. Training Progression

Proceed in three steps.

## Step A — implementation smoke

Tiny subset.

Check:

```text
forward/backward
READ branch shapes
WRITE branch shapes
action mapping
sampler correctness
frozen backbone gradients = none
frozen Stage-1 gradients = none
```

## Step B — small overfit smoke

Use approximately:

```text
32-64 W samples
plus a repeated subset/all of the 39 C preservation samples
```

Question:

> Can the router fit both FULL and non-FULL actions under the proposed sampler?

Track:

```text
FULL recall
non-FULL recall
READ_ONLY recall
WRITE_ONLY recall
IGNORE recall
predicted action distribution
```

Do not proceed if it predicts almost all FULL.

## Step C — full V1 train

Use all:

```text
698 W
39 C
```

with the frozen sampling contract.

---

# 21. V1 Data-Sufficiency Diagnosis

Do not decide that 698 is insufficient from sample count alone.

Use the following diagnostic logic after training.

## Case A — train/overfit itself fails

If the router cannot learn even the small overfit split:

```text
not a data-size problem yet
```

Investigate:

```text
architecture
feature extraction
sampling
label ambiguity
action semantics
```

before adding more samples.

## Case B — train fits, validation action/generalization fails

If training behavior is good but validation transfer is poor:

```text
data diversity / generalization becomes a plausible bottleneck
```

Then authorize V1.5:

```text
+ 209 MCTS_ONLY_FIXABLE samples
```

## Case C — free rollout fails despite decent teacher-forced generalization

Then suspect:

```text
exposure shift
trajectory compounding
policy instability
```

Do not assume more labels alone will fix it.

## Case D — V1 gives positive net correction

Then 698 was sufficient to establish the method direction.

Only add MCTS labels if they are expected to improve coverage meaningfully.

---

# 22. Validation Free Rollout

After full training, evaluate:

```text
Frozen Stage-1
+
learned Stage-2
```

on validation.

Procedure:

```text
FULL while Stage-1 does not trigger

if no trigger:
    stay FULL

if trigger at l*:
    activate Stage-2

for j = l* ... 27:
    obtain current routed V_j and T_j
    predict action
    execute action
    use resulting next state
```

Evaluate final answer using the frozen LMMS-Eval contract.

Teacher-forced action accuracy is secondary.

Free rollout is primary.

---

# 23. Primary Metrics

Report:

```text
Dense baseline accuracy
Stage-1 + Stage-2 accuracy
Δ accuracy
```

Transitions:

```text
W→C
W→W
C→C
C→W
```

Primary net effect:

```text
Net correction count = W→C - C→W
```

Also:

```text
W→C rescue rate
C→C preservation rate
```

The desired V1 behavior is conservative:

```text
some W→C
with low C→W
```

W→C does not need to be maximized at the cost of preservation.

---

# 24. Router Behavior Metrics

During free rollout record:

```text
fraction of triggered samples with any non-FULL
# non-FULL actions/sample
first non-FULL layer
trigger-to-first-non-FULL delay
FULL fraction after trigger
READ_ONLY count
WRITE_ONLY count
IGNORE count
```

Compare with training-corpus statistics.

---

# 25. Collapse Checks

Check explicitly for:

```text
FULL collapse
immediate-intervention collapse
single-action collapse
teacher-forcing vs free-rollout exposure gap
```

Examples:

```text
FULL collapse:
almost all actions = FULL

immediate-intervention collapse:
non-FULL almost always at trigger layer

single-action collapse:
almost every intervention = IGNORE
```

Stop and diagnose before increasing data/model complexity.

---

# 26. Dataset Breakdown

Report validation results separately for:

```text
GQA
ChartQA
TextVQA
```

At minimum:

```text
triggered samples
W→C
C→W
C→C
final accuracy change
non-FULL action frequency
```

Do not train dataset-specific routers.

---

# 27. V1.5 Data Expansion — Not Run Automatically

Only if V1 points to data/generalization limitation, add:

```text
209 MCTS_ONLY_FIXABLE W samples
```

Then corrective-positive W bases become:

```text
907
```

V1.5 should preserve provenance:

```text
single-derived labels
MCTS-derived labels
```

and compare directly against V1.

Do not mix in unresolved W or historical labels without a separate plan.

---

# 28. Why Not Immediately Add Historical W→C Pools

Historical binary/MCTS W→C labels were generated under different routing/search contracts.

The current Stage-2 operates only after the frozen Stage-1 trigger.

Therefore the clean training distribution is:

```text
current frozen trigger
+
current routed post-trigger state
+
current executor-valid corrective action
```

Do not expand to historical W→C samples unless V1/V1.5 shows a clear data bottleneck and those labels are revalidated under the current trigger/runtime contract.

---

# 29. Required Outputs

Use:

```text
analysis/dense_failure_stage2/v1_training_revised/
```

Create:

```text
protocol.md

config/
    stage2_v1_config.yaml
    sampler_config.yaml

smoke/
    implementation_smoke.md
    overfit_metrics.csv
    overfit_action_distribution.csv

training/
    train_log.jsonl
    checkpoint_manifest.json
    selected_checkpoint.json

teacher_forced/
    action_metrics.csv
    confusion_matrix.csv
    predicted_action_distribution.csv

free_rollout/
    per_sample_results.jsonl
    overall_metrics.json
    transition_counts.csv
    action_behavior.csv
    dataset_breakdown.csv

diagnostics/
    data_sufficiency_assessment.md
    collapse_check.json

figures/
    training_loss.png
    action_recall.png
    predicted_action_distribution.png
    transition_counts.png
    dataset_accuracy_change.png
    trigger_to_first_nonfull_delay.png

summaries/
    stage2_v1_summary.md
    stage2_v1_decision.md

artifact_manifest.json
```

---

# 30. `stage2_v1_decision.md` Must Answer

1. Did implementation smoke pass?
2. Did small-overfit smoke pass?
3. Did the router avoid FULL collapse?
4. Did it learn all supported non-FULL actions at nontrivial rates?
5. What are validation W→C and C→W?
6. What is C→C preservation?
7. What is net validation accuracy change?
8. Does the learned policy wait with FULL after trigger when appropriate?
9. Does teacher-forced performance transfer to free rollout?
10. Is failure dominated by:
    - architecture,
    - class/sampling balance,
    - label ambiguity,
    - exposure shift,
    - or likely data diversity?
11. Is 698-sample V1 sufficient to continue?
12. Is V1.5 with the additional 209 MCTS-only W samples justified?

---

# 31. Stop Rule

STOP after V1 validation free rollout and data-sufficiency diagnosis.

Do not automatically:

```text
add the 209 MCTS samples
add historical W→C labels
add layer embeddings
add Stage-1 representations
add focal loss
add class-weighted loss
jointly train Stage-1
retune Stage-1 threshold
run test
```

Every increase in complexity/data should respond to an observed V1 failure mode.

---

# 32. Final V1 Definition

```text
Stage-1:
    frozen
    decides WHEN

Stage-2:
    shared across all post-trigger layers
    decides WHAT

READ branch:
    current text/query state
        attends to
    current routed visual tokens

WRITE branch:
    shared learned visual query
        attends to
    current routed visual tokens

Action head:
    [READ feature ; WRITE feature]
        → small MLP
        → FULL / READ_ONLY / WRITE_ONLY / IGNORE

Training data:
    39 triggered-C preservation samples
    698 single-fixable W samples

Sampling:
    sample-balanced W UID
    one successful route/UID
    mandatory corrective state
    + 3 post-trigger FULL states
    C:W draws = 1:2

Loss:
    plain CE

No:
    layer embeddings
    Stage-1 latent
    MCTS-only labels
    focal loss
```

The first experiment asks only:

> **Can the simplest state-dependent Stage-2 policy learn from the 698 clean corrective samples and generalize under real free rollout?**

If yes, keep the method simple.

If no, diagnose the reason before increasing data or model complexity.
