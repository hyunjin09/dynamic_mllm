# Persistent Corrective Supervision: Matched Router Discriminator Plan

## 1. Goal

Both current four-action routing substrates have collapsed to essentially all-`FULL` deployment:

- **POLAR-style upfront router**
- **Online state-conditioned four-action router**

The next experiment should answer one discriminating question:

> **If corrective W2C supervision is supplied persistently across training, can either router generalize the decision to leave `FULL` on held-out samples while preserving C2C correctness?**

This experiment is not intended to introduce a new router architecture.

The purpose is to distinguish:

1. insufficient corrective supervision mass;
2. lack of held-out predictability/generalization;
3. an actual architecture limitation.

---

## 2. Why This Experiment Is Needed

### A1: Capacity exists under persistent direct exposure

On the fixed overfit pilot, the online router could learn mandatory-deviation states and execute non-`FULL` corrective actions.

Therefore:

```text
current online architecture is not inherently unable to represent non-FULL behavior
```

However, this was memorization/capacity evidence only.

### A2: One guaranteed exposure was insufficient

The full online training run guaranteed exactly one mandatory-boundary-reaching route per W2C sample, but these visits were heavily front-loaded and disappeared after the early epochs.

Held-out behavior still collapsed to almost all-`FULL`.

Therefore:

```text
one mostly early boundary exposure per W2C sample
is not sufficient evidence against persistent corrective supervision
```

### B1: Removing the C2C all-FULL route was insufficient for POLAR

Removing the near-universal exact all-`FULL` C2C route did not break POLAR's all-`FULL` collapse.

Therefore:

```text
C2C all-FULL shortcut is a contributor,
but not the sole sufficient cause
```

### Probe: No first-deviation online advantage

The matched mandatory-boundary probe found approximately equal and weak AUROC for:

```text
upfront initial-input representation
vs
online current hidden-state representation
```

This means there is currently no evidence that current hidden states provide an advantage for predicting the **first mandatory deviation along an all-FULL prefix**.

It does **not** yet rule out online advantages after earlier routing interventions have already changed the trajectory.

---

## 3. Primary Experimental Question

The immediate question is:

> **Can persistent corrective supervision turn either architecture into a held-out W2C-rescuing policy?**

This is the main decision-changing experiment.

The desired distinction is:

```text
memorization/capacity
vs
held-out generalization
```

---

## 4. Experimental Principle

Use a **matched low-budget comparison**.

The two architectures should receive:

- the same train/validation samples;
- the same W2C/C2C sampling ratio;
- the same number of training epochs;
- the same persistent mandatory-boundary supervision frequency;
- the same validation schedule;
- the same checkpoint-selection criterion;
- the same held-out behavioral metrics.

Do not simultaneously introduce:

- new router architecture;
- DAgger;
- RL;
- global inverse-frequency action weighting;
- C2C removal;
- C2C all-FULL removal;
- new route-generation procedure.

The intentional intervention is:

```text
persistent W2C corrective supervision across epochs
```

---

## 5. Compared Substrates

### 5.1 Upfront / POLAR-style router

Input:

```text
initial image/query representation
+
layer-specific learned representation
```

Output:

```text
complete 28-layer four-action program
```

The router does not observe intermediate routed hidden states.

Use the existing best-supported complete-route objective:

```text
exact-set NLL
```

Do not use duplicated per-action BCE as the main model.

### 5.2 Online state-conditioned router

At each layer:

```text
current routed visual state
current routed text state
layer-specific READ/WRITE queries
```

produce:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Use the existing architecture unchanged.

Use the existing set-valued four-action loss unchanged as the base objective.

---

## 6. Mandatory-Boundary Supervision

For every W2C sample, use the precomputed mandatory-deviation boundary.

Let:

```text
l* = latest all-FULL-prefix mandatory deviation layer
```

At this layer:

```text
FULL is invalid
```

and at least one of:

```text
IGNORE
READ_ONLY
WRITE_ONLY
```

is valid.

Let:

```text
A_boundary(x)
```

be the valid action set at `l*`.

Define:

```text
L_boundary(x)
=
-log sum_{a in A_boundary(x)} p(a at l*)
```

The same semantic boundary target must be used for both architectures.

---

## 7. Persistent Supervision Schedule

This is the key change relative to A2.

Every W2C sample must receive mandatory-boundary supervision **in every epoch**.

For a 10-epoch run:

```text
epoch 1  -> boundary supervision
epoch 2  -> boundary supervision
...
epoch 10 -> boundary supervision
```

If all 2,397 W2C samples were used, this would create:

```text
2,397 × 10 = 23,970
```

guaranteed mandatory-boundary supervision events.

For the low-budget discriminator, use the same rule on the selected subset.

Do not front-load all boundary events into the first few epochs.

Boundary supervision should be distributed uniformly across training.

---

## 8. Loss Formulation

### 8.1 Online router

Keep the existing route loss:

```text
L_online_base
```

and add the targeted mandatory-boundary term:

```text
L_online
=
L_online_base
+
lambda_boundary * L_boundary
```

Initial diagnostic setting:

```text
lambda_boundary = 1.0
```

Do not tune `lambda_boundary` in the first matched run.

The boundary term is targeted supervision, not generic inverse-frequency class weighting.

### 8.2 POLAR/upfront router

Keep:

```text
L_polar_base = exact-set route NLL
```

and add the same semantic boundary term:

```text
L_polar
=
L_polar_base
+
lambda_boundary * L_boundary
```

with:

```text
lambda_boundary = 1.0
```

The complete route is still trained with exact-set NLL.

The extra boundary term forces explicit corrective pressure at the same W2C decision point used for the online router.

---

## 9. Low-Budget Matched Dataset

Before a full production retraining, use a fixed discriminator subset.

Recommended starting size:

```text
Train:
512 W2C
512 C2C

Validation:
128 W2C
128 C2C
```

If existing split constraints make this inconvenient, use the nearest fixed balanced subset.

Requirements:

- no train/validation overlap;
- image/group split integrity preserved;
- approximately balanced across GQA, TextVQA, and ChartQA;
- broad W2C boundary-layer coverage;
- include boundary actions across `IGNORE`, `READ_ONLY`, and `WRITE_ONLY`;
- include singleton and multi-valid boundary targets.

Save exact sample IDs.

---

## 10. Training Schedule

Use the same schedule for both architectures.

Initial discriminator:

```text
epochs: 20-30 maximum
optimizer: existing matched optimizer
learning rate: existing architecture-specific validated LR
boundary lambda: 1.0
checkpoint: every epoch
```

Why allow more than 10 epochs?

A1 required persistent exposure over a longer horizon to demonstrate capacity.

This run is meant to determine whether corrective supervision can generalize, not to preserve the previous 10-epoch recipe at all costs.

However:

- train both substrates for the same number of epochs;
- use the same early-stop policy if one is defined;
- do not select the winner by training loss.

---

## 11. Training Sampling

Use matched sample-level balancing:

```text
50% W2C
50% C2C
```

within each epoch.

For W2C:

1. normal base-objective supervision;
2. guaranteed mandatory-boundary loss once per sample per epoch.

For C2C:

- retain original supervision unchanged;
- retain all-FULL routes;
- do not remove C2C;
- do not alter preservation labels.

The purpose is to test whether stronger corrective pressure can coexist with preservation supervision.

---

## 12. Primary Metrics

The architecture decision must be based on **free-running held-out execution**, not teacher-forced node metrics.

### 12.1 W2C rescue

Primary metric:

```text
FULL wrong
->
routed correct
```

Report overall and per dataset.

### 12.2 C2C preservation

Report:

```text
FULL correct
->
routed correct
```

The router should not achieve W2C rescue by broadly damaging FULL-correct examples.

### 12.3 Net validation accuracy

Report:

```text
rescues
regressions
net accuracy change
```

---

## 13. Secondary Metrics

### 13.1 Boundary Valid-Action@1

At held-out W2C mandatory boundaries:

```text
predicted action in A_boundary?
```

### 13.2 Boundary non-FULL recall

Since `FULL` is invalid at the mandatory boundary:

```text
predicted action != FULL
```

### 13.3 First deviation layer

For free rollout:

```text
l_pred = first predicted non-FULL layer
l_boundary = reference mandatory boundary
```

Report exact match, within +/-1, within +/-2, early deviation, late deviation, and no deviation.

### 13.4 Action distribution

Report deployment counts:

```text
FULL
IGNORE
READ_ONLY
WRITE_ONLY
```

A lower training loss with all-FULL deployment is not a successful result.

### 13.5 Teacher-forced vs free-rollout gap

For the online model in particular, compare:

```text
teacher-forced boundary behavior
vs
free-rollout behavior
```

A large gap indicates remaining exposure-bias / on-policy-state mismatch.

---

## 14. Prospective Model Selection Rule

Do not select checkpoints by:

```text
overall route membership
overall Valid-Action@1 alone
training loss
C2C-dominated coverage
```

Use held-out behavioral selection.

Preferred rule:

```text
maximize W2C rescue
subject to C2C preservation >= 95%
```

If the 95% threshold is too restrictive for the small subset, report the full Pareto frontier:

```text
x-axis: C2C regression rate
y-axis: W2C rescue rate
```

Do not retrospectively choose a threshold after seeing results.

---

## 15. Decision Outcomes

### Outcome 1 — Online succeeds, POLAR does not

Interpretation:

> Online state-conditioned routing has a behavioral advantage under matched persistent corrective supervision.

Then scale the online router and continue trajectory-conditioned analysis.

### Outcome 2 — Both succeed similarly

Interpretation:

> Persistent corrective supervision, not current hidden-state conditioning, is the main factor.

Because POLAR is operationally simpler, it becomes attractive unless later trajectory-conditioned analysis shows a specific online advantage.

### Outcome 3 — Both learn training boundaries but fail held-out rescue

Interpretation:

> The mandatory corrective route may be highly sample-specific and poorly predictable from the available representations.

Investigate:

- route-label multimodality;
- inter-sample route consistency;
- whether oracle corrective programs define a learnable function;
- whether labels should be clustered/structured differently;
- whether supervision should target a coarser decision than exact four-action route prediction.

### Outcome 4 — Both still fail even to learn targeted boundaries

Then test:

1. simplified 4-way classification head;
2. stronger targeted boundary weighting;
3. single-boundary binary `FULL vs DEVIATE` objective;
4. gradient magnitude allocation;
5. state-feature probes.

Do not move directly to DAgger yet.

### Outcome 5 — W2C rescue rises but C2C preservation collapses

Interpretation:

> The model can learn deviation but is poorly calibrated about when to preserve FULL.

Then study:

```text
lambda_boundary
deviation penalty
explicit C2C constraint
two-stage continue-vs-deviate controller
```

---

## 16. Follow-Up Representation Question

The current matched probe only tested the **first mandatory deviation along an all-FULL prefix**.

That state has not yet been altered by earlier routing interventions.

Therefore, after the matched persistent-supervision experiment, separately test:

> **After an earlier READ/WRITE intervention has already changed the hidden trajectory, are future actions more predictable from the current routed state than from the initial input?**

Construct matched states where:

```text
previous non-FULL intervention exists
```

and compare:

```text
Upfront features:
initial image/query + layer identity

Online features:
actual post-intervention V_l/T_l + layer identity
```

This is the correct test of whether online state conditioning provides information unavailable to an upfront predictor after trajectory divergence.

---

## 17. What Not to Change in This Experiment

Do not simultaneously:

```text
remove C2C
remove C2C all-FULL
change READ/WRITE architecture
add new layer-specific heads
change four-action semantics
regenerate MCTS labels
introduce RL
introduce DAgger
apply global inverse-frequency class weights
change checkpoint selection independently between models
```

The purpose is to isolate **persistent corrective supervision**.

---

## 18. Suggested Execution Phases

### Phase 0 — Freeze protocol

Write `persistent_supervision_protocol.md` before training.

Record:

- exact subset IDs;
- train/validation split;
- exact mandatory boundaries;
- epochs;
- lambda;
- checkpoint-selection rule;
- random seeds;
- metrics;
- stop rule.

### Phase 1 — Data audit

Verify:

```text
all W2C samples have a mandatory boundary
FULL invalid at every selected boundary
C2C labels unchanged
dataset balance
boundary-layer distribution
boundary-action distribution
```

### Phase 2 — Tiny implementation smoke

Verify:

- boundary loss applied once per W2C sample per epoch;
- no front-loading;
- same semantic valid-action target for both architectures;
- gradients finite;
- C2C path unchanged;
- checkpoint metrics correct.

### Phase 3 — Low-budget matched training

Run:

```text
POLAR persistent corrective supervision
Online persistent corrective supervision
```

on the same fixed subset.

### Phase 4 — Held-out free-rollout evaluation

Evaluate every checkpoint or a prospectively fixed interval.

Primary:

```text
W2C rescue
C2C preservation
```

### Phase 5 — Architecture decision

Use the prospective decision rules above.

Do not scale either model before this decision.

---

## 19. Required Outputs

Suggested root:

```text
analysis/persistent_corrective_supervision/
```

Create:

```text
protocol.md
subset_manifest.json
boundary_stats.md

polar_config.yaml
polar_history.jsonl
polar_execution_eval.jsonl
polar_report.md

online_config.yaml
online_history.jsonl
online_execution_eval.jsonl
online_report.md

matched_comparison.csv
matched_comparison.md
decision_summary.md
```

---

## 20. Required Final Comparison Table

Produce at least:

| Metric | POLAR | Online |
|---|---:|---:|
| Train boundary Valid-Action@1 | | |
| Val boundary Valid-Action@1 | | |
| Val boundary non-FULL recall | | |
| W2C rescue | | |
| C2C preservation | | |
| Net accuracy change | | |
| Exact first-deviation match | | |
| No-deviation fraction | | |
| FULL action fraction | | |
| READ_ONLY fraction | | |
| WRITE_ONLY fraction | | |
| IGNORE fraction | | |

Also report dataset-wise W2C rescue and C2C preservation.

---

## 21. Main Decision Question

At the end of this experiment, answer:

> **Under matched persistent corrective supervision, does either architecture learn a generalizable policy that departs from FULL on held-out W2C samples while preserving C2C?**

Only after answering this should we decide whether to:

- continue with online routing;
- return to POLAR/upfront routing;
- modify the objective;
- introduce on-policy training;
- or reconsider whether the four-action oracle routes define a learnable routing problem at all.

---

## 22. Immediate Next Action

The next authorized run should be a **matched low-budget persistent-supervision discriminator**, not another full production run.

Use:

```text
same fixed train/validation subset
same W2C:C2C balance
same epochs
same mandatory-boundary supervision every epoch
same lambda_boundary = 1.0
same behavioral checkpoint criterion
```

Compare:

```text
POLAR/upfront
vs
Online/current-state
```

using held-out:

```text
W2C rescue
C2C preservation
```

as the primary decision metrics.
