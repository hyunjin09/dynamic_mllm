# Four-Action Router Collapse: Next-Step Experimental Plan

## 1. Goal

Both the POLAR-style upfront router and the online four-action router have collapsed to essentially all-`FULL` deployment.

The next step is **not** to redesign the architecture immediately and **not** to bundle several fixes into another full 10-epoch run.

The goal of this phase is to isolate why collapse occurs and determine whether:

1. the current online router already has enough capacity to recognize when `FULL` must stop, but the training sampler fails to expose the critical states;
2. the C2C all-`FULL` route creates an overly strong complete-route shortcut, especially for the upfront/POLAR-style predictor;
3. action imbalance, exposure bias, or router architecture still requires additional intervention after the first two effects are isolated.

The experiment order must preserve causal interpretability.

## 2. Current Diagnosis

### 2.1 C2C contains a universal complete-route shortcut

Current train split:

```text
C2C train samples: 3,548
C2C samples containing exact all-FULL route: 3,501 / 3,548 = 98.675%
```

No W2C sample contains the exact all-`FULL` route.

This creates a particularly strong shortcut for complete-route/upfront prediction:

```text
FULL, FULL, FULL, ..., FULL
```

Removing only that exact route would still leave at least one non-`FULL` correct route for:

```text
3,513 / 3,548 C2C samples
```

The remaining 35 samples would need exclusion or separate handling.

### 2.2 W2C is still strongly FULL-heavy at the layer/action level

W2C means:

```text
the complete all-FULL route is wrong
```

It does **not** mean that every layer in a correcting W2C route must be non-`FULL`.

The current 10-epoch online sampler produces approximately:

```text
W2C teacher actions:
FULL        76.782%
IGNORE       7.357%
READ_ONLY    9.589%
WRITE_ONLY   6.271%
```

Therefore the relevant learning problem is:

> **Keep `FULL` while it remains safe, but deviate at the correct state before the all-FULL rollout becomes irrecoverable.**

### 2.3 Mandatory all-FULL-prefix deviation boundary

For every W2C sample, define the longest prefix for which repeatedly selecting `FULL` still lies on at least one known-correct route.

Example:

```text
L0   FULL valid
L1   FULL valid
...
L16  FULL valid
L17  FULL invalid  <- mandatory deviation boundary
```

At the mandatory boundary:

```text
FULL is invalid by construction.
```

Known-valid corrective actions are among:

```text
IGNORE
READ_ONLY
WRITE_ONLY
```

The all-FULL prefix can remain valid for a long time:

```text
mean longest valid all-FULL prefix: 14.586 layers
median: 15
P95: 27
max: 27
```

Thus a greedy router can appear locally correct for many layers while drifting toward the globally wrong all-FULL route.

### 2.4 Current teacher forcing under-covers the critical boundary

The current 10-epoch sampler visits:

```text
mean available W2C routes/sample: 30.15
mean teacher routes actually visited/sample: 6.51
```

Critically:

```text
1,045 / 2,397 W2C samples = 43.6%
```

are never teacher-forced through the route that reaches their latest valid all-FULL-prefix deviation boundary.

This creates a direct teacher-forcing / free-rollout mismatch.

The first intervention should therefore target **state/prefix coverage**, not architecture, class weighting, or DAgger.

## 3. Experimental Principle

Change **one supported mechanism at a time**.

Do not combine in the first retry:

- mandatory-boundary coverage repair
- C2C all-FULL removal
- action weighting
- on-policy / DAgger training
- architecture changes
- new loss functions

If multiple changes are applied simultaneously, an improvement would not reveal which defect actually caused the collapse.

## 4. Track A — Online Four-Action Router

### Phase A0 — Prepare mandatory-boundary metadata

For every W2C training sample:

1. enumerate the known-valid four-action routes;
2. find the longest exact prefix consisting only of `FULL` actions that is still shared by at least one known-correct route;
3. define the next layer as the mandatory deviation boundary;
4. identify at least one known-correct route that realizes that longest all-FULL prefix and then takes a valid non-`FULL` action;
5. store:
   - sample ID
   - dataset
   - boundary layer
   - exact all-FULL prefix length
   - valid non-FULL actions at the boundary
   - route IDs that reach the boundary
   - whether the valid action is singleton or multi-valid

Write:

```text
analysis/4action_router/boundary_manifest.jsonl
analysis/4action_router/boundary_audit.md
```

Sanity requirements:

```text
one boundary record per W2C sample
FULL invalid at every stored boundary
at least one known-valid non-FULL action at every stored boundary
```

## 5. Phase A1 — Small Mandatory-Boundary Overfit Pilot

### 5.1 Purpose

Answer:

> **If the current online router is explicitly shown the critical all-FULL-prefix boundary states, can it learn to leave the all-FULL trajectory?**

This is a capacity / state-representation test, not the final training recipe.

### 5.2 Keep everything else unchanged

Do not change:

```text
router architecture
READ branch
WRITE branch
layer-specific queries
structured four-action head
set-valued action loss
optimizer
learning rate
C2C labels
C2C all-FULL route
backbone
executor semantics
```

The only intentional change is:

```text
guaranteed exposure to W2C mandatory-boundary trajectories
```

### 5.3 Pilot subset

Use a fixed small subset:

```text
64-128 W2C samples
```

Prefer:

- coverage across GQA, TextVQA, and ChartQA;
- broad boundary-layer coverage;
- mix of singleton and multi-valid boundary targets;
- mix of `IGNORE`, `READ_ONLY`, and `WRITE_ONLY` valid targets.

Optionally include a small fixed C2C preservation subset, but do not let it dominate the pilot.

Use deterministic sample IDs and save them.

### 5.4 Exact teacher-forced trajectory

For each selected W2C sample, choose a known-correct route that reaches the mandatory boundary through the exact all-`FULL` prefix.

Example:

```text
L0   FULL
L1   FULL
...
L16  FULL
L17  WRITE_ONLY  <- mandatory deviation
L18  ...
...
L27  ...
```

During training:

1. execute `FULL` for all layers before the boundary;
2. obtain the **actual hidden state generated by that all-FULL prefix**;
3. at the boundary, compute the current set-valued four-action loss;
4. teacher-force one known-valid corrective continuation afterward.

Do **not** simulate the boundary by simply assigning a non-FULL label at the same layer under a different prefix.

The hidden state must be the exact state reached by the all-FULL prefix.

## 6. Pilot Loss

Keep the existing set-valued action loss unchanged:

```text
L_l = -log sum_{a in A_valid(s_l)} p(a | s_l)
```

For singleton action:

```text
A_valid = {WRITE_ONLY}
L_l = -log p(WRITE_ONLY)
```

For multi-valid action:

```text
A_valid = {READ_ONLY, WRITE_ONLY}
L_l = -log [p(READ_ONLY) + p(WRITE_ONLY)]
```

The sample route can still produce the usual 28 layer losses.

Do not introduce action weights in this pilot.

## 7. Pilot Training Regime

This is an overfit diagnostic, not a production run.

Therefore:

- do not constrain the experiment to exactly 10 epochs;
- train long enough to determine whether the fixed subset can be fit;
- keep the architecture and loss unchanged;
- save frequent checkpoints;
- use a small number of GPUs if sufficient;
- prioritize rapid iteration over throughput.

Suggested stop condition:

```text
Either:
1. boundary classification and free rollout clearly overfit;

or:
2. training has saturated and the router still cannot reliably leave FULL.
```

## 8. Pilot Metrics

Do not judge the pilot primarily by global node loss.

### 8.1 Mandatory-boundary Valid-Action@1

At the exact teacher-forced mandatory boundary:

```text
predicted action in A_valid(boundary)?
```

Report:

```text
overall
per dataset
per boundary action type
singleton-only
multi-valid
```

### 8.2 Mandatory-boundary non-FULL recall

Because `FULL` is invalid at the boundary:

```text
predicted action != FULL
```

should approach 100% on an overfit subset.

### 8.3 Singleton minority-action recall

For boundary nodes where only one action is valid, report separately:

```text
IGNORE recall
READ_ONLY recall
WRITE_ONLY recall
```

### 8.4 First predicted deviation layer

During **free rollout**, record:

```text
l_pred = first layer where predicted action != FULL
```

Compare with:

```text
l_boundary = mandatory deviation boundary
```

Report:

```text
l_pred - l_boundary
exact-match rate
within +/-1 layer
within +/-2 layers
early deviation fraction
late/no deviation fraction
```

### 8.5 W2C free-rollout rescue

Primary behavioral metric:

```text
FULL wrong
->
router free rollout correct
```

### 8.6 C2C preservation

If a C2C subset is included:

```text
FULL correct
->
router free rollout correct
```

## 9. Pilot Decision Rule

### Outcome A — Current router cannot overfit the mandatory boundaries

Symptoms:

```text
teacher-forced boundary still predicts FULL
singleton READ/WRITE recall stays poor
free rollout remains all-FULL
W2C rescue remains poor
```

Interpretation:

> Boundary exposure alone is insufficient. The current state representation, router architecture, structured head, or optimization may not contain/use the required signal.

Then inspect, in this order:

1. verify gradients reach:
   - READ branch
   - WRITE branch
   - `e_R[l]`
   - `e_W[l]`
   - structured decision head
2. test a simple unrestricted 4-way MLP head;
3. probe whether current routed hidden state separates:
   - safe-FULL states
   - mandatory-deviation states
4. test richer text-state representation if necessary;
5. inspect WRITE visual pooling capacity;
6. only then redesign architecture.

Do **not** launch a full 10-epoch production run.

### Outcome B — Current router can overfit and free-run the subset

Desired pattern:

```text
high boundary Valid-Action@1
high non-FULL recall
high singleton READ/WRITE recall
free rollout actually deviates
meaningful W2C rescue
```

Interpretation:

> The current architecture has sufficient local discrimination capacity. The major defect is training-state / prefix coverage rather than immediate architecture failure.

Proceed to Phase A2.

## 10. Phase A2 — Full Online Training with Guaranteed Boundary Coverage

Keep unchanged:

```text
architecture
loss
C2C population
C2C all-FULL route
optimizer
```

Modify only the W2C route sampler.

Goal:

```text
mandatory-boundary exposure:
56.4% -> 100%
```

Each W2C training sample must receive at least one teacher-forced visit to its mandatory all-FULL-prefix boundary within the planned training schedule.

Possible implementation:

```text
for each W2C sample:
    schedule at least one boundary-reaching route
    use normal valid-route sampling for remaining visits
```

Do not replace all route diversity with only the boundary route.

Preserve ordinary route sampling as additional supervision.

## 11. Phase A2 Training Length

Use the same broad training regime as the existing router for the matched comparison:

```text
10 epochs
```

unless the pilot reveals that substantially more optimization is required.

The first full comparison should remain as matched as possible to the previous failed run.

## 12. Phase A2 Primary Metrics

Primary:

```text
W2C free-rollout rescue
C2C preservation
overall routed accuracy
```

Secondary:

```text
mandatory-boundary Valid-Action@1
first-deviation timing
free-rollout action distribution
teacher-forced action distribution
singleton READ/WRITE recall
mean number of FULL actions
mean READ suppression
mean WRITE suppression
```

The success criterion is not merely lower teacher-forced loss.

The router must actually leave the all-FULL deployment trajectory.

## 13. Track B — POLAR-Style Upfront Router

Treat this as a separate collapse mechanism.

The upfront router does not suffer from the exact same sequential exposure problem because it predicts the complete program before execution.

Its strongest measured shortcut is the near-universal C2C all-FULL route.

## 14. Phase B1 — Matched C2C All-FULL Removal Ablation

Do not remove C2C entirely.

Construct a matched training variant:

```text
Original:
all valid C2C routes

Ablation:
remove only the exact [FULL x 28] route from C2C
```

For the 35 C2C samples that become route-empty:

- exclude them from this ablation, or
- handle them with an explicitly documented preservation rule.

Do not silently invent labels.

Keep everything else matched:

```text
architecture
optimizer
training length
dataset split
objective
route representation
```

Use exact-set NLL as the preferred complete-route objective.

Do not use duplicated per-action BCE as the main scientific model.

## 15. Phase B1 Metrics

Report:

```text
top-1 valid-route coverage
W2C correct-route coverage
C2C correct-route coverage
fraction of predicted exact all-FULL routes
per-dataset results
actual executed routed accuracy
route diversity
```

Key question:

> Does removing the universal complete-route shortcut actually break the all-FULL mode?

This is an ablation, not automatically the final solution.

## 16. Cross-Track Diagnostic — Upfront vs Current-State Boundary Prediction

After the isolated pilot(s), run a direct representational comparison.

### Task

Given a layer/state, predict:

```text
safe to continue FULL
vs
mandatory deviation now
```

### Upfront feature

Use:

```text
initial image/query representation
+ layer identity
```

### Online feature

Use:

```text
current routed visual hidden state
current routed text state
+ layer identity
```

Keep probe capacity approximately matched.

Use balanced train/validation examples.

Primary metric:

```text
mandatory-boundary AUROC / accuracy / F1
```

Optional second stage:

```text
predict valid corrective action
{IGNORE, READ_ONLY, WRITE_ONLY}
```

Scientific question:

> **Is the moment when FULL must stop more predictable from the current routed hidden state than from the original input alone?**

A strong online advantage would directly support the motivation for online state-conditioned routing.

## 17. Only After the Isolated Tests: Additional Remedies

### 17.1 Minority / boundary action weighting

If boundary coverage is fixed but minority actions are still poorly learned, consider bounded weighting derived only from the training split.

Prefer targeted weighting at:

```text
FULL-invalid / mandatory-boundary nodes
```

over indiscriminate global inverse-frequency weighting.

### 17.2 On-policy / DAgger-style training

Use only if:

```text
teacher-forced boundary performance is good
but free rollout still diverges into unseen prefixes
```

Procedure:

1. free-run current router;
2. collect erroneous or unsupported prefixes;
3. evaluate available four actions under the actual reached state;
4. identify valid corrective actions where possible;
5. add these states to training;
6. retrain/fine-tune.

### 17.3 Architecture changes

Architecture changes are justified only if the controlled boundary-overfit pilot fails despite adequate optimization and verified gradients.

Potential diagnostics before redesign:

```text
current text-state probe
visual-state probe
READ-only branch probe
WRITE-only branch probe
generic 4-way head
layer-specific embedding ablation
```

Do not redesign solely because the original production run collapsed.

## 18. Recommended Experiment Order

Execute in this order:

```text
1. Build mandatory-boundary manifest.

2. Run small fixed W2C mandatory-boundary overfit pilot
   with the current online router and current loss.

3. If the router can fit/free-run:
      run full online training with guaranteed boundary coverage.

   If it cannot:
      stop and diagnose state/features/head before full training.

4. Separately run POLAR/upfront C2C exact-all-FULL removal ablation.

5. Run matched upfront-vs-online mandatory-boundary prediction probe.

6. Only afterward consider:
      boundary/action weighting
      scheduled/on-policy rollout
      DAgger-style correction
      architecture redesign.
```

Do not combine Steps 2-4 into one run.

## 19. Required Reports

Create:

```text
analysis/4action_collapse/
```

with:

```text
boundary_manifest.jsonl
boundary_audit.md
pilot_subset.json
mandatory_boundary_overfit_config.yaml
mandatory_boundary_overfit_history.jsonl
mandatory_boundary_overfit_report.md

online_boundary_coverage_v2_config.yaml
online_boundary_coverage_v2_history.jsonl
online_boundary_coverage_v2_report.md

polar_c2c_no_allfull_config.yaml
polar_c2c_no_allfull_history.jsonl
polar_c2c_no_allfull_report.md

upfront_vs_online_boundary_probe_config.yaml
upfront_vs_online_boundary_probe_report.md

decision_summary.md
```

## 20. Decision Summary Template

After each isolated experiment, update:

```text
Question 1:
Can the unchanged online router recognize a mandatory deviation state when explicitly trained on it?

Answer:
YES / NO

Evidence:
...

Question 2:
Does guaranteed mandatory-boundary coverage break online all-FULL free-rollout collapse?

Answer:
YES / NO

Evidence:
...

Question 3:
Does removing the exact C2C all-FULL route break the upfront/POLAR all-FULL mode?

Answer:
YES / NO

Evidence:
...

Question 4:
Are mandatory deviation states more predictable from current routed hidden states than from initial input features?

Answer:
YES / NO

Evidence:
...

Next action:
...
```

## 21. Main Scientific Questions

This phase should answer three questions cleanly:

### Q1

> **Can the model recognize when FULL must stop?**

### Q2

> **Is that decision more predictable from current routed hidden states than from the initial image/query representation?**

### Q3

> **Does exposing those critical states during training convert that predictability into actual W2C rescue under free rollout while preserving C2C?**

These questions should guide the next method decision.

## 22. What Not to Do Yet

Do not:

```text
remove C2C entirely
change architecture and sampler simultaneously
add global inverse-frequency weighting immediately
start RL immediately
start DAgger immediately
run another expensive 10-epoch 8-GPU job before the small boundary pilot
treat lower teacher-forced loss as evidence that collapse is fixed
treat cached-route membership as final execution correctness
```

The next run should maximize **diagnostic information**, not GPU utilization.

## 23. Immediate Action

The next authorized experiment should be:

```text
Small fixed W2C mandatory-boundary overfit pilot

Architecture:
    unchanged online four-action router

Loss:
    unchanged set-valued four-action loss

C2C:
    unchanged

Special intervention:
    every selected W2C sample is explicitly teacher-forced
    through its exact latest all-FULL-prefix mandatory-deviation boundary

Primary readouts:
    boundary Valid-Action@1
    non-FULL recall
    singleton READ_ONLY / WRITE_ONLY / IGNORE recall
    first free-rollout deviation layer
    W2C rescue
    C2C preservation if included
```

Do not launch the full retraining until this pilot resolves whether the current router can learn the critical deviation states at all.
