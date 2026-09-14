# Stage-2 Shared-Union Training Plan
## One Shared Router, P98/P95/P90 Operating-Point Evaluation, Single First and MCTS as a Second Ablation

## 1. Objective

Train **one shared Stage-2 router** using corrective supervision pooled across the three retained robust Stage-1 operating points:

```text
P98
P95
P90
```

and evaluate the **same trained Stage-2 router** under each Stage-1 threshold.

The main scientific question is:

> **Given one fixed Stage-2 policy, which Stage-1 operating point produces the best end-to-end rescue-versus-regression trade-off?**

This avoids confounding threshold quality with different Stage-2 training-set sizes or different Stage-2 models.

Training progression:

```text
Experiment A:
Union Preservation + Union Single

Experiment B:
same setup
+ Union MCTS supervision
```

Do not train separate Stage-2 routers for P98, P95, and P90 in the primary comparison.

---

## 2. Frozen Components

Freeze:

```text
Qwen2.5-VL backbone
robust ALL-source Stage-1 head
Stage-1 feature pipeline
Stage-1 normalization
P98 threshold
P95 threshold
P90 threshold
four-action executor semantics
LMMS-Eval correctness contract
```

Only Stage-2 is trainable.

---

## 3. Retained Stage-1 Operating Points

| Point | Worst-source C preservation | Pooled W recall | Median trigger layer |
|---|---:|---:|---:|
| P98 | 98.08% | 10.54% | 23 |
| P95 | 95.01% | 21.79% | 21.5 |
| P90 | 90.03% | 35.72% | 20 |

None is the final deployment threshold yet.

Final selection must depend on learned Stage-2:

```text
W→C
C→W
C→C
final accuracy
```

---

## 4. Reconstructed Corrective Corpora

Current threshold-specific corpora:

```text
P98:
    preservation bases = 7
    single bases       = 65
    MCTS bases         = 43

P95:
    preservation bases = 30
    single bases       = 143
    MCTS bases         = 100

P90:
    preservation bases = 106
    single bases       = 270
    MCTS bases         = 216
```

A total of:

```text
2,519 exact-replay-valid routes
34,253 routed state rows
```

were retained.

Primary Stage-2 training should use their deduplicated union.

---

## 5. Why Train One Shared Stage-2 Router

Stage-1 threshold determines:

```text
WHEN Stage-2 becomes active
```

Stage-2 determines:

```text
WHAT action to take at the current routed state
```

So the primary comparison should be:

```text
same Stage-2 + P98
same Stage-2 + P95
same Stage-2 + P90
```

This makes Stage-1 threshold the only deployment variable.

---

## 6. Global Union Corpus

Build a deduplicated global route store.

Each unique route should be stored once with:

```text
route_id
uid
dataset
source_regime
route_source
action_trajectory

valid_for_P98
valid_for_P95
valid_for_P90
```

Use:

```text
route_source ∈ {
    preservation_full,
    single,
    mcts
}
```

Do not count the same route three times just because it is valid under three operating points.

---

## 7. Union Preservation Corpus

Create:

```text
UNION_A = all unique triggered-C preservation bases
```

across P98/P95/P90.

For each correct sample preserve:

```text
trigger_P98
trigger_P95
trigger_P90
```

and all replay-valid FULL post-trigger states reachable under at least one retained operating point.

Deduplicate identical states where exact identity is known.

---

## 8. Union Single Corpus

Create:

```text
UNION_B = all unique Dense-W bases with >=1 replay-valid single corrective route
```

across P98/P95/P90.

Preserve for every route:

```text
uid
intervention_layer
intervention_action
route_id
valid_for_P98/P95/P90
```

Training remains sample-balanced at UID level.

---

## 9. Union MCTS Corpus

Create separately:

```text
UNION_C = all unique Dense-W bases with >=1 replay-valid MCTS corrective route
```

across P98/P95/P90.

Preserve:

```text
route_id
action trajectory
non-FULL count
first non-FULL layer
last non-FULL layer
valid_for_P98/P95/P90
```

Do not use UNION_C in Experiment A.

---

## 10. Stage-2 Architecture

Use the existing structured shared Stage-2 router.

### READ branch

```text
q_j = current last text/control hidden state

z_R = CrossAttention(
    query = q_j,
    key   = V_j,
    value = V_j
)
```

Interpretation:

> Should current reasoning directly READ visual evidence at this layer?

### WRITE branch

```text
z_W = CrossAttention(
    query = q_W,
    key   = V_j,
    value = V_j
)
```

where `q_W` is one learned query shared across all layers.

Interpretation:

> Should current visual representation continue to be updated?

### Action head

```text
[z_R ; z_W]
→ small shared MLP
→ FULL / READ_ONLY / WRITE_ONLY / IGNORE
```

---

## 11. Excluded Inputs

Do not add:

```text
learnable layer embedding
layer ID
Stage-1 latent representation
Stage-1 p_wrong
source ID
dataset ID
focal loss
new backbone features
```

Keep the router shared and state-dependent.

---

## 12. Experiment A — Union Preservation + Union Single

Train with:

```text
UNION_A
+
UNION_B
```

Exclude:

```text
UNION_C
```

Purpose:

> Determine whether broader robust-gate single-intervention supervision is sufficient for a shared Stage-2 policy.

---

## 13. Sample-Balanced W Sampling

For each W draw:

```text
1. sample a W UID uniformly from UNION_B
2. sample one successful single route uniformly for that UID
```

Do not sample uniformly from route records.

---

## 14. Positive-Anchored Random-4

For each sampled single route, use approximately:

```text
1 mandatory corrective non-FULL state
1 pre-intervention FULL state
1 post-intervention FULL state
1 additional random reachable FULL state
```

All states must be actual replay-valid routed states.

Do not use states that occur before Stage-2 activation under every retained threshold.

---

## 15. Threshold-Agnostic Training

Do not give the router:

```text
P98 / P95 / P90
threshold value
```

as input.

Threshold validity is metadata only.

The router should learn:

```text
current routed state → current action
```

independent of which Stage-1 threshold caused the handoff.

---

## 16. Preservation Sampling

Sample correct bases uniformly from UNION_A.

For each preservation draw:

```text
sample up to 4 replay-valid FULL states
```

from post-trigger regions reachable under at least one retained operating point.

This gives much more preservation-state diversity than training a P98-specific router on only 7 C bases.

---

## 17. C:W Mixture

Use:

```text
C:W sample draws = 1:2
```

for Experiment A.

Keep this identical in Experiment B.

Do not use natural corpus-size ratios.

---

## 18. Loss

Use:

```text
plain 4-way cross entropy
```

Do not add focal loss, class weights, or set-valued loss unless a diagnosed failure requires it later.

---

## 19. Multi-Valid Single Labels

For W samples with multiple successful single routes:

```text
sample one successful route stochastically per draw/epoch
```

Preserve alternatives but do not weight by route multiplicity.

---

## 20. Training Progression — Experiment A

### A0 — Data-loader audit

Verify:

```text
route deduplication
UID-balanced W sampling
threshold-valid metadata
routed-state references
C:W draw ratio
Random-4 state composition
```

### A1 — Small overfit smoke

Track:

```text
FULL recall
non-FULL recall
READ_ONLY recall
WRITE_ONLY recall
IGNORE recall
predicted action distribution
```

Reject FULL-only solutions.

### A2 — Full UNION_A+B training

Train one shared Stage-2 router.

Keep model size and optimizer recipe aligned with the previous V1 unless a blocking implementation issue appears.

---

## 21. Validation Free Rollout — Same Router, Three Thresholds

Use the **same selected Stage-2 checkpoint** for three evaluations:

```text
robust Stage-1 P98 + same Stage-2
robust Stage-1 P95 + same Stage-2
robust Stage-1 P90 + same Stage-2
```

Do not retrain between evaluations.

---

## 22. Runtime

For each threshold:

```text
FULL while Stage-1 has not triggered

if no trigger:
    remain FULL

if trigger at l*:
    activate shared Stage-2

for j = l* ... 27:
    read actual current routed visual/text states
    predict action
    execute
    continue from resulting state
```

Stage-2 must consume its own routed trajectory.

---

## 23. Primary Experiment-A Metrics

For P98/P95/P90 report:

```text
Dense baseline accuracy
Routed accuracy
Δ accuracy

W→C
W→W
C→C
C→W

W→C rescue rate
C→C preservation rate
```

Primary net metric:

```text
Net correction = W→C - C→W
```

---

## 24. Stage-2 Behavior by Threshold

Report:

```text
Stage-1 triggered count
fraction using >=1 non-FULL
post-trigger FULL fraction
non-FULL actions/sample
first non-FULL layer
trigger→first-non-FULL delay
READ_ONLY count
WRITE_ONLY count
IGNORE count
```

This reveals how one policy behaves when activated under progressively more permissive handoffs.

---

## 25. Threshold Question

The key question is:

> Does extra W admission from P95/P90 create more rescue than extra C admission creates regression?

Expected trade-off:

```text
P98:
low W opportunity, low C risk

P95:
middle

P90:
higher W opportunity, higher C risk
```

Only learned end-to-end routing can choose among them.

---

## 26. Dataset × Source Evaluation

For each threshold report:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

with:

```text
triggered W
triggered C
W→C
C→W
final accuracy change
```

Treat low-support Canonical TextVQA cautiously.

---

## 27. Experiment-A Decision

### A-success

If at least one threshold gives:

```text
positive W→C
low C→W
positive net accuracy
```

proceed to Experiment B.

### A-flat

If the router remains nearly all-FULL with little rescue, diagnose:

```text
insufficient single coverage
training imbalance
state representation
exposure shift
```

before assuming threshold choice is the issue.

### A-regressive

If C→W exceeds W→C, inspect preservation and threshold dependence before adding MCTS.

---

## 28. Why MCTS Is Especially Relevant

The robust-gate missing search found:

```text
newly resolved pairs:
single contribution = 55
MCTS contribution   = 278
```

So many corrections under the repaired Stage-1 handoff required richer trajectory search.

This makes MCTS supervision a meaningful second experiment, but it should remain a separate ablation.

---

## 29. Experiment B — Add Union MCTS Supervision

If Experiment A is technically healthy, train the same architecture with:

```text
UNION_A
+
UNION_B
+
UNION_C
```

The only intended variable is:

```text
+ MCTS trajectory supervision
```

No Stage-1 or architecture changes.

---

## 30. MCTS Sampling

For every MCTS W draw:

```text
1. sample UID uniformly from UNION_C
2. sample one successful MCTS route uniformly
3. sample a bounded set of route states
```

Recommended first policy:

```text
include all non-FULL states up to a small cap
+
sample nearby/intermediate FULL states
```

Cap per-route state count so longer routes do not dominate training.

---

## 31. Single/MCTS W Mixture

In Experiment B, control supervision-type weighting explicitly.

Simple first contract:

```text
Single : MCTS corrective W draws = 1 : 1
```

with C preservation still mixed at the outer C:W ratio.

Do not let raw route counts determine type frequency.

---

## 32. Experiment-B Evaluation

Evaluate the same Experiment-B checkpoint at:

```text
P98
P95
P90
```

with no retraining between thresholds.

Compare:

| Threshold | Single-only ΔAcc | Single+MCTS ΔAcc | Δ(W→C) | Δ(C→W) |
|---|---:|---:|---:|---:|
| P98 | | | | |
| P95 | | | | |
| P90 | | | | |

This isolates whether MCTS supervision adds learned-policy value.

---

## 33. Final Operating-Point Selection

Only after the final planned Stage-2 condition is evaluated should one threshold be selected.

Select on validation using:

```text
final routed accuracy
```

and report:

```text
W→C
C→W
C→C
```

Tie-break:

```text
1. higher final accuracy
2. fewer C→W
3. higher C→C preservation
4. simpler/more conservative threshold
```

Do not choose from Stage-1 W recall or corpus size alone.

---

## 34. No Threshold-Specific Router Training in the Primary Study

Do not train:

```text
Stage2_P98
Stage2_P95
Stage2_P90
```

as the primary comparison.

That would confound threshold effect with training-data size and optimization.

Threshold-specific training may only be a later upper-bound ablation if necessary.

---

## 35. Fairness Contract

Keep fixed across A/B where applicable:

```text
Stage-2 architecture
optimizer family
learning-rate schedule
update budget
checkpoint-selection rule
C:W sample mixture
validation set
free-rollout evaluator
```

Do not give Experiment B substantially more optimization just because it has more supervision.

---

## 36. Required Outputs

Use:

```text
analysis/dense_failure_stage2/shared_union_training/
```

Create:

```text
protocol.md

corpus/
    union_preservation_manifest.jsonl
    union_single_manifest.jsonl
    union_mcts_manifest.jsonl
    route_dedup_summary.csv
    threshold_validity_summary.csv

experiment_A_single/
    config/
    smoke/
    training/
    teacher_forced/
    rollout/
        P98/
        P95/
        P90/
    metrics/
        threshold_comparison.csv
        dataset_source_breakdown.csv
        action_behavior.csv
    summaries/
        experiment_A_summary.md

experiment_B_single_plus_mcts/
    config/
    smoke/
    training/
    rollout/
        P98/
        P95/
        P90/
    metrics/
        threshold_comparison.csv
        dataset_source_breakdown.csv
        action_behavior.csv
    summaries/
        experiment_B_summary.md

comparison/
    A_vs_B.csv
    threshold_final_comparison.csv
    selected_operating_point.json

figures/
    threshold_accuracy_tradeoff.png
    threshold_rescue_regression.png
    threshold_action_behavior.png
    single_vs_mcts_accuracy.png
    dataset_source_accuracy_change.png

summaries/
    shared_stage2_decision.md

artifact_manifest.json
```

---

## 37. `experiment_A_summary.md` Must Answer

1. How many unique preservation bases are in UNION_A?
2. How many unique single-fixable W bases are in UNION_B?
3. Did the router pass overfit and avoid FULL collapse?
4. What are P98/P95/P90 W→C counts?
5. What are P98/P95/P90 C→W counts?
6. Which threshold has the best net correction?
7. Does the same router remain conservative on C under P90?
8. How does behavior vary by dataset/source?
9. Is Experiment A healthy enough to justify adding MCTS supervision?

---

## 38. `experiment_B_summary.md` Must Answer

1. How many unique MCTS-fixable W bases are added?
2. Did MCTS supervision increase non-FULL policy coverage?
3. Did it improve W→C?
4. Did it increase C→W?
5. Is the effect threshold-dependent?
6. Which datasets benefit?
7. Does MCTS supervision improve final validation accuracy beyond Single-only?

---

## 39. `shared_stage2_decision.md`

Choose one:

### Decision A — Shared Stage-2 works; final threshold identified

Requirements:

```text
positive net validation gain
acceptable C preservation
one operating point clearly preferable
```

Next:

```text
freeze Stage-1 threshold + Stage-2
→ final held-out test
```

### Decision B — Shared Stage-2 works but threshold remains tied

Keep 2 candidate thresholds for a narrowly scoped confirmation.

### Decision C — Stage-2 still under-generalizes

If all thresholds show almost no rescue despite strong oracle coverage, investigate:

```text
policy representation
exposure shift
multi-valid objective
training-data coverage
```

before more search.

### Decision D — Regression dominates

If C→W is too large, focus on conservative action selection / preservation before increasing intervention coverage.

---

## 40. Stop Rule

STOP after:

```text
union corpus construction
Experiment A training
P98/P95/P90 free rollout
Experiment B only if A is healthy
P98/P95/P90 free rollout
validation operating-point decision
```

Do not automatically:

```text
run final test
change Stage-1 head
change thresholds outside P98/P95/P90
add layer embeddings
add Stage-1 latent
add focal loss
run new corrective search
```

---

## 41. Core Experimental Logic

```text
                Robust Stage-1 head
                       │
            ┌──────────┼──────────┐
            │          │          │
           P98        P95        P90
            │          │          │
            └──────────┼──────────┘
                       │
              SAME Stage-2 router
                       │
             end-to-end validation
```

Training:

```text
UNION preservation + single
        ↓
shared Stage-2 A
        ↓
evaluate same router at P98/P95/P90

        ↓ if healthy

+ UNION MCTS
        ↓
shared Stage-2 B
        ↓
evaluate same router at P98/P95/P90
```

This cleanly separates:

> **Stage-1 operating-point effect**

from:

> **Stage-2 supervision-complexity effect**
