# Stage-2 MCTS Failure Diagnosis Plan
## Why does single-route supervision produce a small gain while adding MCTS supervision removes it?

## 1. Why This Experiment Matters

The latest Stage-2 result created a specific discrepancy:

```text
Experiment A: Single supervision
P90:
    W→C = 4
    C→W = 1
    net = +3
    routed accuracy = 50.375%

Experiment B: Single + MCTS supervision
P98/P95/P90:
    W→C = 0
    C→W = 0
    routed accuracy = 50.000%
```

Additional behavior:

```text
Teacher-forced non-FULL recall:
    A = 19.20%
    B = 16.87%

P90 free-rollout samples using any non-FULL:
    A = 17.36%
    B = 4.13%
```

The important question is **not**:

> "Does MCTS work?"

The current result cannot answer that.

The important question is:

> **Why does richer MCTS supervision fail to translate into free-rollout corrective behavior, and why does it suppress the small corrective behavior learned from single routes?**

Plausible explanations:

```text
H1. Optimization / action-learning failure
    B does not actually learn the MCTS actions even on oracle routed states.

H2. Exposure shift
    B predicts actions on teacher-forced MCTS states,
    but its earlier mistakes move it to unseen free-rollout states.

H3. Negative transfer
    Adding MCTS supervision interferes with the simpler single-intervention
    behavior already learned by A.

H4. Label ambiguity / conflicting supervision
    Same or similar states admit multiple successful actions,
    while plain single-label CE forces one target.

H5. Representation limitation
    The current READ/WRITE router input is insufficient to tell
    which stage of a multi-intervention correction the model is in.
```

The next phase should discriminate among these explanations before introducing a new training recipe.

---

## 2. What This Phase Can Establish

This phase may establish:

```text
whether B recognizes MCTS actions on oracle states
whether failure begins at the first intervention or later
whether oracle prefix forcing restores later rollout behavior
whether MCTS supervision suppresses A's single-route behavior
whether label ambiguity explains many apparent errors
whether the dominant issue is learning, exposure, interference, or representation
```

---

## 3. What This Phase Cannot Establish

Do not conclude from this phase that:

```text
multi-step corrective routing is impossible
MCTS supervision is fundamentally harmful
single-intervention routing is universally superior
the Stage-2 direction should be abandoned
```

A negative result here only diagnoses the current learned policy under the current training contract.

---

## 4. Frozen Components

Do not retrain anything initially.

Freeze:

```text
robust ALL-source Stage-1 head
P98/P95/P90 thresholds
Stage-2 Experiment-A checkpoint
Stage-2 Experiment-B checkpoint
Stage-2 architecture
single/MCTS/preservation corpora
all replay-valid routed states
four-action executor
LMMS-Eval correctness
```

No new search.
No threshold changes.
No Stage-1 changes.

---

## 5. Primary Diagnostic Operating Point

Use:

```text
P90
```

as the primary free-rollout diagnostic point because A/P90 produced actual answer changes:

```text
W→C = 4
C→W = 1
```

P98/P95 remain secondary checks.

Do not infer that P90 is the final deployment threshold.

---

## 6. Diagnostic A — Oracle-State Action Prediction

### Question

> Does the B router recognize the correct action when shown the exact replay-valid routed state from a successful route?

Use saved states from:

```text
single routes
MCTS routes
preservation FULL routes
```

Evaluate both:

```text
Experiment A checkpoint
Experiment B checkpoint
```

For each state record:

```text
top-1 action
target-action probability
FULL probability
best non-FULL probability
target-vs-FULL margin
```

Report separately for:

```text
single-route states
MCTS-route states
preservation states
```

Metrics:

```text
overall action accuracy
FULL recall
non-FULL recall
READ_ONLY recall
WRITE_ONLY recall
IGNORE recall
mean target probability
mean target-vs-FULL margin
```

For MCTS also stratify by:

```text
route non-FULL count: 1 / 2 / 3 / 4+
intervention index: first / second / third+
```

Interpretation:

```text
B poor even on oracle MCTS states
→ H1/H4/H5 become more likely

B good on oracle states but poor in rollout
→ H2 exposure shift becomes more likely
```

---

## 7. Diagnostic B — First-Deviation Analysis

### Question

> When a known successful route is used as the reference, where does the learned router first disagree?

For each replay-valid successful route:

1. start from the stored trigger state;
2. feed the **oracle routed state** at each layer;
3. query A or B;
4. compare predicted action with stored successful action;
5. record first disagreement.

The state remains oracle/teacher-forced, so this isolates action recognition from state divergence.

Classify the first deviation:

```text
BEFORE_FIRST_INTERVENTION
AT_FIRST_INTERVENTION
BETWEEN_INTERVENTIONS
AT_SECOND_OR_LATER_INTERVENTION
NO_DEVIATION
```

Report:

```text
A on single routes
A on MCTS routes
B on single routes
B on MCTS routes
```

Most important:

```text
A vs B on single routes
B first vs later MCTS interventions
```

---

## 8. Diagnostic C — Controlled Prefix-Forcing Rollout

This is the key exposure-shift diagnostic.

Use replay-valid successful MCTS routes.

Evaluate:

### C0 — Free rollout

```text
router controls all post-trigger actions
```

### C1 — Force through first non-FULL, then free

Teacher-force the successful route through the first required non-FULL action, then release to the router.

### C2 — Force through first two non-FULL actions, then free

Only for routes with at least two interventions.

### C3 — Force through intervention k, then free

For longer routes, progressively force more of the successful prefix.

### C4 — Full oracle route

Replay the full stored successful trajectory as correctness control.

For every forcing depth report:

```text
final correctness
fraction of later oracle actions reproduced
number of non-FULL actions emitted after release
first deviation after release
```

Plot:

```text
rescue rate vs number of forced interventions
```

Interpretation:

```text
C0 fails, C1 often succeeds
→ first corrective decision is the bottleneck

C1 fails, C2 improves strongly
→ sequential coordination/exposure problem

oracle-prefix forcing restores later actions
→ strong evidence for exposure shift

even long forced prefixes do not help after release
→ H1/H4/H5 more plausible
```

---

## 9. Diagnostic D — State Drift After First Free-Rollout Mistake

For MCTS routes where free rollout deviates, compare:

```text
h_j^oracle
h_j^free
```

at and after the first deviation.

Use the existing Stage-2 input representations.

Measure simple:

```text
cosine distance
relative L2 distance
```

for available components such as:

```text
READ text/query representation
WRITE visual representation / pooled input
```

Question:

> How quickly does one wrong action move the policy away from the teacher-forced MCTS state distribution?

This is secondary evidence for H2.

---

## 10. Diagnostic E — What Happened to A's Four Rescues?

Use the exact four validation W UIDs rescued by:

```text
Experiment A + P90
```

Compare layer-by-layer:

```text
A action
B action
Stage-1 trigger
first non-FULL layer
all non-FULL actions
final prediction
```

For each UID classify:

```text
B remains FULL where A intervenes
B picks a different action
B intervenes at another layer
A/B identical but downstream differs
```

This directly tests:

> Did MCTS supervision erase or suppress simple corrective behavior already learned by A?

---

## 11. Diagnostic F — Inspect A's One C→W Regression

For the single A/P90 C→W sample compare:

```text
dense route
A rollout
B rollout
```

Question:

> Is A's regression caused by the same action pattern that rescued W samples, or unrelated over-intervention?

This is descriptive only because N=1.

---

## 12. Diagnostic G — Negative Transfer on Single States

Evaluate A and B on the **same single-route oracle states**.

Compute paired:

```text
Δ target-action probability
Δ non-FULL recall
Δ FULL recall
Δ action accuracy
```

for:

```text
all single states
single corrective states only
single FULL states only
```

If B is systematically worse on single corrective states:

> MCTS supervision caused measurable negative transfer.

This directly tests H3.

---

## 13. Diagnostic H — Label Ambiguity

Use existing observed-action metadata.

For every state where A/B top-1 differs from the chosen route target, record:

```text
number of observed successful actions
observed successful action set
```

Compare error rates for:

```text
single-observed-action states
multi-valid states
```

Also compute:

```text
sum of predicted probability over all observed successful actions
```

If many nominal errors place high mass on another observed-successful action:

> single-label CE may be mischaracterizing valid behavior.

This supports H4 and motivates a set-valued loss later.

---

## 14. Diagnostic I — FULL Bias / Action Confidence

Compare A and B on identical oracle states.

Report:

```text
p(FULL)
max p(non-FULL)
FULL - best_nonFULL margin
```

for:

```text
single corrective states
MCTS first-intervention states
MCTS later-intervention states
preservation states
```

Question:

> Did MCTS supervision globally push the decision boundary toward conservative FULL?

This is motivated by:

```text
P90 any-non-FULL usage:
A = 17.36%
B = 4.13%
```

---

## 15. Minimal Statistical Treatment

Use UID-level bootstrap for major paired comparisons:

```text
A vs B on single corrective-state recall
A vs B target-action probability
prefix-forcing rescue improvement
```

Do not overcomplicate statistical testing.

The purpose is mechanism diagnosis.

---

## 16. Primary Decision Table

Fill at the end:

| Evidence | H1 Learning | H2 Exposure | H3 Negative Transfer | H4 Ambiguity | H5 Representation |
|---|---:|---:|---:|---:|---:|
| B oracle MCTS accuracy low | supports | weak | possible | possible | possible |
| B oracle good, free bad | weak | strong | weak | weak | weak |
| Prefix forcing restores rescue | weak | strong | weak | weak | possible |
| B worse than A on single states | weak | weak | strong | possible | weak |
| Errors concentrated on multi-valid states | weak | weak | possible | strong | weak |
| Later interventions fail even on oracle states | possible | weak | possible | possible | supports |

Do not force one explanation if several remain consistent.

---

## 17. Next-Step Decision Logic

### Case A — Exposure shift dominates

Evidence:

```text
B predicts oracle MCTS states reasonably well
free rollout fails
prefix forcing restores later behavior
```

Next experiment:

```text
small on-policy / DAgger-style corrective-state collection
or
curriculum with partial oracle-prefix forcing
```

Do not discard sequential Stage-2.

### Case B — Negative transfer dominates

Evidence:

```text
B becomes worse than A on exact same single corrective states
A-rescued UIDs lose their interventions under B
```

Next smallest experiment:

```text
initialize from A checkpoint
+
low-rate MCTS fine-tuning
```

Suggested first ratio:

```text
Single : MCTS = 4 : 1
```

Do not jump to RL.

### Case C — Label ambiguity dominates

Evidence:

```text
many nominal errors choose another observed-successful action
multi-valid states fail much more often
```

Next:

```text
observed-valid-set loss
-loss = log(sum p(valid actions))
```

Keep architecture fixed.

### Case D — Action learning itself is weak

Evidence:

```text
B fails even on oracle MCTS states
including first interventions
ambiguity does not explain it
```

Next:

```text
bounded MCTS-specific overfit pilot
```

before changing free-rollout training.

### Case E — Representation limitation is supported

Only consider strongly if:

```text
oracle states are clean
labels are not highly ambiguous
small overfit is weak
later interventions remain inseparable
```

Then test one minimal representation addition.

Do not abandon Stage-2 from the current rollout result alone.

---

## 18. Why No New Training Yet

Different diagnoses imply different fixes:

```text
exposure shift     → on-policy states
negative transfer → supervision weighting
ambiguity          → set-valued objective
learning failure   → overfit/optimization audit
representation     → input/model change
```

Changing training before diagnosis would confound the problem.

---

## 19. Required Outputs

Use:

```text
analysis/dense_failure_stage2/mcts_failure_diagnosis/
```

Create:

```text
protocol.md

oracle_state/
    action_metrics_A.csv
    action_metrics_B.csv
    route_complexity_breakdown.csv
    intervention_index_breakdown.csv

first_deviation/
    per_route_first_deviation.jsonl
    summary.csv

prefix_forcing/
    per_route_results.jsonl
    forcing_depth_summary.csv

state_drift/
    per_route_state_distance.jsonl
    summary.csv

case_studies/
    A_rescued_W_comparison.jsonl
    A_regressed_C_comparison.jsonl

negative_transfer/
    single_state_A_vs_B.csv
    paired_bootstrap.csv

ambiguity/
    error_by_valid_action_count.csv
    observed_set_probability.csv

confidence/
    full_margin_by_state_type.csv

figures/
    oracle_action_recall_A_vs_B.png
    first_deviation_distribution.png
    rescue_vs_forced_prefix.png
    state_drift_after_deviation.png
    single_negative_transfer.png
    full_margin_A_vs_B.png

summaries/
    mcts_failure_diagnosis_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 20. `mcts_failure_diagnosis_summary.md` Must Answer

1. Can A predict its single-route actions on oracle states?
2. Can B predict MCTS actions on oracle states?
3. Does B fail mainly at the first intervention or later?
4. How much does prefix forcing restore successful rollout?
5. How quickly do free states diverge after the first wrong action?
6. Did B lose A's learned single corrective behavior?
7. Why did A's four P90 rescued W samples stop being rescued in B?
8. Is label ambiguity associated with apparent errors?
9. Did B become globally more FULL-biased?
10. Which hypothesis or combination best explains the result?
11. What does the evidence **not** justify concluding?

---

## 21. `next_stage2_recommendation.md`

Recommend **one smallest next experiment** based on the diagnosis.

For the chosen experiment explicitly state:

```text
Why it matters
What hypothesis it tests
What positive result would mean
What negative result would mean
What it still cannot establish
```

Do not recommend multiple large changes simultaneously.

---

## 22. Stop Rule

STOP after diagnosis and one recommended next experiment.

Do not:

```text
retrain Stage-2
run new MCTS search
change Stage-1
change threshold
run held-out test
add RL
add layer embeddings
add Stage-1 latent features
```

unless the diagnosis specifically justifies one of them.

---

## 23. Core Principle

The current negative result does **not** establish:

> "MCTS / sequential corrective routing does not work."

It establishes only:

> **Under the current plain supervised training contract, adding MCTS trajectory supervision did not translate into free-rollout corrections and suppressed the small single-route gain.**

The next experiment must determine **why** before the direction is changed.
