# Stage-2 Observed-Valid-Set Loss Experiment Plan
## Same Single+MCTS Setup, Change Only the Supervision Objective

## 1. Why This Experiment Matters

The latest Stage-2 diagnosis points to one especially direct failure mode:

> **Many replay-valid routed states admit multiple observed successful actions, but training used single-label cross entropy as if only one sampled action were correct.**

This can create contradictory gradients across successful routes.

The latest diagnosis showed:

```text
Experiment A: Single supervision
P90:
    W→C = 4
    C→W = 1
    net = +3

Experiment B: Single + MCTS supervision
P98/P95/P90:
    W→C = 0
    C→W = 0
```

and:

```text
B oracle-state MCTS non-FULL recall = 8.45%
    first intervention = 9.93%
    later intervention = 7.82%

Single-route corrective recall:
    A = 10.78%
    B = 1.36%

Error rate:
    single-valid states = 5.65%
    multi-valid states  = 30.96%
```

Also:

```text
accepting any exact-prefix observed-successful action
recovers 21.63% of multi-valid rows
```

This makes label ambiguity a concrete, testable hypothesis.

The next experiment should therefore ask:

> **If we keep the router, data, sampler, Stage-1, optimizer, and rollout protocol fixed, but stop penalizing alternative observed-successful actions, does corrective-action learning recover?**

This is the smallest experiment that directly tests the diagnosis.

---

## 2. What This Experiment Tests

Primary hypothesis:

```text
H_valid_set:
Single-label CE creates avoidable supervision conflict on multi-valid states.
Using an observed-valid-set objective should improve corrective-action learning
and reduce the negative transfer introduced by MCTS supervision.
```

This experiment is specifically about the **training objective**.

It does not test whether:

```text
MCTS search is globally optimal
multi-step routing is fundamentally necessary
the Stage-2 architecture is sufficient in all settings
exposure shift is fully solved
```

---

## 3. What a Positive Result Would Mean

A positive result would be:

```text
oracle-state corrective recall improves
AND/OR
single-route corrective behavior is retained better
AND
free-rollout corrective behavior improves
```

while architecture and data remain unchanged.

This would support:

> **The current Stage-2 failure was substantially caused by treating one successful action as the unique ground-truth action.**

It would justify keeping the current Stage-2 architecture and continuing with multi-valid supervision.

---

## 4. What a Negative Result Would Mean

If valid-set loss does **not** improve oracle-state corrective learning:

```text
label ambiguity alone is insufficient to explain the failure
```

This would **not** establish that:

```text
MCTS supervision is useless
sequential routing should be abandoned
```

It would instead increase the relative plausibility of:

```text
optimization weakness
representation insufficiency
remaining supervision mismatch
```

If oracle-state metrics improve but free rollout does not:

```text
the label objective was a real problem,
but exposure shift becomes the next bottleneck
```

---

## 5. Frozen Experimental Reference

Use the completed shared-union Experiment-B setup as the matched reference.

Reference B:

```text
Training:
    Union Preservation
    + Union Single
    + Union MCTS

Loss:
    plain 4-way single-label CE
```

Reference free-rollout behavior:

```text
P98: W→C 0, C→W 0
P95: W→C 0, C→W 0
P90: W→C 0, C→W 0
```

The new experiment should replicate this setup except for the loss target construction.

---

## 6. Frozen Components

Keep exactly fixed:

```text
Qwen2.5-VL backbone
robust ALL-source Stage-1 head
P98 / P95 / P90 thresholds
Stage-2 READ branch
Stage-2 WRITE branch
Stage-2 action head
router hidden dimensions
optimizer family
learning-rate schedule
training update budget
checkpoint-selection rule
C:W sample-draw mixture
Single:MCTS supervision-type mixture
route sampler
state sampler
validation set
free-rollout evaluator
random seeds where feasible
```

Do not add:

```text
layer embeddings
Stage-1 representation
source ID
dataset ID
focal loss
class weights
RL
DAgger/on-policy states
new search
```

The intended changed variable is:

```text
single-label CE
→ observed-valid-set loss
```

---

## 7. Core Definition: Exact Observed Valid Set

For a routed state `s`, define:

```text
A_obs(s)
```

as the set of actions that were **actually observed to lead to a replay-valid correct completion from the exact same entering routed state**.

Example:

```text
same routed state s at layer 17

READ_ONLY  → replay-valid correct route
IGNORE     → replay-valid correct route
WRITE_ONLY → no observed correct route
FULL       → no observed correct route
```

Then:

```text
A_obs(s) = {READ_ONLY, IGNORE}
```

Important:

> Only actions supported by an **exact state / exact routed-prefix identity** may be merged.

Do not merge labels merely because:

```text
UID matches
layer matches
hidden states are numerically close
routes look similar
```

Exact provenance/state identity is required.

---

## 8. Preservation-State Valid Sets

For triggered Dense-C preservation states:

```text
FULL
```

is always an observed valid action under the replay-valid dense suffix.

Therefore:

```text
A_obs(s) includes FULL
```

If the existing search corpus also proves another action from the **same exact state** preserves correctness, it may also be included.

However, do not launch new search to enlarge preservation valid sets.

Use only already-observed evidence.

---

## 9. Single-Route Valid Sets

For single-intervention supervision, combine all replay-valid successful actions observed at the exact same state.

Example:

```text
same state at L12

Route A:
    READ_ONLY at L12 → C

Route B:
    IGNORE at L12 → C
```

Then train with:

```text
A_obs = {READ_ONLY, IGNORE}
```

rather than sampling one as the unique target.

---

## 10. MCTS Valid Sets

For MCTS trajectories, exact-state identity is essential.

Two routes may share:

```text
same UID
same layer
```

but differ in earlier actions, creating different routed states.

Do **not** merge their actions unless the entering state/prefix is exactly the same.

For each exact MCTS routed state:

```text
collect all observed replay-valid next actions
```

into:

```text
A_obs(s)
```

---

## 11. State Canonicalization Audit

Before training, create a deterministic state identity.

Preferred identity:

```text
UID
layer
exact routed-prefix action sequence
state shard/index or exact stored state ID
```

If the existing routed-state store already provides a unique exact state ID, reuse it.

Verify:

```text
same state_id → same entering routed state
different prefix → different state_id unless exact state equality is explicitly proven
```

Do not deduplicate by approximate floating-point similarity.

---

## 12. Valid-Set Construction Audit

Build a table:

```text
state_id
uid
layer
route_source
observed_valid_actions
valid_action_count
contains_FULL
contains_nonFULL
```

Report:

```text
# states with 1 valid action
# states with 2 valid actions
# states with 3 valid actions
# states with 4 valid actions
```

and separately for:

```text
preservation
single
MCTS
```

Also report the most common sets:

```text
{FULL}
{RO}
{WO}
{IGNORE}
{RO, IGNORE}
{RO, WO}
{WO, IGNORE}
{FULL, ...}
...
```

---

## 13. Primary Loss

For router probabilities:

```text
p(a | s)
```

use:

```text
L_set(s)
=
-log(
    sum_{a in A_obs(s)} p(a | s)
)
```

This is the primary observed-valid-set loss.

Equivalent logits implementation may use stable `logsumexp`.

Do not renormalize only over valid classes before computing the full probability mass.

The objective is:

> maximize total probability assigned to any observed-successful action.

---

## 14. Numerical Stability

Implement loss using logits `z_a`.

Stable form:

```text
log_p_valid
=
logsumexp(z_a for a in A_obs)
-
logsumexp(z_all_actions)

loss = -log_p_valid
```

Do not compute probability sums in low precision if it risks underflow.

Router computation should use the same stable precision setup that previously avoided BF16 NaNs.

Fail fast on:

```text
NaN
Inf
empty valid set
invalid action index
```

---

## 15. Keep Single-Valid States Equivalent to CE

For:

```text
|A_obs(s)| = 1
```

the loss should be mathematically identical to ordinary CE.

Add an automated equivalence test:

```text
single-valid set loss == CE
```

within tight numerical tolerance.

This ensures the new objective only changes genuinely multi-valid supervision.

---

## 16. Do Not Treat Unobserved as Invalid in Interpretation

The training loss necessarily assigns probability outside `A_obs` as non-success mass.

But in reporting, use the wording:

```text
observed-valid
```

not:

```text
all valid
```

because bounded search may have missed successful alternatives.

The experiment tests whether honoring **known** alternatives improves learning.

---

## 17. Matched Training Experiment

Name the new model:

```text
Experiment C: Single + MCTS + Observed-Valid-Set Loss
```

Train from the same initialization policy as Experiment B.

Preferred:

```text
same base/random initialization seed as B
```

if exact reproducibility is available.

Do not initialize from Experiment A in the primary comparison, because that would add another variable.

Primary matched comparison:

```text
B:
same data + single-label CE

C:
same data + observed-valid-set loss
```

---

## 18. Training Data and Sampler

Use the exact same union corpora and sampling contract as Experiment B:

```text
Union Preservation
Union Single
Union MCTS
```

Keep:

```text
C:W ratio
Single:MCTS W-draw ratio
route sampling
state count per draw
training update count
```

fixed.

When a sampled routed state is drawn, replace its sampled one-hot target with its exact:

```text
A_obs(state_id)
```

set.

---

## 19. No Change to Route Sampling Yet

Even if a sampled route is only one of many successful routes, keep the existing route sampler.

The valid-set lookup handles contradictory next-action labels at shared exact states.

Do not simultaneously redesign:

```text
route weighting
MCTS weighting
curriculum
```

in this experiment.

---

## 20. Smoke Test

Before full training, run a small overfit smoke containing deliberately selected:

```text
single-valid states
2-valid states
3-valid states
preservation FULL states
single corrective states
MCTS corrective states
```

Acceptance checks:

```text
loss decreases
valid-set probability mass rises
non-FULL valid mass rises on corrective states
FULL remains high on preservation states
no NaNs/Infs
single-valid loss matches CE behavior
```

Do not require one specific valid action to dominate on multi-valid states.

---

## 21. Primary Diagnostic Before Free Rollout

The first success/failure decision should be made on **oracle routed states**, because the tested hypothesis is supervision ambiguity.

Evaluate B and C on the exact same oracle-state corpus.

Report:

```text
Observed-valid-set top-1 accuracy:
    top-1 ∈ A_obs

Observed-valid probability mass:
    sum p(a), a ∈ A_obs

non-FULL observed-valid recall

single-route corrective-state recall

MCTS first-intervention recall

MCTS later-intervention recall

preservation FULL recall
```

This evaluation should not force a single target when multiple actions are observed-valid.

---

## 22. Critical Matched Metrics

Compare B vs C:

```text
B oracle MCTS non-FULL recall = 8.45%
C oracle MCTS non-FULL recall = ?

B first-intervention recall = 9.93%
C first-intervention recall = ?

B later-intervention recall = 7.82%
C later-intervention recall = ?

B single-route corrective recall = 1.36%
C single-route corrective recall = ?
```

Also compare against A's single-route reference:

```text
A single-route corrective recall = 10.78%
```

Key question:

> Does C recover MCTS action learning without destroying the simpler single corrective behavior?

---

## 23. Multi-Valid-Specific Metrics

Stratify oracle states by:

```text
|A_obs| = 1
|A_obs| >= 2
```

Report:

```text
top-1 observed-valid accuracy
valid probability mass
FULL margin
non-FULL valid mass
```

The direct hypothesis predicts:

```text
largest improvement on multi-valid states
```

with limited change on single-valid states.

If C improves only single-valid states, the intended mechanism is not supported.

---

## 24. Negative-Transfer Check

Evaluate B and C on the exact same single corrective states.

Use UID-level paired bootstrap for:

```text
non-FULL recall
observed-valid top-1 accuracy
valid probability mass
```

Question:

> Does valid-set loss reduce the negative transfer that B introduced into single-route behavior?

This is a key secondary endpoint.

---

## 25. Free-Rollout Evaluation

Only after oracle-state evaluation is complete, run the same selected C checkpoint under:

```text
P98
P95
P90
```

using the identical free-rollout evaluator.

Do not retrain between thresholds.

Report:

```text
W→C
C→W
C→C
W→W
routed accuracy
net correction
fraction using any non-FULL
post-trigger FULL fraction
```

---

## 26. Primary Development Threshold

Use:

```text
P90
```

as the primary mechanism-development view because Experiment A had actual corrective behavior there.

Still report P98/P95 for completeness.

Do not select P90 as final deployment threshold from this experiment alone.

---

## 27. A/B/C Comparison

Produce:

| Experiment | Loss | P90 W→C | P90 C→W | Net | Any non-FULL | Routed accuracy |
|---|---|---:|---:|---:|---:|---:|
| A | Single-only CE | 4 | 1 | +3 | 17.36% | 50.375% |
| B | Single+MCTS CE | 0 | 0 | 0 | 4.13% | 50.000% |
| C | Single+MCTS valid-set | ? | ? | ? | ? | ? |

Also report oracle-state comparison separately.

---

## 28. Decision Logic

### Case A — Valid-set loss clearly improves oracle-state learning and free rollout

Evidence:

```text
MCTS non-FULL recall rises
single corrective recall recovers
P90 non-FULL use rises
W→C returns/improves without excessive C→W
```

Interpretation:

> Single-label supervision conflict was a major Stage-2 bottleneck.

Next step may focus on remaining exposure shift or threshold selection.

### Case B — Oracle metrics improve, free rollout remains weak

Interpretation:

> Valid-set loss fixes action supervision, but exposure shift remains the next bottleneck.

This would justify a separate, narrowly scoped on-policy / prefix-forcing training experiment.

Do not change representation yet.

### Case C — Single behavior recovers, MCTS behavior remains poor

Interpretation:

> Valid-set loss reduces negative transfer but does not make multi-step corrective actions sufficiently learnable.

Next step should diagnose MCTS optimization/representation more narrowly.

Do not claim MCTS is useless.

### Case D — No meaningful oracle-state improvement

Interpretation:

> Multi-valid supervision conflict was not sufficient to explain the poor corrective-action learning.

Then do not continue elaborating valid-set objectives.

The next discriminating experiment should be a bounded MCTS-state overfit / representation test.

### Case E — C becomes more intervention-heavy and regressions rise

Interpretation:

> Removing label conflict increased corrective confidence but harmed preservation.

Then the problem becomes conservative action selection, not failure to learn actions.

Do not immediately lower Stage-1 thresholds.

---

## 29. What a Negative Result Does Not Mean

Even if Experiment C fails, do not conclude:

```text
Stage-2 routing is impossible
MCTS search is useless
visual READ/WRITE control does not generalize
```

It would only show:

> **Observed-valid-set supervision alone is insufficient under the current router and offline training distribution.**

---

## 30. Required Outputs

Use:

```text
analysis/dense_failure_stage2/observed_valid_set_loss/
```

Create:

```text
protocol.md

valid_sets/
    exact_state_index.jsonl
    observed_valid_actions.jsonl
    valid_set_distribution.csv
    provenance_audit.json

smoke/
    single_valid_ce_equivalence.json
    overfit_metrics.csv
    overfit_valid_mass.csv

training/
    config.yaml
    train_log.jsonl
    checkpoint_manifest.json
    selected_checkpoint.json

oracle_eval/
    per_state_results.jsonl
    overall_metrics.csv
    validity_count_breakdown.csv
    intervention_index_breakdown.csv
    single_negative_transfer.csv
    paired_bootstrap.csv

free_rollout/
    P98/
        per_sample_results.jsonl
        metrics.json
    P95/
        per_sample_results.jsonl
        metrics.json
    P90/
        per_sample_results.jsonl
        metrics.json
    threshold_comparison.csv
    action_behavior.csv

comparison/
    A_B_C_oracle.csv
    A_B_C_rollout.csv

figures/
    valid_set_size_distribution.png
    oracle_nonfull_recall_ABC.png
    multi_valid_accuracy_B_vs_C.png
    single_negative_transfer_B_vs_C.png
    p90_rescue_regression_ABC.png
    valid_probability_mass.png

summaries/
    observed_valid_set_loss_summary.md
    next_stage2_decision.md

artifact_manifest.json
```

---

## 31. `observed_valid_set_loss_summary.md` Must Answer

1. How many exact routed states have 1/2/3/4 observed-valid actions?
2. Does C improve observed-valid top-1 accuracy?
3. Does C improve total probability mass on observed-valid actions?
4. Does MCTS first-intervention recall improve over B's 9.93%?
5. Does later-intervention recall improve over B's 7.82%?
6. Does single corrective recall recover from B's 1.36% toward A's 10.78%?
7. Is improvement concentrated on multi-valid states?
8. Does C reduce FULL bias?
9. What happens under P90 free rollout?
10. Does W→C recover?
11. Does C→W increase?
12. Which original diagnosis is strengthened or weakened?
13. What does the experiment still not establish?

---

## 32. `next_stage2_decision.md`

Recommend exactly **one** next step based on the result.

For that step state:

```text
Why it matters
Which remaining hypothesis it tests
What a positive result means
What a negative result means
What cannot be concluded
```

Do not recommend several architectural/training changes at once.

---

## 33. Stop Rule

STOP after:

```text
valid-set construction
matched C training
oracle-state B-vs-C evaluation
P98/P95/P90 free rollout
diagnostic decision
```

Do not automatically:

```text
add on-policy training
add layer embeddings
add Stage-1 latent
change Stage-1 thresholds
run new search
run held-out test
```

Those require a separate plan justified by the result.

---

## 34. Core Principle

Change exactly the component implicated by the diagnosis:

```text
same router
same data
same sampler
same optimizer
same Stage-1
same thresholds

single-label CE
        ↓
observed-valid-set loss
```

The experiment is important because it can directly test whether the current MCTS failure comes from **incorrectly treating alternative successful actions as mutually exclusive labels**.

A failure of this experiment should narrow the hypothesis space; it should not trigger abandonment of the broader corrective-routing direction.
