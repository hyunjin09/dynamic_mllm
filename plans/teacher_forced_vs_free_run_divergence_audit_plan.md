# Teacher-Forced vs Free-Run Divergence Audit
## Is the closed-loop trajectory-set router failing to learn corrective actions, or failing only after it acts on its own?

## 0. Purpose

Latest full result:

```text
Dense                      : W→C 0, C→W 0,  Net  0
Sequential-A               : W→C 3, C→W 19, Net -16
Open-loop Program          : W→C 3, C→W 8,  Net -5
Closed-loop Trajectory-Set : W→C 0, C→W 8,  Net -8
```

The closed-loop trajectory-set router intervened rarely:

```text
P(any non-FULL | triggered W) = 39 / 496 = 7.86%
P(any non-FULL | triggered C) = 32 / 405 = 7.90%
```

Training diagnostics:

```text
mean top route-responsibility share      = 0.6608
median best-route geometric action prob = 0.7269
```

These sequence-level values may still look good even if the router misses the sparse non-FULL actions that actually define a corrective route.

Before changing the representation or starting on-policy relabeling, answer:

> **Given the exact routed state from a successful trajectory, does the router actually choose the corrective action?**

If yes, then ask:

> **Does it fail only after free rollout causes the state/prefix to drift away from successful expert/search trajectories?**

No new Stage-2 training is allowed in this audit.

---

## 1. Competing hypotheses

### H1 — Corrective-action learning / objective failure

Even on an exact expert routed state:

```text
expert state s_l
→ frozen router
→ wrong action, often FULL
```

especially at the first non-FULL intervention.

Then the current trajectory-set objective did not adequately learn sparse corrective actions.

This is not an exposure problem.

### H2 — On-policy state-distribution / exposure failure

On exact expert states the router predicts corrective actions well, but under free rollout:

```text
policy deviation
→ different routed state
→ subsequent policy changes
→ leaves known successful-route support
→ failure compounds
```

Then on-policy relabeling becomes justified.

### H3 — Generalization / representation failure

The full-refit checkpoint fits seen training UIDs, but the frozen group-disjoint internal-dev checkpoint performs poorly on held-out expert states.

Then the state/action mapping itself does not generalize.

---

## 2. Frozen components

Freeze:

```text
Qwen2.5-VL-7B-Instruct
robust ALL-source Stage-1
P90 threshold
closed-loop trajectory-set Stage-2 architecture
full-refit Stage-2 checkpoint
full 569-UID / 4,948-trajectory corpus
35,565 exact routed-state cache
four action semantics
same-layer READ/WRITE semantics
route-responsibility computation
LMMS correctness evaluator
```

Do not:

```text
retrain Stage-2
change objective
change Stage-1
change thresholds
add a non-FULL bonus
add a reranker
launch DAgger/on-policy relabeling
```

---

## 3. Audit population

Use the full training-side corpus:

```text
569 UIDs
106 Dense-C preservation UIDs
463 Dense-W corrective UIDs

4,948 replay-valid trajectories
35,565 unique exact routed states
69,178 route-state occurrences
```

Do not subsample the scientific audit.

---

## 4. Two evaluation views

### View A — Seen-UID fit audit

Use:

```text
final full-refit checkpoint
→ all 569 UIDs used by that refit
```

Question:

> Can the objective reproduce corrective decisions even on states it trained on?

If no, do not blame exposure.

### View B — Group-disjoint generalization audit

If the exact frozen internal-dev checkpoint and split from the completed experiment still exist, evaluate:

```text
internal-dev checkpoint
→ held-out UID/image-group-disjoint dev UIDs
```

using the same metrics.

Do not reconstruct a new split after seeing results.

If the exact checkpoint is unavailable, report:

```text
not estimable from frozen artifacts
```

---

## 5. Freeze one reference route per W UID

For Dense-W UID `i`, compute for every successful route:

```text
route_logp_ik
=
Σ_l log π_theta(a_l^ik | s_l^ik)
```

and trajectory responsibility:

```text
r_ik = softmax_k(route_logp_ik)
```

Freeze:

```text
tau_i* = highest-responsibility successful trajectory
```

before any hybrid rollout.

Do not choose the reference route based on later rollout success.

---

## 6. Teacher-forced one-step audit on all route states

For every route-state occurrence:

```text
(s_l^ik, a_l^ik)
```

feed the exact cached routed state into the frozen router and record:

```text
p(FULL)
p(READ_ONLY)
p(WRITE_ONLY)
p(IGNORE)

top-1 action
expert-route action
expert-action probability
expert-action rank
expert-action vs FULL logit margin
```

Do not execute the predicted action in this one-step audit.

---

## 7. Separate FULL from corrective action learning

Report separately:

```text
FULL-target top-1 accuracy
non-FULL-target top-1 accuracy

READ_ONLY recall
WRITE_ONLY recall
IGNORE recall
```

Also report:

```text
mean/median expert-action probability
mean/median expert-action-vs-FULL margin
```

Do not report only aggregate action accuracy because FULL dominates route length.

---

## 8. First non-FULL intervention audit

For each successful Dense-W trajectory define:

```text
l_first(τ)
=
first layer >= L*
where expert action != FULL
```

At the exact state `s_l_first` report:

```text
first-nonFULL top-1 recall
target probability
target rank
target-vs-FULL margin
```

Break down by:

```text
all successful routes
highest-responsibility route per UID
single provenance
MCTS / robust / completeness-audit provenance
layer
```

This is the primary corrective-action learning metric.

---

## 9. Highest-responsibility-route metrics

For every W UID's frozen `tau_i*`, report:

```text
FULL-action top-1 accuracy
non-FULL-action top-1 accuracy
first-nonFULL top-1 accuracy
first-nonFULL probability
first-nonFULL vs FULL margin
```

The marginal objective does not have to imitate every route, but it should at least fit the route that receives its largest responsibility.

If even this route's corrective actions are not top-1, the objective is not producing usable specialization.

---

## 10. Prefix-compatible route support

For W UID `i`:

```text
T_i = all known replay-valid successful trajectories
```

Given a free-run action prefix:

```text
p_l = [a_L*, ..., a_l-1]
```

define:

```text
C_i(p_l)
=
{tau in T_i :
 tau has exactly prefix p_l}
```

and supported next actions:

```text
A_i(p_l)
=
{next action of tau at layer l :
 tau in C_i(p_l)}
```

This respects multiple successful routes without collapsing them globally into local labels.

---

## 11. First off-support layer under free rollout

Run the frozen full-refit policy freely from the P90 trigger on every training UID.

For each W UID, track compatible known successful routes.

Define:

```text
l_off
=
first layer where chosen action a_l not in A_i(p_l)
```

If compatibility with at least one known successful route survives to the end:

```text
l_off = NONE
```

Report:

```text
fraction leaving support
trigger-to-off-support delay
off-support layer
off-support chosen action
size of compatible route set before divergence
```

Important:

> Off-support means outside the known successful route corpus, not provably invalid.

---

## 12. Free-run success on training UIDs

Run closed-loop free inference on:

```text
all 463 Dense-W training UIDs
all 106 Dense-C training UIDs
```

Report:

```text
training-W free-run rescue rate
training-C free-run preservation rate
```

Every W UID has at least one replay-valid successful trajectory.

Interpretation:

```text
good expert-state action learning
+ poor training free-run success
→ exposure/on-policy evidence

poor expert-state action learning
+ poor training free-run success
→ objective/action-learning problem comes first
```

---

## 13. Hybrid release experiment

Use the frozen highest-responsibility route `tau_i*` for each Dense-W UID.

### R0 — Free from trigger

```text
start at L*
policy acts freely through L27
```

### R1 — Force expert prefix until first non-FULL state, then release

If first non-FULL is at `l_first`:

```text
force tau_i* from L* through l_first-1
arrive at exact expert state s_l_first
let policy choose at l_first
then free-run to the end
```

This removes all pre-intervention state drift.

### R2 — Force first non-FULL intervention, then release

```text
force tau_i* from L* through l_first
including the first non-FULL action
then free-run from l_first+1
```

This tests whether the first corrective action itself is the main bottleneck.

---

## 14. R0/R1/R2 interpretation

Report final correct counts/rates for each condition.

Interpret:

```text
R1 remains low
→ exact expert corrective state is not enough
→ objective/action-learning issue likely

R1 low, R2 jumps
→ first non-FULL action itself is underlearned

R1 >> R0
→ pre-intervention free rollout causes harmful drift
→ exposure before intervention

R2 high initially but later free rollout fails
→ post-intervention exposure / state-distribution problem
```

---

## 15. First-divergence logit audit

At each W UID's first off-support layer record:

```text
layer
chosen action
supported action set A_i(p)
probability mass on supported actions

P_supported
=
Σ_{a in A_i(p)} π(a | s)

FULL probability
best-supported-action probability
chosen-vs-best-supported margin
```

This tells whether divergence is:

```text
confident
or
near-tie / unstable
```

---

## 16. Prefix-support survival curve

Across all 463 W training UIDs compute:

```text
S(d)
=
fraction still compatible with at least one known
successful trajectory d layers after trigger
```

Also align a survival analysis relative to:

```text
first expert non-FULL intervention
```

Plot both.

---

## 17. Dense-C control

For each of 106 triggered Dense-C training UIDs, canonical expert route is:

```text
FULL, FULL, ..., FULL
```

Teacher-forced:

```text
FULL top-1 accuracy
p(FULL)
FULL-vs-best-nonFULL margin
```

Free-run:

```text
first deviation from all-FULL
whether final answer regresses
```

This controls whether preservation is truly learned and stable.

---

## 18. Layer and provenance breakdown

Report key W metrics by:

```text
Early  : 0-8
Middle : 9-18
Late   : 19-27
```

and, where support permits, exact layer.

Also break down by route provenance:

```text
single
original MCTS
robust search
completeness audit
```

Questions:

```text
Are single-route first interventions easier?
Are MCTS/completeness routes hard even on exact expert states?
Does off-support happen earlier on complex routes?
```

---

## 19. Seen-vs-held-out generalization matrix

If the frozen internal-dev checkpoint exists, produce:

| View | First-nonFULL recall | Non-FULL top1 | Free-run W success | Off-support rate |
|---|---:|---:|---:|---:|
| Full-refit on seen UIDs | | | | |
| Frozen dev checkpoint on held-out UIDs | | | | |

Interpretation:

```text
seen expert fit weak
→ objective / fit problem

seen fit strong, held-out fit weak
→ representation/action generalization problem

seen + held-out expert fit strong, free-run weak
→ on-policy/exposure problem
```

---

## 20. Primary decision logic

### Case A — Corrective-action learning failure

Evidence:

```text
seen-UID first-nonFULL recall low
R1 low
R2 substantially improves over R1
```

Conclusion:

> Sparse corrective actions were not adequately learned by the trajectory marginal objective.

Next experiment should change the objective, not use DAgger.

### Case B — Pre-intervention exposure failure

Evidence:

```text
expert first-nonFULL recall high
R0 poor
R1 strongly improves
off-support often occurs before first intervention
```

Conclusion:

> Free policy states drift before the critical intervention.

On-policy relabeling is justified.

### Case C — Post-intervention exposure failure

Evidence:

```text
first-nonFULL recall high
policy can enter corrective branch
later free rollout leaves support/fails
```

Conclusion:

> The policy cannot stay on a successful corrective state distribution.

On-policy corrective-state relabeling is justified.

### Case D — Generalization failure

Evidence:

```text
seen expert-state recall high
held-out group-disjoint expert-state recall much lower
```

Conclusion:

> The state/action mapping is not generalizing.

Do not jump to DAgger first.

### Case E — Mixed

If multiple failure modes occur, recommend the earliest dominant failure in the rollout.

---

## 21. Decision-grade metrics

The final recommendation must rely mainly on:

```text
1. seen first-nonFULL top-1 recall
2. seen all-nonFULL top-1 recall
3. target non-FULL probability vs FULL
4. R0 success
5. R1 success
6. R2 success
7. first off-support delay
8. supported-action probability mass at divergence
9. held-out expert-state recall if available
```

Do not use overall geometric route probability alone.

---

## 22. External-label firewall

Do not use the 19,960 external benchmark labels for:

```text
route selection
threshold tuning
metric selection
release-point selection
model changes
```

This audit is training/internal-dev based.

The external full result motivates the audit but does not train anything here.

---

## 23. What this audit can establish

This audit can distinguish whether the current router primarily fails because:

```text
A. corrective actions were never learned,
B. free rollout leaves successful expert-state support,
C. state/action mapping fails to generalize,
D. or a mixture.
```

That determines whether the next training experiment should target:

```text
objective
on-policy relabeling
or representation/generalization
```

---

## 24. What this audit cannot establish

Do not claim:

```text
off-support means invalid
known search routes are exhaustive
expert route action is uniquely correct
DAgger will necessarily work
training free-run success guarantees external success
```

Known routes are only observed successful trajectories.

---

## 25. No new full training

Do not launch:

```text
new Stage-2 training
new MCTS search
new label collection
new external full evaluation
```

until this audit is complete.

---

## 26. Required artifacts

Use:

```text
analysis/dense_failure_stage2/teacher_forced_free_run_audit/
```

Create:

```text
protocol.md

contracts/
    checkpoint_contract.json
    corpus_contract.json
    internal_dev_availability.md

teacher_forced/
    all_route_state_logits.jsonl
    action_class_metrics.csv
    first_nonfull_metrics.csv
    highest_responsibility_route_metrics.csv
    layer_breakdown.csv
    provenance_breakdown.csv

free_run/
    training_uid_rollouts.jsonl
    support_tracking.jsonl
    first_off_support.csv
    prefix_support_survival.csv
    dense_c_control.csv

release/
    selected_reference_routes.jsonl
    r0_free_from_trigger.jsonl
    r1_release_at_first_nonfull.jsonl
    r2_force_first_nonfull_then_release.jsonl
    release_success_summary.csv

generalization/
    seen_vs_heldout_summary.csv
    heldout_teacher_forced_metrics.csv
    heldout_free_run_metrics.csv

metrics/
    decision_metrics.csv
    supported_action_mass_at_divergence.csv
    action_probability_summary.csv

figures/
    first_nonfull_recall_by_layer.png
    target_nonfull_vs_full_probability.png
    prefix_support_survival.png
    first_off_support_distribution.png
    r0_r1_r2_success.png
    seen_vs_heldout_expert_recall.png
    dense_c_preservation_control.png

summaries/
    teacher_forced_free_run_audit_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 27. `teacher_forced_free_run_audit_summary.md` must answer

1. How many UIDs, trajectories, and route-state occurrences were audited?
2. On seen expert states, what is FULL top-1 accuracy?
3. What is non-FULL top-1 accuracy?
4. What is first-nonFULL top-1 recall?
5. What are first-nonFULL target probabilities and margins versus FULL?
6. On highest-responsibility routes, is the first corrective action actually learned?
7. How well are READ_ONLY / WRITE_ONLY / IGNORE learned?
8. What is free-run rescue on the 463 training W UIDs?
9. What fraction leaves known successful-route support?
10. How many layers after trigger does first off-support occur?
11. Does off-support happen before or after first expert intervention?
12. What is supported-action probability mass at divergence?
13. What are R0/R1/R2 final-correct rates?
14. Does removing pre-intervention drift improve success?
15. Does forcing the first non-FULL action improve success?
16. Does the policy remain stable after the first intervention?
17. How stable is Dense-C all-FULL preservation?
18. If available, how much does expert-state recall drop on group-disjoint held-out UIDs?
19. Is the dominant bottleneck objective/action learning, pre-intervention exposure, post-intervention exposure, generalization, or mixed?
20. What result justifies on-policy relabeling?
21. What does the audit not justify concluding?

---

## 28. `next_stage2_recommendation.md`

Recommend exactly one next experiment.

If objective/action learning dominates:

```text
one minimal corrective-action-sensitive objective change
```

If exposure dominates:

```text
one bounded on-policy state collection + expert/search relabeling experiment
```

If generalization dominates:

```text
one minimal routed-state representation/generalization experiment
```

If mixed:

```text
choose the earliest quantitatively dominant failure
```

For the recommendation state:

```text
why it is the smallest discriminating next experiment
what positive evidence means
what negative evidence means
what must not be overinterpreted
```

---

## 29. Stop rule

STOP after:

```text
1. full seen-UID teacher-forced expert-state audit
2. first-nonFULL corrective-action audit
3. full training-UID free rollout
4. successful-route support tracking
5. R0/R1/R2 hybrid release experiment
6. Dense-C preservation control
7. frozen held-out internal-dev audit if available
8. exactly one evidence-based next-stage recommendation
```

Do not modify or retrain Stage-2 in this phase.

---

## 30. Core principle

The current full result:

```text
W→C = 0
C→W = 8
```

does not by itself prove exposure bias.

The correct diagnostic order is:

```text
First:
    can the policy choose the corrective action on the exact expert state?

Then:
    if yes, can it maintain successful behavior after it acts on its own?
```

Only if the first answer is yes and the second is no should the project conclude that **on-policy state-distribution shift** is the primary next bottleneck.
