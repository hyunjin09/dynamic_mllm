# Trigger-Conditioned Corrective Search Pilot Plan

## 1. Objective

Run a bounded pilot to determine whether Stage-1-triggered Dense-W samples can actually be corrected by visual-computation routing after the trigger layer, and to estimate the search budget needed before scaling to all 1,881 triggered Dense-W train samples.

This phase is search/label-feasibility only. Do not train Stage-2.

Main question:

> Given a Dense-W sample whose frozen Stage-1 gate first triggers at layer `l*`, can we find a corrective four-action trajectory over layers `l*...27`?

---

## 2. Frozen inputs

Use the frozen Stage-1 trigger map and the current `Shared Random-4` gate outputs.

For every sampled training example, keep fixed:

```text
uid
dataset
dense correctness = W
first_trigger_layer = l*
```

Do not retrain Stage-1, change thresholds, move trigger layers, or use validation/test trajectories as training-label sources.

---

## 3. Pilot subset

Sample approximately 120 triggered Dense-W training examples.

Stratify by:

```text
Dataset:
- GQA
- ChartQA
- TextVQA

Trigger depth:
- L0
- Early non-L0: L1-L8
- Middle: L9-L18
- Late: L19-L27
```

Because L0 triggers are very common, do not allow the pilot to become almost entirely L0. Ensure enough examples from each supported depth bin and dataset to diagnose failure modes.

Freeze the pilot UID list before search.

---

## 4. Action space

At every treatment layer:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Use the existing verified executor and unchanged READ/WRITE semantics.

For a sample triggering at `l*`:

```text
L0 ... L(l*-1) = original dense FULL prefix
search begins from the actual native state entering L*
```

Do not restart from L0 for the dynamic pilot.

---

## 5. Phase A: exhaustive single-intervention baseline

Before MCTS, test whether one local intervention anywhere after the trigger is sufficient.

For every `j = l* ... 27`, evaluate:

```text
READ_ONLY at j, FULL elsewhere
WRITE_ONLY at j, FULL elsewhere
IGNORE at j, FULL elsewhere
```

The all-FULL suffix is the Dense-W control.

Maximum non-control routes for an L0 trigger:

```text
3 × 28 = 84
```

A sample is `SINGLE-FIXABLE` if at least one such route produces a correct final LMMS-Eval answer.

Save all successful single routes.

Do not send SINGLE-FIXABLE samples into the primary MCTS pipeline; MCTS should focus on unresolved samples.

---

## 6. Phase B: trigger-conditioned MCTS

Run MCTS only for samples that remain wrong after exhaustive single-intervention search.

Root:

```text
actual routed state entering layer l*
```

Tree horizon:

```text
l* ... 27
```

At layer `j`, branch over:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

After action execution, use the resulting routed hidden state as the state for layer `j+1`.

The search must therefore be trajectory-conditioned. Do not score actions at later layers using the original dense trajectory.

---

## 7. MCTS reward

Use a simple terminal reward:

```text
1.0 if final LMMS-Eval answer is correct
0.0 otherwise
```

Reference answers are allowed because this is training-label discovery.

For successful routes, use post-hoc preference:

```text
1. correct > incorrect
2. fewer non-FULL actions preferred
3. if still tied, simpler route preferred
```

Do not add a complicated learned reward in this pilot.

---

## 8. MCTS iteration budget

Hard cap:

```text
max_iterations = 300 per unresolved sample
```

Record search state at:

```text
100 iterations
200 iterations
300 iterations
```

Report:

```text
Fixable@100
Fixable@200
Fixable@300
Gain 100→200
Gain 200→300
```

If clean checkpointing is inconvenient, run to at most 300 iterations and log the iteration of the first successful route so the three rates can be reconstructed.

Do not exceed 300 iterations/sample in this pilot.

The purpose is explicitly to decide whether the full label-generation phase should use roughly 200 or 300 MCTS iterations/sample.

---

## 9. Success handling

When a correct route is first found, record:

```text
first_success_iteration
full suffix action trajectory
final prediction
number of non-FULL actions
```

Do not necessarily terminate immediately. If simple to implement, allow up to 25-50 additional iterations after first success, still respecting the global 300-iteration cap, to find alternative successful routes.

Retain up to 8 distinct successful routes per sample after deduplication.

This is useful because multiple first actions may be valid for the same trigger state.

---

## 10. Outcome classes

Define:

```text
SINGLE-FIXABLE
    rescued by exhaustive one-intervention search

MCTS-FIXABLE
    not single-fixable, but rescued by MCTS within <=300 iterations

UNRESOLVED
    no correct route found under either frozen search budget
```

Important:

```text
UNRESOLVED != provably unfixable
```

It only means no route was found under the current bounded search.

---

## 11. Primary scientific comparison

Compute:

```text
Single rescue rate
= SINGLE-FIXABLE / all pilot triggered-W
```

```text
Additional MCTS rescue rate
= MCTS-FIXABLE / all pilot triggered-W
```

```text
Total bounded correctability
= (SINGLE-FIXABLE + MCTS-FIXABLE) / all pilot triggered-W
```

The key quantity is:

```text
MCTS additional rescue over the single-intervention baseline
```

If this is substantial, it supports the need for a sequential trajectory-conditioned Stage-2.

If single interventions account for most rescue, keep a simpler Stage-2 formulation under consideration.

---

## 12. Route-complexity analysis

For every successful route, compute:

```text
number of non-FULL actions
first non-FULL layer
last non-FULL layer
intervention span
READ_ONLY count
WRITE_ONLY count
IGNORE count
```

For each FIXABLE sample, identify the successful route with the fewest non-FULL actions found.

Report the fraction requiring:

```text
1 intervention
2 interventions
3 interventions
4+ interventions
```

This tells us how complex Stage-2 likely needs to be.

---

## 13. First-action analysis

For MCTS-FIXABLE samples, inspect the action at the trigger layer.

Observed successful first-action set:

```text
A_valid(l*) =
{trigger-layer actions appearing in at least one discovered successful route}
```

Report:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

and the fraction of samples with:

```text
one observed valid first action
multiple observed valid first actions
```

Do not call this exhaustive validity; it is only validity observed under bounded search.

---

## 14. Trigger-depth breakdown

Report separately for:

```text
L0
L1-L8
L9-L18
L19-L27
```

For each group:

```text
N
single rescue rate
MCTS additional rescue
total bounded correctability
Fixable@100/200/300
median non-FULL count
median first-success iteration
```

Do not claim earlier triggering is causally better. Earlier triggers also leave a longer treatment horizon.

---

## 15. Dataset breakdown

Report separately for:

```text
GQA
ChartQA
TextVQA
```

Metrics:

```text
N
single rescue rate
MCTS additional rescue
total bounded correctability
Fixable@100
Fixable@200
Fixable@300
median successful non-FULL count
```

Do not use dataset-specific search rules.

---

## 16. Future Stage-2 supervision capture

For every successful route, replay the exact route under the current executor and save trajectory-conditioned states.

For every layer `j >= l*`, where feasible:

```text
uid
dataset
trigger_layer
current_layer
routed state / feature at layer j
chosen action at layer j
route_id
final_correct = true
```

The intended future training sequence is:

```text
state at Lj
→ action at Lj
→ actual routed state at Lj+1
→ action at Lj+1
→ ...
```

Do not train Stage-2 in this pilot.

---

## 17. Triggered Dense-C

Do not run MCTS on Dense-C in this pilot.

Their conservative preservation trajectory is already known:

```text
FULL → FULL → ... → FULL
```

from the trigger layer onward.

Keep the 39 triggered Dense-C train samples for future preservation supervision.

This pilot focuses only on triggered Dense-W correctability.

---

## 18. Compute logging

For each sample record:

```text
single routes evaluated
MCTS iterations used
terminal evaluations
first-success iteration
wall-clock time
GPU time if available
```

Report:

```text
mean/median runtime per sample
runtime by trigger depth
runtime by dataset
projected cost for all 1,881 triggered-W train samples
```

---

## 19. Budget decision

Use the pilot to choose the full-search budget.

If:

```text
Fixable@200 ≈ Fixable@300
```

freeze approximately:

```text
max_iterations = 200
```

for the full train label-generation phase.

If:

```text
Fixable@300 materially > Fixable@200
```

use:

```text
max_iterations = 300
```

or inspect search efficiency before scaling.

If the curve is still strongly increasing at 300, do not immediately increase MCTS to thousands of iterations. First inspect:

```text
tree policy
rollout policy
reward sparsity
duplicate routes
executor overhead
```

---

## 20. Required outputs

Use:

```text
analysis/dense_failure_stage2/corrective_search_pilot/
```

Create:

```text
protocol.md
pilot_manifest.jsonl

single_search/
    execution_rows.jsonl
    successful_routes.jsonl
    summary.csv

mcts/
    search_rows.jsonl
    successful_routes.jsonl
    per_sample_summary.csv

metrics/
    overall_correctability.csv
    budget_saturation.csv
    trigger_depth_breakdown.csv
    dataset_breakdown.csv
    route_complexity.csv
    first_action_distribution.csv
    compute_summary.csv

future_stage2_labels/
    routed_training_states.jsonl
    successful_route_manifest.jsonl

figures/
    fixability_vs_mcts_budget.png
    single_vs_mcts_rescue.png
    correctability_by_trigger_depth.png
    correctability_by_dataset.png
    intervention_count_distribution.png

decision_summary.md
artifact_manifest.json
```

---

## 21. `decision_summary.md` must answer

1. What fraction of pilot triggered-W samples are rescued by a single intervention?
2. What additional fraction is rescued only by MCTS?
3. What is total bounded correctability?
4. What are Fixable@100, Fixable@200, and Fixable@300?
5. Is 200 iterations sufficient, or does 300 materially improve coverage?
6. How many non-FULL interventions do successful routes usually require?
7. How often are multiple successful first actions observed?
8. Does correctability differ strongly by trigger depth?
9. Does correctability differ strongly by dataset?
10. What is the projected compute for all 1,881 triggered-W training samples?
11. Does the evidence justify a sequential layer-wise Stage-2 policy?
12. What MCTS iteration cap should be frozen for the full corrective-label phase?

---

## 22. Stop rule

STOP after the pilot and decision summary.

Do not:

```text
run MCTS on all 1,881 triggered-W samples
train Stage-2
change Stage-1
change the frozen trigger threshold
use validation/test search results as training labels
run external evaluation
```

The next phase should only be authorized after deciding:

```text
whether multi-layer trajectory search is actually necessary
and
whether the full label-generation budget should be 200 or 300 MCTS iterations/sample
```
