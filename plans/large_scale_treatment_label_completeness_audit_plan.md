# Large-Scale Treatment-Label Completeness Audit Plan
## Before blaming the Stage-2 representation, verify that KEEP / INTERVENE labels are complete enough

## 1. Why this experiment matters

The latest Stage-2 treatment-selectivity diagnostic found weak generalizable separability:

- READ + WRITE OOF AUROC = 0.562
- Optional RW-MLP AUROC = 0.558
- Precision@5/10/20% = 0.123 / 0.135 / 0.123
- Recall@90% precision = 0.001

This suggests the current frozen `z_R / z_W` representation does not obviously contain a strong treatment-selectivity signal.

However, the labels used in that diagnostic were search-derived:

```text
KEEP_REQUIRED:
    A_obs(s) = {FULL}

INTERVENE_REQUIRED:
    FULL not observed successful
    and >=1 non-FULL action observed successful
```

The key limitation is:

> **absence from A_obs(s) does not prove an action is invalid.**

A state labeled KEEP_REQUIRED may actually permit a successful non-FULL continuation that prior search never found.

A state labeled INTERVENE_REQUIRED may actually permit:

```text
FULL now
→ intervene later
→ final correct
```

even though no such continuation was previously observed.

Before changing the Stage-2 representation, ask:

> **How often do supposedly clean KEEP / INTERVENE labels become MIXED when every currently unobserved first action is given a fresh bounded suffix search?**

This phase is a label-identification audit, not a model-improvement experiment.

---

## 2. What this experiment can establish

This audit can establish:

- how incomplete current observed-successful action sets are;
- how often KEEP_REQUIRED is invalidated by a newly discovered non-FULL action;
- how often INTERVENE_REQUIRED is invalidated by a newly discovered FULL continuation;
- which datasets / sources / layers / route types have the highest incompleteness;
- whether the 200-iteration bounded search appears saturated;
- whether treatment-selectivity probe performance improves after contaminated labels are removed.

---

## 3. What this experiment cannot establish

Even after this audit, the new action set remains a bounded-search lower bound.

Failure to discover a successful continuation does not prove impossibility.

Do not claim:

```text
an action is truly invalid
the audited action set is exhaustive
routing cannot work for unresolved branches
```

Use the terminology:

```text
bounded-observed successful action set
```

not:

```text
true valid action set
```

---

## 4. Frozen components

Freeze:

```text
Qwen2.5-VL backbone
robust ALL-source Stage-1
P98 / P95 / P90 trigger maps
four-action executor semantics
LMMS-Eval correctness
existing exact routed-state store
existing route provenance
existing Stage-2 representations z_R / z_W
```

Do not:

```text
retrain Stage-1
retrain Stage-2
change thresholds
change action semantics
use external full-benchmark rescue/regression outcomes for sampling
```

---

## 5. Source population

Start from the full treatment-selectivity census:

```text
569 unique UIDs
21,071 unique exact routed prefix-states

18,438 old KEEP_REQUIRED
1,657 old INTERVENE_REQUIRED
976 old MIXED
```

Use all 21,071 states for the census and sampling frame.

The expensive deep audit runs on a large frozen subset.

---

## 6. Deep-audit sample size

Freeze:

```text
1,200 unique exact routed states
```

before any new action search.

Target allocation:

```text
500 old KEEP_REQUIRED
500 old INTERVENE_REQUIRED
200 old MIXED
```

This gives both clean-label classes enough support for separate incompleteness estimates.

Do not reduce the scientific audit to a 50–100 state subset.

A tiny smoke subset may be used only for implementation validation.

---

## 7. UID diversity constraint

Prevent route-rich UIDs from dominating.

Default rule:

```text
maximum 4 deep-audit states per UID
```

unless this makes the frozen 1,200-state target impossible.

Prefer broad UID coverage and report:

```text
# unique UIDs
states/UID distribution
```

Target several hundred unique UIDs.

---

## 8. Stratification

Within each old-label class, stratify across:

```text
dataset:
    GQA
    ChartQA
    TextVQA

source regime:
    Historical
    Canonical

layer:
    Early  0-8
    Middle 9-18
    Late   19-27

route source:
    preservation
    single
    MCTS
```

where support exists.

Preserve minimum support for rare but important strata, especially:

```text
INTERVENE_REQUIRED
MCTS states
Canonical states
late-layer states
```

Do not force impossible cells.

Freeze the allocation before search.

---

## 9. Outcome-blind sampling rule

The 1,200 states may be selected using only:

```text
old KEEP / INTERVENE / MIXED label
dataset
source regime
layer
route source
UID/group identity
```

Do not use:

```text
Stage-2 probe score
z_R / z_W score
current action margin
newly discovered action success
external full-benchmark transition
```

for selection.

---

## 10. Exact state identity

Every audited row must correspond to one exact entering state:

```text
state_id
UID
current layer j
exact routed-prefix action sequence
stored hidden-state reference/hash
dataset
source
old A_obs(s)
old label
```

Do not merge states with different routed prefixes.

---

## 11. Quantity to estimate

For each state `s`, define the old observed action set:

```text
A_old(s)
```

The audit attempts to expand it:

```text
A_audit(s)
=
A_old(s)
∪
{newly discovered first actions with >=1 successful bounded suffix}
```

For each currently unobserved first action:

```text
a ∈ {FULL, READ_ONLY, WRITE_ONLY, IGNORE} \ A_old(s)
```

ask:

> If action `a` is forced now, does there exist a successful continuation under the frozen bounded suffix-search protocol?

---

## 12. Test the first action, search the suffix

Do not test only:

```text
forced action a
→ all remaining layers FULL
```

because that would reject actions requiring later coordination.

Instead:

```text
exact state s at layer j
↓
force candidate first action a
↓
actual resulting state at j+1
↓
bounded suffix search over later layers
↓
does any continuation reach final correct?
```

---

## 13. Do not re-search already observed successful actions

For:

```text
a ∈ A_old(s)
```

do not launch a new expensive search.

Replay one existing provenance-valid successful continuation as a parity check.

The expensive search focuses only on:

```text
previously unobserved first actions
```

---

## 14. Action-conditioned bounded search

For each previously unobserved first action `a`:

### Step 0 — exact forced branch

Replay the exact prefix to state `s`.

Force:

```text
action a at layer j
```

Use the actual resulting state for all later computation.

### Step 1 — direct FULL suffix

Evaluate:

```text
a at j
FULL at j+1 ... 27
```

If correct, mark `a` newly successful and stop this action branch.

### Step 2 — exhaustive one-additional-intervention suffix

If direct FULL suffix is wrong, test every later layer:

```text
k = j+1 ... 27
```

with:

```text
READ_ONLY at k
WRITE_ONLY at k
IGNORE at k
```

and FULL elsewhere after the forced first action.

If any route is correct, mark `a` newly successful and stop this branch.

### Step 3 — MCTS suffix

If still unresolved, run MCTS from the actual post-action state.

Use:

```text
max_iterations = 200
```

with the previously validated:

```text
tree policy
reward
executor
action ordering
seed contract
```

The current action `a` remains fixed.

MCTS controls only later layers.

If a correct suffix is found, mark `a` newly successful.

Otherwise record:

```text
unresolved_at_budget
```

not invalid.

---

## 15. Why keep the 200-iteration cap

Prior experiments showed only small marginal coverage gain from 200 to 300 MCTS iterations.

Use 200 to keep the audit comparable to the existing label-generation contract.

Log first-success iteration for saturation analysis.

---

## 16. Stop search after first success per action

The audit needs existence, not full route enumeration.

Once one replay-valid successful suffix is found:

```text
mark the first action successful
stop that action branch
```

Retain:

```text
first successful route
search stage that found it
first-success iteration
```

---

## 17. Exact replay validation

Every newly discovered successful action must pass exact replay:

```text
exact prefix
forced first action
stored successful suffix
final LMMS correctness
```

Only replay-valid discoveries enter `A_audit(s)`.

Quarantine:

```text
replay mismatch
runtime error
provenance mismatch
```

---

## 18. Primary label transitions

Because `A_audit` only expands successful sets, the important outcomes are clean-label invalidations to MIXED.

### Old KEEP_REQUIRED

Old:

```text
A_old = {FULL}
```

Audited:

```text
KEEP_STABLE:
    A_audit = {FULL}

KEEP_INVALIDATED_TO_MIXED:
    FULL plus >=1 newly successful non-FULL
```

### Old INTERVENE_REQUIRED

Old:

```text
FULL not in A_old
>=1 non-FULL in A_old
```

Audited:

```text
INTERVENE_STABLE:
    FULL still not discovered successful

INTERVENE_INVALIDATED_TO_MIXED:
    FULL now has a successful bounded continuation
```

This second case is crucial: immediate non-FULL was not actually required under the audited search budget.

### Old MIXED

Old MIXED remains MIXED, though its action set may expand.

---

## 19. Primary completeness metrics

Report:

```text
KEEP invalidation rate
=
# KEEP_INVALIDATED_TO_MIXED
/
# audited old KEEP

INTERVENE invalidation rate
=
# INTERVENE_INVALIDATED_TO_MIXED
/
# audited old INTERVENE
```

Also report:

```text
fraction of states with any newly discovered action
mean action-set cardinality before/after
distribution of newly discovered action types
```

Use UID-level bootstrap 95% CIs.

---

## 20. Action-specific discovery rates

Among previously unobserved actions, report new-success discovery rates for:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Question:

> Which action type was most often incorrectly treated as absent by the previous corpus?

---

## 21. Search-stage attribution

For each new successful action, record whether it was found by:

```text
direct FULL suffix
one-additional-intervention exhaustive search
MCTS@200
```

Report contribution by search stage.

This distinguishes simple missed alternatives from genuinely coordinated suffixes.

---

## 22. Budget-saturation diagnostic

For MCTS-discovered actions, report cumulative discovery:

```text
Found@50
Found@100
Found@150
Found@200
```

If many discoveries appear near 200, the audit remains search-limited.

Do not extend beyond 200 in this phase.

---

## 23. Breakdown analyses

Report KEEP and INTERVENE invalidation by:

```text
GQA / ChartQA / TextVQA
Historical / Canonical
Early / Middle / Late
preservation / single / MCTS route source
```

Do not infer causality from stratum differences.

---

## 24. Is MCTS supervision more incomplete?

The previous separability probe showed:

```text
single-route RW AUROC ≈ 0.664
MCTS-route RW AUROC  ≈ 0.572
```

Compare:

```text
action-set expansion / invalidation rate on single states
vs
MCTS states
```

If MCTS states are substantially more incomplete, poor separability may partly reflect weaker label identification.

---

## 25. Recompute audited labels

Construct:

```text
AUDITED_KEEP
AUDITED_INTERVENE
AUDITED_MIXED
```

from `A_audit`.

Rules:

```text
AUDITED_KEEP:
    A_audit = {FULL}

AUDITED_INTERVENE:
    FULL not in A_audit
    and >=1 non-FULL in A_audit

AUDITED_MIXED:
    FULL in A_audit
    and >=1 non-FULL in A_audit
```

These remain bounded-search labels.

---

## 26. Secondary representation re-evaluation

This is a decision bridge, not a new router experiment.

On the audited 1,200 states:

1. exclude `AUDITED_MIXED`;
2. use 5-fold UID/group-disjoint evaluation;
3. reuse the same frozen representation families:

```text
M0 margin
M1 four logits
R
W
RW
```

4. train only the same lightweight probes as before.

Compare against:

```text
RW old-label AUROC = 0.562
```

Primary question:

> Does better label completeness materially increase treatment-need separability?

---

## 27. Audited-probe metrics

Report:

```text
AUROC
AUPRC
Precision@5%
Precision@10%
Precision@20%
Recall@90% precision
Recall@95% precision
```

A modest-recall high-precision region is still useful.

---

## 28. Decision logic

### Case A — Large invalidation + probe improves

Interpretation:

> Previous treatment-selectivity analysis was substantially label-limited.

Next step:

```text
improve / expand route-label construction
before redesigning representation
```

### Case B — Large invalidation + probe remains weak

Interpretation:

> Label incompleteness is real but does not explain weak separability.

Next step should target representation/training-state diversity.

### Case C — Labels mostly stable + probe remains weak

Interpretation:

> Evidence for a Stage-2 representation limitation becomes much stronger.

Only then prioritize a richer treatment representation.

This still does not prove routing is impossible.

### Case D — Labels mostly stable + probe improves strongly

Interpretation:

> Sampling/stratum composition may have driven the earlier weak estimate.

Inspect group/stratum effects before redesigning the method.

### Case E — Search remains unsaturated at 200

Interpretation:

> The audit cannot support a strong completeness conclusion.

Do not treat stable labels as true negatives.

---

## 29. No external benchmark fitting

Do not use ChartQA/TextVQA/MMMU-Pro/POPE full-eval rescue/regression labels for:

```text
state sampling
label construction
probe fitting
threshold selection
```

Those remain external method-level evidence.

---

## 30. Implementation smoke

Before launching all 1,200 states, run a tiny smoke only to validate:

```text
exact state reconstruction
forced first-action semantics
direct suffix execution
single-later search
MCTS root after forced action
replay parity
artifact writing
```

The smoke result must not change the frozen 1,200-state manifest or scientific criteria.

---

## 31. Compute logging

Record:

```text
states audited
unobserved action branches searched
direct suffix evaluations
single-later evaluations
MCTS roots launched
MCTS iterations
new successful actions
GPU-hours
wall time
```

Also estimate:

```text
compute per state
compute per newly discovered action
```

---

## 32. Required outputs

Use:

```text
analysis/dense_failure_stage2/treatment_label_completeness/
```

Create:

```text
protocol.md

sampling/
    full_state_census.csv
    frozen_1200_state_manifest.jsonl
    sampling_strata.csv
    uid_coverage.csv

search/
    per_state_action_audit.jsonl
    direct_suffix_results.jsonl
    single_later_results.jsonl
    mcts_results.jsonl
    replay_validation.jsonl

labels/
    old_vs_audited_action_sets.jsonl
    audited_clean_labels.jsonl
    audited_mixed_states.jsonl
    label_transition_summary.csv
    action_expansion_summary.csv

metrics/
    overall_completeness.csv
    uid_bootstrap_ci.csv
    dataset_breakdown.csv
    source_breakdown.csv
    layer_breakdown.csv
    route_source_breakdown.csv
    action_discovery_rates.csv
    search_stage_attribution.csv
    budget_saturation.csv
    compute_summary.csv

probe_recheck/
    fold_manifest.jsonl
    probe_summary.csv
    high_precision_metrics.csv
    old_vs_audited_probe_comparison.csv

figures/
    keep_invalidation_rate.png
    intervene_invalidation_rate.png
    action_set_cardinality_before_after.png
    newly_discovered_actions.png
    discovery_by_search_stage.png
    mcts_budget_saturation.png
    completeness_by_route_source.png
    old_vs_audited_rw_probe.png

summaries/
    treatment_label_completeness_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 33. `treatment_label_completeness_summary.md` must answer

1. How many exact states and unique UIDs were audited?
2. How many previously unobserved action branches were searched?
3. What fraction of old KEEP labels became MIXED?
4. What fraction of old INTERVENE labels became MIXED because FULL was newly found?
5. Which action types were most often newly discovered?
6. Were discoveries direct, one-additional-intervention, or MCTS-dependent?
7. Did MCTS discovery saturate by 200?
8. Which datasets/sources/layers had the highest incompleteness?
9. Were MCTS-route labels more incomplete than single-route labels?
10. How much did action-set cardinality increase?
11. How many audited states remain clean KEEP / INTERVENE / MIXED?
12. Does RW probe performance improve on audited labels?
13. Is there now a useful high-precision INTERVENE region?
14. Does the evidence support a label-limited or representation-limited interpretation?
15. What does the bounded audit still not prove?

---

## 34. `next_stage2_recommendation.md`

Recommend exactly one next direction.

If label-limited:

```text
expand/refine treatment-label search
before changing Stage-2 representation
```

If representation-limited:

```text
one minimal richer treatment representation experiment
```

If search-limited:

```text
improve the search-identification protocol
rather than treating unresolved branches as negatives
```

For the chosen recommendation state:

```text
Why it matters
Which hypothesis it tests
What a positive result means
What a negative result means
What should not be overinterpreted
```

---

## 35. Stop rule

STOP after:

```text
frozen 1,200-state deep audit
action-conditioned bounded suffix search
exact replay validation
label-transition analysis
audited-label probe recheck
one next-stage recommendation
```

Do not:

```text
retrain Stage-2
change Stage-1
rerun external benchmark evaluation
change P90
train a deployment head
```

---

## 36. Core principle

The previous Stage-2 probe result was:

```text
RW AUROC ≈ 0.56
```

but a weak probe is only meaningful if the target is trustworthy.

Before concluding:

> "the Stage-2 representation lacks treatment information"

first ask:

> **Were KEEP_REQUIRED and INTERVENE_REQUIRED clean enough labels, or were many states labeled too strongly because alternative successful suffixes were never searched?**

This audit answers that question with a large, prospectively frozen sample rather than a handful of cases.
