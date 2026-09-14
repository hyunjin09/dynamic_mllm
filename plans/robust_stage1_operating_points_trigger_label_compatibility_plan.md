# Robust Stage-1 Operating-Point + Trigger/Label Compatibility Plan

## 1. Objective

The ALL-source Stage-1 head is now the robust Stage-1 candidate.

The next step is **not** to retrain Stage-1 again and **not** to immediately train Stage-2.

Instead:

1. characterize the robust Stage-1 preservation/recall frontier;
2. keep several conservative operating points rather than prematurely fixing one final threshold;
3. regenerate the Stage-1 trigger map under the robust head;
4. compare the new trigger population against the historical Stage-2 label population;
5. determine which existing single/MCTS corrective labels remain directly reusable and which samples would require new search.

The main question is:

> **After replacing the source-sensitive old Stage-1 gate with the robust ALL-source gate, what population now reaches Stage-2, and how much of the existing corrective supervision remains compatible?**

Stop before new corrective search or Stage-2 retraining.

---

## 2. Frozen Stage-1 Model

Freeze the Phase-63 robust ALL-source Stage-1 head.

Do not change:

```text
Qwen2.5-VL backbone
Stage-1 feature definition
Shared Random-4 architecture
old normalization artifact
ALL-source trained head weights
score definition
strict trigger comparison: score > tau
```

The head itself is now frozen.

Only the **operating threshold** is analyzed in this phase.

---

## 3. Current Conservative Reference

Preserve the Phase-63 threshold as a named reference point:

```text
tau_98_reference = 0.9711347410314399
```

Observed behavior:

```text
Historical C preservation: 98.75%
Canonical C preservation:  98.08%

Historical W recall: 6.50%
Canonical W recall:  12.40%

Pooled W recall:      10.54%
Pooled precision:     67.34%
Median trigger layer: 23
```

This is a **conservative reference**, not yet the final method threshold.

The final deployment threshold should eventually depend on actual:

```text
W→C rescue
C→W regression
final accuracy
```

after Stage-2 is reconnected.

---

## 4. Why Threshold Analysis Is Still Needed

The robust head fixed the catastrophic source-shift problem, but the conservative 98% operating point triggers only about 10.5% of W samples.

Therefore the current remaining trade-off is:

```text
higher threshold:
    C→C protection ↑
    W admission ↓
    later triggers

lower threshold:
    W admission ↑
    intervention horizon ↑ potentially
    false C admission ↑
```

Do not assume the 98% point is optimal for the final pipeline.

---

## 5. Phase A — Build a Robust Operating-Point Frontier

Use only the already frozen held-out / cross-fit Stage-1 score predictions used for Phase-63 calibration.

Do not use Stage-2 outcomes.

For a dense grid of threshold values, compute separately:

```text
Historical C preservation
Canonical C preservation

Historical W recall
Canonical W recall

Historical trigger precision
Canonical trigger precision

pooled W recall
pooled trigger precision

median first-trigger layer
trigger-layer distribution
```

Also compute:

```text
worst-source C preservation
= min(Historical C preservation, Canonical C preservation)

worst-source W recall
= min(Historical W recall, Canonical W recall)
```

The threshold frontier must be source-aware.

---

## 6. Reference Preservation Operating Points

Derive a small set of named operating points from the robust held-out calibration scores.

Suggested targets:

```text
P99
P98
P97
P95
P90
```

For target preservation `p`, define the threshold as:

> the least conservative threshold that still satisfies the target C-preservation constraint on **both** Historical and Canonical calibration populations.

That is:

```text
Historical C preservation >= p
AND
Canonical C preservation >= p
```

If exact target satisfaction is impossible because of discrete scores, choose the closest more-conservative threshold.

Do not choose thresholds separately by source.

Do not choose thresholds separately by dataset.

---

## 7. Threshold Stability

For each named operating point, report threshold stability across the existing cross-fit/fold structure.

Report:

```text
median tau
min tau
max tau
IQR
```

A useful operating point should not require wildly different thresholds across folds.

Preserve the Phase-63 cross-fit stability analysis as the reference for P98.

---

## 8. Do Not Pick a Final Threshold Yet

At the end of Phase A, keep a **small candidate set**, not one final winner.

Recommended candidate set after frontier inspection:

```text
one very conservative point
one medium-conservative point
one more permissive point
```

For example:

```text
P98
P95
P90
```

if their score distributions and fold stability are reasonable.

The exact set should be chosen from the observed frontier, not fixed blindly.

These are future Stage-2 operating points, not independent models.

---

## 9. Phase B — Generate Robust Trigger Maps

Using the **single frozen full ALL-source Stage-1 head**, generate trigger maps for training populations under each retained operating point.

At minimum include:

```text
Historical training population
Canonical training population
```

For every sample and operating point save:

```text
uid
dataset
source_regime
dense_correct
dense_wrong

threshold_name
threshold_value

triggered
first_trigger_layer
score_at_trigger
max_score
```

If the 28-layer score trajectory already exists, reuse it rather than rerunning Qwen.

---

## 10. Trigger Map Summary

For each operating point, report:

```text
C/no-trigger
C/trigger
W/no-trigger
W/trigger
```

separately for:

```text
Historical
Canonical
GQA
ChartQA
TextVQA
```

Also report:

```text
median trigger layer
L0 trigger fraction
Early: 0-8
Middle: 9-18
Late: 19-27
```

This establishes the actual Stage-2 admission population under the repaired Stage-1 head.

---

## 11. Phase C — Compare New Trigger Map with Old Trigger Map

For every training UID present in the prior Stage-2 search corpus, compare:

```text
old first_trigger_layer
new first_trigger_layer
```

under each candidate threshold.

Classify each sample as:

```text
SAME_TRIGGER
NEW_EARLIER
NEW_LATER
OLD_TRIGGER_ONLY
NEW_TRIGGER_ONLY
NEITHER_TRIGGER
```

For NEW_EARLIER / NEW_LATER also record:

```text
delta_layer = new_trigger - old_trigger
```

Report transition matrices separately for Dense-C and Dense-W.

---

## 12. Why Trigger Compatibility Matters

Existing Stage-2 corrective labels were generated under the old Stage-1 trigger contract.

A successful route is only useful under the new Stage-1 handoff if the required corrective intervention(s) still occur **at or after the new trigger**.

Therefore do not blindly merge the old Stage-2 corpus with the robust Stage-1 gate.

---

## 13. Compatibility Rule — Single-Intervention Labels

For an old successful single route, define:

```text
old trigger layer = l_old
corrective intervention layer = c
new trigger layer = l_new
```

The route is a **candidate reusable route** if:

```text
l_new <= c
```

Reason:

- if `l_new < l_old`, prepend FULL from `l_new` to `l_old-1`;
- if `l_old < l_new <= c`, drop old pre-handoff FULL supervision before `l_new`;
- the one non-FULL intervention at `c` still occurs after handoff.

If:

```text
l_new > c
```

the old corrective intervention would occur before Stage-2 is activated, so the route is incompatible with the new handoff.

This rule is only a structural pre-filter.

Exact replay validation is still required.

---

## 14. Compatibility Rule — MCTS Routes

For each successful MCTS route identify:

```text
first_non_FULL_layer = f
```

A route is a candidate reusable route if:

```text
l_new <= f
```

If the new trigger occurs after the route's first required intervention:

```text
l_new > f
```

the route is incompatible.

Again, this is a structural pre-filter only.

---

## 15. Exact Replay Compatibility Audit

For all candidate reusable routes, replay the route from the new Stage-1 trigger contract.

Procedure:

```text
1. FULL through L0 ... L(new_trigger-1)
2. FULL from new_trigger until the first stored non-FULL action if needed
3. execute the stored corrective route
4. evaluate final answer under current LMMS contract
```

Save:

```text
REPLAY_COMPATIBLE_CORRECT
REPLAY_INCOMPATIBLE_WRONG
REPLAY_ERROR
```

Do not assume structural compatibility guarantees correctness.

No route enters the future repaired Stage-2 corpus without exact replay confirmation.

---

## 16. Compatibility Categories per W Sample

For each newly triggered Dense-W training sample classify:

```text
A. EXISTING_SINGLE_REUSABLE
B. EXISTING_MCTS_REUSABLE
C. EXISTING_LABEL_BUT_INCOMPATIBLE
D. NEW_TRIGGER_NO_EXISTING_LABEL
E. EXISTING_UNRESOLVED
```

A sample can have both reusable single and MCTS routes; preserve both labels.

The main output is the number of newly admitted W samples already covered by verified old supervision.

---

## 17. Compatibility Categories per C Sample

For newly triggered Dense-C:

```text
FULL suffix
```

is always the natural preservation trajectory because dense FULL is correct.

Therefore every newly triggered C can receive:

```text
PRESERVATION_FULL
```

without corrective search, subject to standard dense replay/provenance checks.

Count how much preservation supervision changes under each threshold.

---

## 18. Search Workload Estimate

For each candidate threshold, estimate how many newly triggered Dense-W samples require new search:

```text
new triggered W
-
W with replay-compatible existing corrective routes
```

Break this into:

```text
no prior search
prior unresolved
prior route incompatible with new trigger
```

Also report by:

```text
dataset
source regime
trigger depth
```

This lets us understand the cost of reconnecting Stage-2 at different thresholds.

Do not run the new search yet.

---

## 19. Existing-Label Lower-Bound Treatment Opportunity

Using **only exact replay-compatible existing labels**, compute a descriptive lower bound:

```text
known_repairable_recall
=
# all Dense-W that are:
    newly triggered
    AND have >=1 replay-compatible correct route
/
# all Dense-W
```

Also compute conditional coverage:

```text
P(known repairable | newly triggered W)
```

Do this for each candidate threshold.

Important:

> These are lower bounds determined by existing search coverage.

Do not use lack of an existing compatible label as evidence that a sample is unfixable.

---

## 20. Preservation / Treatment Opportunity Table

Produce:

| Operating point | Hist C preserve | Canon C preserve | Pooled W recall | Median trigger | Known repairable W recall | New W search required |
|---|---:|---:|---:|---:|---:|---:|
| P99 | | | | | | |
| P98 | | | | | | |
| P97 | | | | | | |
| P95 | | | | | | |
| P90 | | | | | | |

This table is not for final threshold selection.

It is for choosing which 2-3 operating points deserve actual Stage-2 treatment evaluation.

---

## 21. Important Interpretation

Do not conclude:

```text
P98 is best because preservation is highest
```

or:

```text
P90 is best because W recall is highest
```

The final choice depends on:

```text
W→C after learned Stage-2
C→W after learned Stage-2
```

Stage-1-only metrics cannot determine the final optimum.

---

## 22. Recommended Decision at the End

Select a **small operating-point set for downstream Stage-2 repair**, ideally:

```text
conservative
middle
permissive
```

A reasonable default may be:

```text
P98 / P95 / P90
```

but only if their actual frontier and cross-fit stability justify them.

Do not select more than 3 primary downstream thresholds unless there is a clear reason.

---

## 23. What Happens After This Phase

Only after the trigger/compatibility audit should a new search phase be authorized.

Future sequence:

```text
Robust Stage-1 head
    ↓
candidate thresholds
    ↓
new trigger maps
    ↓
reuse compatible labels
    ↓
search only missing newly-triggered W samples
    ↓
rebuild Stage-2 corpus
    ↓
Stage-2 retraining / free rollout
    ↓
choose final threshold using W→C - C→W
```

This minimizes expensive repeated search.

---

## 24. Required Outputs

Use:

```text
analysis/dense_failure_stage1/robust_operating_points_and_compatibility/
```

Create:

```text
protocol.md

thresholds/
    frontier.csv
    named_operating_points.csv
    fold_stability.csv

trigger_maps/
    historical_train.jsonl
    canonical_train.jsonl
    trigger_summary.csv
    dataset_source_breakdown.csv
    trigger_depth_breakdown.csv

compatibility/
    old_vs_new_trigger_map.jsonl
    single_structural_compatibility.jsonl
    mcts_structural_compatibility.jsonl
    replay_results.jsonl
    per_sample_compatibility.jsonl

metrics/
    trigger_transition_matrix.csv
    route_reuse_summary.csv
    preservation_population.csv
    known_repairable_lower_bound.csv
    new_search_workload.csv
    operating_point_summary.csv

figures/
    preservation_vs_wrong_recall.png
    preservation_vs_trigger_depth.png
    threshold_frontier_by_source.png
    old_vs_new_trigger_layer.png
    route_reuse_by_threshold.png
    search_workload_by_threshold.png

summaries/
    robust_operating_point_summary.md
    stage2_label_compatibility_summary.md
    next_search_recommendation.md

artifact_manifest.json
```

---

## 25. `robust_operating_point_summary.md` Must Answer

1. What does the Stage-1 preservation/recall frontier look like?
2. What are P99/P98/P97/P95/P90 thresholds?
3. How stable are those thresholds across folds?
4. How do Historical and Canonical C preservation differ at each point?
5. How do Historical and Canonical W recall differ?
6. How does median trigger depth move as the threshold is relaxed?
7. Which 2-3 operating points should be carried into actual Stage-2 evaluation?
8. Why is none of them yet the final deployment threshold?

---

## 26. `stage2_label_compatibility_summary.md` Must Answer

For each retained operating point:

1. How many W samples newly trigger?
2. How many retain the same trigger as the old gate?
3. How many trigger earlier?
4. How many trigger later?
5. How many old single routes are structurally compatible?
6. How many old MCTS routes are structurally compatible?
7. How many replay-compatible correct routes remain?
8. How many newly triggered W samples have at least one reusable corrective route?
9. How many newly triggered W samples require new corrective search?
10. How many newly triggered C samples add preservation supervision?

---

## 27. `next_search_recommendation.md`

Recommend the smallest new corrective-search workload required to rebuild Stage-2 supervision under the robust Stage-1 gate.

Prefer:

```text
reuse exact-replay compatible routes
+
search only missing triggered-W samples
```

Do not recommend rerunning all historical searches from scratch unless replay compatibility is unexpectedly poor.

If multiple thresholds are retained, identify whether search can be shared.

Do not execute search in this phase.

---

## 28. Stop Rule

STOP after:

```text
robust threshold frontier
named operating points
new robust trigger maps
old/new trigger comparison
existing-label structural compatibility
exact replay compatibility
new-search workload estimate
```

Do not:

```text
rerun single search
rerun MCTS
train Stage-2
retune the robust Stage-1 head
change normalization
use Stage-2 outcomes to choose a final threshold
run test deployment
```

---

## 29. Core Principle

The robust Stage-1 **head** can now be frozen.

The Stage-1 **threshold** should remain a treatment-dependent operating point.

The next task is therefore:

```text
freeze robust head
        ↓
map preservation/recall frontier
        ↓
generate robust trigger populations
        ↓
reuse what old Stage-2 supervision still supports
        ↓
identify only the missing search workload
```

Only after Stage-2 is reconnected should the project choose the final threshold based on:

```text
final accuracy
=
dense accuracy
+ W→C
- C→W
```
