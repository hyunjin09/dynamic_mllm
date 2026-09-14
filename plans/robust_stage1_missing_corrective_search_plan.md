# Robust Stage-1 Missing Corrective Search Plan
## P98 / P95 / P90 — Shared Search, Threshold-Specific Replay Validation

## 1. Objective

Rebuild the Stage-2 corrective supervision under the newly frozen robust ALL-source Stage-1 head.

The robust Stage-1 operating-point audit retained three downstream thresholds:

```text
P98
P95
P90
```

These thresholds trade C preservation against W admission differently.

The next task is:

> **For every newly triggered Dense-W sample that does not already have a replay-compatible corrective route, find new corrective supervision using the same validated search pipeline, while sharing search work across P98/P95/P90 whenever possible.**

This phase performs **corrective search and corpus reconstruction only**.

Do not train Stage-2 yet.

Do not choose the final deployment threshold yet.

---

# 2. Frozen Stage-1 Inputs

Freeze the robust ALL-source Stage-1 head and all three retained thresholds.

Use:

```text
P98
P95
P90
```

with their exact frozen threshold values from the previous operating-point audit.

Do not:

```text
retrain Stage-1
change normalization
change threshold values
change strict comparison semantics
change trigger-map logic
```

The trigger layer for each sample/operating point is already frozen.

---

# 3. Frozen Search Population

The compatibility audit identified the missing search workload.

Per operating point:

```text
P98: 277 W samples require new search
P95: 581 W samples require new search
P90: 1,048 W samples require new search
```

Across all three operating points:

```text
1,104 unique Dense-W samples
```

require at least one new search result.

These populations are not fully nested.

Freeze the exact union manifest before launching search.

Create:

```text
missing_search_union.jsonl
```

with, for every UID:

```text
uid
dataset
source_regime
dense_correct = false

trigger_P98
trigger_P95
trigger_P90

needs_search_P98
needs_search_P95
needs_search_P90

existing_reusable_single
existing_reusable_mcts
existing_replay_compatible_routes
```

---

# 4. Core Efficiency Principle

Do **not** run three independent full searches per sample.

Search discovery should be shared across operating points whenever exact executor semantics allow.

However:

> **A discovered route is only valid for an operating point if it is replay-correct under that operating point's trigger contract.**

Therefore separate:

```text
route discovery
```

from:

```text
threshold-specific route validation
```

---

# 5. Trigger Ordering

For a given sample, lower thresholds generally trigger earlier:

```text
l_P90 <= l_P95 <= l_P98
```

but do not assume this without checking the frozen trigger map.

For each UID, verify and save:

```text
trigger ordering
```

If an unexpected non-monotonic ordering is found, do not force a correction.

Use the actual frozen trigger layers.

---

# 6. Search Root Strategy

For each UID, define:

```text
l_min = earliest trigger among the operating points that require new search
```

Example:

```text
P90 trigger = L8
P95 trigger = L14
P98 trigger = L19
```

then:

```text
l_min = L8
```

Use the dense FULL prefix through:

```text
L0 ... L(l_min-1)
```

as the earliest permissible dynamic handoff state.

All route discovery starts from `l_min`.

---

# 7. Why Search from the Earliest Needed Trigger

Any route discovered from the earliest trigger may potentially be reusable for later thresholds if its first required intervention occurs after those later thresholds.

Example:

```text
P90 trigger = L8
P95 trigger = L14
P98 trigger = L19

discovered intervention:
L23 WRITE_ONLY
```

This route is potentially valid for all three operating points.

But:

```text
L12 READ_ONLY
```

can only serve operating points whose trigger is at or before L12.

Therefore one earliest-root search can discover supervision for several thresholds.

---

# 8. Phase A — Exhaustive Single-Intervention Search

For every UID in the 1,104-sample union, perform exhaustive single-intervention search from:

```text
l_min ... L27
```

At each layer `j`, test:

```text
READ_ONLY at j, FULL elsewhere
WRITE_ONLY at j, FULL elsewhere
IGNORE at j, FULL elsewhere
```

Pre-`l_min` layers remain FULL.

This is the same validated single-search semantics used previously.

---

# 9. Retain All Successful Single Routes

For every correct single route, save:

```text
uid
dataset
source_regime
search_root = l_min
intervention_layer
intervention_action
suffix trajectory
final prediction
LMMS correctness
route_source = new_single
```

Replay every retained successful route exactly.

Do not keep only one preferred single route.

Multiple successful single routes remain useful for later Stage-2 supervision.

---

# 10. Threshold-Specific Single Compatibility

For every successful single route with intervention layer `c`, validate it separately for every operating point `P` that requires search.

Structural condition:

```text
trigger_P <= c
```

If false:

```text
route incompatible with P
```

If true:

1. run FULL through `trigger_P - 1`;
2. continue FULL until intervention layer `c`;
3. execute the discovered non-FULL action at `c`;
4. continue stored route/full suffix;
5. evaluate final answer.

Save:

```text
P98_compatible
P95_compatible
P90_compatible
```

and exact replay result.

Do not infer later-threshold compatibility from the earliest-root execution alone.

---

# 11. Single-Fixable Resolution per Operating Point

For each sample/operating-point pair:

```text
SINGLE_RESOLVED_P
```

if at least one newly discovered or previously reusable single route is replay-correct under that operating point.

Once a pair is single-resolved:

```text
no MCTS is needed for that pair
```

However, MCTS may still be needed for the same UID under a later operating point if all successful single interventions occur before that later trigger.

---

# 12. Phase B — Identify Remaining Pair-Level Unresolved Cases

After single search and threshold-specific replay, create:

```text
remaining_unresolved_pairs.jsonl
```

Each row is:

```text
(uid, operating_point)
```

where:

```text
newly triggered Dense-W
AND
no replay-compatible existing route
AND
no replay-compatible new single route
```

MCTS should operate only on these remaining pair-level cases.

---

# 13. MCTS Sharing Strategy

Avoid blindly running independent MCTS for every unresolved pair.

For each UID:

1. group unresolved operating points by trigger layer;
2. begin with the **earliest unresolved trigger**;
3. run trigger-conditioned MCTS from that layer;
4. validate every discovered successful route against all later unresolved thresholds for the same UID.

If a route resolves later thresholds, mark those pairs resolved without separate MCTS.

Only launch another MCTS search from a later trigger if:

```text
the earlier-trigger MCTS did not discover any route usable after that later trigger
```

and the pair remains unresolved.

This gives maximum search reuse while preserving exact trigger semantics.

---

# 14. MCTS Budget

Use the previously validated cap:

```text
max_iterations = 200
```

per actually launched MCTS search root.

Do not increase to 300.

Do not change:

```text
tree policy
reward
action ordering
executor semantics
seed contract
termination logic
```

unless a blocking implementation defect is discovered.

---

# 15. MCTS Reward

Terminal reward remains:

```text
1 if final LMMS-Eval answer is correct
0 otherwise
```

No reward shaping.

No learned value model.

No Stage-2 policy assistance.

This phase discovers supervision; it does not learn the router.

---

# 16. MCTS Successful Route Retention

For each launched search, retain up to:

```text
8 distinct replay-valid successful trajectories
```

per UID/search root.

Save:

```text
uid
search_root
first_success_iteration
full suffix action trajectory
non_FULL_count
READ_ONLY count
WRITE_ONLY count
IGNORE count
first_non_FULL_layer
last_non_FULL_layer
final prediction
route_source = new_mcts
```

Deduplicate exact trajectories.

---

# 17. Threshold-Specific MCTS Compatibility

For each successful MCTS route, let:

```text
f = first_non_FULL_layer
```

For operating point `P`, structural compatibility requires:

```text
trigger_P <= f
```

If structurally compatible, perform exact replay from `trigger_P`.

Only replay-correct routes may enter that operating point's Stage-2 corpus.

---

# 18. Final Pair-Level Outcome Classes

Every newly triggered Dense-W `(uid, operating_point)` pair must end in exactly one of:

```text
EXISTING_SINGLE_REUSED
EXISTING_MCTS_REUSED

NEW_SINGLE_FIXABLE
NEW_MCTS_FIXABLE

UNRESOLVED_AT_BUDGET
```

Do not assign fake labels to unresolved pairs.

Important:

```text
UNRESOLVED_AT_BUDGET != provably unfixable
```

---

# 19. Preserve Existing Reusable Supervision

Do not discard the old replay-compatible labels from the compatibility audit.

For each operating point, future W corrective corpus should include:

```text
existing replay-compatible single routes
existing replay-compatible MCTS routes
new single routes
new MCTS routes
```

Keep provenance fields explicit.

---

# 20. Dense-C Preservation Corpus

For each operating point, every newly triggered Dense-C sample receives:

```text
FULL suffix from trigger_P ... L27
```

as preservation supervision.

Current triggered-C counts from the audit are:

```text
P98: 7
P95: 30
P90: 106
```

Rebuild these preservation corpora under the robust trigger map.

No corrective search is needed for Dense-C.

---

# 21. Build Threshold-Specific Stage-2 Corpora

Construct independent corpora for:

```text
P98
P95
P90
```

Each should contain:

## Corpus A_P — Preservation

```text
triggered Dense-C
→ FULL suffix
```

## Corpus B_P — Single Corrective

```text
existing reusable single
+
new single-fixable
```

## Corpus C_P — MCTS Corrective

```text
existing reusable MCTS
+
new MCTS-fixable
```

## Unresolved_P

```text
triggered Dense-W with no successful route under current bounded search
```

Do not merge thresholds into a single ambiguous corpus.

---

# 22. Shared Route Store, Threshold-Specific Views

To avoid data duplication, store unique routes once:

```text
global_route_store.jsonl
```

and create threshold-specific manifests that reference route IDs.

For each route save:

```text
valid_for_P98
valid_for_P95
valid_for_P90
```

This allows later training datasets to share identical trajectories while maintaining correct threshold provenance.

---

# 23. State Capture for New Routes

For every retained new successful route, replay it and save the actual routed Stage-2 input states.

For each layer from the relevant trigger onward:

```text
uid
route_id
current_layer
current visual-token states
current text/control-token states
action
```

Use the same state schema expected by the existing Stage-2 READ/WRITE router.

Do not save only pooled summaries if the router requires token-level visual/text states.

If token states are expensive, preserve the exact feature references/shards required by the existing training loader.

---

# 24. Search Coverage Metrics

For each operating point report:

```text
Triggered W
Existing W covered
New single-fixable W
New MCTS-fixable W
Total W with known corrective supervision
Unresolved W
```

Also compute:

```text
P(corrective supervision | triggered W)
```

and population lower-bound:

```text
P(triggered AND known-correctable | all W)
```

These remain bounded-search lower estimates.

---

# 25. Dataset / Source Breakdown

For each operating point report by:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

Metrics:

```text
triggered W
single-fixable
MCTS-fixable
unresolved
known corrective coverage
```

This matters because the robust Stage-1 recall remains dataset/source dependent.

---

# 26. Trigger-Depth Breakdown

For:

```text
L0-L8
L9-L18
L19-L27
```

report:

```text
triggered W
new search count
single-fixable rate
MCTS-only rate
total bounded correctability
```

Also preserve exact-layer summaries where support is adequate.

Question:

> Does the later trigger distribution of the robust Stage-1 reduce corrective capacity?

Do not infer causality from depth alone.

---

# 27. Search-Reuse Efficiency Metrics

Because this phase explicitly shares search across thresholds, report:

```text
naive pair-level search count
actual single-search UID count
actual MCTS search-root count
MCTS searches avoided by cross-threshold reuse
route replays performed
```

Estimate compute saved relative to three independent searches.

---

# 28. No Stage-2 Training Yet

Even after all threshold-specific corpora are reconstructed, stop.

Do not train:

```text
P98 Stage-2
P95 Stage-2
P90 Stage-2
```

yet.

First inspect corpus sizes and corrective coverage.

The following training phase should use the **same Stage-2 architecture** across thresholds.

---

# 29. Future Training Strategy

The next phase after this search should compare:

```text
P98 Stage-2
P95 Stage-2
P90 Stage-2
```

using the same:

```text
READ/WRITE router architecture
optimizer
sampling logic
loss
validation evaluator
```

The final threshold should then be selected by end-to-end validation:

```text
W→C
C→W
C→C
final accuracy
```

not by Stage-1 metrics alone.

---

# 30. Required Outputs

Use:

```text
analysis/dense_failure_stage2/robust_gate_corrective_search/
```

Create:

```text
protocol.md

manifests/
    missing_search_union.jsonl
    remaining_unresolved_pairs.jsonl

single_search/
    per_uid_results.jsonl
    successful_routes.jsonl
    threshold_compatibility.jsonl

mcts/
    launched_searches.jsonl
    successful_routes.jsonl
    threshold_compatibility.jsonl
    per_root_summary.csv

routes/
    global_route_store.jsonl
    replay_results.jsonl

states/
    routed_state_manifest.jsonl
    shards/

threshold_corpora/
    P98/
        preservation.jsonl
        single.jsonl
        mcts.jsonl
        unresolved.jsonl
    P95/
        preservation.jsonl
        single.jsonl
        mcts.jsonl
        unresolved.jsonl
    P90/
        preservation.jsonl
        single.jsonl
        mcts.jsonl
        unresolved.jsonl

metrics/
    overall_search_summary.csv
    threshold_coverage.csv
    dataset_source_breakdown.csv
    trigger_depth_breakdown.csv
    search_reuse_efficiency.csv
    compute_summary.csv

figures/
    corrective_coverage_by_threshold.png
    single_vs_mcts_by_threshold.png
    dataset_source_corrective_coverage.png
    trigger_depth_correctability.png
    search_reuse_savings.png

summaries/
    robust_gate_corrective_search_summary.md
    threshold_corpus_summary.md
    next_stage2_training_recommendation.md

artifact_manifest.json
```

---

# 31. `robust_gate_corrective_search_summary.md` Must Answer

1. How many unique W UIDs were searched?
2. How many single searches were launched?
3. How many MCTS roots were actually launched?
4. How much cross-threshold search reuse was achieved?
5. For P98, how many triggered W now have known corrective supervision?
6. Same for P95?
7. Same for P90?
8. How many remain unresolved at each point?
9. How much did single search contribute?
10. How much did MCTS add?
11. How strongly does correctability vary by dataset/source?
12. How strongly does it vary by trigger depth?
13. Are all retained routes exact-replay valid?

---

# 32. `threshold_corpus_summary.md` Must Answer

For each P98/P95/P90:

```text
# preservation C bases
# single-fixable W bases
# MCTS-fixable W bases
# unresolved W bases
# retained routes
# routed state rows
```

Also report overlap across thresholds.

---

# 33. `next_stage2_training_recommendation.md`

Recommend the cleanest next training comparison.

Default if all corpora are healthy:

```text
Train the same Stage-2 V1 architecture separately for:
    P98
    P95
    P90
```

Start with:

```text
preservation + single
```

and only then add MCTS supervision if needed.

Do not change the router architecture between thresholds.

The downstream threshold winner should be selected by:

```text
validation final accuracy
W→C
C→W
C→C
```

not by label-corpus size alone.

---

# 34. Stop Rule

STOP after:

```text
1,104-UID union search
single exhaustive search
shared MCTS@200 where needed
threshold-specific replay compatibility
new route/state capture
P98/P95/P90 corpus reconstruction
coverage/compute audit
```

Do not:

```text
train Stage-2
change Stage-1
change thresholds
run test
run external benchmarks
```

---

# 35. Core Principle

The expensive part should be shared:

```text
one UID
    ↓
earliest needed trigger
    ↓
discover route once
    ↓
replay against P90 / P95 / P98
    ↓
reuse where exact semantics permit
```

The scientific comparison must remain threshold-specific:

```text
P98:
high preservation, low admission

P95:
middle trade-off

P90:
higher admission, more preservation risk
```

Only after all three have adequate corrective supervision should the project ask:

> **Which operating point produces the best end-to-end rescue-minus-regression trade-off under the same learned Stage-2 router?**
