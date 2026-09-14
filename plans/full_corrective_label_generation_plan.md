# Full Corrective Label Generation Plan

## 1. Objective

Scale corrective-label generation to the full frozen Stage-1 triggered Dense-W training population while preserving two supervision types separately:

1. **Single-intervention corrective routes**
   - exactly one non-FULL action after trigger
   - simpler and potentially easier Stage-2 supervision

2. **MCTS-discovered corrective trajectories**
   - searched only when exhaustive single-intervention search fails
   - captures trajectory-dependent multi-layer corrections

This phase builds the reusable Stage-2 label corpus. Do **not** train Stage-2 yet.

---

## 2. Motivation

The pilot found:

```text
Single-fixable:        35/120 = 29.17%
Additional MCTS-only:  22/120 = 18.33%
Total bounded fixable: 57/120 = 47.50%
```

Therefore single-intervention supervision is substantial and must not be discarded. MCTS also adds meaningful additional coverage.

The corpus must support later staged experiments:

```text
Stage-2 V1:
Triggered-C FULL supervision
+ single-intervention W supervision

Stage-2 V2:
V1
+ MCTS trajectory supervision
```

---

## 3. Frozen Population

Use the frozen `Shared Random-4` Stage-1 trigger map.

Training populations:

```text
Triggered Dense-W: 1,881
Triggered Dense-C: 39
```

Corrective search in this phase is only for the 1,881 triggered Dense-W samples.

The 39 triggered Dense-C samples are retained separately for future C→C preservation supervision using the known-safe FULL suffix.

Do not change Stage-1, thresholds, trigger layers, dataset calibration, or split membership.

---

## 4. Reuse the Completed Pilot

Reuse the existing 120-sample pilot if provenance matches:

```text
same trigger map
same executor
same model/runtime semantics
same LMMS-Eval correctness definition
same replay contract
```

Expected new workload:

```text
1,881 total
- 120 pilot
= 1,761 remaining
```

Quarantine and regenerate only rows with provenance mismatches.

---

## 5. Prefix and Search Contract

For each triggered Dense-W sample:

```text
first_trigger_layer = l*
```

Keep:

```text
L0 ... L(l*-1) = FULL
```

exactly as in the frozen dense trajectory.

Search starts from the actual state entering `L(l*)`.

Available actions at each suffix layer:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

All later states must be trajectory-conditioned on earlier chosen actions.

---

## 6. Phase A — Exhaustive Single-Intervention Search

For every triggered-W sample, test every one-intervention suffix route.

For every `j = l* ... 27`:

```text
READ_ONLY at j, FULL elsewhere
WRITE_ONLY at j, FULL elsewhere
IGNORE at j, FULL elsewhere
```

For L0 trigger, maximum non-control routes:

```text
3 × 28 = 84
```

If at least one route is correct:

```text
fixability_type = SINGLE_FIXABLE
```

Save **all successful single routes**, not only one.

For each successful single route save:

```text
uid
dataset
trigger_layer
intervention_layer
intervention_action
full suffix action sequence
final prediction
LMMS correctness
route_source = single
non_FULL_count = 1
```

Replay every retained successful route exactly before accepting it.

---

## 7. Stop Single-Fixable Samples Before MCTS

For compute efficiency:

```text
SINGLE_FIXABLE
→ do not run main-pipeline MCTS
```

This preserves a natural curriculum:

```text
simple correction when simple correction exists
complex search only when needed
```

Do not generate unnecessary complex routes for samples already solved by one intervention.

---

## 8. Phase B — MCTS for Single-Unresolved Samples

Only samples with no successful single-intervention route enter MCTS.

Use trigger-conditioned MCTS over:

```text
l* ... 27
```

with:

```text
max_iterations = 200
```

This cap is frozen from the pilot:

```text
Fixable@200 = 46.67%
Fixable@300 = 47.50%
200→300 gain = 0.83 percentage points
```

Do not increase above 200 in this phase.

---

## 9. MCTS Semantics

Root:

```text
actual routed state entering layer l*
```

At each layer branch over:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

After action execution, use the resulting routed state for the next layer.

Terminal reward:

```text
1 if final LMMS-Eval answer is correct
0 otherwise
```

Do not add a learned reward model.

---

## 10. MCTS Route Retention

If MCTS finds a correct trajectory:

```text
fixability_type = MCTS_ONLY_FIXABLE
```

Retain multiple distinct successful routes, but cap storage:

```text
up to 8 successful MCTS routes/sample
```

Deduplicate by exact suffix action trajectory.

For each route save:

```text
uid
dataset
trigger_layer
full suffix action sequence
first_success_iteration
final prediction
LMMS correctness
route_source = mcts
non_FULL_count
READ_ONLY count
WRITE_ONLY count
IGNORE count
first non-FULL layer
last non-FULL layer
```

Replay every retained successful MCTS route exactly before accepting it.

---

## 11. Preferred Route Metadata

Define one preferred route per fixable sample for diagnostics only:

```text
1. correct
2. minimum non-FULL count
3. earliest discovered if tied
```

Do not overwrite the richer successful-route set with the preferred route.

---

## 12. Final Fixability Classes

Every triggered Dense-W sample must end in exactly one class:

```text
SINGLE_FIXABLE
MCTS_ONLY_FIXABLE
UNRESOLVED
```

`UNRESOLVED` means only that no correct route was found under the frozen bounded search. It does not mean provably unfixable.

Do not invent corrective labels for unresolved samples.

---

## 13. Preserve Route Source Explicitly

Every label must retain:

```text
route_source ∈ {single, mcts, preservation_full}
```

This enables later controlled experiments:

```text
single only
mcts only
single + mcts
```

Do not merge the two W supervision types without provenance.

---

## 14. Triggered Dense-C Preservation Labels

Do not run corrective search on the 39 triggered Dense-C training samples.

For each one save the known safe suffix:

```text
FULL, FULL, ..., FULL
```

from trigger layer onward.

Mark:

```text
route_source = preservation_full
dense_correct = true
final_correct = true
```

These are future C→C preservation examples.

---

## 15. Trajectory-Conditioned State Capture

For every retained successful W route, replay the exact route and save the actual state used before each action.

For each `j = trigger_layer ... 27`:

```text
uid
dataset
trigger_layer
current_layer
route_id
route_source
routed Stage-2 feature/state at layer j
chosen action at layer j
final_correct = true
```

If already available and inexpensive, also preserve:

```text
Stage-1 trigger score
Stage-1 failure representation z_fail at trigger
layer index/embedding
```

The critical requirement is that each saved state corresponds to the actual prior actions in that route.

---

## 16. Multiple Successful Actions

If multiple retained routes pass through the exact same routed state and use different actions that both lead to correctness, preserve that information.

Use wording:

```text
observed successful actions
```

not:

```text
all valid actions
```

because bounded search may miss alternatives.

---

## 17. Build Three Explicit Corpora

### Corpus A — Preservation

```text
Triggered Dense-C
→ FULL suffix
```

Purpose: C→C preservation.

### Corpus B — Simple Corrective

```text
SINGLE_FIXABLE Dense-W
→ all successful single-intervention routes
```

Purpose: sparse/easier W→C supervision.

### Corpus C — Trajectory Corrective

```text
MCTS_ONLY_FIXABLE Dense-W
→ retained successful MCTS trajectories
```

Purpose: multi-layer trajectory-dependent W→C supervision.

Do not collapse these into one unlabeled corpus.

---

## 18. Full-Cohort Label Audit

After processing all 1,881 triggered-W samples, report:

```text
# SINGLE_FIXABLE
# MCTS_ONLY_FIXABLE
# UNRESOLVED
```

and corresponding rates.

Also report:

```text
single-only contribution
MCTS-only additional contribution
total bounded correctability
```

Compare full-cohort results with the pilot.

---

## 19. Route Complexity Audit

For preferred successful routes report:

```text
1 intervention
2 interventions
3 interventions
4+ interventions
```

Also report:

```text
median
IQR
95th percentile
```

Repeat separately for MCTS-only samples.

---

## 20. Layer and Action Audit

Report:

```text
intervention frequency by layer
READ_ONLY frequency by layer
WRITE_ONLY frequency by layer
IGNORE frequency by layer
FULL frequency along successful trajectories
```

For single-fixable samples specifically report:

```text
which layer contains the single corrective intervention?
which action is used?
```

---

## 21. Dataset Audit

Report separately for:

```text
GQA
ChartQA
TextVQA
```

Metrics:

```text
SINGLE_FIXABLE rate
MCTS_ONLY_FIXABLE rate
total bounded correctability
UNRESOLVED rate
route complexity
action distribution
```

Do not change search logic by dataset.

---

## 22. Trigger-Depth Audit

Report separately for:

```text
L0
L1-L8
L9-L18
L19-L27
```

Metrics:

```text
single rescue rate
MCTS-only additional rescue
total bounded correctability
median route complexity
action distribution
```

This is descriptive only.

---

## 23. Future Stage-2 Experiments Enabled

Do not train yet, but preserve the corpus so the next phase can test:

### Stage-2 V1

```text
Corpus A
+
Corpus B
```

Question:

> Can a simple layer-wise router learn mostly-FULL trajectories with sparse corrective interventions while preserving C→C?

### Stage-2 V2

```text
Corpus A
+
Corpus B
+
Corpus C
```

Question:

> Does MCTS trajectory supervision add learned W→C beyond the simpler single-intervention training?

This V1→V2 comparison should be kept as a core ablation.

---

## 24. Sampling Caveat for Future Training

Do not let route multiplicity automatically determine training weight.

A sample with 8 successful routes should not automatically count 8× more than a sample with 1 route.

Later training should explicitly choose sample-balanced or route-balanced sampling.

Do not decide that policy in this phase.

---

## 25. Compute and Integrity Logging

Record:

```text
single routes evaluated
MCTS iterations
terminal evaluations
successful routes retained
wall-clock time
GPU time
replay pass/fail
```

Quarantine:

```text
failed replay
runtime mismatch
dense-prefix mismatch
provenance mismatch
```

No questionable route should enter the label corpus.

---

## 26. Required Outputs

Use:

```text
analysis/dense_failure_stage2/full_corrective_labels/
```

Create:

```text
protocol.md

manifests/
    full_triggered_wrong_manifest.jsonl
    single_fixable.jsonl
    mcts_only_fixable.jsonl
    unresolved.jsonl
    triggered_correct_preservation.jsonl

routes/
    successful_single_routes.jsonl
    successful_mcts_routes.jsonl
    preferred_routes.jsonl

states/
    single_route_states/
    mcts_route_states/
    preservation_full_states/

metrics/
    overall_fixability.csv
    dataset_breakdown.csv
    trigger_depth_breakdown.csv
    route_complexity.csv
    action_by_layer.csv
    action_distribution.csv
    compute_summary.csv

stage2_corpora/
    corpus_A_preservation.jsonl
    corpus_B_single_corrective.jsonl
    corpus_C_mcts_corrective.jsonl
    corpus_manifest.json

figures/
    fixability_composition.png
    route_complexity_distribution.png
    single_intervention_layer_distribution.png
    action_by_layer.png
    dataset_fixability.png
    trigger_depth_fixability.png

decision_summary.md
artifact_manifest.json
```

Reuse the 120 pilot records through provenance-safe import where possible.

---

## 27. `decision_summary.md` Must Answer

1. Across all 1,881 triggered-W samples, what fraction is SINGLE_FIXABLE?
2. What additional fraction is MCTS_ONLY_FIXABLE?
3. What is the final bounded correctability rate?
4. How close is the full-cohort result to the pilot projection?
5. How many successful single-intervention routes were found?
6. How many successful MCTS routes were retained?
7. How many non-FULL actions do successful routes usually require?
8. Where do single interventions occur by layer and action type?
9. How strongly do labels differ by dataset?
10. How strongly do labels differ by trigger depth?
11. How many training states belong to Corpus A, B, and C?
12. Does the corpus support a simple `preservation + single` Stage-2 V1?
13. Is there enough MCTS-only coverage to justify V2?
14. Are all retained routes exact-replay valid and provenance clean?

---

## 28. Stop Rule

STOP after:

```text
full triggered-W corrective search
route replay
corpus construction
label audit
```

Do not:

```text
train Stage-2
retune Stage-1
change threshold
search validation/test for training supervision
run downstream final routing evaluation
run external benchmarks
```

The next decision should be:

```text
Stage-2 V1:
preservation FULL + single-intervention corrective supervision
```

followed, only if justified, by:

```text
Stage-2 V2:
add MCTS trajectory supervision
```

Core principle:

> Preserve simple single-intervention labels and richer MCTS trajectory labels separately so Stage-2 complexity increases only when the data and learned-policy results justify it.
