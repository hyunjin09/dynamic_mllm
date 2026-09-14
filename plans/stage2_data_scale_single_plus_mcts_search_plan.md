# Stage-2 Data-Scale Search Plan
## Expand Once, Run Single Search First, Then MCTS on the Remaining Triggered-W Samples

## 1. Objective

Expand the current Stage-2 supervision with a new, pre-frozen training candidate pool and perform the expensive corrective search only once.

For every new candidate:

```text
Dense FULL evaluation
→ frozen Stage-1 trigger
→ if triggered Dense-W:
       exhaustive single-intervention search
       → if still unresolved:
              trigger-conditioned MCTS, max 200 iterations
→ if triggered Dense-C:
       save FULL suffix as preservation supervision
```

Keep the resulting supervision types strictly separate:

```text
SINGLE_FIXABLE
MCTS_ONLY_FIXABLE
UNRESOLVED
PRESERVATION_C
```

This phase is **data expansion + label generation only**.

Do **not** train a new Stage-2 router in this phase.

The later training comparison should be:

```text
Current V1-698 baseline
        ↓
Scaled-Single
        ↓
Scaled-Single + MCTS
```

so that the effect of more clean single-intervention supervision can be separated from the effect of richer multi-layer MCTS supervision.

---

## 2. Current Frozen Reference

Keep the completed Stage-2 V1 result frozen.

Current V1 training data:

```text
Triggered-C preservation bases: 39
SINGLE_FIXABLE W bases:         698
Successful single routes:       7,628
```

Current validation result:

```text
Dense accuracy:   50.000%
Routed accuracy:  50.625%
Δ accuracy:       +0.625 percentage points

W→C: 5
C→W: 0
C→C: 400
W→W: 395
```

Do not retrain or replace this baseline during the search phase.

---

## 3. Why Expand Data Now

The current V1 produced:

```text
nonzero free-rollout correction
zero observed C→W regression
positive net accuracy change
```

but behavior remained narrow:

```text
5 W→C over 800 validation samples
54/235 triggered samples used any non-FULL action
98.40% of post-trigger actions were FULL
```

This is enough to justify testing whether broader executor-consistent corrective supervision increases coverage.

Do not change the router architecture before answering that question.

---

## 4. New Candidate Pool Size

Freeze approximately:

```text
4,000 new raw training candidates
```

before running any label-dependent search.

The exact number may differ slightly if dataset availability or group-disjoint constraints require it, but the candidate list must be frozen before observing:

```text
Dense C/W
Stage-1 trigger
single fixability
MCTS fixability
```

Do not stop early because a desired number of positive labels has been reached.

---

## 5. Disjointness

The new pool must be disjoint from:

```text
all current train examples already used for Stage-1 / Stage-2
all validation examples
all test examples
all existing Stage-2 label-generation bases
```

Use the same group-disjoint identity logic already used by the project.

Freeze:

```text
new_candidate_manifest.jsonl
```

before dense evaluation.

---

## 6. Dataset Composition

Prefer the same general GQA / ChartQA / TextVQA mixture used by the current training pool unless source availability prevents it.

Do not use dataset-specific:

```text
Stage-1 thresholds
corrective search algorithms
MCTS budgets
reward rules
```

Record label yield by dataset.

---

## 7. Frozen Runtime Contract

Freeze:

```text
Qwen2.5-VL backbone
current executor/runtime
LMMS-Eval correctness implementation
Stage-1 Shared Random-4 predictor
Stage-1 threshold
Stage-1 trigger rule
four-action semantics
MCTS implementation
```

The goal is to generate labels compatible with the current deployment contract.

---

## 8. Step A — Fresh Dense FULL Evaluation

For every new candidate, run the current dense FULL model.

Save:

```text
uid
dataset
group_id
dense_prediction
dense_correct
dense_wrong
provenance/hash fields
```

Use the same LMMS-Eval correctness definitions as the existing corpus.

---

## 9. Step B — Frozen Stage-1 Triggering

Apply the frozen Stage-1 gate to every new candidate.

Save:

```text
triggered
first_trigger_layer
score_at_trigger
optional full 28-layer score trajectory
```

Partition into:

```text
Dense-C, no trigger
Dense-C, trigger
Dense-W, no trigger
Dense-W, trigger
```

---

## 10. New Triggered Dense-C

For every:

```text
Dense-C + Stage-1 trigger
```

save the known-safe post-trigger suffix:

```text
FULL, FULL, ..., FULL
```

as:

```text
route_source = preservation_full
```

Do not run corrective search or MCTS on these C samples.

---

## 11. New Triggered Dense-W — Single Search First

For every:

```text
Dense-W + Stage-1 trigger at l*
```

run exhaustive single-intervention search over:

```text
l* ... 27
```

For each layer `j` test:

```text
READ_ONLY at j, FULL elsewhere
WRITE_ONLY at j, FULL elsewhere
IGNORE at j, FULL elsewhere
```

The prefix:

```text
L0 ... L(l*-1)
```

must remain the frozen dense FULL prefix.

All successful single routes must be exact-replay validated.

---

## 12. SINGLE_FIXABLE Retention

If at least one single-intervention route is correct:

```text
fixability_type = SINGLE_FIXABLE
```

Retain all replay-valid successful single routes.

For each route save:

```text
uid
dataset
trigger_layer
intervention_layer
intervention_action
suffix action sequence
final prediction
final correctness
route_source = single
```

Do not run MCTS on SINGLE_FIXABLE samples in the main pipeline.

---

## 13. Single-Unresolved → MCTS

Only samples with no successful single route enter MCTS.

Use trigger-conditioned MCTS.

Root:

```text
actual state entering first_trigger_layer l*
```

Search horizon:

```text
l* ... 27
```

Actions:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Every next state must be the actual routed state produced by previous search actions.

---

## 14. MCTS Budget

Use:

```text
max_iterations = 200 per single-unresolved sample
```

Do not increase to 300 or beyond in this phase.

Use the same validated:

```text
tree policy
rollout behavior
reward
action ordering
seed contract
termination rules
```

as the previous corrective-label search unless a blocking bug is found.

---

## 15. MCTS Reward

Terminal reward:

```text
1 if final LMMS-Eval answer is correct
0 otherwise
```

Do not add learned value models or reward shaping.

---

## 16. MCTS Route Retention

If MCTS finds a correct route:

```text
fixability_type = MCTS_ONLY_FIXABLE
```

Retain up to:

```text
8 distinct replay-valid successful routes/sample
```

Save:

```text
uid
dataset
trigger_layer
suffix action trajectory
first_success_iteration
non_FULL_count
READ_ONLY count
WRITE_ONLY count
IGNORE count
final prediction
route_source = mcts
```

Exact replay validation is mandatory.

---

## 17. UNRESOLVED

If neither search succeeds:

```text
fixability_type = UNRESOLVED
```

Do not invent action labels.

`UNRESOLVED` means only that no route was found under the frozen bounded search.

---

## 18. Preserve Provenance

Distinguish:

```text
old Phase-56 labels
new scale-up labels
```

and independently:

```text
single
mcts
preservation_full
```

Every route/base should retain:

```text
source phase
candidate pool
search type
search budget
executor/model hash
Stage-1 trigger contract
```

---

## 19. Build Expanded Corpora

Construct:

### Expanded Corpus A — Preservation

```text
old triggered-C preservation
+
new triggered-C preservation
```

### Expanded Corpus B — Single

```text
old 698 SINGLE_FIXABLE W
+
new SINGLE_FIXABLE W
```

### Expanded Corpus C — MCTS

```text
old 209 MCTS_ONLY_FIXABLE W
+
new MCTS_ONLY_FIXABLE W
```

Also retain the expanded unresolved manifest.

---

## 20. Do Not Train During This Phase

Even though both single and MCTS labels are generated now, keep future training experiments separate.

### Future Experiment A — Scaled-Single

Train with:

```text
Expanded A + Expanded B
```

Exclude MCTS labels.

Question:

> Does broader clean single-intervention supervision increase free-rollout correction coverage?

### Future Experiment B — Scaled-Single + MCTS

Train the same router with:

```text
Expanded A + Expanded B + Expanded C
```

Question:

> Does multi-layer trajectory supervision add performance beyond the larger single-intervention corpus?

---

## 21. Keep Stage-2 Architecture Frozen Later

For the later comparisons, initially keep the current V1 architecture unchanged:

```text
READ:
    current text/query attends to current visual tokens

WRITE:
    shared learned WRITE query attends to current visual tokens

shared 4-action head

no explicit layer embedding
no Stage-1 representation
```

Do not change architecture at the same time as data scale.

---

## 22. Keep Stage-1 Frozen Later

Do not retune:

```text
Stage-1 threshold
Stage-1 predictor
trigger persistence
trigger timing
```

during the data-scale comparison.

---

## 23. Expanded-Corpus Audit

Report:

```text
new raw candidates
new Dense-C
new Dense-W
new triggered-C
new triggered-W
new SINGLE_FIXABLE
new MCTS_ONLY_FIXABLE
new UNRESOLVED
```

Also report final combined sizes:

```text
Expanded Corpus A
Expanded Corpus B
Expanded Corpus C
```

---

## 24. Label-Yield Metrics

Compute for new, old, and combined pools:

```text
P(trigger | W)
P(trigger | C)
P(single-fixable | triggered W)
P(MCTS-only-fixable | triggered W)
P(total bounded fixable | triggered W)
```

---

## 25. Dataset Yield Audit

For:

```text
GQA
ChartQA
TextVQA
```

report:

```text
raw new candidates
Dense-W count
triggered-W count
single-fixable count/rate
MCTS-only count/rate
total bounded correctability
triggered-C preservation count
```

Explicitly check whether GQA corrective supervision increases materially.

Do not change search rules by dataset.

---

## 26. Trigger-Depth Audit

For:

```text
L0
L1-L8
L9-L18
L19-L27
```

report:

```text
new triggered-W
single-fixable rate
MCTS-only rate
total bounded correctability
search cost
```

---

## 27. Action / Route Audit

For new single labels report:

```text
READ_ONLY count
WRITE_ONLY count
IGNORE count
intervention-layer distribution
trigger-to-intervention delay
routes/sample
```

For new MCTS labels report:

```text
non-FULL count distribution
action distribution
first-success iteration
routes/sample
```

Compare old vs new.

---

## 28. Compute Logging

Record:

```text
dense-eval GPU time
Stage-1 scoring time
single-search terminal routes
MCTS iterations
MCTS terminal routes
wall-clock time
GPU-hours
replay-validation cost
```

Report single search and MCTS cost separately.

---

## 29. Integrity Checks

Require:

```text
fresh dense reproduction
correct frozen Stage-1 trigger
FULL prefix unchanged before trigger
single-route exact replay
MCTS-route exact replay
no duplicate UID processing
no validation/test contamination
artifact/hash validation
```

Quarantine failed rows.

---

## 30. Required Outputs

Use:

```text
analysis/dense_failure_stage2/data_scale_search/
```

Create:

```text
protocol.md

manifests/
    new_candidate_manifest.jsonl
    new_dense_results.jsonl
    new_trigger_map.jsonl
    new_triggered_correct.jsonl
    new_triggered_wrong.jsonl
    new_single_fixable.jsonl
    new_mcts_only_fixable.jsonl
    new_unresolved.jsonl

routes/
    new_successful_single_routes.jsonl
    new_successful_mcts_routes.jsonl

combined_corpora/
    expanded_corpus_A_preservation.jsonl
    expanded_corpus_B_single.jsonl
    expanded_corpus_C_mcts.jsonl
    expanded_unresolved.jsonl
    corpus_manifest.json

metrics/
    scaleup_summary.csv
    old_vs_new_yield.csv
    dataset_breakdown.csv
    trigger_depth_breakdown.csv
    single_action_distribution.csv
    mcts_route_complexity.csv
    compute_summary.csv

figures/
    label_yield_old_vs_new.png
    dataset_label_yield.png
    fixability_composition.png
    trigger_depth_yield.png
    single_action_distribution.png
    mcts_route_complexity.png

summaries/
    data_scale_search_summary.md
    next_training_recommendation.md

artifact_manifest.json
```

---

## 31. `data_scale_search_summary.md` Must Answer

1. How many new candidates were processed?
2. How many new triggered Dense-W samples were obtained?
3. How many are SINGLE_FIXABLE?
4. How many additional samples are MCTS_ONLY_FIXABLE?
5. How many remain UNRESOLVED?
6. How many new triggered Dense-C preservation samples were obtained?
7. What are the final sizes of Expanded A, B, and C?
8. Did the new pool reproduce the old label-yield pattern?
9. Did GQA corrective supervision materially increase?
10. How different are labels by trigger depth?
11. How much compute did single search consume?
12. How much additional compute did MCTS consume?
13. Are all retained routes replay-valid and provenance clean?

---

## 32. `next_training_recommendation.md`

If the expanded corpus is healthy, recommend:

```text
Experiment A:
Scaled-Single
= Expanded A + Expanded B

Experiment B:
Scaled-Single + MCTS
= Expanded A + Expanded B + Expanded C
```

Do not recommend changing Stage-1, router architecture, layer embeddings, Stage-1 latent features, or loss unless a blocking incompatibility is discovered.

---

## 33. Future Training Fairness Contract

For later comparisons keep:

```text
same Stage-2 architecture
same optimizer
same loss
same Stage-1
same validation set
same free-rollout evaluator
```

Prefer a matched optimizer-update budget to the frozen V1-698 run, or explicitly report any difference.

---

## 34. Future Decision Logic

If Scaled-Single improves W→C while C→W stays low:

```text
clean corrective data coverage was a real bottleneck
```

Then test `+MCTS`.

If Scaled-Single does not improve despite much more single data:

```text
data quantity alone is unlikely to be the main bottleneck
```

Investigate representation/policy/exposure issues.

If `+MCTS` improves beyond Scaled-Single:

```text
multi-layer trajectory supervision adds useful information
```

If `+MCTS` does not help:

```text
keep the simpler single-supervision method
```

---

## 35. Stop Rule

STOP after:

```text
new candidate dense evaluation
frozen Stage-1 trigger mapping
single search
MCTS@200 on single-unresolved samples
route replay validation
expanded corpus construction
label-yield audit
```

Do not:

```text
train Scaled-Single
train Scaled-Single + MCTS
retune Stage-1
change Stage-2 architecture
search validation/test for labels
run test evaluation
```

Training is the next separately authorized phase.

---

## 36. Core Principle

Search once, preserve both supervision types, and separate their effects later:

```text
New training data
    ↓
Frozen Stage-1
    ↓
Triggered-W
    ↓
Single exhaustive search
   / \
  /   \
Single  unresolved
labels       ↓
         MCTS@200
             ↓
        MCTS labels

        ↓

Store both separately

        ↓ later

Scaled-Single
vs
Scaled-Single + MCTS
```

This avoids repeating expensive search while preserving a clean comparison between:

> **more clean corrective data**

and

> **richer multi-layer corrective supervision**.
