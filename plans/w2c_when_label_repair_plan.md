# W2C Route-Cache Repair and WHEN-Label Rebuild Plan

## 1. Goal

The next step is to repair the **W→C route cache** before training any new `CONTINUE / DEVIATE` gate.

The current problem is that the previous "mandatory deviation boundary" was defined from the discovered correcting-route cache:

```text
longest known all-FULL prefix
→ next layer has no known FULL continuation
→ label as DEVIATE
```

However, the expanded audit showed that many such boundaries were not truly mandatory under execution: forcing `FULL` at the supposed boundary and attaching a compatible known suffix still produced a correct answer.

Therefore the current `DEVIATE` labels are not sufficiently trustworthy.

The purpose of this phase is to:

1. repair W→C correcting-route coverage;
2. iteratively push the candidate FULL boundary as far as execution allows;
3. rebuild cleaner `CONTINUE / DEVIATE` supervision;
4. re-audit label completeness before any Stage-1 gate training.

Do **not** train a new router in this phase.

---

## 2. Scope

Repair **W→C samples only**.

Reason:

```text
W→C:
all-FULL = wrong
some non-FULL route = correct
```

These are the samples for which a `DEVIATE` state may exist.

Do not repair C→C in this phase.

C→C will later be used as:

```text
CONTINUE / preserve-FULL supervision
```

and for preservation evaluation.

---

## 3. Terminology

Avoid treating search-derived boundaries as globally proven necessities.

### 3.1 FULL_RESCUABLE

At candidate layer `l`, after forcing `FULL`, at least one tested continuation yields the correct final answer.

```text
prefix
→ FULL at l
→ tested continuation
→ correct
```

Interpretation:

> `FULL` is still recoverable at this layer under the tested continuation set.

This layer must **not** be labeled `DEVIATE`.

### 3.2 FULL_UNRESCUED_UNDER_BUDGET

At candidate layer `l`, no tested continuation yields the correct final answer under the frozen bounded search budget.

Interpretation:

> No correct FULL continuation was found under the tested budget.

This is a candidate `DEVIATE` label.

Do not call it globally "FULL impossible."

### 3.3 UNRESOLVED

Execution could not determine the state because of missing/incompatible suffixes, runtime failure, or incomplete search bookkeeping.

Do not use unresolved states as clean supervision.

---

## 4. Authoritative Inputs

Use the current authoritative four-action W→C labels and exact unified executor.

Before repair, audit and freeze:

```text
source label root
train/validation manifests
sample IDs
source route IDs
four-action executor commit
model revision
evaluator contract
answer-normalization contract
random seeds
search budget
```

Write:

```text
analysis/w2c_when_repair/repair_protocol.md
```

before executing the repair.

The repair must be reproducible.

---

## 5. Core Repair Principle

For each W→C sample:

1. start from the current discovered correct four-action routes;
2. determine the longest all-`FULL` prefix supported by at least one known correct route;
3. inspect the next candidate boundary;
4. force `FULL` at that boundary;
5. test all compatible known correct suffixes;
6. if any FULL continuation is correct:
   - add that complete route to the valid route cache;
   - recompute the longest all-FULL prefix;
   - move the candidate boundary later;
   - repeat;
7. if no known suffix rescues:
   - optionally run a bounded continuation search;
8. stop only when no FULL continuation is found under the frozen repair budget.

The repair is **iterative**.

Do not simply flip one old `DEVIATE` label to `CONTINUE`.

---

## 6. Example

Suppose the current known route is:

```text
L0  FULL
L1  FULL
L2  FULL
L3  READ_ONLY
L4  FULL
L5  WRITE_ONLY
...
L27 FULL
→ correct
```

The old label says:

```text
candidate boundary = L3
```

Test:

```text
L0  FULL
L1  FULL
L2  FULL
L3  FULL
L4  FULL
L5  WRITE_ONLY
...
L27 FULL
```

If this route is correct:

```text
L3 = FULL_RESCUABLE
```

Add the new route to the cache.

Now recompute the boundary.

The next candidate might be:

```text
L5
```

Then test FULL at L5 with compatible continuations.

Continue until no tested FULL continuation succeeds.

---

## 7. Phase 0 — Repair Audit

Before launching repair, report:

```text
number of W→C train samples
number of W→C validation samples
mean/median valid routes per sample
old mandatory-boundary depth distribution
number of compatible known suffixes per boundary
number of samples with only one known suffix
number of samples with multiple suffixes
```

Output:

```text
analysis/w2c_when_repair/pre_repair_audit.md
```

---

## 8. Phase 1 — Known-Suffix Repair

This is the first and cheapest repair pass.

### 8.1 For each W→C sample

At the current candidate boundary:

```text
prefix = all FULL up to l-1
candidate action at l = FULL
```

Construct all compatible suffixes from discovered correct routes.

A compatible suffix must:

- belong to the same sample;
- begin strictly after the candidate layer;
- satisfy the unified four-action route format;
- be executable under the current prefix;
- use the frozen evaluator contract.

Do not mix suffixes across samples.

### 8.2 Execute all unique candidate routes

For each:

```text
all-FULL prefix
+
FULL at candidate layer
+
compatible known suffix
```

run the exact executor.

Deduplicate identical complete routes before execution.

Cache by:

```text
sample ID
complete 28-action route
model revision
executor revision
```

### 8.3 If any continuation is correct

Mark:

```text
FULL_RESCUABLE
```

Then:

1. add every newly verified correct route to the valid-route cache;
2. recompute the sample's longest supported all-FULL prefix;
3. derive the new candidate boundary;
4. repeat known-suffix repair at the new boundary.

The process may advance multiple layers for the same sample.

### 8.4 If none are correct

Mark the state as:

```text
known_suffix_exhausted
```

Do not yet call it clean `DEVIATE`.

Pass it to Phase 2.

---

## 9. Phase 2 — Bounded Continuation Search

Run only for samples that remain unresolved after Phase 1.

The goal is to test whether the absence of a correct known suffix is merely a cache-coverage artifact.

Do not rerun full original MCTS blindly.

Use a bounded, explicitly frozen continuation search.

### 9.1 Search state

Start from:

```text
exact all-FULL prefix through candidate layer
```

The search only chooses actions for:

```text
L_{l+1} ... L_27
```

### 9.2 Search action space

Use:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

for remaining layers.

If existing repair infrastructure supports pruning or beam search, freeze it prospectively.

Do not change the search budget after seeing per-sample results.

### 9.3 Suggested bounded search strategy

Prefer reusing the validated search implementation already used for route discovery.

If a lighter repair search is needed, use:

```text
known suffixes first
→ local suffix variants
→ bounded beam / MCTS continuation search
```

Keep the budget small enough to make full W→C repair feasible but large enough to improve route-cache coverage.

Record exact per-state budgets.

### 9.4 Search outputs

If a correct continuation is found:

```text
FULL_RESCUABLE
```

Add the new route to the cache and restart iterative boundary advancement for that sample.

If no correct continuation is found:

```text
FULL_UNRESCUED_UNDER_BUDGET
```

This becomes the repaired candidate `DEVIATE` state.

---

## 10. Phase 3 — Iterate Until Convergence

For each W→C sample, continue:

```text
candidate boundary
→ FULL insertion
→ known suffix test
→ bounded search if needed
→ cache update
→ recompute boundary
```

until one of the following occurs.

### Stop condition A

```text
FULL_UNRESCUED_UNDER_BUDGET
```

at the current candidate boundary.

### Stop condition B

The all-FULL prefix reaches layer 27.

Because these are W→C samples, this requires explicit investigation.

### Stop condition C

Runtime/evaluator inconsistency.

Quarantine the sample.

Do not silently assign a label.

---

## 11. Repaired WHEN Labels

After repair, build a new W→C WHEN-label dataset.

### CONTINUE states

States where execution verifies that `FULL` can still be followed by at least one correct continuation:

```text
CONTINUE
```

This includes newly discovered `FULL_RESCUABLE` states.

### DEVIATE-candidate states

The first state after the maximal repaired all-FULL prefix where:

```text
FULL_UNRESCUED_UNDER_BUDGET
```

Label:

```text
DEVIATE_CANDIDATE
```

Do not use the stronger phrase `mandatory deviation` in the raw label schema.

### UNRESOLVED

Exclude from clean gate supervision.

---

## 12. Preserve All Newly Verified Correct Routes

The repair should not output only one new boundary label.

Every newly verified correct route is valuable.

Store:

```text
sample ID
full 28-action route
source of discovery
  - original cache
  - known-suffix repair
  - bounded continuation repair
candidate boundary that generated it
execution result
answer
correctness
model/executor revision
```

Deduplicate exact routes.

Do not overwrite the original route cache.

Create a versioned repaired cache.

---

## 13. Train and Validation Repair

Apply the **same repair algorithm and search budget** to both train and validation W→C samples.

Do not repair only validation.

Otherwise training and validation labels would follow different contracts.

The repair may be executed separately by split, but:

```text
algorithm
budget
executor
evaluator
```

must be identical.

---

## 14. Sample Counts

Repair all available W→C samples in the authoritative train/validation sets if computationally feasible.

At minimum:

```text
all W→C validation samples
```

must be repaired.

For later Stage-1 training, target at least:

```text
>= 512 repaired train DEVIATE candidates
>= 128 repaired validation DEVIATE candidates
```

If fewer remain after repair, stop and report that the clean-positive pool is insufficient.

Do not relax the threshold after observing the counts.

---

## 15. Post-Repair Completeness Re-Audit

After rebuilding the cache and WHEN labels, repeat the FULL-insertion audit prospectively.

Use at least:

```text
128 repaired validation DEVIATE candidates
```

if available.

If fewer than 128 exist, use all and stop Stage-1 training under the current contract.

### 15.1 Audit procedure

For each repaired candidate:

1. force `FULL`;
2. test all newly compatible known suffixes from the repaired cache;
3. apply the same bounded continuation search budget used in repair;
4. record whether any correct FULL continuation is found.

Report:

```text
FULL rescue rate
95% UID-bootstrap CI
per dataset
per depth bin
per mechanism family
```

Use:

```text
10,000 UID-group bootstrap draws
fixed seed
```

---

## 16. Depth Analysis

Report before vs after repair:

| Depth group | Old FULL rescue rate | Repaired FULL rescue rate |
|---|---:|---:|
| Early | | |
| Middle | | |
| Late | | |

Also report boundary shift:

```text
new_boundary - old_boundary
```

with:

```text
mean
median
P25
P75
P95
distribution by dataset
```

This tests whether early interventions were often replaceable by later ones.

---

## 17. Route Substitutability Analysis

For each repaired sample, record:

```text
old boundary
new boundary
number of layers boundary moved
old action family
new action family
number of newly discovered correct routes
```

Report:

```text
fraction boundary unchanged
fraction shifted by 1-2 layers
fraction shifted by 3-5 layers
fraction shifted by >5 layers
```

This is diagnostic only.

Do not make a causal claim from boundary movement alone.

---

## 18. Mechanism Statistics After Repair

At the repaired DEVIATE candidate, report the known valid non-FULL action set:

```text
READ_ONLY
WRITE_ONLY
IGNORE
MULTI
```

Compare old vs repaired distributions.

Repair may change not only WHEN but also WHAT.

Do not train Stage 2 yet.

---

## 19. C2C Handling

Do not alter C→C labels/routes during this repair.

C2C remains:

```text
FULL-correct preservation population
```

Later Stage-1 gate training should use C2C states as conservative `CONTINUE` examples.

For now:

```text
no C2C route repair
no C2C re-search
no C2C Stage-2 decomposition
```

---

## 20. GPU / Execution Strategy

This workload is parallel across samples.

Use a deterministic shard/work-queue setup.

Suggested topology:

```text
8 GPUs
1 process / GPU initially
```

Benchmark before increasing replicas per GPU.

Primary throughput metric:

```text
completed route executions / second
```

or:

```text
completed repaired samples / hour
```

Cache all route evaluations so repeated suffixes or rediscovered routes are not re-executed.

---

## 21. Smoke Test

Before full repair, use:

```text
8-16 W→C samples
```

covering:

```text
all three datasets
early/middle/late old boundaries
single-suffix and multi-suffix cases
previously FULL-cache-incomplete examples
previously FULL-confirmed-invalid examples
```

Verify:

1. exact old route replay;
2. FULL insertion at candidate boundary;
3. compatible suffix enumeration;
4. deduplication;
5. correct cache update;
6. candidate boundary moves after rescue;
7. iterative re-evaluation;
8. bounded continuation search only after known suffix exhaustion;
9. resume/restart consistency;
10. output determinism.

---

## 22. Required Artifacts

Suggested root:

```text
analysis/w2c_when_repair/
```

Create:

```text
repair_protocol.md
pre_repair_audit.md

smoke/
  smoke_manifest.json
  smoke_executions.jsonl
  smoke_report.md

repair/
  repaired_routes.jsonl
  route_execution_cache.jsonl
  repaired_when_labels.jsonl
  quarantined_samples.jsonl
  repair_history.jsonl

post_repair/
  post_repair_audit_manifest.json
  post_repair_full_insertion_results.jsonl
  post_repair_completeness_report.md
  boundary_shift.csv
  mechanism_shift.csv

decision_summary.md
```

---

## 23. Required Summary Tables

### Table 1 — Repair yield

| Split | W2C samples | FULL-rescuable discoveries | Final DEVIATE candidates | Unresolved/quarantined |
|---|---:|---:|---:|---:|
| Train | | | | |
| Validation | | | | |

### Table 2 — Boundary shifts

| Metric | Train | Validation |
|---|---:|---:|
| Mean shift | | |
| Median shift | | |
| P95 shift | | |
| Unchanged | | |
| Shifted >=1 | | |
| Shifted >5 | | |

### Table 3 — Post-repair completeness

| Group | States | FULL bounded rescue | Rescue rate | 95% CI |
|---|---:|---:|---:|---:|
| Overall | | | | |
| GQA | | | | |
| TextVQA | | | | |
| ChartQA | | | | |
| Early | | | | |
| Middle | | | | |
| Late | | | | |

### Table 4 — Repaired mechanism distribution

| Mechanism | Old count | New count |
|---|---:|---:|
| READ_ONLY | | |
| WRITE_ONLY | | |
| IGNORE | | |
| MULTI | | |

---

## 24. Decision Rule

### Case A — Post-repair FULL rescue remains high

Interpretation:

> The bounded search/cache is still too incomplete to support a clean binary `CONTINUE / DEVIATE` target.

Do not train Stage 1.

Next step should be broader route-search repair or a different supervision formulation.

### Case B — Post-repair FULL rescue is low and enough clean positives remain

Requirements:

```text
>= 128 validation DEVIATE candidates
>= 512 train DEVIATE candidates
```

and no major executor inconsistency.

Then proceed to:

```text
train a simple CONTINUE / DEVIATE gate
```

using:

```text
W→C repaired DEVIATE candidates
W→C verified CONTINUE states
C→C CONTINUE states
```

Do not train READ_OFF / WRITE_OFF / BOTH_OFF yet.

### Case C — Too few DEVIATE candidates remain

Interpretation:

> Corrective interventions are highly substitutable or the current search contract does not identify a sufficiently large clean WHEN population.

Stop before gate training.

Reconsider the Stage-1 target definition.

---

## 25. Questions This Phase Must Answer

### Q1

> How incomplete was the old W→C route cache with respect to continuing `FULL`?

### Q2

> How far do candidate deviation boundaries move after repair?

### Q3

> Is early-layer incompleteness reduced after iterative repair?

### Q4

> How many clean DEVIATE candidates remain after repair?

### Q5

> Are there enough repaired train/validation examples to support the planned selective gate?

### Q6

> Does repair materially change the READ_ONLY / WRITE_ONLY / IGNORE mechanism distribution?

---

## 26. Stop Rule

Do not start:

```text
CONTINUE / DEVIATE gate training
Stage-2 mechanism training
external evaluation
full router retraining
```

until:

1. W→C repair is complete;
2. repaired WHEN labels are rebuilt;
3. post-repair completeness audit is complete;
4. the clean train/validation DEVIATE pools satisfy the frozen minimum counts.

The immediate research task is:

```text
repair W→C supervision first,
then decide whether Stage 1 is a valid learning problem.
```
