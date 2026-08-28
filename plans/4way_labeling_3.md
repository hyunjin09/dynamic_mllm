# Four-Action Label Conversion Plan

## 1. Goal

Convert the existing binary MCTS positive route labels into four-action visual-computation labels without rerunning MCTS from scratch.

The existing binary search already identified successful visual execution trajectories. The conversion should reuse those trajectories and refine only the layers that were `OFF`.

The four actions are:

| Action | READ | WRITE | Meaning |
|---|---:|---:|---|
| `FULL` | 1 | 1 | Native visual participation |
| `READ_ONLY` | 1 | 0 | Text/control can read visual K/V, but visual rows are not updated |
| `WRITE_ONLY` | 0 | 1 | Visual rows are updated, but text/control cannot directly read visual K/V |
| `IGNORE` | 0 | 0 | Neither READ nor WRITE is enabled |

Binary-to-four-action mapping:

- binary `ON` -> `FULL`
- binary `OFF` -> `IGNORE`

For W->C routes, the goal is to determine, within an already successful correcting trajectory, whether each binary `OFF` position really needs READ suppression, WRITE suppression, both, or no suppression at all.

This is route-conditioned sequential refinement, not a new MCTS search.

## 2. Source Label Sets

### 2.1 GQA / TextVQA / ChartQA

Use only:

`datasets/mcts_labels/gqa_textvqa_chartqa_v1/`

Do not use or merge `datasets/mcts_v2/`.

### 2.2 WeMath2.0 Standard / Pro

Inspect:

`datasets/math_labels`

Identify the current authoritative MCTS label artifacts for:
- WeMath2.0 Standard
- WeMath2.0 Pro

Do not guess filenames. Record the selected source artifacts and counts before conversion.

## 3. Scope

Process all positive binary labels in the selected authoritative supervision view.

Keep route types separated:

### W->C

`FULL = wrong`, binary route = correct.

Interpretation: corrective / answer-alignment supervision. This is the main target for four-action refinement.

### C->C

`FULL = correct`, binary route = correct.

Interpretation: correctness-preserving / redundancy / efficiency supervision. Do not interpret its suppression as answer-unaligned computation.

Also tag W->C routes whose source binary route is ALL-OFF.

## 4. Executor Audit

Reuse the already validated unified four-action executor from the previous FULL-context four-action analysis and route-conditioned READ/WRITE decomposition.

Verify:
1. `FULL = READ on, WRITE on`
2. `READ_ONLY = READ on, WRITE off`
3. `WRITE_ONLY = READ off, WRITE on`
4. `IGNORE = READ off, WRITE off`
5. binary `ON` reproduces unified `FULL`
6. binary `OFF` reproduces unified `IGNORE`
7. arbitrary multi-layer four-action routes can be executed
8. existing benchmark-specific evaluators and answer scoring are reused
9. deterministic generation settings are preserved

Do not create another independent executor unless required.

## 5. Source Route Replay

Before converting any source positive binary route:

1. Map `ON -> FULL`, `OFF -> IGNORE`.
2. Replay the complete route using the current unified executor.
3. Verify evaluator correctness.

If the route no longer reproduces:
- mark it as `source_route_replay_failure`
- do not refine it
- do not invent a replacement
- do not rerun MCTS automatically

Recompute current unified FULL correctness and classify the route as current-runtime `W2C` or `C2C`.

## 6. W->C Sequential Four-Action Refinement

### 6.1 Starting Point

Suppose a known correct binary route is:

- L2 = OFF
- L4 = OFF
- L7 = OFF
- all others = ON

Initial four-action route:

- L2 = IGNORE
- L4 = IGNORE
- L7 = IGNORE
- all others = FULL

This complete trajectory is already known to be correct.

Process the original binary-OFF layers in a fixed deterministic order.

Recommended production order: `early -> late`.

Do not tune the order to maximize results.

### 6.2 Per-Layer Decision Rule

For every currently surviving route branch and target layer `l`:

#### Step 1: Try FULL restoration

Change only the target layer to `FULL`.

Keep all earlier decisions and all remaining unprocessed binary-OFF layers exactly as they are in the current route branch.

If the resulting complete trajectory is still correct:
- set layer `l = FULL`
- do not run READ_ONLY / WRITE_ONLY for that branch at that layer

Interpretation: the binary OFF at this layer was unnecessary in the current correcting trajectory.

#### Step 2: If FULL becomes wrong, test partial suppressions

Evaluate:
- `READ_ONLY`
- `WRITE_ONLY`

`IGNORE` does not need to be rerun because the current branch with `IGNORE` is already known correct.

Use:

| FULL | READ_ONLY | WRITE_ONLY | Next action |
|---|---|---|---|
| correct | - | - | `FULL` |
| wrong | correct | wrong | `READ_ONLY` |
| wrong | wrong | correct | `WRITE_ONLY` |
| wrong | wrong | wrong | `IGNORE` |
| wrong | correct | correct | branch into both `READ_ONLY` and `WRITE_ONLY` |

Interpretation:
- `READ_ONLY` retained -> WRITE must remain suppressed
- `WRITE_ONLY` retained -> READ must remain suppressed
- `IGNORE` retained -> both must remain suppressed
- both partial actions valid -> multiple valid refinements exist

Do not force a unique READ-vs-WRITE explanation when both are valid.

## 7. Branching

If both `READ_ONLY` and `WRITE_ONLY` are correct, keep both branches.

Example:

Branch A:
- L2 READ_ONLY
- L4 IGNORE
- L7 IGNORE

Branch B:
- L2 WRITE_ONLY
- L4 IGNORE
- L7 IGNORE

Then process L4 independently under each branch.

This is necessary because later-layer actions may depend on earlier decisions.

Do not independently select a best action at each layer and combine them without executing the full trajectory.

Every surviving route must be evaluator-correct as a complete trajectory.

## 8. Search Policy

Do not run:
- binary MCTS
- four-action MCTS
- global `4^28` search
- the previous beam-width optimization

Use sequential verified branching only.

Initially keep all valid branches.

Record:
- active branch count per sample
- maximum branch count
- total route evaluations per sample
- branch-count median / P90 / P99 / max

If branch explosion becomes a real practical problem, stop only pathological cases and design a separate bounded-cap policy.

Do not introduce a beam width before showing branching is actually intractable.

## 9. W->C ALL-OFF Seeds

Include W->C routes whose binary source route is ALL-OFF.

Tag `all_off_seed = true`.

Start from 28 `IGNORE` actions and apply the same sequential refinement.

Keep these samples separately identifiable in analysis.

Do not merge ALL-OFF-rescued W->C with positive-vision W->C when making mechanism claims.

## 10. C->C Conversion

For C->C routes, do not use the W->C restoration procedure.

Because FULL is already correct, repeatedly restoring binary OFF positions would trivially collapse C->C routes back toward FULL and destroy their efficiency/redundancy supervision.

For the first conversion version:
- binary ON -> FULL
- binary OFF -> IGNORE

Store `label_semantics = preserving_c2c`.

For W->C store `label_semantics = corrective_w2c`.

A future efficiency-oriented C->C READ/WRITE refinement can be designed separately if needed.

## 11. Multiple Binary Routes per Sample

A sample may contain many valid source binary routes.

Process all positive routes in the selected authoritative supervision view.

Within one sample:
1. preprocess/load the image once where possible
2. cache unified FULL
3. cache every evaluated complete four-action route by exact 28-layer action sequence
4. reuse cached results across all source binary routes
5. deduplicate identical final four-action routes

If multiple source routes refine to the same final route, preserve all source route IDs as provenance.

## 12. Four-Action Label Output

For every final valid route store at least:
- dataset
- sample_id
- image_id
- split
- current_full_answer
- current_full_correctness
- route_type = W2C / C2C
- label_semantics
- all_off_seed
- source_binary_route_id
- source_binary_route
- source_binary_off_count
- four_action_route[28]
- num_FULL
- num_READ_ONLY
- num_WRITE_ONLY
- num_IGNORE
- read_suppression_count
- write_suppression_count
- generated_answer
- evaluator_correctness
- answer score / margin if available
- source provenance
- executor version/hash
- worker/shard provenance

For each sample create:
1. raw source-to-converted mapping
2. unique valid four-action route set
3. later-training view
4. source provenance mapping

Do not overwrite original binary labels.

## 13. Smoke Test

### 13.1 Size

Use only 8 samples total.

This is an implementation/logic smoke test, not population-level calibration.

Choose deliberately to cover as many paths as possible across:
- GQA
- TextVQA
- ChartQA
- WeMath2.0 Standard
- WeMath2.0 Pro
- W->C
- C->C
- short OFF route
- longer OFF route
- multi-source-route sample
- ALL-OFF W->C if available

Use synthetic/unit tests for branching logic if a desired ambiguity case does not naturally occur.

### 13.2 Smoke-Test GPU Configuration

Use:

`8 GPUs x 1 process/GPU = 8 concurrent worker processes`

Each process:
- loads one model replica once
- is pinned to one GPU
- processes one sample at a time
- reuses the model across work
- does not reload the model per route

The smoke-test goal is correctness, not maximum utilization.

### 13.3 Smoke-Test Validation

Verify:
1. source binary positive route replay
2. ON -> FULL parity
3. OFF -> IGNORE parity
4. FULL restoration logic
5. READ_ONLY logic
6. WRITE_ONLY logic
7. IGNORE fallback logic
8. branch creation when both partial actions are valid
9. later-layer processing uses the branch's updated trajectory context
10. every stored route is evaluator-correct
11. route-cache reuse works
12. deduplication works
13. resume works
14. outputs from different workers do not collide
15. no evaluator target changes across route variants
16. no unexpected numerical/executor inconsistency

If any semantic or correctness issue is found:
- stop
- fix
- rerun the 8-sample smoke

Do not launch the full conversion until the smoke test is clean.

No beam-width stability test is required.

## 14. Full Conversion GPU Configuration

After the 8-sample smoke passes, launch the full conversion using all 8 GPUs.

Preferred production configuration:

`8 GPUs x 2 processes/GPU = 16 concurrent worker processes`

Each process:
- owns one model replica
- processes one sample at a time
- loads the model once and reuses it

Conceptually:

- GPU0: worker0 + worker1
- GPU1: worker2 + worker3
- ...
- GPU7: worker14 + worker15

Use 2 processes/GPU as the production default.

If technically unstable, OOMs, or causes severe throughput problems:
1. diagnose and try to fix it first
2. document the issue
3. only then fall back to 1 process/GPU if necessary

Do not silently change the execution topology.

## 15. Work Scheduling and Load Balancing

Do not make exactly 16 static equal-sample shards unless sample costs are known to be balanced.

Conversion cost depends on:
- number of source positive routes
- number of original OFF positions
- number of branches created
- route-cache overlap

Prefer many small deterministic shards or a shared work queue.

Keep all 16 workers fed until remaining work is too small.

Track:
- samples completed
- source routes processed
- four-action route evaluations
- valid branches
- wall-clock time
- GPU utilization
- peak memory

Primary throughput metric: completed valid work per wall-clock second.

GPU utilization is secondary.

## 16. Resumability

Requirements:
- append-only or atomic per-sample outputs
- deterministic sample IDs
- per-worker progress
- completed samples skipped on restart
- route-evaluation cache preserved if practical
- no duplicate final routes
- provenance retained
- failed sample does not invalidate other completed work

## 17. Required Analysis

Report separately for:
- GQA
- TextVQA
- ChartQA
- WeMath2.0 Standard
- WeMath2.0 Pro

For W->C report:
1. source positive binary routes
2. source-route replay success/failure
3. final valid four-action routes
4. deduplication ratio
5. fraction of binary OFF positions restored to FULL
6. fraction refined to READ_ONLY
7. fraction refined to WRITE_ONLY
8. fraction remaining IGNORE
9. depth distribution of each action
10. average final branches/routes per source route
11. branch-count median/P90/P99/max
12. number of ALL-OFF seeds
13. how ALL-OFF seeds refine
14. differences across datasets

For C->C report separately:
- number of labels
- FULL/IGNORE distribution
- source OFF-count distribution
- efficiency/redundancy summary

Do not combine W->C and C->C mechanism interpretations.

## 18. Scientific Interpretation

W->C labels support route-conditioned statements such as:

- within this correcting trajectory, suppression at layer l is unnecessary
- within this correcting trajectory, WRITE must remain suppressed while READ can be restored
- within this correcting trajectory, READ must remain suppressed while WRITE can be restored
- within this correcting trajectory, either READ or WRITE suppression is sufficient

Do not interpret these labels as globally unique root causes.

An operation may be harmful because of previous trajectory decisions.

The labels are valid route-conditioned corrective programs, not unique causal origins.

## 19. Output Root

Create a new root such as:

`datasets/mcts_labels_4action/`

Suggested structure:

- `gqa_textvqa_chartqa_v1/`
- `wemath2_standard/`
- `wemath2_pro/`
- `reports/implementation_audit.md`
- `reports/smoke_test_report.md`
- `reports/four_action_label_conversion_report.md`

Preserve source datasets unchanged.

## 20. Final Decision Questions

At completion explicitly answer:

1. Which authoritative source artifact was used for each dataset?
2. How many positive binary routes replayed successfully?
3. How many W->C and C->C routes were processed?
4. How many unique valid four-action routes were produced?
5. How often were binary OFF positions:
   - unnecessary -> FULL
   - WRITE suppression only -> READ_ONLY
   - READ suppression only -> WRITE_ONLY
   - both suppression required -> IGNORE
6. How often did branching occur?
7. Was branch explosion manageable without beam search?
8. How different were the five datasets?
9. Did ALL-OFF W->C routes remain globally suppressed or refine into selective visual programs?
10. Are the resulting valid four-action sets clean enough for downstream router training?
11. Is any fresh four-action MCTS still scientifically or practically necessary?

## 21. Execution Order

1. audit authoritative sources
2. audit/reuse unified four-action executor
3. implement sequential branching converter
4. run unit/synthetic logic tests
5. run 8-sample smoke test with `8 GPUs x 1 process/GPU`
6. if issue: fix and rerun smoke
7. if clean: run full conversion with `8 GPUs x 2 processes/GPU = 16 workers`
8. merge
9. deduplicate
10. validate every final route
11. build output views
12. final analysis/report

Do not rerun MCTS.
Do not use beam search by default.
Do not overwrite source binary labels.
