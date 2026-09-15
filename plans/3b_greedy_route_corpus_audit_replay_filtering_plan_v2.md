# 3B Greedy Route Corpus Audit & Current-Server Replay Filtering Plan — V2
## Include current-server executor parity, Dense/FULL replay, routed replay, and current-label cohort reconstruction before Routing Trajectory Representation Geometry

## 0. Purpose

Before any hidden-state geometry analysis, first make the 3B greedy-search corpus trustworthy on the **current server/runtime**.

Canonical package:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
```

Canonical merged route corpus:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
02_GREEDY_SEARCH/vqa_10k/final_phase1_phase2/
```

Current paired dataset:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
03_PAIRED_DATASETS/vqa_correctness_first_v31/
```

The package is read-only.

This phase must establish four things before geometry:

```text
A. Current-server executor parity:
   Standard HF FULL == custom executor all-FULL.

B. Current-server Dense/FULL outcome:
   every sample's dense answer/correctness must be replayed now.

C. Current-server routed outcome:
   every saved route must be replayed now.

D. Geometry cohort reconstruction:
   C-C / C-W / W-W and Dense-C/W→Route-C/W must be defined
   using CURRENT-SERVER labels only.
```

Old labels are preserved for transition analysis but are never used as the final geometry ground truth.

---

# 1. Why Dense/FULL replay is mandatory

The original server and the current server may disagree.

Therefore a sample that was historically:

```text
old Dense = Wrong
old Route A = Correct
```

may become:

```text
current Dense = Correct
current Route A = Correct
```

and is no longer a correction case.

Likewise:

```text
old Dense = Correct
old Route = Wrong
```

can become a current-server W→C case if both outcomes flip differently.

Therefore all future categories must be reconstructed from:

```text
current_dense_correctness
current_route_correctness
```

not from historical saved labels.

---

# 2. Two different replay checks must be separated

There are two conceptually different comparisons.

## Gate A — Current-server implementation parity

On the **same current server**, same model/input:

```text
Standard Hugging Face generation
vs
Custom route executor with all layers FULL
```

must match under the frozen generation contract.

Required invariant:

```text
HF_STANDARD_FULL
==
CUSTOM_ALL_FULL
```

This gate must pass before any route comparison is trusted.

## Audit B — Old-server vs current-server replay drift

After Gate A passes:

```text
old saved Dense/route result
vs
current verified custom-executor result
```

may differ.

Those differences are exactly what this audit is meant to measure.

Do not confuse:

```text
executor parity failure
```

with:

```text
server/runtime replay drift.
```

---

# 3. Immediate rule after a Gate-A failure

If:

```text
Standard HF FULL != custom executor all-FULL
```

then:

```text
STOP full corpus replay.
```

Do not launch full multi-GPU replay again.

Instead run a focused single-example parity diagnostic on one failing anchor.

Compare token-by-token:

```text
prefill outputs
first generation-step logits
selected first generated token
post-token KV cache
position_ids
cache_position
attention_mask
RoPE positions
visual token positions
second-step logits
selected second generated token
```

Record the **first numerical divergence point**.

If the observed pattern is:

```text
first generated token matches
second generated token diverges
```

prioritize decode-state/KV-cache/position handling, but do not assume the cause.

Environment differences such as different PyTorch/CUDA versions may contribute, but they are not accepted as the root cause without evidence.

Create:

```text
parity/current_executor_parity_diagnostic.md
```

Do not resume corpus replay until Gate A passes.

---

# 4. Canonical source contract

Read first:

```text
README.md
00_METADATA/PACKAGE_SUMMARY.json
00_METADATA/PATH_MAP.json
```

Then inspect:

```text
02_GREEDY_SEARCH/vqa_10k/
```

for config, replay gate, raw Phase 1/2, aggregate files, and `final_phase1_phase2`.

Use `final_phase1_phase2` as canonical route corpus if schema/lineage checks pass.

Raw Phase1/2 are lineage/resume references only.

---

# 5. Hard invariants

Never modify the canonical package.

Preserve:

```text
old_dense_answer
old_dense_correctness
old_route_answer
old_route_correctness
```

and add:

```text
current_dense_answer
current_dense_correctness
current_route_answer
current_route_correctness
```

Old and current labels must coexist.

Do not overwrite V3.1 pairs.

---

# 6. Output root

Use:

```text
analysis/3b_greedy_route_corpus_replay_audit_v2/
```

---

# Stage A — Source inventory and corpus census

## 7. File inventory

Create `source_inventory/file_inventory.csv` with path, role, size, sha256, and canonical/raw status.

Include README, metadata, path map, generation config, gate files, final merged route files, and V3.1 pair manifests.

---

## 8. Canonical route schema

Create `source_inventory/route_schema.md`.

Map actual source fields to:

```text
sample_uid
dataset
image/content_group
question
gold answer
route ID
route action sequence
route provenance
phase origin
old generated answer
old correctness
dense identity
trigger/start metadata if any
generation metadata
```

---

## 9. Route identity contract

Define:

```text
route_key =
    sample_uid
    + exact action sequence
    + any required start/domain metadata
```

Preserve original immutable route IDs if present.

Store the exact definition in `frozen_contract.json`.

---

## 10. Existing corpus census

Measure directly:

```text
unique samples
unique image groups
unique routes
raw records
duplicate route records
```

Create `census/global_census.json`.

---

## 11. Routes per sample

Create `census/routes_per_sample.csv`.

Report:

```text
mean
std
min
p10
p25
median
p75
p90
p95
p99
max
```

---

## 12. Old saved-label census

Route-level:

```text
old Correct
old Wrong
missing/invalid
```

Sample-level:

```text
ALL_C
ALL_W
MIXED_CW
```

Preliminary geometry counts:

```text
>=1C & >=1W
>=2C
>=2W
>=2C & >=1W
>=2W & >=1C
>=2C & >=2W
```

Historical only.

---

# Stage B — Dense route identity and old baseline

## 13. Identify exactly one Dense/FULL route per sample

Report:

```text
exactly one dense route
zero dense route
multiple dense candidates
```

Any ambiguity blocks later cohort reconstruction.

Create `dense/original_dense_manifest.jsonl`.

---

# Stage C — Reconstruct current-server runtime contract

## 14. Frozen generation contract

Document:

```text
exact model revision/snapshot
tokenizer/processor revision
chat template
image preprocessing
dtype
attention backend
torch / transformers versions
generation parameters
greedy decoding settings
max_new_tokens
stopping criteria
evaluation normalization
dataset-specific scoring
```

Create:

```text
replay_gate/replay_contract.md
replay_gate/current_server_env.json
```

Mark missing fields `NOT RECORDED`.

---

# Stage D — Gate A: Standard HF vs custom all-FULL parity

## 15. Parity anchor set

Use the existing package anchor set if available; otherwise freeze one stratified set before outcome inspection.

For every anchor run:

```text
Standard HF generation
Custom executor with FULL on every layer
```

Compare generated token IDs, text, correctness, and per-step logits where practical.

Create:

```text
parity/current_hf_vs_custom_full.csv
parity/gate_a_report.md
```

---

## 16. Gate-A pass criterion

Primary requirement:

```text
identical generated token sequence
```

under the frozen deterministic generation contract.

Do not silently loosen this criterion.

---

## 17. Focused parity debugging if Gate A fails

On one failing example compare prefill and decode step-by-step.

At each step compare:

```text
hidden-state checksum/norm
logits top-k
selected token
KV shapes/checksums
position_ids
cache_position
attention mask
visual positions
```

Find first divergence, fix the cause, then rerun the complete anchor set.

No full replay until Gate A passes.

---

# Stage E — Current-server Dense/FULL replay

## 18. Replay Dense/FULL for every sample

After Gate A passes, replay every sample's Dense/FULL route using the **verified custom executor**.

This is mandatory.

Create `dense/current_dense_manifest.jsonl` with:

```text
sample_uid
dense_route_key
old_dense_answer
old_dense_correctness
current_dense_answer
current_dense_correctness
dense_transition
replay_status
```

---

## 19. Dense transition matrix

Report:

```text
old Dense C -> current Dense C
old Dense C -> current Dense W
old Dense W -> current Dense C
old Dense W -> current Dense W
```

overall and per dataset.

Create `transitions/dense_transition_matrix.csv`.

---

# Stage F — Current-server replay of every routed trajectory

## 20. Replay every unique route exactly once

For each route:

```text
replay via verified custom executor
generate current answer
score current correctness
```

Store `routes/current_route_replay.jsonl`.

Fields include old/current answers, old/current correctness, transition, status, runtime, model/runtime identifiers.

---

## 21. Routed transition matrix

Report:

```text
old Route C -> current Route C
old Route C -> current Route W
old Route W -> current Route C
old Route W -> current Route W
```

overall and by dataset, provenance, phase, route length, non-FULL count, and action composition.

---

## 22. Replay errors and resumability

Create `routes/replay_errors.jsonl`.

Replay must be deterministic and resumable without duplicate completed records.

---

# Stage G — Current-server cohort reconstruction

## 23. Final sample anchor uses CURRENT Dense only

Define:

```text
Dense-C:
    current_dense_correctness = Correct

Dense-W:
    current_dense_correctness = Wrong
```

---

## 24. Final route label uses CURRENT route only

For sample `i`:

```text
C_i = all replay-valid routes currently Correct
W_i = all replay-valid routes currently Wrong
```

Historical route correctness is retained only for replay-transition analysis.

---

## 25. Rebuild the four current categories

Using current Dense + current route labels:

```text
Dense-C -> Route-C
Dense-C -> Route-W
Dense-W -> Route-C
Dense-W -> Route-W
```

Interpret as:

```text
preservation
regression
correction
unresolved failure
```

Create:

```text
filtering/current_route_manifest.jsonl
filtering/current_route_categories.csv
```

This is the only valid basis for geometry cohorts.

---

## 26. Old-vs-current category transition audit

Record semantic changes such as:

```text
old W→C correction
→ current C→C preservation

old C→W regression
→ current W→C correction
```

Create `transitions/semantic_route_category_transition.csv`.

---

# Stage H — Geometry eligibility

## 27. Current C-C / C-W / W-W eligibility

Using current labels only, count:

```text
|C_i| >= 1 and |W_i| >= 1
|C_i| >= 2
|W_i| >= 2
|C_i| >= 2 and |W_i| >= 1
|W_i| >= 2 and |C_i| >= 1
|C_i| >= 2 and |W_i| >= 2
```

Create `geometry_readiness/geometry_eligibility.csv`.

Break down by dataset and current Dense-C/W.

---

## 28. W→C geometry cohort

Define:

```text
current Dense-W
AND
at least one current Route-C
```

Create `geometry_readiness/w_to_c_samples.jsonl`.

Report correct/wrong route counts and unique action sequences per sample.

---

## 29. C→C geometry cohort

Define:

```text
current Dense-C
AND
at least one current Route-C
```

Create `geometry_readiness/c_to_c_samples.jsonl`.

---

# Stage I — Pair-count and route-action diversity

## 30. Theoretical pair counts

For every sample:

```text
C-C pairs = C(|C_i|,2)
W-W pairs = C(|W_i|,2)
C-W pairs = |C_i|*|W_i|
```

Report totals, median/sample, p90/sample, max/sample.

Create `geometry_readiness/pair_count_estimates.csv`.

---

## 31. Route-action distance

Compute within-sample route Hamming-distance distributions for C-C, W-W, and C-W.

Create `geometry_readiness/route_action_diversity.csv`.

Future hidden-state comparisons must control for route/action distance.

---

# Stage J — Hidden-state storage planning

## 32. Last-question-token hidden states

Future geometry must at minimum capture the last question-token hidden state at every decoder layer.

Estimate exact storage for BF16/FP16/FP32:

```text
# selected routes × #layers × hidden_dim × bytes
```

Create `storage/last_question_token_storage_estimate.md`.

---

## 33. Visual-token hidden states

Estimate:

```text
sum_routes(
  #visual_tokens × #layers × hidden_dim × bytes
)
```

for BF16/FP16/FP32.

Create `storage/visual_hidden_storage_estimate.md`.

Do not automatically store all raw visual tensors.

---

## 34. Hidden-state capture decision

Recommended:

```text
last question token:
    capture all layers if manageable

visual tokens:
    defer raw all-layer capture unless explicitly approved
```

Possible later alternatives:

```text
mean pooled visual states
fixed random projection
selected layers
selected route subset
streaming pairwise statistics
```

Create `storage/hidden_capture_decision.md`.

---

# Stage K — Audit V3.1 pair supervision

## 35. Re-evaluate pair validity

Do not modify V3.1.

For every pair classify:

```text
both labels stable
preferred/rejected reversed
preferred changed only
rejected changed only
route replay error
```

Create `filtering/v31_pair_replay_audit.csv`.

If later needed, create a new versioned current-server pair dataset. Do not rebuild it now.

---

# 36. Replay-stability categories

## RS-A — Stable

Gate A passes; old→current changes are rare; Dense labels mostly stable.

## RS-B — Moderate drift

Non-trivial old→current changes, but current replay still yields sufficient geometry cohorts.

## RS-C — Severe drift

Large Dense and/or route label changes; old cohort meanings change substantially.

Do not invent post-hoc thresholds. Report actual transition rates and practical impact.

---

# 37. Geometry-readiness decision

READY only if:

```text
1. Standard HF == custom all-FULL parity passes;
2. Dense/FULL replay succeeds for essentially all samples;
3. routed replay coverage is sufficiently complete;
4. Dense anchors are unambiguous;
5. current C-C / C-W / W-W cohorts are sufficiently large;
6. route/action identities are exact;
7. hidden-state storage/extraction is feasible.
```

Otherwise report the exact blocker.

---

# 38. Required top-level summary

Produce:

```text
Current-Server 3B Route Corpus Audit

Unique samples: ?
Unique image groups: ?
Unique routes: ?

Routes/sample:
  mean:
  median:
  p90:
  max:

OLD route labels:
  Correct:
  Wrong:

CURRENT Dense transitions:
  old C -> new C:
  old C -> new W:
  old W -> new C:
  old W -> new W:

CURRENT route transitions:
  old C -> new C:
  old C -> new W:
  old W -> new C:
  old W -> new W:

CURRENT Dense:
  Dense-C:
  Dense-W:

CURRENT route categories:
  Dense-C -> Route-C:
  Dense-C -> Route-W:
  Dense-W -> Route-C:
  Dense-W -> Route-W:

Geometry eligibility:
  >=1C & >=1W:
  >=2C:
  >=2W:
  >=2C & >=1W:
  >=2W & >=1C:
  >=2C & >=2W:
```

---

# 39. Dataset-specific summary

Repeat key counts for GQA, ChartQA, DocVQA, and TextVQA.

---

# 40. Required summary files

Create:

```text
summaries/corpus_audit_summary.md
summaries/executor_parity_summary.md
summaries/replay_filtering_summary.md
summaries/geometry_readiness_summary.md
summaries/next_phase_recommendation.md
```

`executor_parity_summary.md` must explicitly state whether current HF FULL and custom all-FULL matched, where any initial divergence occurred, what fixed it, and whether full replay was authorized.

`replay_filtering_summary.md` must separately report Dense old→current transitions and routed old→current transitions.

`geometry_readiness_summary.md` must use current-server labels only.

---

# 41. No geometry claims yet

Do not yet claim:

```text
correct routes converge
wrong routes diverge
C/W manifolds separate
visual states converge
text states converge
```

This phase only certifies the corpus.

---

# 42. No finetuning yet

Do not start LoRA, adapters, distillation, preference training, or representation alignment until executor parity, current replay filtering, and geometry analysis are complete.

---

# 43. Execution order

Run exactly:

```text
1. read package README/metadata/path map
2. inventory canonical files
3. audit merged-route schema
4. census samples/routes/old labels
5. identify canonical dense route per sample
6. reconstruct current runtime/generation contract
7. run Gate A: current HF FULL vs custom all-FULL
8. if Gate A fails:
       STOP full replay
       run focused token/KV/cache parity diagnostic
       fix cause
       rerun Gate A
9. after Gate A passes:
       replay Dense/FULL for every sample
10. build Dense old→current transition matrix
11. replay every unique routed trajectory
12. build routed old→current transition matrix
13. reconstruct current Dense-C/W and Route-C/W categories
14. audit old category → current category semantic changes
15. compute geometry eligibility
16. compute pair counts and action-distance distributions
17. estimate hidden-state storage
18. audit V3.1 pair validity
19. write RS-A/B/C result
20. write geometry-readiness decision
```

---

# 44. Required artifacts

```text
analysis/3b_greedy_route_corpus_replay_audit_v2/
├── protocol.md
├── frozen_contract.json
├── source_inventory/
│   ├── file_inventory.csv
│   └── route_schema.md
├── census/
│   ├── global_census.json
│   ├── routes_per_sample.csv
│   ├── saved_label_summary.csv
│   └── dataset_breakdown.csv
├── parity/
│   ├── current_hf_vs_custom_full.csv
│   ├── current_executor_parity_diagnostic.md
│   └── gate_a_report.md
├── replay_gate/
│   ├── replay_contract.md
│   └── current_server_env.json
├── dense/
│   ├── original_dense_manifest.jsonl
│   └── current_dense_manifest.jsonl
├── routes/
│   ├── current_route_replay.jsonl
│   └── replay_errors.jsonl
├── transitions/
│   ├── dense_transition_matrix.csv
│   ├── route_transition_matrix.csv
│   ├── route_transition_by_dataset.csv
│   └── semantic_route_category_transition.csv
├── filtering/
│   ├── current_route_manifest.jsonl
│   ├── current_route_categories.csv
│   └── v31_pair_replay_audit.csv
├── geometry_readiness/
│   ├── geometry_eligibility.csv
│   ├── w_to_c_samples.jsonl
│   ├── c_to_c_samples.jsonl
│   ├── pair_count_estimates.csv
│   └── route_action_diversity.csv
├── storage/
│   ├── last_question_token_storage_estimate.md
│   ├── visual_hidden_storage_estimate.md
│   └── hidden_capture_decision.md
├── summaries/
│   ├── corpus_audit_summary.md
│   ├── executor_parity_summary.md
│   ├── replay_filtering_summary.md
│   ├── geometry_readiness_summary.md
│   └── next_phase_recommendation.md
└── artifact_manifest.json
```

---

# 45. Stop rule

STOP after:

```text
1. corpus census
2. executor parity fully resolved
3. Dense/FULL current replay
4. routed current replay
5. old→current transition analysis
6. current cohort reconstruction
7. geometry eligibility census
8. route-pair/action-diversity estimates
9. hidden-state storage decision
10. V3.1 audit
11. RS-A/B/C decision
12. READY / NOT READY decision
```

Do not start hidden-state geometry itself.

Do not start finetuning.

---

# 46. Core principle

The geometry phase must compare representations under labels that are actually true on the current server.

Therefore the order is:

```text
first:
    verify the current executor itself

then:
    replay Dense/FULL and every saved route

then:
    rebuild all C/W route categories from current labels

only then:
    analyze routing-trajectory representation geometry.
```

Historical correctness remains valuable for drift analysis, but it is not trusted as the final geometry label until current-server replay confirms it.
