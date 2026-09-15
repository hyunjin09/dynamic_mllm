# 3B Greedy Route Corpus Audit & Current-Server Replay Filtering Plan
## Prepare the canonical route corpus before Routing Trajectory Representation Geometry

## 0. Purpose

Before analyzing routing-trajectory hidden-state geometry, first establish exactly what the existing 3B greedy-search corpus contains and which saved route labels remain valid on the current server/runtime.

Canonical package:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
```

Canonical greedy-search candidate corpus:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
02_GREEDY_SEARCH/vqa_10k/final_phase1_phase2/
```

Current paired dataset:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/
03_PAIRED_DATASETS/vqa_correctness_first_v31/
```

The source package is immutable.

Do not modify, relabel, overwrite, or delete canonical files.

The objective of this phase is:

```text
1. census the existing route population;
2. verify the exact 3B generation/replay contract;
3. replay the routes on the current server;
4. preserve old labels and add current-server labels in a NEW derived manifest;
5. rebuild Dense-C / Dense-W and route-C / route-W categories;
6. quantify which samples are usable for future C-C / C-W / W-W geometry;
7. estimate hidden-state storage requirements before the geometry phase.
```

This phase is preparation for:

```text
Routing Trajectory Representation Geometry
```

It is not yet the geometry experiment itself.

---

# 1. Canonical package contract

Read first:

```text
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/README.md
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/00_METADATA/PACKAGE_SUMMARY.json
datasets/Qwen_3B_7B/Qwen2.5-VL-3B-Instruct/00_METADATA/PATH_MAP.json
```

Then inspect only the files needed to reconstruct:

```text
exact Qwen2.5-VL-3B-Instruct model revision
prompt/chat template
image preprocessing contract
generation config
greedy-decoding parameters
evaluation/correctness normalization
route/action semantics
Phase-1 / Phase-2 merge semantics
generation-anchor replay gate
canonical result schema
```

Also inspect:

```text
02_GREEDY_SEARCH/vqa_10k/
```

for:

```text
config
replay gate
raw Phase 1
raw Phase 2
Phase-1 aggregate
final_phase1_phase2
```

Use `final_phase1_phase2` as the canonical candidate corpus.

Raw Phase 1/2 are for:

```text
schema verification
lineage checks
resume/audit
```

not for primary counting if the final merged corpus is valid.

---

# 2. Hard invariants

Preserve these rules throughout the phase:

```text
canonical package files are read-only
current-server labels are written only to a new analysis root
saved old labels are never destroyed
route identity is preserved
sample identity is preserved
dense identity is preserved
old and new correctness are both retained
```

Do not silently replace:

```text
old_correctness
```

with:

```text
current_correctness
```

Instead retain both.

---

# 3. Output directory

Use:

```text
analysis/3b_greedy_route_corpus_replay_audit/
```

Create all new artifacts there.

Suggested structure:

```text
analysis/3b_greedy_route_corpus_replay_audit/
├── protocol.md
├── frozen_contract.json
├── source_inventory/
├── census/
├── replay_gate/
├── replay/
├── dense/
├── filtering/
├── geometry_readiness/
├── storage/
├── summaries/
└── artifact_manifest.json
```

---

# Stage A — Corpus inventory and schema audit

## 4. Inventory every relevant canonical file

Build:

```text
source_inventory/file_inventory.csv
```

with:

```text
relative_path
file_type
size_bytes
sha256
role
canonical_or_raw
```

At minimum include:

```text
README
metadata summary
path map
greedy config
replay gate files
final_phase1_phase2 files
current paired-dataset manifests
```

Do not hash model weights unless necessary.

---

## 5. Determine the canonical route-record schema

Document in:

```text
source_inventory/route_schema.md
```

For each candidate route record identify fields corresponding to:

```text
sample UID
dataset
image/content identity
question
answer target
dense correctness if present
route correctness
generated answer
route action sequence
route provenance
Phase-1 / Phase-2 origin
trigger/start layer if applicable
search rank / score if present
model revision
generation config references
```

Do not assume field names.

Map actual source fields to canonical names.

---

## 6. Resolve route identity

Define a canonical route key prospectively.

Preferred:

```text
route_key =
    sample_uid
    + exact action sequence
    + any required start-layer / route-domain metadata
```

If the original collector provides an immutable route ID, preserve it too.

Write:

```text
frozen_contract.json
```

containing the exact route-key definition.

---

# Stage B — Existing corpus census

## 7. Count samples and routes

Produce:

```text
census/global_census.json
```

with:

```text
unique sample count
unique image/content-group count
unique route count
raw record count
duplicate record count
duplicate canonical route count
```

Do not infer that `vqa_10k` means exactly 10,000 samples or routes.

Measure it.

---

## 8. Routes per sample

Produce:

```text
census/routes_per_sample.csv
```

and summary statistics:

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

Also report histogram bins:

```text
1 route
2-4
5-9
10-19
20-49
50+
```

If counts vary strongly by dataset, report per dataset.

---

## 9. Existing saved correctness distribution

Using the canonical saved labels only, report:

```text
route-level saved Correct count
route-level saved Wrong count
route-level missing/invalid label count
```

Also per dataset:

```text
GQA
ChartQA
DocVQA
TextVQA
```

Do not pool source datasets without a separate macro view.

---

## 10. Sample-level saved outcome composition

For each sample compute:

```text
# saved-correct routes
# saved-wrong routes
# total routes
```

Classify samples as:

```text
ALL_C
ALL_W
MIXED_CW
```

Report counts.

Also report preliminary geometry eligibility under saved labels:

```text
>=1 C and >=1 W
>=2 C
>=2 W
>=2 C and >=1 W
>=2 W and >=1 C
>=2 C and >=2 W
```

These are preliminary only until current-server replay is complete.

---

## 11. Dense-route audit

Identify the dense/full route for each sample.

Do not assume one exists until verified.

Report:

```text
samples with exactly one identifiable dense route
samples with zero dense routes
samples with multiple dense candidates
```

Any ambiguity must be resolved before replay categorization.

Create:

```text
dense/dense_route_manifest_original.jsonl
```

with:

```text
sample_uid
dense_route_key
old_dense_answer
old_dense_correctness
```

---

## 12. Existing route-category census

Under original saved labels classify route records relative to the original dense label:

```text
old Dense-C -> old Route-C
old Dense-C -> old Route-W
old Dense-W -> old Route-C
old Dense-W -> old Route-W
```

Report both:

```text
route counts
unique sample counts
```

This is historical reference only.

---

## 13. Dataset / phase / provenance breakdown

Report census by:

```text
dataset
Phase-1 vs Phase-2 provenance
route source/provenance type
route length
# non-FULL actions
action composition if available
```

Do not yet interpret hidden-state geometry.

This is only to understand corpus composition and future sampling bias.

---

# Stage C — Replay contract reconstruction

## 14. Recover exact model/runtime contract

Create:

```text
replay_gate/replay_contract.md
```

Document:

```text
model name
exact revision / snapshot hash
torch version
transformers version if recorded
Qwen-VL code revision if recorded
tokenizer revision
processor revision
image preprocessing
chat template
prompt formatting
generation parameters
greedy decoding settings
max_new_tokens
stopping criteria
dtype
attention implementation
evaluation normalization
benchmark correctness rules
```

If the package does not support a field, explicitly mark:

```text
NOT RECORDED
```

Do not guess.

---

## 15. Current-server environment report

Create:

```text
replay_gate/current_server_env.json
```

including:

```text
hostname or machine alias
GPU model
CUDA version
torch version
transformers version
Python version
model snapshot path/hash
processor/tokenizer snapshot
critical environment variables
```

Do not modify the source package to match current paths.

Use relocated manifests.

---

## 16. Generation-anchor replay gate

Before replaying the full corpus, reproduce the package's existing generation-anchor gate.

Use the exact frozen anchor set specified by the source artifacts.

If the source gate defines exact token parity, check exact token parity.

Otherwise reproduce its documented acceptance rule.

Report:

```text
# anchor samples
exact generated-answer matches
exact token-sequence matches if available
correctness matches
mismatch details
```

Write:

```text
replay_gate/gate_result.json
replay_gate/gate_report.md
```

---

## 17. Hard gate

If the replay gate fails:

```text
STOP full replay.
```

Diagnose first.

Possible causes to inspect:

```text
wrong model revision
chat-template drift
processor drift
generation config mismatch
different image path/content
different evaluator
dtype/backend nondeterminism
code-version mismatch
```

Do not proceed by simply accepting the drift.

---

# Stage D — Full current-server replay

## 18. Replay each unique route exactly once

After the gate passes, replay every canonical unique route on the current server.

For each route preserve:

```text
sample_uid
route_key
action sequence
dataset
old_generated_answer
old_correctness
current_generated_answer
current_correctness
replay_status
runtime
model revision
evaluation contract
```

Save:

```text
replay/current_replay_routes.jsonl
```

---

## 19. Never overwrite old labels

Required fields:

```text
old_correctness
current_correctness
old_answer
current_answer
```

Also add:

```text
label_transition
```

with one of:

```text
C_TO_C
C_TO_W
W_TO_C
W_TO_W
UNKNOWN
```

---

## 20. Replay failures

For routes that cannot be replayed:

```text
replay_status = ERROR
```

Record:

```text
error type
message
sample UID
route key
attempt count
```

Do not drop failed routes silently.

Create:

```text
replay/replay_errors.jsonl
```

---

## 21. Replay progress/resume

The replay must be resumable.

Use deterministic route ordering.

Checkpoint after a fixed number of routes.

On resume:

```text
skip routes whose complete current-server record already exists
```

Validate existing record hash before skipping.

Do not append duplicate replay records.

---

## 22. Replay transition matrix

After complete replay, report route-level:

```text
old C -> current C
old C -> current W
old W -> current C
old W -> current W
```

Create:

```text
replay/replay_transition_matrix.csv
```

Report counts and percentages overall and by:

```text
dataset
phase/provenance
route length
# interventions
```

---

## 23. Sample-level replay instability

For each sample report:

```text
fraction of its routes whose correctness changed
# C->W
# W->C
```

Create:

```text
replay/sample_replay_stability.csv
```

Question:

> Is replay drift concentrated in a few samples, datasets, or route families?

---

# Stage E — Current dense anchor

## 24. Re-evaluate dense route on current server

Dense correctness must be current-server based.

For every sample create:

```text
dense/current_dense_manifest.jsonl
```

with:

```text
sample_uid
dense_route_key
current_dense_answer
current_dense_correctness
old_dense_correctness
dense_label_transition
```

Do not reuse old dense correctness as the geometry anchor.

---

## 25. Current sample class

Define:

```text
Dense-C:
    current dense route is correct

Dense-W:
    current dense route is wrong
```

Report:

```text
# Dense-C samples
# Dense-W samples
```

overall and per dataset.

---

# Stage F — Current-server filtered route corpus

## 26. Define current route sets

For every sample `i`:

```text
C_i =
    routes with current_correctness = correct

W_i =
    routes with current_correctness = wrong
```

Do not remove original labels.

Add derived current-server fields.

---

## 27. Current route categories

Classify all replay-valid routes relative to the current dense anchor:

```text
Dense-C -> Route-C
Dense-C -> Route-W
Dense-W -> Route-C
Dense-W -> Route-W
```

These correspond to:

```text
preservation
regression
correction
unresolved failure
```

Create:

```text
filtering/current_route_categories.csv
filtering/current_route_manifest.jsonl
```

---

## 28. Geometry-eligible sample census

Using current labels, count:

```text
C-W eligible:
    |C_i| >= 1 and |W_i| >= 1

C-C eligible:
    |C_i| >= 2

W-W eligible:
    |W_i| >= 2

C-C + C-W:
    |C_i| >= 2 and |W_i| >= 1

W-W + C-W:
    |W_i| >= 2 and |C_i| >= 1

full pairwise geometry:
    |C_i| >= 2 and |W_i| >= 2
```

Create:

```text
geometry_readiness/geometry_eligibility.csv
```

Report:

```text
sample counts
route counts
dataset breakdown
Dense-C / Dense-W breakdown
```

---

## 29. W->C correction cohort

Define:

```text
current Dense = W
and
at least one current Route-C
```

For each sample report:

```text
# current correct routes
# current wrong routes
# unique correct action sequences
# unique wrong action sequences
```

Create:

```text
geometry_readiness/w_to_c_samples.jsonl
```

This is a primary future geometry cohort.

---

## 30. C->C preservation cohort

Define:

```text
current Dense = C
and
at least one current Route-C
```

For each sample report:

```text
# preserved-correct routes
# wrong/regression routes
```

Create:

```text
geometry_readiness/c_to_c_samples.jsonl
```

---

## 31. Geometry pair-count estimation

Before hidden-state extraction, estimate possible pair counts per sample:

```text
C-C pairs = C(|C_i|, 2)
W-W pairs = C(|W_i|, 2)
C-W pairs = |C_i| * |W_i|
```

Report:

```text
total theoretical C-C pairs
total theoretical W-W pairs
total theoretical C-W pairs
median pairs/sample
p90 pairs/sample
max pairs/sample
```

This determines whether full pairwise geometry is computationally feasible or needs balanced sampling.

---

## 32. Route-action diversity audit

For each sample compute:

```text
pairwise route Hamming-distance distribution
# unique action sequences
action-count distribution
```

Separately for:

```text
C-C
W-W
C-W
```

where possible.

This is preparation for the future geometry control:

> hidden-state similarity must later be conditioned on route/action similarity.

No hidden-state analysis yet.

---

# Stage G — Hidden-state storage planning

## 33. Mandatory future hidden states

The future geometry study requires at least:

```text
last question token hidden state
for every decoder layer
```

for every replay-valid route selected into the geometry corpus.

If hidden size is `d`, estimated BF16 storage:

```text
#routes × 28 × d × 2 bytes
```

Compute exact estimate from the actual model config.

Create:

```text
storage/last_question_token_storage_estimate.md
```

---

## 34. Visual-token storage estimate

For visual hidden states estimate:

```text
sum_over_routes(
    #layers × #visual_tokens(route) × d × bytes_per_value
)
```

Report for:

```text
BF16 raw
FP16 raw
FP32 raw
```

Do not commit to raw visual storage before seeing this estimate.

Create:

```text
storage/visual_hidden_storage_estimate.md
```

---

## 35. Optional capture during replay

Because full route replay is expensive, after the census and storage estimate are known, decide whether to capture last-question-token hidden states during the same replay pass.

Recommended rule:

```text
if projected last-question-token storage is comfortably manageable:
    capture all 28-layer last-question-token states during replay

else:
    replay labels first
    extract hidden states only for the geometry-eligible filtered corpus later
```

For visual tokens:

```text
do not capture raw all-layer visual states by default
unless the storage estimate is explicitly approved.
```

Possible later alternatives:

```text
streaming pairwise statistics
mean-pooled visual states
fixed low-dimensional random projection
selected layers only
selected route subset
```

Do not choose among these until the current corpus size is known.

---

## 36. If hidden states are captured now

Store them outside the canonical package under:

```text
analysis/3b_greedy_route_corpus_replay_audit/hidden_cache/
```

Each record/file must map exactly to:

```text
sample_uid
route_key
layer
current replay record
```

No orphan hidden tensors.

Validate random readback and route-hash alignment.

---

# Stage H — Current-server pair-dataset implications

## 37. Audit existing V3.1 pair dataset against replay

Do not modify:

```text
03_PAIRED_DATASETS/vqa_correctness_first_v31/
```

Instead determine how many existing pairs remain semantically valid under current-server labels.

Report:

```text
both pair labels stable
preferred/rejected labels reversed
one/both labels changed
route missing/replay error
```

Create:

```text
filtering/v31_pair_replay_audit.csv
```

---

## 38. Do not rebuild supervision yet

This phase may produce a current-server filtered route manifest, but it should **not** automatically train or rebuild the preference router.

If later needed, construct a new versioned pair dataset such as:

```text
v32_current_server
```

Do not overwrite V3.1.

---

# 39. Required summary table

Produce a top-level census:

```text
Canonical 3B Greedy Route Corpus Audit

Unique samples: ?
Unique image groups: ?
Unique routes: ?

Routes/sample:
  mean:
  median:
  p90:
  max:

Saved route labels:
  Correct:
  Wrong:

Current-server replay:
  C->C:
  C->W:
  W->C:
  W->W:

Current dense:
  Dense-C samples:
  Dense-W samples:

Current route categories:
  Dense-C -> Route-C:
  Dense-C -> Route-W:
  Dense-W -> Route-C:
  Dense-W -> Route-W:

Geometry eligibility:
  >=1 C & >=1 W:
  >=2 C:
  >=2 W:
  >=2 C & >=1 W:
  >=2 W & >=1 C:
  >=2 C & >=2 W:
```

Report sample-level and route-level counts where applicable.

---

## 40. Dataset-specific summary

Repeat key counts for:

```text
GQA
ChartQA
DocVQA
TextVQA
```

At minimum:

```text
samples
routes
routes/sample
current Correct routes
current Wrong routes
Dense-C
Dense-W
mixed C/W samples
>=2C & >=2W samples
```

---

# 41. Replay-stability categories

## RS-A — Stable replay

Pattern:

```text
anchor gate passes
full replay label changes are rare
dense labels are mostly stable
geometry eligibility remains broadly consistent
```

Interpretation:

> Existing corpus is largely reusable after current-server relabeling.

## RS-B — Moderate replay drift

Pattern:

```text
non-trivial C<->W changes
but enough mixed/correct-route coverage remains
```

Interpretation:

> Use only current-server labels for all future geometry/training.

## RS-C — Severe replay drift

Pattern:

```text
large label-transition rates
dense labels change substantially
route eligibility changes strongly
```

Interpretation:

> Treat saved labels as historical only; future experiments must use a fully reconstructed current-server corpus.

Do not invent arbitrary thresholds after seeing the data.

Report actual transition rates and practical impact.

---

# 42. Geometry-readiness decision

The corpus is READY for Routing Trajectory Representation Geometry only if:

```text
1. replay gate passes;
2. current replay coverage is high enough;
3. dense anchors are unambiguous;
4. enough C-C / C-W / W-W samples exist;
5. route/action identities are exact;
6. hidden-state storage/extraction strategy is feasible.
```

If not ready, state the exact blocker.

---

# 43. No geometry conclusions in this phase

Do not yet claim:

```text
correct routes converge
wrong routes diverge
correct/wrong manifolds separate
visual representations converge
text representations converge
```

Those are next-phase questions.

This phase only prepares a trustworthy route corpus and measures its eligibility.

---

# 44. No finetuning in this phase

Do not start:

```text
LoRA
adapter training
counterfactual distillation
preference training
representation alignment
```

until the replay-filtered corpus and geometry analysis are complete.

---

# 45. Recommended execution order

Run exactly:

```text
1. read README + metadata package contract
2. inventory canonical files
3. audit final_phase1_phase2 schema
4. census samples/routes/old labels
5. identify exact dense route per sample
6. reconstruct exact replay contract
7. run generation-anchor replay gate
8. STOP if gate fails
9. run full unique-route current-server replay
10. replay current dense anchors
11. build old->current transition matrices
12. build current route categories
13. compute geometry eligibility
14. estimate pair counts and route-action diversity
15. estimate last-token / visual hidden-state storage
16. decide whether to capture last-token hidden states now
17. audit V3.1 pair validity
18. write final corpus-audit summary
19. make RS-A/B/C and geometry-readiness decision
```

---

# 46. Required artifacts

Create:

```text
analysis/3b_greedy_route_corpus_replay_audit/
├── protocol.md
├── frozen_contract.json
├── source_inventory/
│   ├── file_inventory.csv
│   └── route_schema.md
├── census/
│   ├── global_census.json
│   ├── routes_per_sample.csv
│   ├── saved_label_summary.csv
│   ├── saved_sample_composition.csv
│   ├── original_route_categories.csv
│   └── dataset_breakdown.csv
├── replay_gate/
│   ├── replay_contract.md
│   ├── current_server_env.json
│   ├── gate_result.json
│   └── gate_report.md
├── replay/
│   ├── current_replay_routes.jsonl
│   ├── replay_errors.jsonl
│   ├── replay_transition_matrix.csv
│   ├── dataset_transition_matrix.csv
│   └── sample_replay_stability.csv
├── dense/
│   ├── dense_route_manifest_original.jsonl
│   ├── current_dense_manifest.jsonl
│   └── dense_transition_summary.csv
├── filtering/
│   ├── current_route_manifest.jsonl
│   ├── current_route_categories.csv
│   ├── v31_pair_replay_audit.csv
│   └── filtering_report.md
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
│   ├── replay_filtering_summary.md
│   ├── geometry_readiness_summary.md
│   └── next_phase_recommendation.md
└── artifact_manifest.json
```

---

# 47. `corpus_audit_summary.md` must answer

1. How many unique samples exist?
2. How many unique image/content groups?
3. How many unique routes?
4. What is the routes/sample distribution?
5. How many saved Correct/Wrong routes?
6. How many samples are saved ALL-C / ALL-W / MIXED?
7. How many samples have >=2C / >=2W / >=2C&>=2W under saved labels?
8. Is there exactly one dense route per sample?
9. What are the original Dense-C/Route-C etc. counts?
10. How do these statistics vary by dataset and search phase?

---

# 48. `replay_filtering_summary.md` must answer

1. What exact model revision/runtime contract was reconstructed?
2. Did the generation-anchor replay gate pass?
3. How many routes replayed successfully?
4. How many replay errors?
5. What are C->C / C->W / W->C / W->W counts and rates?
6. Which datasets have the largest replay drift?
7. Is drift concentrated in particular route lengths/provenance?
8. How many dense labels changed?
9. How many current Dense-C / Dense-W samples exist?
10. How many V3.1 pair labels remain valid?
11. Is replay stability RS-A / RS-B / RS-C?

---

# 49. `geometry_readiness_summary.md` must answer

Using current-server labels only:

1. How many samples have >=1C & >=1W?
2. How many have >=2C?
3. How many have >=2W?
4. How many have >=2C & >=1W?
5. How many have >=2W & >=1C?
6. How many have >=2C & >=2W?
7. How many current Dense-W samples have at least one corrective Route-C?
8. How many current Dense-C samples have preservation Route-C?
9. How many theoretical C-C / C-W / W-W pairs exist?
10. What is the route-Hamming-distance distribution?
11. Is full pairwise geometry computationally feasible?
12. What is estimated last-question-token storage?
13. What is estimated raw visual-token storage?
14. Should last-question-token states be captured during replay?
15. Should raw visual states be deferred?
16. Is the corpus READY for Routing Trajectory Representation Geometry?

---

# 50. `next_phase_recommendation.md`

Recommend exactly one next phase.

If READY:

```text
Routing Trajectory Representation Geometry
```

The next phase must use the measured current-server:

```text
C-C eligible count
C-W eligible count
W-W eligible count
route action-distance distribution
storage budget
```

If NOT READY:

recommend only the minimal blocker-resolution phase.

Do not recommend finetuning before the geometry analysis.

---

# 51. Stop rule

STOP after:

```text
1. canonical corpus census
2. replay-contract reconstruction
3. generation-anchor gate
4. full current-server route replay
5. dense-anchor replay
6. current-server filtering
7. geometry eligibility census
8. route pair-count/action-diversity estimates
9. hidden-state storage decision
10. V3.1 pair audit
11. RS-A/B/C decision
12. geometry-readiness decision
```

Do not start the geometry analysis itself.

Do not start finetuning.

---

# 52. Core principle

The next scientific question is representation geometry, but that analysis is only meaningful if the route labels and route identities are trustworthy on the current runtime.

Therefore first establish:

```text
what routes actually exist,
which routes are currently correct/wrong,
which dense samples are currently correct/wrong,
and how many within-sample C-C / C-W / W-W comparisons are truly available.
```

Only then analyze whether:

```text
different correct trajectories converge,
correct and wrong trajectories separate,
or multiple correct representation modes persist across layers.
```
