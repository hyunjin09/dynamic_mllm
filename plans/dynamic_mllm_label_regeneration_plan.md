# Dynamic MLLM Label Regeneration Plan

Status: active, amended 2026-08-10.

The amendment freezes a minimal 15-record smoke test followed immediately by
full 8K extraction when it passes. Image-group-disjoint predictor splits are
constructed after extraction and before predictor training. It supersedes the
earlier large-pilot and split-before-search ordering in this document.

## 1. Objective

Regenerate route-supervision labels under a single frozen, reproducible execution contract for the Dynamic MLLM project.

The new labels will be used later to compare:

1. single-route supervision;
2. binary 28-bit routing with exact valid-set NLL;
3. multi-route / top-K prediction;
4. POLAR-style structured segment prediction as a derived baseline.

This plan is **only for label generation and label-quality validation**. Do not start predictor training until the label-generation gates below pass.

---

## 2. Key Decision

### Keep label generation layer-wise and unrestricted

Use the existing primitive route space:

```text
m = (m_0, ..., m_27),  m_l ∈ {0, 1}
```

where:

- `1` = visual-on: run the original Qwen decoder layer on the full text + visual token sequence;
- `0` = visual-off: run the layer on text/control tokens only and carry visual hidden states forward unchanged.

The final label is a **complete 28-bit route**, evaluated by actual greedy generation and the benchmark-specific answer metric.

Do **not** constrain MCTS to POLAR-style segments during label generation.

Reason:

- the executor itself is layer-wise;
- unrestricted 28-bit labels preserve the full search space;
- POLAR-style segment labels can always be derived later from the same full masks;
- segment-constrained search would inject the unverified assumption that good routes are contiguous blocks;
- using one common raw label source allows a clean comparison between binary and structured predictors later.

POLAR-style structure should therefore be treated as a **predictor representation / decoding baseline**, not as the primary label-generation search space.

---

## 3. Data Scope

Source pool:

```text
/data/projects/hyunjin/MLLM/dynamic_mllm/datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713
```

Use all currently selected samples:

| Dataset | Historical all-ON correct | Historical all-ON wrong | Total |
|---|---:|---:|---:|
| GQA | 2,000 | 2,000 | 4,000 |
| TextVQA | 1,000 | 1,000 | 2,000 |
| ChartQA | 1,000 | 1,000 | 2,000 |
| **Total** | **4,000** | **4,000** | **8,000** |

Do not add DocVQA in this phase.

### Important

The `correct` / `wrong` directory membership is historical metadata only.

Because the execution contract is being regenerated, **recompute the authoritative all-ON prediction and correctness for every sample under the new frozen executor**. Never copy the historical correct/wrong label into the new cache.

Record both:

```text
historical_all_on_status
current_all_on_status
```

so any contract-induced flips can be audited.

---

## 4. Split After Label Extraction

Do not delay route extraction to construct predictor splits. MCTS operates
independently per sample and does not learn across samples, so generate the raw
8K label cache first.

After raw extraction is complete, create and freeze image-group-disjoint
predictor manifests before any predictor training or model selection.

Target split:

| Dataset | Train | Validation | Total |
|---|---:|---:|---:|
| GQA | 3,500 | 500 | 4,000 |
| TextVQA | 1,750 | 250 | 2,000 |
| ChartQA | 1,750 | 250 | 2,000 |
| **Total** | **7,000** | **1,000** | **8,000** |

Post-extraction split requirements:

- no image may appear in more than one split;
- preserve the historical 50/50 correct-vs-wrong balance exactly in validation:
  GQA 250/250, TextVQA 125/125, and ChartQA 125/125;
- report the **actual** correct/wrong balance under the new contract;
- construct the split using identifiers and grouping metadata, not route
  success, valid-route count, or downstream predictor outcomes;
- do not move examples between train/validation after the predictor manifest is
  frozen merely to improve results.

The balanced dataset is for router development and controlled generalization analysis. It is **not** the final natural-distribution benchmark evaluation set.

---

## 5. Freeze the Minimal Execution Contract

Before generating labels, record the execution contract in one compact
artifact. This is a reproducibility record, not an additional validation
project. Do not delay the smoke test for extensive provenance reconstruction,
layer tracing, or broad hardware diagnostics.

At minimum record:

```text
model_name
model_snapshot / revision
model snapshot path or hash
code commit hash when available, otherwise deterministic hashes of the active
executor/MCTS source files
local dvr_qwen implementation path
processor snapshot / revision
tokenizer revision
chat / prompt template
image preprocessing configuration
dtype
attention implementation/backend
generation configuration
benchmark metric implementation
correctness threshold
GPU / CUDA / PyTorch environment used for the smoke and full run
```

### Visual-token policy

Use **Qwen's native/default processor behavior**.

Do not apply the project-specific `max_image_tokens` override.

In particular:

```text
custom max_image_tokens = None
```

Do not silently introduce a visual-token cap to solve memory or speed problems.

If native processing causes a resource problem:

1. stop the pilot;
2. report the visual-token distribution and memory failure;
3. propose one fixed alternative contract;
4. do not continue label generation until that contract is explicitly approved.

For every sample, save the actual:

```text
image dimensions
num_text_tokens
num_visual_tokens
```

---

## 6. Mandatory Executor Gates

### Gate A — all-ON native parity

Before MCTS, verify:

```text
G_binary(x, all_on) == G_HF(x)
```

under deterministic greedy generation.

Require exact generated-token parity on the replay/pilot fixtures.

Also verify first-token/final logits where the existing parity infrastructure supports it.

Do not loosen the gate to make the run pass.

### Gate B — route determinism sanity check

On a small pilot set, execute representative:

- all-ON masks;
- all-OFF masks;
- mixed masks;

more than once under the same frozen environment.

Confirm that generated token IDs and benchmark scores are stable.

If they are not stable, stop and diagnose before full label generation.

---

## 7. Minimal Smoke Before the 8K Run

Select deterministically approximately five records each from GQA, TextVQA,
and ChartQA: 15 records total.

The smoke has only two gates:

1. exact generated-token parity for binary ALL-ON versus native Qwen on all
   15/15 records;
2. exact repeated generated tokens and benchmark scores for a few frozen
   representative mixed ON/OFF masks.

If both gates pass, immediately start full 8K label extraction. Do not add a
large pilot, broad provenance audit, hidden-state tracing, route-yield study,
or hardware diagnostic.

If either gate fails, stop before full extraction and diagnose the concrete
failure. Implementation repair directly required to make the frozen smoke
valid is allowed; do not loosen parity or determinism criteria.

---

## 8. MCTS Search Space

Keep the current graph-MCTS semantics.

Root:

```text
all-ON = (1, ..., 1)
```

Also evaluate:

```text
all-OFF = (0, ..., 0)
```

A search action is:

```text
(layer_index, binary_visual_action)
```

where the layer may be any undecided layer.

Do not impose an early-to-late layer order.

Each completed mask must be evaluated through **actual route-conditioned greedy generation**.

The route reward remains benchmark correctness:

```text
reward(x, m) = 1 if benchmark_score >= correctness_threshold else 0
```

Save the raw benchmark score in addition to the binary reward.

---

## 9. Search Budget

Use more search budget for current all-ON-wrong examples because a correcting route must first be discovered.

Default:

```text
current all-ON correct:
    200 MCTS simulations

current all-ON wrong:
    400 MCTS simulations
```

Adaptive extension for all-ON-wrong samples:

```text
up to 600 simulations
```

may be used when:

- no valid route has been found yet; or
- valid-route diversity remains clearly insufficient.

Do not spend extra budget simply to force every sample to have 20 valid routes.

A sample with only a few valid routes is still useful.

A sample with zero valid routes must still be saved as a zero-positive sample.

---

## 10. Valid-Route Target

The goal is **not** to manufacture an exact number of labels per query.

The target is:

```text
~20 diverse valid routes per image-query when the search naturally finds them
```

For predictor training later:

```text
training cap = 50 valid routes per image-query
```

The 50-route cap is the primary POLAR-matched supervision cap. A 32-route view
may be retained only as a later secondary ablation.

### Rules

- `0 valid routes`: save the sample; exclude it from positive set-NLL training, but preserve it for negative/reranker analysis.
- `1–50 valid routes`: keep all valid routes in the derived training view.
- `>50 valid routes`: retain all routes in the raw cache, but construct a diverse 50-route training subset.

Do not discard a sample because it has fewer than 20 valid routes.

---

## 11. Diversity Matters More Than Route Count

Twenty near-duplicate routes are less useful than a smaller set of structurally different routes.

For every successful mask, compute at least:

```text
num_visual_on_layers
num_visual_off_layers
number_of_ON/OFF_transitions
Hamming distance to all-ON
Hamming distance to minimum-budget valid route
```

For the derived 50-route training subset, preserve diversity across:

1. visual-ON count / compute budget;
2. Hamming distance;
3. transition count / segment structure.

Recommended selection when more than 50 valid routes exist:

1. always include the minimum-visual-ON successful route;
2. include all-ON and/or all-OFF if they are valid and useful as anchors;
3. stratify candidates by visual-ON count;
4. greedily add routes that maximize Hamming distance from already selected routes;
5. avoid filling the set with tiny one-bit variants of the same mode.

The raw cache must still retain **all evaluated routes**, not only the 50-route view.

---

## 12. Store Positive and Negative Routes

For every evaluated route, save:

```text
sample_id
dataset
split
image_id / image-group id
question / query id
28-bit mask
generated token IDs
decoded prediction
raw benchmark score
correctness threshold
valid / invalid
num_visual_on_layers
num_visual_off_layers
num_transitions
text token count
visual token count
generation metadata
```

Also retain:

```text
all successful route IDs
minimum-budget successful route
all-ON result
all-OFF result
MCTS simulation trace
expanded layer/action
visit statistics
transposition-table information
```

Do not save only positive routes.

Negative evaluated masks are important for later:

- route-success scoring;
- reranking;
- hard-negative training;
- analysis of route sensitivity.

---

## 13. Old Cache Policy

The old cache must not be treated as ground-truth supervision.

It may be used only as an optional **candidate proposal / warm start** if this reduces search cost.

If an old mask is reused:

```text
old mask -> current frozen executor -> new answer/score -> new label
```

The old valid/invalid status must never be copied.

The new executor is the sole authority.

---

## 14. Label-Quality Diagnostics

After generation, report separately for:

- GQA;
- TextVQA;
- ChartQA;
- current all-ON-correct samples;
- current all-ON-wrong samples.

Required statistics:

### Coverage

```text
number of samples
samples with >=1 valid route
samples with >=5 valid routes
samples with >=10 valid routes
samples with >=20 valid routes
median / mean / percentile count of valid routes
```

### Correction behavior

For current all-ON-wrong samples:

```text
P(MCTS finds >=1 correcting route)
```

Also report how many correcting routes are found per successful sample.

### Preservation / efficiency behavior

For current all-ON-correct samples:

```text
distribution of minimum visual-ON count among valid routes
compute saving of minimum-budget valid route
```

### Route diversity

Report:

```text
pairwise Hamming-distance statistics
ON-count distribution
transition-count distribution
segment-length distribution derived from full masks
```

This diagnostic is important for deciding later whether POLAR-style segment structure is a justified inductive bias.

### Contract drift

Report:

```text
historical all-ON correct -> current wrong flips
historical all-ON wrong -> current correct flips
```

Do not silently merge these with router effects.

---

## 15. Post-Generation Views

The raw label cache should support multiple later training formulations without rerunning MCTS.

Create derived manifests for at least:

### A. Single best route

```text
one minimum-budget valid mask per sample
```

Used only as a baseline reproducing the old one-route formulation.

### B. Full valid set

```text
all valid masks, or the diverse 50-route cap
```

For exact valid-set NLL.

### C. Candidate-ranking data

```text
positive and negative evaluated masks
```

For a learned route-success scorer / reranker.

### D. POLAR-style structured view

Convert the same 28-bit masks into contiguous ON/OFF segment representations after label generation.

Do not rerun a segment-constrained MCTS for the initial comparison.

This keeps:

```text
label source fixed
predictor representation variable
```

which is necessary for a clean ablation.

---

## 16. Why This Design

The previous predictor experiments did not generalize well.

The new label set must allow us to distinguish several possible causes:

1. **single-target ambiguity**  
   A query has many valid routes, but training forces one arbitrary minimum-budget route.

2. **multi-modality**  
   A predictor may know several plausible routing modes but fail at top-1 mode selection.

3. **factorization / representation**  
   Independent 28-bit prediction may generalize worse than a structured segment representation.

4. **route predictability itself**  
   The input may not contain enough information for the predictor to identify a useful route.

The regenerated full-mask multi-route labels must therefore support later comparisons of:

```text
single best mask
vs.
binary + exact valid-set NLL
vs.
binary top-K / multi-route decoding
vs.
POLAR-style structured top-K
vs.
learned route reranking
```

Do not assume in advance that POLAR-style segmentation is superior.

The label-generation phase should preserve enough information to test that hypothesis cleanly.

---

## 17. Hard Stop Conditions

Stop and report instead of continuing if any of the following occurs:

1. all-ON binary execution does not match native Qwen generation;
2. the same mask is not deterministic under the frozen environment;
3. native image processing causes an unhandled memory/resource failure;
4. benchmark scoring differs from the frozen scoring contract;
5. split leakage is discovered;
6. route metadata needed for reproducibility is missing;
7. code changes alter route semantics after label generation has started.

Do not relax gates or silently change preprocessing to keep the run moving.

---

## 18. Deliverables

Produce:

```text
1. frozen_execution_contract.md
2. smoke_manifest.jsonl
3. smoke_report.md
4. raw_route_cache/
5. per_sample_route_summary.jsonl
6. split_manifest.jsonl
7. derived_single_best_manifest.jsonl
8. derived_valid_set_manifest.jsonl
9. derived_route_ranking_manifest.jsonl
10. derived_polar_segment_manifest.jsonl
11. label_generation_report.md
```

The final report should include:

- exact dataset counts;
- current all-ON correct/wrong counts;
- search-budget usage;
- route count distributions;
- correction-route discovery;
- minimum-budget route statistics;
- route-diversity statistics;
- contract-drift statistics;
- failures / incomplete samples;
- exact commands and code commit used.

---

## 19. Execution Order

Use this order:

```text
P0. Inspect existing MCTS v2 implementation/documentation and freeze the
    minimal code/model/processor/executor/evaluator contract
P1. Freeze the deterministic 15-record smoke manifest
P2. Run the 15/15 ALL-ON parity and mixed-route determinism smoke
P3. If and only if P2 passes, run full 8K unrestricted MCTS extraction;
    recompute authoritative all-ON outcomes inside each sample run
P4. Verify 8K raw-cache completeness and rerun only failed/incomplete records
    under the identical frozen contract
P5. Build per-sample summaries and current all-ON/correction-route statistics
P6. Run route-diversity and transition-structure analysis
P7. Build and freeze the exact image-group-disjoint 7K/1K predictor split,
    with validation balanced across the historical correct/wrong source strata
P8. Build single-best, diverse valid-set, route-ranking, and derived POLAR
    supervision views from the unchanged raw cache
P9. Freeze checksums, contract-drift results, commands, failures, and the final
    label-generation report
P10. Only after separate approval, begin predictor experiments
```

Do not begin predictor training before P9 is complete.

---

## 20. Short Instruction to the Research Agent

> Regenerate the Dynamic MLLM routing labels for the fixed 8K GQA/TextVQA/ChartQA pool under one frozen native-Qwen execution contract. First inspect the existing MCTS v2 implementation and run only a deterministic 15-record smoke: exact binary ALL-ON/native token parity on five samples per dataset plus repeated mixed-route token/score equality. If it passes, immediately run unrestricted 28-bit layer-wise MCTS over all 8,000 samples using native Qwen processing with no project `max_image_tokens` override. Recompute current all-ON correctness per sample; use 200 simulations when correct, 400 by default when wrong, and at most 600 adaptively when no correction has been found or more search is clearly useful. Save every evaluated positive and negative route with generated IDs, answer, raw score, current validity, mask, route/token geometry, and available search trace. Aim for about 20 diverse valid routes when naturally found, retain all raw routes, and truncate only later derived training views to a diverse maximum of 50. After extraction, compute quality/diversity summaries, freeze the approved image-disjoint predictor split, and derive single-best, valid-set, ranking, and POLAR-segment views from the same raw masks. Do not start predictor training without separate approval.
