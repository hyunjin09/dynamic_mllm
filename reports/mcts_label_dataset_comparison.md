# MCTS Label Dataset Comparison

**Compared directories**

- `datasets/mcts_labels/gqa_textvqa_chartqa_v1`
- `datasets/mcts_v2`

**Date:** 2026-08-24

## Bottom line

`datasets/mcts_labels/gqa_textvqa_chartqa_v1` is the current canonical,
training-authoritative route-label collection for GQA, TextVQA, and ChartQA.

`datasets/mcts_v2` is the older historical MCTS cache. It is preserved as
provenance and negative-result evidence, but it should not be merged into or
used in place of the regenerated labels. Some old cached outcomes and positive
masks did not reproduce under the repaired target executor.

For current predictor training, use:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1/
```

Do not combine it with:

```text
datasets/mcts_v2/
```

## High-level comparison

| Property | Regenerated `gqa_textvqa_chartqa_v1` | Historical `mcts_v2` |
|---|---:|---:|
| Project role | Current canonical labels | Historical cache |
| Total image-query records | 8,000 | 4,000 |
| Benchmarks | GQA, TextVQA, ChartQA | GQA, TextVQA, ChartQA, DocVQA |
| GQA records | 4,000 | 1,000 |
| TextVQA records | 2,000 | 1,000 |
| ChartQA records | 2,000 | 1,000 |
| DocVQA records | 0 | 1,000 |
| Search simulations | 200, 400, or bounded 600 | Fixed 200 |
| Evaluated routes | 2,642,998 | 808,000 |
| Valid routes | 528,047 | 184,785 |
| Records with at least one valid route | 6,917 | 3,408 |
| Records with no valid route | 1,083 | 592 |
| Frozen predictor split | Yes | No current authoritative split |
| Derived max-50 valid-set view | Yes | No current authoritative view |
| Full provenance/checksum chain | Yes | Limited historical provenance |
| Approximate storage | 23 GB | 9 GB |

## Dataset composition

### Regenerated collection

The regenerated collection contains:

- 4,000 GQA records:
  - 2,000 historically ALL-ON correct;
  - 2,000 historically ALL-ON wrong.
- 2,000 TextVQA records:
  - 1,000 historically ALL-ON correct;
  - 1,000 historically ALL-ON wrong.
- 2,000 ChartQA records:
  - 1,000 historically ALL-ON correct;
  - 1,000 historically ALL-ON wrong.

Historical correctness was used only to balance the source pool. The
authoritative ALL-ON result was recomputed under the new frozen executor.

### Historical `mcts_v2` collection

The old collection contains eight 500-record cells:

- GQA easy/correct and hard/wrong;
- TextVQA easy/correct and hard/wrong;
- ChartQA easy/correct and hard/wrong;
- DocVQA easy/correct and hard/wrong.

This gives 1,000 records per benchmark and 4,000 total.

### Record overlap

Exact benchmark/sample-ID comparison shows:

- all 1,000 old GQA records occur in the regenerated source pool;
- all 1,000 old TextVQA records occur in the regenerated source pool;
- all 1,000 old ChartQA records occur in the regenerated source pool;
- the 1,000 old DocVQA records do not occur in the regenerated pool.

The regenerated collection therefore contains all 3,000 non-DocVQA old
samples plus 5,000 additional samples:

- 3,000 additional GQA;
- 1,000 additional TextVQA;
- 1,000 additional ChartQA.

This overlap is one reason the two caches must not be concatenated for
training: doing so would duplicate 3,000 inputs under potentially conflicting
executor-conditioned labels.

## Route semantics

Both collections represent a complete 28-bit route:

```text
m = (m_0, ..., m_27), m_l in {0,1}
```

The intended semantics are:

- `ON`: run the native decoder layer on text/control and visual rows;
- `OFF`: run the decoder layer on compacted text/control rows and carry visual
  rows through unchanged.

Visual rows removed at an OFF layer can re-enter at a later ON layer. Text
tokens still execute every decoder layer.

Although the abstract action definition is shared, a route's validity depends
on the complete runtime: preprocessing, visual-token geometry, executor,
numerical kernels, generation settings, and benchmark scorer.

## Why regeneration was necessary

The old cache was tested against the repaired binary executor before predictor
training. The audit found that the cached outcomes were not fully portable.

Confirmed issues included:

- record-specific image-processing differences, especially for DocVQA;
- different visual-token counts and later-text MRoPE layouts;
- exact generated-token mismatches;
- previously positive routes that became incorrect under replay;
- incomplete hardware/kernel and vision-utility provenance for the old run.

Some token differences preserved benchmark correctness, but other old positive
masks changed from valid to invalid. Therefore the project could not safely
repair the old cache by removing only the known failures.

The selected resolution was to regenerate every authoritative route outcome
under one pinned and reproducible execution contract.

## Execution-contract comparison

Both collections used the same nominal base model snapshot:

```text
Qwen/Qwen2.5-VL-7B-Instruct
cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Both also recorded Transformers 5.3.0, PyTorch 2.6, BF16, and SDPA. These
shared fields were not enough to guarantee identical route outcomes.

### Old execution contract

The old records identify the implementation as:

```text
portable_mcts_v2_layer_location_expansion
```

The old label runtime used the packaged `dvr_qwen` executor. Records could
carry record-specific visual-token/image-resizing constraints. Important
runtime details such as the exact GPU/kernel path and vision-utility version
were not completely recoverable.

### Regenerated execution contract

The new collection freezes:

```text
binary_policy.executor.BinaryQwen25VL
```

The contract specifies:

- native/default Qwen image processing;
- no project-specific maximum-image-token override;
- Qwen2.5-VL snapshot `cc594...cfb5`;
- BF16 and SDPA;
- deterministic greedy generation;
- current benchmark-specific scorers;
- repaired ALL-ON/native behavior;
- source-code hashes;
- per-record hashes and a final checksum ledger.

Before full generation, the new executor passed:

- exact generated-token ALL-ON/native parity on 15/15 smoke records;
- deterministic repeat execution for representative mixed routes.

The frozen contract is stored at:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1/frozen_execution_contract.json
```

## MCTS algorithm and budget differences

The main graph-MCTS structure is similar in both collections:

- an action chooses `(layer_index, ON/OFF)`;
- any undecided layer can be chosen next;
- no fixed early-to-late expansion order;
- transposition-table search;
- exploration constant `1.8`;
- length penalty `3.0`;
- random-action probability `0.1`;
- rollout OFF probability `0.5`;
- search does not stop after the first success.

### Old budget

Every old sample receives:

```text
200 simulations + ALL-ON anchor + ALL-OFF anchor
```

This normally produces 202 evaluated masks per sample and 808,000 total route
evaluations.

### Regenerated budget

The new run recomputes ALL-ON correctness first and assigns:

```text
current ALL-ON correct: 200 simulations
current ALL-ON wrong:   400 simulations
no correction at 400:  bounded extension to 600
```

Final sample counts by budget were:

| Simulations | Records |
|---:|---:|
| 200 | 4,045 |
| 400 | 2,775 |
| 600 | 1,180 |

The larger population and deeper search for current-wrong samples explain the
larger regenerated cache.

## Route yield

### Old cache

- 808,000 evaluated masks;
- 184,785 deduplicated valid masks;
- 3,408 records with at least one valid route;
- 592 records without a valid route;
- approximately 54.2 valid masks per positive record.

### Regenerated cache

- 2,642,998 evaluated masks;
- 528,047 valid masks;
- 6,917 records with at least one valid route;
- 1,083 records without a valid route;
- approximately 76.3 valid masks per positive record;
- mean/median valid masks over all 8,000 records: 66.01/39.

The raw regenerated cache retains both positive and negative evaluated masks.
It is not truncated to 50 routes.

## Training artifacts

The regenerated tree is a complete training-data package. It contains:

- an immutable source manifest;
- 8,000 raw per-sample route records;
- positive and negative route outcomes;
- a per-record checksum index;
- per-sample route counts and diversity summaries;
- a deterministic image-group-disjoint split;
- a max-50 diverse valid-route supervision view;
- a single-minimum-budget valid-route view;
- grouped valid-set supervision;
- positive/negative route-ranking data;
- derived POLAR-style segment representations;
- audit reports and SHA-256 sidecars.

The frozen overall source split is:

```text
7,000 train
1,000 validation
```

with zero cross-split image overlap.

Only records with at least one discovered valid route enter the positive-route
objectives:

| Split | GQA | TextVQA | ChartQA | Total |
|---|---:|---:|---:|---:|
| Train positive | 2,957 | 1,525 | 1,561 | 6,043 |
| Validation positive | 429 | 221 | 224 | 874 |

The derived max-50 view contains 237,802 selected route occurrences. The raw
528,047 valid masks remain preserved separately.

The main current predictor manifest is:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1/
  post_generation/binary_predictor_manifest_v1.jsonl
```

The old `mcts_v2` directory contains raw per-sample records and final audit
summaries but not the current split, current derived supervision, or a current
executor-validity guarantee.

## Compatibility path

The historical configuration path:

```text
outputs/label_regeneration/v1
```

is a compatibility symlink to:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1
```

These are two paths to the same regenerated artifact, not two label sets.

## Training decision

For GQA/TextVQA/ChartQA training:

```text
USE: datasets/mcts_labels/gqa_textvqa_chartqa_v1
```

Do not use or merge `datasets/mcts_v2` because:

1. its shared 3,000 records duplicate records already regenerated;
2. old and new route outcomes are tied to different executor contracts;
3. some old positive masks did not reproduce under the target executor;
4. mixing caches would give duplicated records extra and inconsistent weight;
5. the old DocVQA population was not admitted into the current regenerated
   training protocol;
6. only the new package has the frozen split, max-50 selection, derived views,
   and checksum-complete training contract.

`datasets/mcts_v2` should remain untouched as historical provenance and as
evidence motivating the label regeneration.

## Evidence paths

- Regeneration summary: `reports/label_generation_report.md`
- Old-label mismatch analysis:
  `reports/binary_mcts_label_mismatch_analysis.md`
- Executor repair: `reports/binary_polar_bp1_executor_repair.md`
- Input-contract repair:
  `reports/binary_polar_bp1_input_contract_repair.md`
- Relocation/integrity report: `reports/mcts_label_relocation_20260819.md`
- New final audit:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/post_generation/p9_final_audit_v1.json`
- New cache audit:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/post_generation/cache_audit_v1.json`
- New predictor split audit:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/post_generation/predictor_split_audit_v1.json`
- New derived-supervision audit:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/post_generation/derived_supervision_audit_v1.json`
- Old complete audit: `datasets/mcts_v2/final/audit_summary_full_v2.json`
