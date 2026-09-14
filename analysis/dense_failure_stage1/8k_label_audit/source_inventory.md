# Stage-1 Dense-Failure 8K Source Inventory

Audit date: 2026-08-30 KST  
Plan SHA-256: `1e645d2f59df580606be067812d78d4e2c3e19e8973ce78387e9cc727fcdcc6d`

## Decision-relevant result

The complete authoritative 8K label population is not present on this server.
All 8,000 selected images are present, but the six source JSONLs, the canonical
8K source manifest, the frozen dense execution contract, and the regenerated
8K label/index artifacts were deliberately not included in the transferred
asset set. The available four-action source inventory is a positive-route-only
derivative containing 6,917 VQA rows; it is not a replacement for the missing
8K source.

## Present assets

| Asset | Absolute path | Records/files | SHA-256 or status |
|---|---|---:|---|
| Selected images | `/home/aix7101/hyemin/0830/dynamic_mllm/datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images` | 8,000 | present; 1,043,038,742 bytes |
| Positive-route source inventory | `/home/aix7101/hyemin/0830/dynamic_mllm/datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl` | 6,917 VQA rows | `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7` |
| Source-inventory summary | `/home/aix7101/hyemin/0830/dynamic_mllm/datasets/mcts_labels_4action/source_inventory_v1/source_inventory_summary_v1.json` | one summary | `216561f54f2512ce73922a8ab08764beeb14397d86f09227ef4e29c538f508dd` |
| Sequential four-action records | `/home/aix7101/hyemin/0830/dynamic_mllm/datasets/mcts_labels_4action/sequential_branching_v1/full/records` | 6,917 JSON records plus sidecars | present; positive-route population only |
| Qwen snapshot | `/home/aix7101/hyemin/0830/dynamic_mllm/eval/reference/shared_prefix_eval_20260812/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5` | five weight shards plus processor/tokenizer/config | present |

All 6,917 positive-inventory image basenames resolve to the transferred image
tree. No positive-inventory image is missing.

## Missing authoritative assets

The repository loader `label_regeneration/data.py` requires these six files
under `complete_correct_wrong_pools_20260713/`; all six are absent:

| Expected file | Expected rows | Status |
|---|---:|---|
| `gqa_complete_correct_2000.jsonl` | 2,000 | missing |
| `gqa_complete_wrong_2000.jsonl` | 2,000 | missing |
| `textvqa_complete_correct_1000.jsonl` | 1,000 | missing |
| `textvqa_complete_wrong_1000.jsonl` | 1,000 | missing |
| `chartqa_complete_correct_1000.jsonl` | 1,000 | missing |
| `chartqa_complete_wrong_1000.jsonl` | 1,000 | missing |

The canonical label bundle referenced throughout the repository is also
absent:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1/
outputs/label_regeneration/v1/
```

Consequently, these contract-bound files are unavailable here:

```text
source_manifest_v1.jsonl
frozen_execution_contract.json
post_generation/binary_predictor_manifest_v1.jsonl
post_generation/cache_record_index_v1.jsonl
post_generation/predictor_split_audit_v1.json
```

## Available field schema

The 6,917-row positive-route inventory contains:

```text
dataset / benchmark
uid / sample_id
image_id / image_group_id / image_path
question / prompt / answer / all_answer_norms
metric_name / correctness_threshold / max_new_tokens
source_current_all_on_status
source_positive_routes
```

It does **not** retain the original dense generated answer:
`source_current_all_on_prediction` is null for all 6,917 VQA rows. The 1,083
zero-positive samples have no transferred source row at all, so their question,
prompt, answer, image identifier, and stored dense generated answer cannot be
audited from physical files on this server.

## Producing code and recoverable contract

| Component | Path | Current SHA-256 |
|---|---|---|
| Source normalization | `label_regeneration/data.py` | `ba57808c943361540c835a645f95d2f8633b0f34d7fec41c7f0be97dfa5758f7` |
| Dense/native runtime | `label_regeneration/runtime.py` | `1739b9d0f696ee3da3f601849f54c4d3f2077aa3cb72dea544f11b2ff796f201` |
| Extraction runner | `experiments/run_label_regeneration.py` | `5aacc8935e1510a4d1f2055e54512cf811c37ec6e7b1ef984a1d2114221f09d3` |
| Evaluator | `reference/dvr_qwen/eval_metrics.py` | `c2ce4974fb03110841d55af61d26307a1b8cde31cd214906e2b5bafc5ed5373e` |
| Package lock | `requirements-lock.txt` | `fbdd81149fe70ae97deebcf6baaa565a7ff7d524b65121b7810895cd8915eed0` |

Verified from code and preserved state:

- model revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- `AutoProcessor`, `use_fast=False`, native image processing, no custom visual-token cap;
- image-first chat content followed by the frozen prompt and generation prompt;
- native `generate`, greedy decoding (`do_sample=False`), `use_cache=True`,
  `max_new_tokens=16`;
- BF16 with SDPA, deterministic algorithms, TF32 disabled;
- GQA exact-match-ignore-case/punctuation at threshold 1.0;
- ChartQA relaxed accuracy at threshold 1.0;
- TextVQA EvalAI consensus at threshold 0.5;
- original dense extraction ran first with four RTX A6000 workers and completed
  through an eight-RTX-A6000 resumable run.

Unresolved because the frozen artifacts are absent:

- exact frozen contract JSON and its complete source-hash inventory;
- exact historical CUDA driver/kernel environment;
- the original dense generated tokens/answers for all 8,000 records;
- complete 8K UID/question/answer/image-group metadata.

