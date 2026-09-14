# Stage-1 Dense-Failure 8K Label Audit Decision

## Decision

`EXECUTION_CONTRACT_UNRESOLVED`

The audit stops before dense replay, hidden-state extraction, split freezing,
or Stage-1 training. No GPU scientific job was started.

## Q1 — Population

The selected physical population is exactly 8,000 images and the historical
filename buckets are exactly 4,000 correct / 4,000 wrong. The authoritative
later dense outcomes are reported and reconstruct to 4,045 correct / 3,955
wrong, not 4,000 / 4,000. The complete 8K label records are absent, so that
reconstruction cannot be promoted to a fresh record-level audit.

## Q2 — Source contract

The producing code and scientific settings are recoverable: pinned
Qwen2.5-VL-7B revision `cc594898...`, native processor defaults, greedy
`use_cache=True` generation for at most 16 tokens, BF16 SDPA, and the frozen
benchmark evaluators/thresholds. The extraction ran on RTX A6000 GPUs. The
contract-bound JSON, complete source manifest, exact original dense outputs,
and driver/kernel record are not present on this server.

## Q3 — Runtime compatibility

Unmeasured. A valid 96–128 record replay cannot be frozen from the incomplete
transferred source, and stored token/answer parity cannot be computed because
the available derivative omits the original dense predictions.

## Q4 — Dataset shortcut risk

The reconstructed current wrong rates are 50.00% GQA, 49.45% ChartQA, and
48.30% TextVQA. This does not indicate a strong dataset-label shortcut, but it
remains descriptive until the 8K source is restored.

## Q5 — Leakage risk

Use image-group-disjoint splitting with equivalence classes formed from both
the authoritative image-group ID and exact image-content SHA-256. The physical
audit found 513 repeated-content groups covering 1,036 records.

## Q6 — Split feasibility

Likely feasible from 7,477 physical image-content groups, but not safely
freezable without the missing 1,083 source rows and complete authoritative
labels. No manifest was created.

## Q7 — Authority decision

`EXECUTION_CONTRACT_UNRESOLVED`

Neither `STORED_LABELS_AUTHORITATIVE` nor
`REBUILD_LABELS_FROM_CURRENT_DENSE_RUNTIME` is justified by this audit. The
first lacks replay evidence; the second would be a new full 8K labeling action
and the exact prompts/answers for 1,083 records are unavailable.

## Smallest unblock

Restore a metadata-complete copy of the canonical 8K bundle, at minimum:

```text
outputs/label_regeneration/v1/source_manifest_v1.jsonl
outputs/label_regeneration/v1/frozen_execution_contract.json
outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl
outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl
```

with their SHA-256 sidecars, or the corresponding paths under
`datasets/mcts_labels/gqa_textvqa_chartqa_v1/`. The six original pool JSONLs
should also be restored if the canonical source manifest does not embed every
question, prompt, answer, image group, and historical prediction.

Once restored, rerun this plan's single 128-record dense-only parity smoke on
four GPUs and stop again for the authority decision.

