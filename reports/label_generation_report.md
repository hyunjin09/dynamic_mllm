# Dynamic MLLM Binary Routing Label Generation Report

P9 status: **PASS**

This report closes P0–P9 for the regenerated unrestricted 28-bit visual ON/OFF route cache. It does not contain predictor-training or external-evaluation results.

## Why the labels were regenerated

The prior MCTS cache was not portable to the repaired binary executor: cached outputs and some previously positive masks failed exact reproduction. The project therefore regenerated every authoritative route outcome under one pinned Qwen2.5-VL execution contract rather than deleting only known mismatches or copying historical validity labels.

## Frozen execution contract

- Model: `Qwen/Qwen2.5-VL-7B-Instruct` at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Executor: `binary_policy.executor.BinaryQwen25VL`; 28 independent layer actions.
- ON: native full text/control plus visual decoder layer.
- OFF: compacted text/control decoder layer; visual rows bypass unchanged.
- Precision/attention: `bfloat16` / `sdpa`.
- Image processing: native Qwen processor defaults; custom max-image-token cap: `None`.
- Generation: deterministic greedy, `max_new_tokens=16`.
- Environment: Python `3.12.7`, PyTorch `2.6.0`, Transformers `5.3.0`.
- Frozen contract SHA-256: `64f525f5d0a4333e1aeae27f41b9055c8da19a9a0fc566ab3c7db270ea37fc7d`.
- Git revision: unavailable because the workspace was not a Git checkout. The contract and P3 resume amendment record deterministic source-file hashes instead.

## Validity and completeness gates

- P2 ALL-ON/native generated-token parity: `15/15`.
- P2 repeated mixed-route determinism: `True`.
- P4 terminal raw records: `8,000/8,000`.
- Missing/unexpected/duplicate/invalid/error/temp/zero-byte records: `0/0/0/0/0/0/0`.
- P4 raw-record checksum ledger: `outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl` (`f61eb0ac6c40e0498cdfaa53c328b3de34cbb67733a8fbd44c5bd590db051ebe`).

## Dataset and current ALL-ON outcomes

Historical source strata are metadata only; current results below were recomputed by the frozen executor.

| Dataset | Samples | Current correct | Current wrong | Wrong with correcting route | Fraction |
|---|---:|---:|---:|---:|---:|
| GQA | 4,000 | 2,000 | 2,000 | 1,386 | 69.30% |
| TEXTVQA | 2,000 | 1,034 | 966 | 712 | 73.71% |
| CHARTQA | 2,000 | 1,011 | 989 | 774 | 78.26% |
| **Total** | **8,000** | **4,045** | **3,955** | **2,872** | **72.62%** |

Contract drift versus historical metadata:

- Stable correct: `3,978`; stable wrong: `3,933`.
- Historical correct → current wrong: `22`.
- Historical wrong → current correct: `67`.

## Search budgets and label yield

- Samples at 200/400/600 MCTS simulations: `4,045` / `2,775` / `1,180`.
- Evaluated routes: `2,642,998`; mean/median per sample: `330.37` / `202.0`.
- Valid routes: `528,047`; mean/median per sample: `66.01` / `39.0`.
- Samples with ≥1/≥5/≥10/≥20 valid routes: `6,917` / `6,012` / `5,516` / `4,877`; zero-positive: `1,083`.

For the 4,045 current-ALL-ON-correct records, the minimum-budget successful route uses a mean/median `7.58` / `9.0` visual-ON layers. These are oracle label statistics, not learned-policy or latency results.

## Route structure

- Valid masks analyzed: `528,047` across `6,917` samples.
- Sample-balanced mean transitions: `13.20`.
- Sample-balanced mean within-sample pairwise Hamming distance: `13.36/28`.
- Exact within-sample unordered mask pairs: `36,163,535`.
- Masks with ≤3 transitions: `5,389` (1.02%); with ≥14 transitions: `268,174` (50.79%).

## Predictor split and P8 supervision

- Frozen split: `7,000` train / `1,000` validation; cross-split image groups: `0`.
- Validation historical strata: GQA 250/250, TextVQA 125/125, ChartQA 125/125.
- Selected max-50 valid routes: `237,802`; samples capped: `3,616`.
- Shared duplicated-BCE/exact-set-NLL route-set digest: `eafd8bb9dd66b2a800850e6f8e778eb68ad493c24c2861b921e91bb387c2bc0b`.
- Independent P8 verification passed: `True`; POLAR masks reconstructed: `237,802`.

## Operational amendments and failures

- P3 job 99741 was intentionally stopped after 2,291 atomically published records for a user-approved 4→8 GPU migration. Job 99758 contributed the remaining 5,709 records. The resume amendment changed only cross-shard discovery; the scientific contract remained unchanged.
- P8 had three cancelled, unpublished performance-only attempts. The supported bottlenecks were repeated tuple Hamming work and serialized decoding of the 21 GB trace cache. Exact XOR bit-count Hamming and bounded process decoding completed the same selection contract. No raw or published artifact was deleted or altered.
- One proposed P8 CPU request was rejected before submission because 32 GB exceeded the scheduler's node cap; the streaming job required only 24 GB.
- Scientific failures/incomplete records in the final cache: none.

## Reproducibility and checksums

- Final artifact inventory SHA-256: `815ca74ec4572bb0d62934b9b7b0840afa42531b8d67d5e82e60869e2a3f7b0f`.
- Command/provenance record SHA-256: `8955dbf20a8f8c0b4e745bda437df21a2a104164eaa0b532cb68dbe1c77c1a23`.
- Raw records are frozen individually by the P4 record index; copying only the aggregate files without that index does not reproduce the raw-cache integrity chain.
- Exact P0/P1 reproduction command and every scheduled P2–P8 command are saved in `p9_run_provenance_v1.json`.

## Final gate

P9 passes: the raw route cache, outcomes, diversity summaries, split identities, derived views, source/runtime provenance, operational amendments, and checksum chain are complete. Predictor training remains a separate P10 action requiring explicit approval. The first permitted P10 action is the bounded matched duplicated-BCE versus exact-set-NLL smoke—not full training.
