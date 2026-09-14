# Predictability Step-A measurement summary

Contract: `6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d`

This phase constructs outcomes only. It makes no Stage-1 or Stage-2 predictability claim.

## Census and validation

1. **Internal samples:** 10,399 (6,399 Historical-train + 4,000 Canonical) across 9,982 image groups.
2. **Stage-1 states:** 291,172 `(sample, layer)` rows with exact references to the established three-block raw BF16 representation.
3. **Dense baseline parity:** passed. Every source artifact is hash-bound, and 12 stratified live native replays matched stored tokens, LMMS correctness, image SHA, and all three 28-layer feature tensors exactly.
4. **P90 triggers:** 1,413 samples under strict `p > 0.9061332901863008`.
5. **Primary dense Stage-2 states:** 15,185.
6. **Primary dense branches:** 60,740; every state has exactly one result for each of the four actions.

## Primary dense utility

| Metric | Mean | Median | Std | Q05 | Q95 | Positive | Negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| u_read | 0.010914 | 0.002864 | 0.278266 | -0.356632 | 0.395253 | 7962 | 7223 |
| u_write | 0.008955 | 0.000000 | 0.200840 | -0.270241 | 0.292842 | 7353 | 6419 |
| u_interaction | -0.000426 | 0.000000 | 0.089307 | -0.121024 | 0.123023 | 6762 | 7010 |

7. The table reports the complete continuous `U_READ`, `U_WRITE`, and interaction distributions.
8. READ sign counts are shown in the `u_read` row; zero is retained rather than thresholded.
9. WRITE sign counts are shown in the `u_write` row; zero is retained rather than thresholded.
10. Strong controlled correctness flips: READ=1,291; WRITE=981. Detailed directional counts are in `stage2_dense/correctness_flip_labels.csv`.
11. Local single-layer rescue states: 1,048. Local regression states: 83; all-four-correct=874; all-four-wrong=13,180.
12. Layer variation is frozen in `stage2_dense/layer_breakdown.csv`.
13. Dataset/source/outcome variation is frozen in `stage2_dense/dataset_source_breakdown.csv`.
14. Dense-C versus Dense-W is included explicitly in that same breakdown; no outcomes were filtered.

## Secondary routed-state utility

15. **Exact routed states:** 35,565, deduplicated from 69,178 route occurrences; 142,260 four-action branches.

| Metric | Mean | Median | Std | Q05 | Q95 | Positive | Negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| u_read | 0.042380 | 0.012074 | 0.272375 | -0.267004 | 0.465496 | 20127 | 15438 |
| u_write | 0.015463 | 0.000000 | 0.162200 | -0.192435 | 0.250477 | 17143 | 13743 |
| u_interaction | 0.000715 | 0.000000 | 0.094787 | -0.111578 | 0.117340 | 15673 | 15213 |

16. Routed-state and dense-state distributions are kept separate above and in their raw label files; differences are descriptive measurement differences, not predictability evidence.
17. **Validation:** action semantics, independent FULL+dense parity, repeated branch execution, live routed-state hashes, complete anchor continuation replay, global completeness, and utility algebra all passed.
18. **Limitations:** continuous q is a teacher-forced annotated-answer score rather than an evaluator score. In particular, ChartQA's relaxed ±5% numeric acceptance interval cannot be represented by one finite answer string; q uses the literal annotation while discrete branch correctness uses exact LMMS semantics. Routed states cover the exact existing 569-UID successful/preservation corpus, not all possible routed states. Measurement does not establish that utility or failure is predictable.

## Frozen target artifacts

- Stage-1: `stage1/dense_state_manifest.jsonl`, `stage1/dense_outcome_labels.csv`, `stage1/state_feature_manifest.jsonl`.
- Primary Stage-2: `stage2_dense/exact_state_manifest.jsonl`, `four_branch_results.jsonl`, `utility_labels.csv`, `correctness_flip_labels.csv`, and `state_feature_manifest.jsonl`.
- Secondary Stage-2: the corresponding files under `stage2_routed/`.
