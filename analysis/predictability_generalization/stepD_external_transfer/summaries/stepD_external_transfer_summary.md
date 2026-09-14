# Predictability Step-D external-transfer summary

Contract: `efa342f0a8c6dd5b3d38de5642de87d4b405208a70affac7d1242332e3e571cf`

## Required answers

1. **Full refits:** yes; all frozen Stage-1/Stage-2 M0-X, M1, and M3 tasks trained on all internal data for the prospectively frozen Step-B-median epochs.
2. **Dense parity:** yes; all 19,960 UIDs matched the established generated tokens, task score/correctness, images, and robust trigger trace.
3. **P90 domain:** 901 external UIDs entered Stage 2.
4. **Trigger parity:** yes; the exact prior P90 trigger census was reproduced, including zero POPE triggers.
5. **Stage-1 external M3:** ChartQA AUROC/AUPRC 0.4998/0.1574; TextVQA 0.6794/0.2747; MMMU-Pro 0.5399/0.6737; POPE 0.5582/0.1356.
6. **Stage-1 macro AUROC:** 0.5693.
7. **Peak layers:** chartqa=L24 (0.6051), textvqa=L20 (0.7710), mmmu_pro=L19 (0.5977), pope=L21 (0.6832).
8. **Layer 20:** exact layer-20 values are in `stage1/layer_metrics.csv`; no benchmark-specific deployment layer was selected.
9. **Calibration drift:** internally calibrated 95/98/99% points are reported without external repair in `stage1/operating_point_transfer.csv`.
10. **Existing robust P90:** external funnel is preserved in `stage1/robust_p90_external_funnel.csv`; total triggers=901.
11. **M3 versus nuisance:** per-benchmark differences are in the main table/control file; no nuisance feature contains source or dataset ID.
12. **M3 versus question-kNN:** per-benchmark comparisons are in `stage1/model_control_comparison.csv`.
13. **Semantic similarity:** benchmark mean/median/P10/P90 are in `question_semantics/benchmark_similarity_summary.csv`.
14. **Similarity-conditioned transfer:** lower/upper external halves are in `stage1/semantic_similarity_metrics.csv`.
15. **External Stage-2 measurement:** 8,442 dense post-trigger states.
16. **Four-branch validity:** FULL token/score/correctness/q parity and utility algebra passed for all 8,442 states.
17. **Local opportunities:** per-benchmark rescue/regression/all-correct/all-wrong counts are in `stage2_measurement/local_rescue_summary.csv`.
18. **READ utility:** ChartQA/TextVQA/MMMU-Pro Spearman 0.08484881713946812/0.10950275659308661/0.02464337669770887; POPE is N/A because it never triggered.
19. **WRITE utility:** ChartQA/TextVQA/MMMU-Pro Spearman 0.03346685638149732/0.015844179412111574/0.0203551695123434; POPE is N/A.
20. **Stage-2 niche:** fixed rule classification is `D2-A`; detailed uncertainty and controls prevent selecting an isolated favorable cell.
21. **Stage-2 controls:** M3, transfer-safe nuisance, M1, and exact-layer question-kNN are compared in `stage2_predictability/control_comparison.csv`.
22. **Asymmetry:** Stage-1 category `D1-C` and Stage-2 category `D2-A` provide the frozen external answer.
23. **Stage-1 category:** `D1-C`.
24. **Stage-2 category:** `D2-A`.
25. **Not established:** deployment routing gain, causal mechanism, impossibility of richer-history/counterfactual Stage 2, or transfer of a redesigned Stage-1 representation.

## Main Stage-1 table

| Benchmark | N | Dense C | Dense W | M0-X AUROC | M1 AUROC | M3 AUROC | M3 AUPRC | kNN AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2500 | 2147 | 353 | 0.4429 | 0.5302 | 0.4998 | 0.1574 | 0.4573 |
| textvqa | 5000 | 4288 | 712 | 0.4997 | 0.6706 | 0.6794 | 0.2747 | 0.5863 |
| mmmu_pro | 3460 | 1225 | 2235 | 0.5718 | 0.5188 | 0.5399 | 0.6737 | 0.5065 |
| pope | 9000 | 7920 | 1080 | 0.4830 | 0.5447 | 0.5582 | 0.1356 | 0.4971 |
| macro | 19960 | 15580 | 4380 | 0.4994 | 0.5661 | 0.5693 | 0.3103 | 0.5118 |
| pooled | 19960 | 15580 | 4380 | 0.6674 | 0.6701 | 0.6980 | 0.3928 | 0.5567 |
