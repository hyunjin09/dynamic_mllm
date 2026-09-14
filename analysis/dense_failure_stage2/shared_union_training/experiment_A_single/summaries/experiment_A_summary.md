# Experiment A: union preservation + single

1. UNION_A contains **106** unique preservation bases.
2. UNION_B contains **270** unique single-fixable W bases and 1688 routes.
3. Implementation/overfit passed: **True / True**. Final train non-FULL recall: 0.1920.
4. P98/P95/P90 W-to-C: **0 / 0 / 4**.
5. P98/P95/P90 C-to-W: **0 / 0 / 1**.
6. Best Historical-validation net correction is **P90 = +3**.
7. P90 C preservation: **0.9975**.
8. Dataset/source results are in `metrics/dataset_source_breakdown.csv`; Canonical cells are explicitly unavailable because the Canonical OOF population supplies Stage-2 union supervision.
9. Experiment A health gate for proceeding to B: **True**.
