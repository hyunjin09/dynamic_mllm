# Experiment B: union preservation + single + MCTS

1. UNION_C adds **216** unique MCTS-fixable W bases and 725 routes.
2. Final train non-FULL recall is **0.1687**; threshold action behavior is in `metrics/action_behavior.csv`.
3. P98/P95/P90 W-to-C: **0 / 0 / 0**.
4. P98/P95/P90 C-to-W: **0 / 0 / 0**.
5. The threshold-dependent comparison is frozen in `metrics/threshold_comparison.csv`.
6. Dataset effects are in `metrics/dataset_source_breakdown.csv`; Canonical held-out evidence is unavailable.
7. Best Experiment-B Historical-validation point: **P98**, routed accuracy **0.500000**.
