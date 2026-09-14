# Closed-loop trajectory-set full evaluation summary

- Contract: `00639e7e5638d766ca9d83d912e827456185bfa778fda4ab702407dc6c835f6a`
- Full corpus: **569 UIDs / 4948 replay-valid trajectories**.
- Dense-C/Dense-W UIDs: **106 / 463**.
- Routed-state cache: **35565 unique states / 69178 route occurrences**, zero quarantines.
- Exact marginal objective numerical/gradient tests: **passed**.
- Full-corpus refit: **completed**.
- Full external evaluation: **19,960/19,960 rows**, exact Dense/Stage-1 baseline parity and exact state-feedback trace checks.

| Method | Accuracy | W→C | C→W | Net |
|---|---:|---:|---:|---:|
| Dense | 0.780561 | 0 | 0 | 0 |
| Sequential-A | 0.779760 | 3 | 19 | -16 |
| Open-loop Program | 0.780311 | 3 | 8 | -5 |
| Closed-loop Trajectory-Set | 0.780160 | 0 | 8 | -8 |

## Required answers

1. Training population: 569 UIDs and 4948 trajectories.
2. Dense-C preservation / Dense-W corrective UIDs: 106 / 463.
3. Provenance occurrences: preservation 106, single 1442, original MCTS 75, robust search 896, completeness audit 2429.
4. Routed-state replay parity: passed for every retained route; quarantined routes: 0.
5. Exact marginal objective numerical and brute-force gradient parity: passed.
6. Full-corpus refit: completed from exact Sequential-A initialization.
7. All four method accuracies are in the table above.
8. Closed-loop W→C/C→W/Net: 0/8/-8 pooled; per benchmark is in `metrics/benchmark_breakdown.csv`.
9. Pooled Net positive: **False**.
10. Closed-loop accuracy >= Dense: **False**.
11. W rescue improved over Sequential-A and Open-loop: **False**.
12. C preservation improved over Sequential-A: **True**.
13. Triggered-W receiving non-FULL: 39/496 (0.0786).
14. Triggered-C receiving non-FULL: 32/405 (0.0790).
15. Mean trigger-to-first-intervention delay: W 5.076923076923077; C 7.75.
16. TextVQA rescues: 0.
17. MMMU-Pro rescues: 0; improved beyond two: **False**.
18. POPE triggers: 0; Net 0.
19. Route responsibility mean top share: 0.6608; median best-route geometric action probability: 0.7269.
20. Evidence supports this formulation as a deployment winner: **False**.
21. If unsuccessful, the next bottleneck is most consistent with: **on-policy state-distribution shift** under the fixed diagnostic rule.
22. This result does not establish that dynamic READ/WRITE routing, MCTS, Stage-1, richer representations, or on-policy relabeling are generally ineffective.

Paired bootstrap intervals are in `metrics/paired_bootstrap.csv`.
