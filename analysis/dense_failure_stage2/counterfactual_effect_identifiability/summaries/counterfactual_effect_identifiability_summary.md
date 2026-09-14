# Counterfactual effect identifiability summary

## Validity and population

1. Primary population: **15,185 states / 1,413 UIDs / 1,385 image groups**. Secondary routed population: **35,565 states / 569 UIDs**.
2. Exact same-prestate/action/repeat parity passed for the frozen smoke and complete extraction.
3. FULL post-state exactly reproduced the canonical Dense next-layer state for all 15,185 primary states, including layer 27's actual final-layer output; no synthetic layer 28 was created.

## READ primary OOF Spearman

| PRE | FULL post | WO post | Pair | Delta | Pair+Delta | Token comparator |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0626 | 0.0533 | 0.0552 | 0.0523 | 0.0773 | 0.0568 | 0.0694 |

## WRITE primary OOF Spearman

| PRE | FULL post | RO post | Pair | Delta | Pair+Delta | Token comparator |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0354 | 0.0364 | 0.0399 | 0.0421 | 0.0380 | 0.0399 | 0.0383 |

## Answers to the remaining protocol questions

11. The single-post versus explicit-counterfactual conclusion follows the frozen cases below; no architecture is credited without its same-capacity controls.
12. READ is **Case D** (one-step conditions remain weak or the positive pattern is incomplete); WRITE is **Case D** (one-step conditions remain weak or the positive pattern is incomplete).
13. Linear and MLP results are reported side-by-side in `read/metrics.csv` and `write/metrics.csv`.
14. The token-aware comparison is reported above and uses a fixed shared projection/attention pair comparator.
15-16. Text-only, visual-only, and combined delta ablations are in each target's `text_visual_delta_ablation.csv`; they are treated as hypotheses, not assumed stream specialization.
17. Dense-C and Dense-W results are in `controls/dense_cw_conditional.csv`.
18-19. Exact-layer, depth-bin, and trigger-relative results are in `controls/layer_breakdown.csv` and `controls/trigger_relative_breakdown.csv`.
20. The matched random-pair control is in `controls/random_pair_control.csv`; its registry records all match relaxations prospectively.
21. The swapped-order diagnostic is in `controls/branch_order_control.csv`; primary signs remain FULL-minus-OFF and are never averaged across orders.
22. Routed-state OOF results are in `routed_secondary/metrics_read.csv` and `metrics_write.csv`.
23. Dense-trained routed-state transfer is in `routed_secondary/dense_to_routed_transfer.csv` with the inherited group-disjoint folds.
24. The joint result is **Case D**: both targets agree. Per-target categories are retained rather than hidden by pooling.
25. This experiment does not establish benchmark gain, compute savings, external transfer, causal correctness, or global optimality of one-step probing.
