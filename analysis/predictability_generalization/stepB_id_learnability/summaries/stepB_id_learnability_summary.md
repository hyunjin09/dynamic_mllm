# Step-B in-domain learnability summary

Contract: `792cc760bb41ce8c9b11776f21f43a88e54409ef1543471709000f83ec21bba4`  
Evidence category: **D — Stage-1 strong / Stage-2 weak**

## Main results

| Stage-1 model | AUROC | AUPRC | W recall @95% C-preserve | W recall @98% C-preserve |
|---|---:|---:|---:|---:|
| M0 nuisance | 0.6814 | 0.5081 | 0.0909 | 0.0290 |
| M1 linear state | 0.7665 | 0.6595 | 0.2655 | 0.1511 |
| M2 two-layer MLP | 0.7874 | 0.6863 | 0.2970 | 0.1751 |
| M3 current head | 0.7869 | 0.6860 | 0.3029 | 0.1609 |

| Dense Stage-2 target / M3 joint | Spearman | Harmful AUROC | Harmful AUPRC | Precision top 5% | Precision top 10% |
|---|---:|---:|---:|---:|---:|
| READ `q_F-q_WO` | 0.0416 | 0.5193 | 0.4915 | 0.4816 | 0.4937 |
| WRITE `q_F-q_RO` | 0.0347 | 0.5115 | 0.4768 | 0.4630 | 0.4775 |

## Answers required by the plan

1. Stage-1 evaluated **9,982 image groups, 10,399 UIDs, and 291,172 states** OOF.
2. Dense Stage-2 evaluated **1,385 image groups, 1,413 UIDs, and 15,185 states** OOF.
3. All five folds were image-group-disjoint and passed the frozen support audit; every expected state appears once per model after seed ensembling.
4. Stage-1 nuisance/linear/MLP/current-head AUROCs were 0.681/0.766/0.787/0.787.
5. Current-head Stage-1 AUROC was maximal at layer 20 (0.826); the complete emergence curve is in `stage1/layer_metrics.csv`.
6. At 95/98/99% calibration-fold C preservation, current-head held-out UID W recall was 0.303/0.161/0.122.
7. Primary READ utility joint-router OOF Spearman was 0.042.
8. Primary WRITE utility joint-router OOF Spearman was 0.035.
9. At top-10% predicted harmful coverage, READ/WRITE precision was 0.494/0.478 versus natural harmful prevalence 0.480/0.472.
10. READ specificity: z_R ρ=0.047 versus z_W ρ=0.050.
11. WRITE specificity: z_W ρ=0.031 versus z_R ρ=0.036.
12. Joint z_RW READ/WRITE ρ=0.042/0.035; this is comparative predictive evidence, not causal branch proof.
13. Joint-router Dense-C versus Dense-W Spearman was READ 0.153/0.038, WRITE 0.006/0.038.
14. Nuisance comparisons are frozen in `stage2_dense/nuisance_control_comparison.csv`; the evidence category accounts for whether nuisance matched full-state performance.
15. Text-only, visual-only, concatenated linear, MLP, and router controls are reported in `stage2_dense/text_visual_control_comparison.csv`.
16. Exact-layer and trigger-relative results are in `stage2_dense/layer_metrics.csv` and `trigger_relative_metrics.csv`.
17. All five secondary factorial/context targets were fit with the preregistered reduced ladder; results are in `secondary_factorial_targets.csv`.
18. Selected routed-state joint-router READ/WRITE Spearman was 0.104/0.062 over 35,565 states; this is secondary selection-biased evidence.
19. State-regime transfer was estimable without group relaxation: stage2_dense→stage2_routed read/m3_z_RW ρ=0.081; stage2_dense→stage2_routed write/m3_z_RW ρ=0.063; stage2_routed→stage2_dense read/m3_z_RW ρ=0.053; stage2_routed→stage2_dense write/m3_z_RW ρ=0.034.
20. The evidence pattern is **Case D (Stage-1 strong / Stage-2 weak)** under the plan's qualitative taxonomy; 95% group-bootstrap intervals are in `statistics/uid_bootstrap_ci.csv`.
21. Stage-1 AUROC and Stage-2 utility correlation answer different questions, so their raw magnitudes are not directly commensurate; the observed pattern is summarized by Case D.
22. Step B does **not** establish semantic/template-independent generalization, dataset LODO robustness, external-benchmark transfer, causal branch mechanism, or deployable routing improvement.

Three-seed fold-concatenated variability for the reference models was: Stage-1 M3 AUROC SD 0.0038, dense READ M3 Spearman SD 0.0050, and dense WRITE M3 Spearman SD 0.0147. Full seed means/SDs are in each domain's `seed_metrics.csv`.

All results use current dense outcomes and the frozen Step-A utility measurements; no target was redesigned after observing performance.
