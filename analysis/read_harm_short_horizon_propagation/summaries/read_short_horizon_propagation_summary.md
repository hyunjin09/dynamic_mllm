# READ short-horizon counterfactual propagation summary

## Outcome

The frozen decision is **H-READ-D**. no short horizon satisfies the complete prospective emergence pattern The strongest preregistered pooled-delta point estimate was H=8 (Spearman 0.1097); material horizons were [].

## Common-support main tables

Spearman:

| Input | PRE | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| ON-only | 0.0322 | 0.0617 | 0.0478 | 0.0459 | 0.0623 |
| OFF-only | 0.0322 | 0.0663 | 0.0540 | 0.0418 | 0.0704 |
| Pair | — | 0.0670 | 0.0468 | 0.0395 | 0.0567 |
| Delta | — | 0.0756 | 0.0688 | 0.0646 | 0.1097 |
| Token comparator | — | 0.0722 | 0.0723 | 0.0626 | 0.0854 |

Harmful READ AUROC:

| Input | PRE | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| ON-only | 0.5134 | 0.5187 | 0.5133 | 0.5139 | 0.5222 |
| OFF-only | 0.5134 | 0.5210 | 0.5168 | 0.5124 | 0.5274 |
| Pair | — | 0.5229 | 0.5089 | 0.5103 | 0.5217 |
| Delta | — | 0.5316 | 0.5287 | 0.5314 | 0.5483 |
| Token comparator | — | 0.5335 | 0.5376 | 0.5318 | 0.5457 |

Precision at top 10%:

| Input | PRE | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| ON-only | 0.5520 | 0.5506 | 0.5549 | 0.5217 | 0.5405 |
| OFF-only | 0.5520 | 0.5202 | 0.5376 | 0.5318 | 0.5260 |
| Pair | — | 0.5318 | 0.5376 | 0.5275 | 0.5361 |
| Delta | — | 0.5347 | 0.5506 | 0.5462 | 0.5462 |
| Token comparator | — | 0.5217 | 0.5592 | 0.5043 | 0.5578 |

## Required questions

1. Native eligibility is H1 **15,185 states / 1,413 UIDs**, H2 **13,772 / 1,291**, H4 **11,260 / 1,204**, and H8 **6,916 / 872**.
2. **Yes.** Stored H=1 ON/OFF hashes exactly reproduced Phase-82 (`True`); the native delta-MLP Spearman is 0.0773 versus 0.0773 previously.
3. **Yes.** Every recorded trace passed the fixed intervention-then-FULL contract (`True`).
4. The median pooled ON/OFF norm ratio at H8 is 2.268× H1; stream-wise curves are in `features/effect_growth_statistics.csv`.
5. Median H8/H1 pooled growth is 2.253× for harmful and 2.286× for beneficial states; this is descriptive, not a predictability claim.
6. ON-only: H1=0.0617, H2=0.0478, H4=0.0459, H8=0.0623.
7. OFF-only: H1=0.0663, H2=0.0540, H4=0.0418, H8=0.0704.
8. PAIR: H1=0.0670, H2=0.0468, H4=0.0395, H8=0.0567.
9. DELTA: H1=0.0756, H2=0.0688, H4=0.0646, H8=0.1097.
10. Token comparator: H1=0.0722, H2=0.0723, H4=0.0626, H8=0.0854.
11. Pooled-delta Spearman monotonicity is **False**; AUROC monotonicity is **False**.
12. Materiality gate passed at: **[]**. A gate requires the fixed absolute gain and a strictly positive image-group bootstrap lower bound.
13. At H=8, delta exceeds the best single branch by +0.0393 Spearman.
14. Random-pair Spearman values are H1=0.0287, H4=0.0094, H8=-0.0091.
15. UID-permuted-target Spearman values are H1=0.0124, H8=-0.0220.
16. Dense-W delta results are H1: rho=0.0789, AUROC=0.5332, H2: rho=0.0742, AUROC=0.5326, H4: rho=0.0676, AUROC=0.5320, H8: rho=0.1142, AUROC=0.5504.
17. Strong harmful-flip AUROC/AUPRC and cohort score medians at every horizon are in `metrics/strong_flip_metrics.csv`.
18. Exact-layer, depth-bin, and trigger-relative results are in `metrics/layer_breakdown.csv` and `metrics/trigger_relative_breakdown.csv`; they are descriptive subgroup checks.
19. Historical/canonical GQA, ChartQA, and TextVQA cells are all reported without best-cell selection in `metrics/dataset_source_breakdown.csv`.
20. The supported category is **H-READ-D**: no short horizon satisfies the complete prospective emergence pattern.
21. This phase does not establish benchmark gain, deployment utility, compute savings, external transfer, causal optimality, or WRITE identifiability. Routed-state evidence was not promoted into the primary dense decision.

## Validity

The 16-state fresh-cache smoke, swapped-order H8 check, exact H1 parent parity, exact ON canonical parity, global branch census, and cache readback hashes all passed before training. Primary comparisons use the same H8-common population, fixed model capacity, inherited image-group folds, UID-balanced Huber training, and three fixed seeds.
