# READ-harm learnability summary

## Baselines and feature ablations

- Nuisance-only B0 reaches Spearman/AUROC `0.0673/0.5364`; the inherited Step-B generic pre-state B1 reaches `0.0416/0.5193`; and the Phase-82 one-step delta B2 reaches `0.0773/0.5321`.
- Image-group-disjoint OOF results for every READ group are below. Cells are `Spearman / harmful AUROC`.

| Features | Linear | Two-layer MLP |
|---|---:|---:|
| F1 update magnitude | 0.0409 / 0.5208 | 0.0442 / 0.5201 |
| F2 update direction | 0.0178 / 0.5090 | 0.0449 / 0.5261 |
| F3 token concentration | 0.0075 / 0.4974 | 0.0225 / 0.5079 |
| F4 attention structure | 0.0563 / 0.5299 | 0.0526 / 0.5260 |
| F5 query-key compatibility | -0.0004 / 0.4973 | 0.0103 / 0.5038 |
| F6 output/value statistics | 0.0486 / 0.5213 | 0.0456 / 0.5211 |
| F7 visual concentration | 0.0519 / 0.5278 | 0.0304 / 0.5145 |
| F_ALL | 0.0594 / 0.5292 | **0.0697 / 0.5338** |

- The prospectively selected dense winner is **F_ALL / MLP**. Its Pearson/MAE/RMSE are `0.0710/0.1540/0.2806`.
- It improves on generic pre-state by only `+0.0281` Spearman and `+0.0145` AUROC. Although both group-bootstrap intervals are above zero (`[0.0065, 0.0493]` and `[0.0021, 0.0267]`), the gains are far below the frozen material thresholds `+0.10/+0.08`.
- It does not beat the one-step delta on the primary metric (`0.0697` versus `0.0773` Spearman); AUROC is essentially tied (`0.5338` versus `0.5321`).

## Ranking, Dense-W, and restricted regimes

- F_ALL precision@top5/10/20% is `0.4947/0.5161/0.5219` versus harmful prevalence `0.4797`; recall at 90% and 95% precision is zero. This is not a useful high-precision harmful subset.
- Harmful-flip ranking is worse than chance (AUROC `0.3484`, AUPRC `0.0309` for 625 positives), so the continuous predictor does not prioritize the decisive FULL-wrong to WRITE_ONLY-correct events.
- Dense-W-only performance survives only at the same weak level: 14,228 states, Spearman `0.0697`, AUROC `0.5341`, precision@top10% `0.5193`. Thus the result is not merely a Dense-C/Dense-W shortcut, but it is also not practically predictive.
- Early/middle/late Spearman is `0.0673/0.0805/0.0619` and AUROC `0.5366/0.5486/0.5258`. Trigger-relative Spearman is strongest at the trigger (`0.1063`), then `0.0678`, `0.0773`, and `0.0615` at +1-2, +3-5, and +6 or later. No broad computational phase is strongly learnable.
- The strongest dataset/source cell is canonical TextVQA (`0.4063` Spearman, `0.6493` AUROC), but it contains only 68 states. Historical TextVQA is `0.1015/0.5283`; the other four cells have Spearman `0.0166-0.0626`. This is a small niche, not stable cross-source evidence.
- Generic-prestate + F_ALL MLP reaches Spearman `0.0734` and AUROC `0.5317`, a negligible primary-metric gain over F_ALL alone and a lower AUROC.

## Generalization and routed secondary evidence

- Semantic Q1/Q5 Spearman is `0.0821/0.0426` and AUROC `0.5333/0.5235`.
- Five question-cluster OOD folds span Spearman `0.0355-0.1057` and AUROC `0.5211-0.5397`.
- Historical-to-canonical transfer is Spearman/AUROC `0.0307/0.5180`; canonical-to-historical is `0.0435/0.5151`.
- Dataset LODO is weak: held-out ChartQA `0.0570/0.5252`, GQA `-0.0210/0.4937`, and TextVQA `0.0649/0.5227`.
- The frozen external-transfer gate is **false**, so no external branch measurement was launched.
- Routed results are secondary and selection-qualified because the population is route-selected. Using the dense-selected F_ALL/MLP only, routed OOF Spearman/AUROC is `0.1623/0.5614`, while dense-to-routed transfer is `0.1385/0.5537`. They do not override the dense-primary and weak-transfer conclusions.

## Decision and limits

**R-LEARN-C — not locally solvable under the frozen feature/capacity family.** READ-specific features neither materially beat the generic representation nor beat the one-step delta, do not yield a high-precision harmful subset, and do not generalize robustly across sources/datasets. This phase does not establish causal optimality, benchmark improvement, compute savings, WRITE behavior, or the impossibility of using history or short-horizon propagation/planning information.
