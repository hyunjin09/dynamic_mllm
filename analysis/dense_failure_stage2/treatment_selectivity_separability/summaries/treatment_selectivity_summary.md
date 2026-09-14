# Stage-2 treatment-selectivity separability summary

## Decision

- Case **D**.
- The frozen Stage-2 representations do not meet the prospective generalizable/high-precision evidence gates.
- Contract: `082c9f459da798c419b50b637c0afb1c1b388b403ea2b51c73092b1df3e44edd`.
- All numbers below are out-of-fold under five UID/image-group-disjoint folds.

## Answers to the plan questions

1. **Unique UIDs:** 569.
2. **Exact routed states:** 21,071 unique exact prefix-states from 34,253 route-state occurrences.
3. **KEEP_REQUIRED:** 18,438.
4. **INTERVENE_REQUIRED:** 1,657.
5. **MIXED:** 976; excluded from fitting and scored OOF only.
6. **Scalar margin:** AUROC 0.4632, AUPRC 0.1050; this is not meaningfully above the frozen weak-signal reference.
7. **Four logits versus margin:** M1 AUROC 0.4640 (+0.0007 versus M0).
8. **z_R:** AUROC 0.5370, AUPRC 0.0931.
9. **z_W:** AUROC 0.5219, AUPRC 0.1088.
10. **[z_R;z_W]:** AUROC 0.5620, AUPRC 0.1086; optional MLP was run.
11. **High-precision subset:** RW precision at top 5/10/20% coverage is 0.123/0.135/0.123; recall at 90/95% precision is 0.001/0.001.
12. **Group-disjoint survival:** yes in evaluation design; RW OOF AUROC is 0.5620, so signal is useful by the frozen criterion.
13. **Matched survival:** RW matched AUROC/AUPRC is 0.5763/0.6008, versus natural 0.5620/0.1086; AUROC change -0.0143 drop. Support is 18936 states and 463 UIDs (Kish state ESS 459.0).
14. **Route consistency:** single AUROC 0.664 (I=247, K=12873), mcts AUROC 0.572 (I=1410, K=4690).
15. **Learned KEEP-vs-INTERVENE head justified:** no under the prospective decision rule.
16. **Not justified:** This does not show that routing cannot work, that READ/WRITE control is invalid, that unobserved actions are truly harmful, or that richer/diversified representations cannot help. KEEP_REQUIRED means uniquely observed FULL among replay-valid successful routes, not exhaustive proof against every unsearched intervention.

## Additional controls

- Nuisance-only AUROC: 0.8150 using dataset, source regime, layer bin, and route-source signature only.
- RW dataset/source AUROCs where supported: canonical:chartqa 0.651, canonical:gqa 0.521, historical:chartqa 0.600, historical:gqa 0.537, historical:textvqa 0.533.
- MIXED-state RW score median 0.4759 (q25–q75 0.3733–0.5854); this is descriptive, not a binary target.
