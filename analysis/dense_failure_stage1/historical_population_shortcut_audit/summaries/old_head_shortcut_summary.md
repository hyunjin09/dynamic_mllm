# Old-Head Shortcut Audit Summary

| Diagnostic | GQA | ChartQA | TextVQA |
|---|---:|---:|---:|
| Source-ID probe AUROC on C, L21 | 0.586 | 0.938 | 0.937 |
| Best nuisance probe AUROC | 0.994 | 0.957 | 0.973 |
| Nuisance-only old test C/W AUROC | 0.673 | 0.788 | 0.630 |
| Frozen head old test AUROC | 0.747 | 0.960 | 0.972 |
| Frozen head token-matched old test AUROC | 0.756 | 0.824 | 0.970 |
| Frozen head new AUROC | 0.745 | 0.347 | 0.685 |

## Findings

- Historical-vs-canonical source is evaluated with a train-standardized, group-held-out centroid linear probe using exactly the Shared Random-4 input blocks. At L21 the Dense-C AUROCs are {'gqa': 0.5863240110859158, 'chartqa': 0.9378815628815629, 'textvqa': 0.936543995153718}. This establishes feature availability, not causal head use.
- The strongest cross-validated nuisance-property AUROCs are {'gqa': 0.9943729987327058, 'chartqa': 0.9571976993044409, 'textvqa': 0.9726097569014641}. Frozen-score associations by source, outcome, layer, and measured nuisance are preserved in `metrics/score_nuisance_association.csv`.
- The nuisance-only old-test/new AUROCs are respectively {'gqa': 0.6730375, 'chartqa': 0.78825, 'textvqa': 0.6296} and {'gqa': 0.6772538869014859, 'chartqa': 0.38141675768450617, 'textvqa': 0.5632812919148023}; this tests shortcut opportunity without hidden states.
- Exact visual-token matching changes per-dataset old-test max-score AUROC from {'gqa': 0.746675, 'chartqa': 0.9602, 'textvqa': 0.972} to {'gqa': 0.7561873849130801, 'chartqa': 0.8244170096021948, 'textvqa': 0.9702380952380952}; mean drop is 0.043. Remaining performance is evidence that visual-token count alone is insufficient.
- At L21 the new-correct projections on the historical C→W direction are {'gqa': -3.174720048904419, 'chartqa': 14.697389602661133, 'textvqa': 4.876222610473633}; historical wrong reference means are {'gqa': 5.334926128387451, 'chartqa': 17.47273063659668, 'textvqa': 7.485122203826904}. This is descriptive decision-geometry shift.
- After conditioning on dataset, correctness, layer, and feature block, canonical-vs-historical-validation RMS normalized mean differences have median 0.063 and maximum 1.323. This measures raw feature/source shift in frozen-normalized units; it does not show that the normalization transform itself is the dominant cause.

## Decision

The evidence is most consistent with **Case B with Case-D geometry: source is strongly encoded for ChartQA/TextVQA, but measured nuisance matching does not explain most old ranking**. The old population provided abundant selection/source shortcut opportunity and the fitted boundary is source-sensitive. However, feature availability and score association are observational; visual-token matching cannot prove a causal shortcut, and residual same-regime AUROC means transferable failure information may coexist with nuisance/source dependence.
