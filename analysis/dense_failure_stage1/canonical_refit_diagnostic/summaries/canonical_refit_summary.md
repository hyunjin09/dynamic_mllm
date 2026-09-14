# Canonical Refit Summary

## Outcome

The exact historical Shared Random-4 architecture, refit with canonical current-runtime labels and the frozen old normalization, achieved **0.8178 OOF AUROC** versus **0.4056** for the frozen old head on the same 4,000 samples. The paired difference was **+0.4123** (95% bootstrap CI [+0.3862, +0.4376]).

| Head | Overall AUROC | GQA | ChartQA | TextVQA | Overall AUPRC |
|---|---:|---:|---:|---:|---:|
| Frozen old head | 0.4056 | 0.7453 | 0.3469 | 0.6846 | 0.1705 |
| Canonical refit, old norm | 0.8178 | 0.7214 | 0.7021 | 0.4640 | 0.5089 |

ChartQA inversion disappeared (OOF AUROC 0.7021). The five largest layerwise AUROC gains were L0 (+0.424), L1 (+0.419), L10 (+0.414), L12 (+0.411), L11 (+0.410).

Canonical Dense-C maximum-risk scores changed from mean 0.8056 / p95 1.0000 to mean 0.4677 / p95 0.9186; this reduces the extreme canonical-correct high-risk behavior descriptively.

TextVQA has only 19 wrong samples. Its OOF AUROC is 0.4640, and its paired refit-minus-old interval is [-0.4226, +0.0131]; the point estimate should not be treated as precise.

## Historical cross-evaluation

On the frozen historical validation+test population, the full-canonical fit achieved AUROC 0.5511; the old historical head achieved 0.8885. This cross-regime result did not select the canonical model.

## Interpretation

Prospective decision: **A — Same architecture is adequate**. Canonical OOF recovery meets every frozen criterion, supporting the historical fitted-boundary/population-regime explanation over an inherent representational inability of this head.

No threshold was calibrated and no trigger map, Stage-2 dataset, corrective search, or routing artifact was changed.
