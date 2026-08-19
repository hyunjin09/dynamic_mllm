# Full10 POLAR-Style Duplicated-BCE External Evaluation

## Integrity

- Status: **PASS**; all 22,307 frozen active UIDs completed exactly once.
- DocVQA was excluded prospectively. Core VQA, multiple choice, and POPE are not pooled into one overall metric.
- The checkpoints were selected on internal validation before these external outcomes were evaluated.
- The scientific baseline is current live ALL-ON execution. The historical bundle cache is audit-only because 485/22,307 predictions/scores/correctness tuples differed under the current runtime, including 168 correctness labels.
- Reported compute is visual-ON decoder-layer count, not measured wall-clock acceleration.

## Route behavior

- Question selected ALL-ON for 13,120/22,307 records and a non-ALL-ON mask for 9,187. The predicted execution changed prediction, score, or correctness relative to current live ALL-ON on 410 records.
- Image Question selected ALL-ON for 10,044/22,307 records and a non-ALL-ON mask for 12,263. The predicted execution changed prediction, score, or correctness relative to current live ALL-ON on 598 records.

### Question

| Benchmark | N | All-ON acc. | Router acc. | Ratio (base=100%) | Harm | Rescue | Mean ON layers | Unique masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2,500 | 0.8600 | 0.8580 | 99.77% | 14 | 9 | 27.02 | 113 |
| DocVQA | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| TextVQA | 5,000 | 0.8546 | 0.8534 | 99.86% | 13 | 7 | 27.70 | 68 |
| POPE adversarial | 3,000 | 0.8690 | 0.8693 | 100.04% | 12 | 13 | 25.95 | 13 |
| POPE popular | 3,000 | 0.8773 | 0.8773 | 100.00% | 11 | 11 | 25.93 | 13 |
| POPE random | 3,000 | 0.8893 | 0.8900 | 100.07% | 9 | 11 | 26.32 | 13 |
| SEEDBench-lite | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MMStar | 1,500 | 0.6200 | 0.6027 | 97.20% | 39 | 13 | 25.86 | 143 |
| MMMU | 847 | 0.5277 | 0.5254 | 99.55% | 8 | 6 | 27.30 | 46 |
| MMMU-Pro standard | 1,730 | 0.3688 | 0.3665 | 99.37% | 10 | 6 | 27.46 | 47 |
| MMMU-Pro vision | 1,730 | 0.3370 | 0.3382 | 100.34% | 14 | 16 | 27.70 | 17 |

### Image Question

| Benchmark | N | All-ON acc. | Router acc. | Ratio (base=100%) | Harm | Rescue | Mean ON layers | Unique masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2,500 | 0.8600 | 0.8588 | 99.86% | 3 | 0 | 27.52 | 43 |
| DocVQA | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| TextVQA | 5,000 | 0.8546 | 0.8534 | 99.86% | 10 | 4 | 27.63 | 52 |
| POPE adversarial | 3,000 | 0.8690 | 0.8647 | 99.50% | 45 | 32 | 23.06 | 95 |
| POPE popular | 3,000 | 0.8773 | 0.8803 | 100.34% | 24 | 33 | 23.05 | 88 |
| POPE random | 3,000 | 0.8893 | 0.8900 | 100.07% | 29 | 31 | 23.29 | 99 |
| SEEDBench-lite | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MMStar | 1,500 | 0.6200 | 0.5767 | 93.01% | 99 | 34 | 22.92 | 312 |
| MMMU | 847 | 0.5277 | 0.5183 | 98.21% | 17 | 9 | 26.72 | 67 |
| MMMU-Pro standard | 1,730 | 0.3688 | 0.3630 | 98.43% | 25 | 15 | 26.08 | 188 |
| MMMU-Pro vision | 1,730 | 0.3370 | 0.3382 | 100.34% | 4 | 6 | 27.93 | 8 |

`Ratio (base=100%)` is `router accuracy / current live All-ON accuracy × 100`.
`Harm` is All-ON-correct → router-wrong; `Rescue` is All-ON-wrong →
router-correct. `Unique masks` counts distinct predicted 28-bit masks within
the benchmark. DocVQA was explicitly excluded and SEEDBench-lite was not in
the frozen evaluation suite, so both are reported as `N/A`.

## Interpretation boundary

These are deterministic static-mask executions of validation-selected factorized predictors. External correctness changes and visual-ON counts describe their behavior; they do not by themselves establish deployable latency gains or causal routing mechanisms.
