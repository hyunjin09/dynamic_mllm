# Reference Results From The Original Server

## Core VQA: All-on vs SW31

The frozen SW31 router was evaluated on the project prompt over ChartQA test, TextVQA validation, and DocVQA validation.

| Benchmark | n | All-on correct | SW31 correct | Delta | All-on score | SW31 score | Mean ON |
|---|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2,500 | 85.92% | 79.32% | -6.60%p | 0.8592 | 0.7932 | 22.95 |
| TextVQA | 5,000 | 85.74% | 80.44% | -5.30%p | 0.8489 | 0.7979 | 20.94 |
| DocVQA | 5,349 | 95.38% | 90.60% | -4.79%p | 0.9389 | 0.8806 | 22.01 |
| **Micro-average** | **12,849** | **89.79%** | **84.45%** | **-5.34%p** | **0.8883** | **0.8314** | **21.78** |

There are 889 harm cases and 203 rescue cases. TextVQA/DocVQA score is fractional, so score and thresholded correctness are not interchangeable. The full CI table is in `results/reference_core_vqa/report.md`.

Reference artifacts:

- `results/reference_core_vqa/heldout_generation_rows.jsonl`
- `results/reference_core_vqa/summary.json`
- `results/reference_core_vqa/report.md`
- `results/reference_core_vqa/source_manifest_alignment.json`
- `baseline/core_vqa_all_on_generation_rows.jsonl`

## External Shared-Prefix Admission

The original 4-GPU generation, frozen admission scoring, and one-pass audit completed on 2026-08-12. These files are copied under results/reference_original and are the target for reproduction.

| Policy | Accuracy | Delta | Routed | Harm / rescue | Mean ON |
|---|---:|---:|---:|---:|---:|
| All-on | 44.62% | 0.00%p | 0.00% | 0 / 0 | 28.00 |
| Ungated K=8 SW31 | 43.95% | -0.67%p | 100.00% | 167 / 128 | 24.72 |
| Learned admission | 44.05% | -0.57%p | 92.32% | 153 / 120 | 25.01 |
| Oracle admission | 46.82% | +2.20%p | 97.12% | 0 / 128 | 24.82 |

Learned admission paired accuracy delta bootstrap 95% CI is [-1.12, 0.00] percentage points.

| Benchmark | n | All-on | Learned | Delta | Mean ON |
|---|---:|---:|---:|---:|---:|
| MMMU-Pro standard test | 1,730 | 36.71% | 36.65% | -0.06%p | 25.26 |
| MMMU-Pro vision test | 1,730 | 33.82% | 32.20% | -1.62%p | 25.29 |
| MMMU validation | 847 | 52.18% | 53.36% | +1.18%p | 24.80 |
| MMStar validation | 1,500 | 61.93% | 61.00% | -0.93%p | 24.51 |

Runtime equivalence passed on 64 samples with zero admission, prediction, score, correctness, or mask mismatches.

The correct interpretation is negative for the learned gate: oracle route availability exists, but the canonical-calibrated early-prefix ranking shifts strongly on external tasks and does not preserve all-on accuracy. Reproduction should match the numbers; it should not reinterpret this as a successful deployed gate.

Reference artifacts:

- results/reference_original/summary.json
- results/reference_original/external_predictions.jsonl
- results/reference_original/runtime_equivalence.json
- results/reference_original/score_shift_diagnostic.json
- results/reference_original/report.md
- results/reference_original/result.png

## POPE: All-on vs SW31

| Split | n | All-on | SW31 | Delta | Mean ON | Harm / rescue |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 3,000 | 87.03% | 85.10% | -1.93%p | 17.40 | 112 / 54 |
| popular | 3,000 | 87.87% | 85.83% | -2.03%p | 17.36 | 104 / 43 |
| random | 3,000 | 89.03% | 87.60% | -1.43%p | 17.66 | 83 / 40 |
| **Micro-average** | **9,000** | **87.98%** | **86.18%** | **-1.80%p** | **17.47** | **299 / 137** |

Reference artifacts:

- `results/reference_pope/heldout_generation_rows.jsonl`
- `results/reference_pope/summary.json`
- `results/reference_pope/report.md`
- `baseline/pope_all_on_generation_rows.jsonl`
