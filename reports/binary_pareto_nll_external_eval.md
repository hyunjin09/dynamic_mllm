# Pareto exact_set_nll Image+Question External Evaluation

- Integrity: **PASS**; 22,307 frozen active UIDs completed exactly once.
- DocVQA remains excluded; this is the unchanged earlier BCE/NLL suite.
- The scientific baseline is current live ALL-ON execution.
- Compute is visual-ON decoder-layer count, not measured latency.

| Population | N | ALL-ON acc | Router acc | Ratio | Harm | Rescue | Mean ON | Unique masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2,500 | 0.8600 | 0.1216 | 14.14% | 1866 | 20 | 1.62 | 7 |
| textvqa | 5,000 | 0.8546 | 0.1078 | 12.61% | 3755 | 21 | 1.53 | 50 |
| mmstar_val | 1,500 | 0.6200 | 0.2740 | 44.19% | 608 | 89 | 0.19 | 4 |
| mmmu_val | 847 | 0.5277 | 0.4345 | 82.33% | 145 | 66 | 0.36 | 6 |
| mmmu_pro_standard_test | 1,730 | 0.3688 | 0.2301 | 62.38% | 326 | 86 | 0.31 | 5 |
| mmmu_pro_vision_test | 1,730 | 0.3370 | 0.2179 | 64.67% | 375 | 169 | 0.90 | 5 |
| pope_adversarial | 3,000 | 0.8690 | 0.5000 | 57.54% | 1179 | 72 | 0.00 | 1 |
| pope_popular | 3,000 | 0.8773 | 0.5000 | 56.99% | 1178 | 46 | 0.00 | 1 |
| pope_random | 3,000 | 0.8893 | 0.5000 | 56.22% | 1178 | 10 | 0.00 | 1 |
| core_vqa | 7,500 | 0.8564 | 0.1124 | 13.12% | 5621 | 41 | 1.56 | 52 |
| external_multiple_choice | 5,807 | 0.4474 | 0.2676 | 59.82% | 1454 | 410 | 0.46 | 6 |
| pope | 9,000 | 0.8786 | 0.5000 | 56.91% | 3535 | 128 | 0.00 | 1 |
| pope_image_disjoint | 8,982 | 0.8783 | 0.5000 | 56.93% | 3526 | 128 | 0.00 | 1 |

External correctness and local visual-ON counts are descriptive execution results; they do not establish wall-clock acceleration.
