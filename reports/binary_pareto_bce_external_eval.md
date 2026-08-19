# Pareto duplicated_bce Image+Question External Evaluation

- Integrity: **PASS**; 22,307 frozen active UIDs completed exactly once.
- DocVQA remains excluded; this is the unchanged earlier BCE/NLL suite.
- The scientific baseline is current live ALL-ON execution.
- Compute is visual-ON decoder-layer count, not measured latency.

| Population | N | ALL-ON acc | Router acc | Ratio | Harm | Rescue | Mean ON | Unique masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2,500 | 0.8600 | 0.1232 | 14.33% | 1863 | 21 | 1.82 | 42 |
| textvqa | 5,000 | 0.8546 | 0.1178 | 13.78% | 3708 | 24 | 1.91 | 78 |
| mmstar_val | 1,500 | 0.6200 | 0.2720 | 43.87% | 608 | 86 | 0.07 | 4 |
| mmmu_val | 847 | 0.5277 | 0.4286 | 81.21% | 147 | 63 | 0.22 | 7 |
| mmmu_pro_standard_test | 1,730 | 0.3688 | 0.2272 | 61.60% | 329 | 84 | 0.16 | 6 |
| mmmu_pro_vision_test | 1,730 | 0.3370 | 0.2145 | 63.64% | 373 | 161 | 0.31 | 5 |
| pope_adversarial | 3,000 | 0.8690 | 0.5000 | 57.54% | 1179 | 72 | 0.00 | 1 |
| pope_popular | 3,000 | 0.8773 | 0.5000 | 56.99% | 1178 | 46 | 0.00 | 1 |
| pope_random | 3,000 | 0.8893 | 0.5000 | 56.22% | 1178 | 10 | 0.00 | 1 |
| core_vqa | 7,500 | 0.8564 | 0.1196 | 13.97% | 5571 | 45 | 1.88 | 104 |
| external_multiple_choice | 5,807 | 0.4474 | 0.2643 | 59.08% | 1457 | 394 | 0.19 | 9 |
| pope | 9,000 | 0.8786 | 0.5000 | 56.91% | 3535 | 128 | 0.00 | 1 |
| pope_image_disjoint | 8,982 | 0.8783 | 0.5000 | 56.93% | 3526 | 128 | 0.00 | 1 |

External correctness and local visual-ON counts are descriptive execution results; they do not establish wall-clock acceleration.
