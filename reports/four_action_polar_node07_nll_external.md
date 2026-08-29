# Four-Action POLAR NLL External Evaluation

## Integrity

- PASS: all 14,960 prospectively selected UIDs completed exactly once.
- The checkpoint was selected using internal four-action validation before any external outcomes were evaluated.
- The scientific baseline is current live unified FULL; no imported historical output cache enters the comparison.
- ChartQA, MMMU-Pro, and POPE are not pooled into a cross-suite metric.

## Results

| Population | N | Unified FULL correct | Predicted correct | Delta (95% image-cluster CI) | W→C | C→W | IGNORE | READ_ONLY | WRITE_ONLY | FULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2,500 | 0.8600 | 0.8600 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| mmmu_pro_standard_test | 1,730 | 0.3688 | 0.3688 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| mmmu_pro_vision_test | 1,730 | 0.3370 | 0.3370 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| pope_adversarial | 3,000 | 0.8690 | 0.8690 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| pope_popular | 3,000 | 0.8773 | 0.8773 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| pope_random | 3,000 | 0.8893 | 0.8893 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| mmmu_pro | 3,460 | 0.3529 | 0.3529 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |
| pope | 9,000 | 0.8786 | 0.8786 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 0.00 | 0.00 | 0.00 | 28.00 |

## Interpretation boundary

These are deterministic complete four-action route executions from an Image+Question factorized predictor. The layer-action counts describe selected routes; they are not measured latency or memory savings.
