# Full10 Best-Checkpoint External Evaluation

## Integrity

- Status: **PASS**; all 22,307 frozen active UIDs completed exactly once.
- DocVQA was excluded prospectively. Core VQA, multiple choice, and POPE are not pooled into one overall metric.
- The checkpoints were selected on internal validation before these external outcomes were evaluated.
- The scientific baseline is current live ALL-ON execution. The historical bundle cache is audit-only because 485/22,307 predictions/scores/correctness tuples differed under the current runtime, including 168 correctness labels.
- Reported compute is visual-ON decoder-layer count, not measured wall-clock acceleration.

## Route behavior

- Question-only selected ALL-ON for 22,263/22,307 records and a non-ALL-ON mask for 44. None changed the generated prediction, benchmark score, or correctness relative to current live ALL-ON.
- Image+Question selected ALL-ON for 22,307/22,307 records. It therefore exactly reproduced current live ALL-ON throughout.
- Consequently, neither selected checkpoint produced a behavioral improvement or meaningful visual-layer compute reduction on this external evaluation.

### Question

| Population | N | FULL correct | Predicted correct | Delta (95% clustered CI) | W→C | C→W | Mean ON | ON reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2,500 | 0.8600 | 0.8600 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 27.98 | 0.06% |
| textvqa | 5,000 | 0.8546 | 0.8546 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmstar_val | 1,500 | 0.6200 | 0.6200 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_val | 847 | 0.5277 | 0.5277 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_pro_standard_test | 1,730 | 0.3688 | 0.3688 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_pro_vision_test | 1,730 | 0.3370 | 0.3370 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_adversarial | 3,000 | 0.8690 | 0.8690 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_popular | 3,000 | 0.8773 | 0.8773 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_random | 3,000 | 0.8893 | 0.8893 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| core_vqa | 7,500 | 0.8564 | 0.8564 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 27.99 | 0.02% |
| external_multiple_choice | 5,807 | 0.4474 | 0.4474 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope | 9,000 | 0.8786 | 0.8786 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_image_disjoint | 8,982 | 0.8783 | 0.8783 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |

### Image Question

| Population | N | FULL correct | Predicted correct | Delta (95% clustered CI) | W→C | C→W | Mean ON | ON reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 2,500 | 0.8600 | 0.8600 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| textvqa | 5,000 | 0.8546 | 0.8546 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmstar_val | 1,500 | 0.6200 | 0.6200 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_val | 847 | 0.5277 | 0.5277 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_pro_standard_test | 1,730 | 0.3688 | 0.3688 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| mmmu_pro_vision_test | 1,730 | 0.3370 | 0.3370 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_adversarial | 3,000 | 0.8690 | 0.8690 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_popular | 3,000 | 0.8773 | 0.8773 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_random | 3,000 | 0.8893 | 0.8893 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| core_vqa | 7,500 | 0.8564 | 0.8564 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| external_multiple_choice | 5,807 | 0.4474 | 0.4474 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope | 9,000 | 0.8786 | 0.8786 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |
| pope_image_disjoint | 8,982 | 0.8783 | 0.8783 | +0.0000 [+0.0000, +0.0000] | 0 | 0 | 28.00 | 0.00% |

## Interpretation boundary

These are deterministic static-mask executions of validation-selected factorized predictors. External correctness changes and visual-ON counts describe their behavior; they do not by themselves establish deployable latency gains or causal routing mechanisms.
