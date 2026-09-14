# Independent Layer-Wise Sequential Gate Decision Summary

Frozen protocol: `6b1e4812a0a1e51f3dec822964b848a3a0410451dfe7a289d25c4b378653831d`.

| Validation target | Alpha | Validation preservation | Validation wrong recall | Test preservation | Test wrong recall | Test precision | Median test trigger |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 99% | 0.00000000 | 1.0000 | 0.4100 | 0.9575 | 0.4175 | 0.9076 | 3.0 |
| 98% | 0.00000000 | 1.0000 | 0.4100 | 0.9575 | 0.4175 | 0.9076 | 3.0 |
| 95% | 0.00250627 | 0.9675 | 0.4750 | 0.9550 | 0.4775 | 0.9139 | 2.0 |

## Q1. Can the 28 probes form a conservative sequential gate?

Not under the complete frozen 99% usefulness criterion. At least one of preservation, nontrivial wrong recall, or sufficiently early median triggering fails.

## Q2. Detection at the three preservation targets

The table above gives both validation-selected and untouched-test results. Counts are available in `validation_results.csv` and `test_results.csv`.

## Q3. Where do failures first trigger?

- 99% target: early `116`, middle `37`, late `14`, never `233` wrong samples.
- 98% target: early `116`, middle `37`, late `14`, never `233` wrong samples.
- 95% target: early `128`, middle `44`, late `19`, never `209` wrong samples.

## Q4. Does sequential gating outperform one fixed strong layer?

| Target | Sequential recall | Best fixed gate | Best fixed recall | Sequential gain |
|---:|---:|---|---:|---:|
| 99% | 0.4175 | layer_27 | 0.3975 | +0.0200 |
| 98% | 0.4175 | layer_21 | 0.4400 | -0.0225 |
| 95% | 0.4775 | layer_21 | 0.5275 | -0.0500 |

## Q5. Is the common calibration rule consistent across datasets?

- 99% target: dataset preservation spread `0.0250`, wrong-recall spread `0.7250`.
- 98% target: dataset preservation spread `0.0250`, wrong-recall spread `0.7250`.
- 95% target: dataset preservation spread `0.0300`, wrong-recall spread `0.6650`.

The frozen 0.10 spread warning is reached, so the shared quantile rule has a material dataset calibration mismatch. No dataset-specific threshold was added.

## Q6. Is the independent gate sufficient?

The independent gate is not sufficient as a robust final design under the frozen criteria. The observed limitation motivates—but does not authorize—a separately planned shared predictor or global risk-budget formulation.

## Scope limits

- This is an in-domain Phase-48 split analysis, not evidence that calibration transfers OOD.
- Test was evaluated once after all thresholds were frozen from validation.
- Triggering does not demonstrate that any treatment would improve the answer.
- No shared predictor, global risk-budget model, or intervention was trained or executed.
