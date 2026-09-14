# Leave gqa Out

- Source datasets: `chartqa, textvqa`
- OOD target: `gqa` (`4,000` rows)
- Source-validation-selected representative layer: `20`
- Representative OOD AUROC / AUPRC: `0.6160` / `0.5953`
- Native pre-language-decoder OOD AUROC / AUPRC: `0.5432` / `0.5334`
- Representative hidden-minus-input AUROC: `+0.0729`

| Source preservation target | Transferred threshold | OOD actual preservation | OOD wrong recall | OOD failure precision |
|---:|---:|---:|---:|---:|
| 99% | 0.871923 | 0.1380 | 0.9385 | 0.5212 |
| 98% | 0.839797 | 0.1240 | 0.9490 | 0.5200 |
| 95% | 0.480201 | 0.0420 | 0.9830 | 0.5064 |

The target dataset was not used for fitting, checkpoint selection, threshold calibration, probability calibration, or representative-layer selection.
