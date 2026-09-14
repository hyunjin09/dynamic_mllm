# Leave textvqa Out

- Source datasets: `gqa, chartqa`
- OOD target: `textvqa` (`2,000` rows)
- Source-validation-selected representative layer: `22`
- Representative OOD AUROC / AUPRC: `0.7236` / `0.7345`
- Native pre-language-decoder OOD AUROC / AUPRC: `0.5219` / `0.5473`
- Representative hidden-minus-input AUROC: `+0.2017`

| Source preservation target | Transferred threshold | OOD actual preservation | OOD wrong recall | OOD failure precision |
|---:|---:|---:|---:|---:|
| 99% | 0.905516 | 0.9950 | 0.0920 | 0.9485 |
| 98% | 0.889643 | 0.9930 | 0.1220 | 0.9457 |
| 95% | 0.835789 | 0.9690 | 0.2010 | 0.8664 |

The target dataset was not used for fitting, checkpoint selection, threshold calibration, probability calibration, or representative-layer selection.
