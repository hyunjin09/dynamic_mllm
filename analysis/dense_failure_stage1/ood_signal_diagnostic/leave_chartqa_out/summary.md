# Leave chartqa Out

- Source datasets: `gqa, textvqa`
- OOD target: `chartqa` (`1,999` rows)
- Source-validation-selected representative layer: `26`
- Representative OOD AUROC / AUPRC: `0.4502` / `0.5026`
- Native pre-language-decoder OOD AUROC / AUPRC: `0.5472` / `0.5368`
- Representative hidden-minus-input AUROC: `-0.0970`

| Source preservation target | Transferred threshold | OOD actual preservation | OOD wrong recall | OOD failure precision |
|---:|---:|---:|---:|---:|
| 99% | 0.947070 | 0.3053 | 0.5840 | 0.4570 |
| 98% | 0.932262 | 0.2392 | 0.6540 | 0.4625 |
| 95% | 0.898526 | 0.1552 | 0.7640 | 0.4751 |

The target dataset was not used for fitting, checkpoint selection, threshold calibration, probability calibration, or representative-layer selection.
