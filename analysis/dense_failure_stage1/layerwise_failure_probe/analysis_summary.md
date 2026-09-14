# Layer-Wise Dense-Failure Predictability Analysis

## Contract and validity

- Frozen contract: `3cf49a46d0a47a0ae1955f8a518f5a74bef65b7de8736980a6e3d26e170234cd`
- Population: `7,999` current native-dense LMMS records (`3,999` correct / `4,000` wrong).
- Split: train `6,399`, validation `800`, test `800`; UID overlap `0`, image-group overlap `0`.
- Input at every layer: train-normalized concatenation of `text_final`, `text_mean`, and `visual_mean` (10,752 values). No dataset ID, answer, historical label, W→C, or route input was used.
- Twenty-eight independent linear probes used identical settings. Selective thresholds and the candidate region were fixed from validation before test evaluation; test was evaluated once.

## Test metrics

| Layer | AUROC | AUPRC | Balanced Acc |
|---:|---:|---:|---:|
| 0 | 0.8421 | 0.8519 | 0.7612 |
| 1 | 0.8522 | 0.8510 | 0.7788 |
| 2 | 0.8528 | 0.8549 | 0.7775 |
| 3 | 0.8537 | 0.8523 | 0.7875 |
| 4 | 0.8561 | 0.8571 | 0.7775 |
| 5 | 0.8518 | 0.8588 | 0.7750 |
| 6 | 0.8639 | 0.8578 | 0.7850 |
| 7 | 0.8538 | 0.8555 | 0.7825 |
| 8 | 0.8673 | 0.8677 | 0.7900 |
| 9 | 0.8636 | 0.8633 | 0.7900 |
| 10 | 0.8668 | 0.8687 | 0.7725 |
| 11 | 0.8669 | 0.8700 | 0.7875 |
| 12 | 0.8691 | 0.8706 | 0.7963 |
| 13 | 0.8718 | 0.8711 | 0.7987 |
| 14 | 0.8756 | 0.8836 | 0.8000 |
| 15 | 0.8759 | 0.8752 | 0.8013 |
| 16 | 0.8829 | 0.8876 | 0.7975 |
| 17 | 0.8836 | 0.8839 | 0.7975 |
| 18 | 0.8912 | 0.8945 | 0.8050 |
| 19 | 0.8945 | 0.8977 | 0.8137 |
| 20 | 0.8921 | 0.8909 | 0.8025 |
| 21 | 0.8992 | 0.8974 | 0.8100 |
| 22 | 0.8895 | 0.8908 | 0.7950 |
| 23 | 0.8884 | 0.8867 | 0.8000 |
| 24 | 0.8906 | 0.8899 | 0.8000 |
| 25 | 0.8871 | 0.8879 | 0.7900 |
| 26 | 0.8892 | 0.8891 | 0.8075 |
| 27 | 0.8919 | 0.8918 | 0.7937 |

## Answers to the plan questions

### Q1. At what depth does failure first become meaningfully predictable?

The validation-frozen descriptive region is layers **0-27**. At its first layer, test AUROC is `0.8421` and AUPRC is `0.8519`; mean test AUROC across the region is `0.8737`. Rationale frozen before test: Validation shows useful, neighboring-layer-stable failure signal from layer 0: overall AUROC 0.8660, AUPRC 0.8658, and wrong recall 0.2825 at 99% correct preservation; layer-0 dataset AUROCs are 0.7242/0.9677/0.9370 for GQA/ChartQA/TextVQA. Later layers improve modestly and layers 16-27 form a stronger plateau, but no joint AUROC/AUPRC/selective-recall/dataset-consistency evidence supports discarding layers 0-15.

### Q2. Is useful failure predictability present before answer emergence at layers 25-27?

Yes. The validation-selected region begins at layer 0, before layer 25. The strongest pre-25 test layer is 21 with AUROC `0.8992`.

### Q3. What wrong detection is possible at fixed dense-correct preservation?

Thresholds were chosen on validation and transferred unchanged to test. Because preservation can drift on held-out data, the table below reports the best wrong recall only among layers whose transferred threshold still met the stated preservation level on test. Full per-layer results—including layers that fell below the target—are in `layerwise_selective_metrics.csv` and `test_metrics.csv`.

| Validation preservation target | Best test wrong recall | Layer | Test correct preservation | Failure precision |
|---:|---:|---:|---:|---:|
| 99% | 0.3350 | 14 | 0.9950 | 0.9853 |
| 98% | 0.4275 | 19 | 0.9800 | 0.9553 |
| 95% | 0.5225 | 27 | 0.9500 | 0.9127 |

### Q4. Is failure-awareness depth consistent across datasets?

Dataset-wise held-out peaks are shown below; the full curves are in `dataset_layerwise_metrics.csv`.

| Dataset | Best test AUROC | Layer | AUROC at selected start |
|---|---:|---:|---:|
| gqa | 0.7713 | 21 | 0.6981 |
| chartqa | 0.9769 | 25 | 0.9392 |
| textvqa | 0.9833 | 20 | 0.9377 |

The onset is consistent: every dataset is already above chance at layer 0 and remains informative across the stack. The strength is not consistent—GQA is substantially harder (`0.6981` at layer 0; peak `0.7713`) than ChartQA and TextVQA, which are already near `0.94` at layer 0 and peak above `0.97`.

### Q5. What depth range is defensible for later Stage-1 supervision and gating?

Layers **0-27** are the validation-frozen informative range. Layers **16-27** form a stronger descriptive plateau, but the validation evidence does not justify withholding supervision from layers 0-15. This is evidence of regularized linear accessibility, not authorization to train the shared predictor and not proof that every layer is equally causal or useful for intervention.

### Q6. What should the next training experiment compare?

The originally proposed three-way comparison is not identifiable as written: informative-depth-only supervision on layers 0-27 is identical to all-layer supervision. A separately authorized next experiment should therefore prioritize **all-layer supervision versus random-k sampling across 0-27**. If a distinct depth-restricted sensitivity arm is desired, prospectively define layers 16-27 as the stronger-plateau arm; treat it as an efficiency/ablation test, not as evidence that early layers are uninformative. Keep the split, labels, compact feature definition, optimizer budget, and evaluation protocol fixed.

## Important interpretation limits

- The corrected answer-emergence curve is an unlearned final-head token readout at the true assistant answer position; the failure curve is a learned held-out decoder from compact states. Their depth ordering is informative, but the y-axes are not the same quantity.
- The concatenated probe does not attribute signal to text-final, text-mean, or visual-mean components individually.
- The test maximum is descriptive. Probe settings, thresholds, and the candidate region were not retuned on test.
- AUPRC and failure precision reflect this intentionally near-50:50 candidate population and should not be transferred to a natural deployment prevalence without recalibration.

Best overall held-out AUROC is `0.8992` at layer `21`.
