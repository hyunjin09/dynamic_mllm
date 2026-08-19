# Pareto-Filtered Binary Predictor: Training-Set Fitting Analysis

## Executive conclusion

Neither matched objective learned the Pareto-efficient training supervision.
The best exact train-set Pareto-route Hit@1 was **18.27% for duplicated BCE**
and **17.95% for exact valid-set NLL**, compared with a frozen weighted BCE
label-oracle Hit@1 of **73.92%** on the same 6,043 training inputs. At their
best train checkpoints, predictions remained about eight bits from the nearest
Pareto route on average.

The primary diagnosis is therefore **training-fit failure**, not primarily
held-out generalization failure. Two secondary effects coexist:

- a modest generalization gap grows late in training;
- residual multimodal/objective difficulty is severe, because exact Hit@1 is
  nearly zero for multi-route Pareto sets, but this is not the whole problem:
  even singleton Pareto sets reach only about 24% train Hit@1.

Exact set NLL does not materially fit the training labels better than
duplicated BCE. Falling loss and increasing mask diversity do not translate
into coherent complete-mask learning.

## Scope and evidence contract

- Benchmarks: GQA, TextVQA, and ChartQA.
- Frozen positive training set: 6,043 image-query inputs.
- Frozen positive validation set: 874 image-group-disjoint inputs.
- Supervision: one checksum-identical Pareto manifest for both objectives,
  capped prospectively at 50 routes before Pareto filtering.
- Pareto geometry: 9,905 routes over 6,917 positive train/validation inputs;
  mean 1.432 routes per positive.
- Training multiplicity: 4,459 singleton, 937 doubleton, and 647 sets with at
  least three routes.
- Validation multiplicity: 608 singleton, 173 doubleton, and 93 sets with at
  least three routes.
- Decoder: unchanged thresholded 28-bit factorized binary head.
- Checkpoints: every saved epoch 1–10 checkpoint from both completed runs.
- No training, optimizer update, label change, split change, or model-state
  mutation was performed for this analysis.

The original A6000 training histories are authoritative for validation. The
missing train metrics were reconstructed read-only on A4000 BF16 hardware.
Cross-hardware checks reproduced Pareto/original-valid Hit@1 exactly through
the checked early epochs, while small continuous-loss and threshold-count
differences appeared. We therefore stopped cross-hardware validation parity
tuning and used only the original A6000 validation logs. This uncertainty can
affect a few thresholded masks, but it cannot plausibly explain the roughly
56-percentage-point gap between learned train Hit@1 and the label oracle.

## All-epoch trajectories

“Online loss” is the loss accumulated while training that epoch. “Checkpoint
loss” is a read-only full-train-set evaluation of the saved end-of-epoch
checkpoint. Mask entropy is empirical top-1 mask entropy in nats.

### Duplicated BCE training trajectory

| Ep | Online loss | Checkpoint loss | Pareto Hit | Original Hit | Nearest Hamming | Mean ON | ALL-ON | ALL-OFF | Unique | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6231 | 0.5941 | 15.99% | 15.99% | 8.448 | 1.919 | 0.00% | 51.02% | 53 | 1.836 |
| 2 | 0.5920 | 0.5849 | 16.35% | 16.35% | 8.333 | 2.708 | 0.00% | 50.19% | 124 | 2.287 |
| 3 | 0.5867 | 0.5854 | 16.55% | 16.55% | 8.388 | 4.354 | 0.00% | 48.14% | 277 | 2.803 |
| 4 | 0.5807 | 0.5769 | **18.27%** | **18.27%** | 8.382 | 0.892 | 0.00% | 59.52% | 79 | 1.592 |
| 5 | 0.5749 | 0.5675 | 17.56% | 17.56% | 8.110 | 2.962 | 0.00% | 51.48% | 610 | 3.109 |
| 6 | 0.5668 | 0.5540 | 17.62% | 17.62% | 7.990 | 3.195 | 0.03% | 48.22% | 1,018 | 3.572 |
| 7 | 0.5569 | 0.5472 | 17.46% | 17.46% | 7.875 | 4.079 | 0.02% | 42.43% | 1,658 | 4.315 |
| 8 | 0.5463 | 0.5342 | 17.61% | 17.61% | 7.662 | 3.608 | 0.00% | 35.76% | 2,001 | 4.970 |
| 9 | 0.5360 | 0.5283 | 17.69% | 17.69% | 7.600 | 3.889 | 0.00% | 34.73% | 2,237 | 5.159 |
| 10 | 0.5311 | 0.5275 | 17.69% | 17.69% | 7.582 | 3.861 | 0.00% | 34.39% | 2,250 | 5.185 |

### Duplicated BCE train versus validation

The gap is train minus validation in percentage points. Validation values are
the frozen original A6000 metrics.

| Ep | Hit T/V (gap pp) | Hamming T/V | Mean ON T/V | ALL-ON T/V | Unique T/V |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.99% / 14.07% (+1.91) | 8.448 / 8.672 | 1.919 / 1.986 | 0.00% / 0.00% | 53 / 33 |
| 2 | 16.35% / 14.87% (+1.48) | 8.333 / 8.563 | 2.708 / 2.681 | 0.00% / 0.00% | 124 / 66 |
| 3 | 16.55% / 14.19% (+2.36) | 8.388 / 8.707 | 4.354 / 4.232 | 0.00% / 0.00% | 277 / 111 |
| 4 | 18.27% / 15.68% (+2.59) | 8.382 / 8.635 | 0.892 / 0.875 | 0.00% / 0.00% | 79 / 30 |
| 5 | 17.56% / 14.65% (+2.91) | 8.110 / 8.643 | 2.962 / 2.949 | 0.00% / 0.00% | 610 / 198 |
| 6 | 17.62% / 14.42% (+3.21) | 7.990 / 8.622 | 3.195 / 3.079 | 0.03% / 0.00% | 1,018 / 249 |
| 7 | 17.46% / 13.96% (+3.50) | 7.875 / 8.759 | 4.079 / 4.050 | 0.02% / 0.00% | 1,658 / 327 |
| 8 | 17.61% / 12.59% (+5.02) | 7.662 / 8.838 | 3.608 / 3.566 | 0.00% / 0.00% | 2,001 / 374 |
| 9 | 17.69% / 12.81% (+4.88) | 7.600 / 8.834 | 3.889 / 3.826 | 0.00% / 0.00% | 2,237 / 400 |
| 10 | 17.69% / 12.81% (+4.88) | 7.582 / 8.864 | 3.861 / 3.769 | 0.00% / 0.00% | 2,250 / 412 |

### Exact valid-set NLL training trajectory

| Ep | Online loss | Checkpoint loss | Pareto Hit | Original Hit | Nearest Hamming | Mean ON | ALL-ON | ALL-OFF | Unique | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.3490 | 16.6803 | 15.99% | 15.99% | 8.495 | 1.054 | 0.00% | 54.56% | 21 | 1.360 |
| 2 | 16.6242 | 16.3602 | 16.45% | 16.45% | 8.343 | 2.511 | 0.00% | 50.67% | 81 | 2.084 |
| 3 | 16.5051 | 16.2976 | 15.70% | 15.70% | 8.340 | 2.475 | 0.00% | 47.48% | 96 | 2.307 |
| 4 | 16.3222 | 16.2366 | 17.52% | 17.52% | 8.417 | 0.783 | 0.00% | 55.12% | 30 | 1.324 |
| 5 | 16.2623 | 16.0396 | 17.14% | 17.14% | 8.243 | 2.529 | 0.00% | 52.62% | 228 | 2.527 |
| 6 | 16.0338 | 15.7457 | 17.57% | 17.57% | 8.189 | 2.002 | 0.00% | 53.14% | 373 | 2.504 |
| 7 | 15.7560 | 15.4992 | 17.92% | 17.92% | 8.061 | 3.055 | 0.00% | 50.92% | 776 | 3.322 |
| 8 | 15.5271 | 15.2680 | 17.84% | 17.84% | 7.955 | 3.299 | 0.00% | 47.01% | 1,123 | 3.815 |
| 9 | 15.3127 | 15.1376 | **17.95%** | **17.95%** | 7.912 | 3.211 | 0.00% | 46.90% | 1,258 | 3.909 |
| 10 | 15.1801 | 15.1125 | 17.86% | 17.86% | 7.902 | 3.480 | 0.00% | 44.60% | 1,412 | 4.142 |

### Exact valid-set NLL train versus validation

| Ep | Hit T/V (gap pp) | Hamming T/V | Mean ON T/V | ALL-ON T/V | Unique T/V |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.99% / 14.07% (+1.91) | 8.495 / 8.669 | 1.054 / 1.102 | 0.00% / 0.00% | 21 / 16 |
| 2 | 16.45% / 14.99% (+1.46) | 8.343 / 8.547 | 2.511 / 2.525 | 0.00% / 0.00% | 81 / 46 |
| 3 | 15.70% / 14.07% (+1.63) | 8.340 / 8.546 | 2.475 / 2.469 | 0.00% / 0.00% | 96 / 43 |
| 4 | 17.52% / 15.56% (+1.96) | 8.417 / 8.616 | 0.783 / 0.762 | 0.00% / 0.00% | 30 / 12 |
| 5 | 17.14% / 14.99% (+2.16) | 8.243 / 8.572 | 2.529 / 2.505 | 0.00% / 0.00% | 228 / 97 |
| 6 | 17.57% / 14.42% (+3.16) | 8.189 / 8.588 | 2.002 / 1.912 | 0.00% / 0.00% | 373 / 116 |
| 7 | 17.92% / 14.76% (+3.16) | 8.061 / 8.654 | 3.055 / 2.978 | 0.00% / 0.00% | 776 / 200 |
| 8 | 17.84% / 14.30% (+3.54) | 7.955 / 8.704 | 3.299 / 3.149 | 0.00% / 0.00% | 1,123 / 257 |
| 9 | 17.95% / 14.30% (+3.65) | 7.912 / 8.697 | 3.211 / 3.055 | 0.00% / 0.00% | 1,258 / 278 |
| 10 | 17.86% / 13.96% (+3.90) | 7.902 / 8.744 | 3.480 / 3.317 | 0.00% / 0.00% | 1,412 / 284 |

## Did the models memorize the Pareto targets?

At the objective-specific best-train and final checkpoints:

| Objective/checkpoint | Exact Pareto | Exact original valid | Outside Pareto | Nearest Pareto Hamming |
|---|---:|---:|---:|---:|
| BCE epoch 4 (best train) | 18.27% | 18.27% | 81.73% | 8.382 |
| BCE epoch 10 | 17.69% | 17.69% | 82.31% | 7.582 |
| NLL epoch 9 (best train) | 17.95% | 17.95% | 82.05% | 7.912 |
| NLL epoch 10 | 17.86% | 17.86% | 82.14% | 7.902 |

Pareto Hit@1 and original-valid Hit@1 are identical at every epoch for both
objectives. Thus none of the extra hits come from learning a dominated route
that remained in the original cache. Conversely, a cache miss is not proof
that an executed route would be behaviorally wrong; it is evidence that the
network did not reproduce the cached training supervision.

### BCE predictor versus the exact Pareto BCE label oracle

The ideal weighted bit-marginal BCE target on the frozen train set has:

- Pareto Hit@1: 73.92%;
- nearest-Pareto Hamming: 1.413;
- mean ON layers: 9.729;
- ALL-OFF: 19.31%;
- ALL-ON: 0.61%.

| BCE comparison | Predictor–oracle exact agreement | Predictor–oracle Hamming | Predictor mean ON | Predictor Pareto Hit |
|---|---:|---:|---:|---:|
| Epoch 4 | 18.19% | 9.337 | 0.892 | 18.27% |
| Epoch 10 | 17.66% | 8.384 | 3.861 | 17.69% |

The oracle target itself is imperfect but substantially coherent. The trained
BCE network does not approach it. Therefore BCE's failure here is principally
network/optimization/input-fit failure, not evidence that the ideal Pareto BCE
target necessarily remains unusable.

## Pareto-set multiplicity

This split is the strongest discriminator between general fit failure and
residual multimodal supervision. Values below are train/validation.

| Objective/checkpoint | Multiplicity | n train/val | Hit@1 T/V | Hamming T/V | Mean ON T/V |
|---|---|---:|---:|---:|---:|
| BCE epoch 4 | 1 | 4,459 / 608 | 24.65% / 22.53% | 7.783 / 8.061 | 0.82 / 0.80 |
| BCE epoch 4 | 2 | 937 / 173 | 0.43% / 0.00% | 10.096 / 9.861 | 1.14 / 1.01 |
| BCE epoch 4 | ≥3 | 647 / 93 | 0.15% / 0.00% | 10.031 / 10.108 | 1.01 / 1.15 |
| BCE epoch 10 | 1 | 4,459 / 608 | 23.93% / 18.42% | 7.009 / 8.405 | 3.63 / 3.60 |
| BCE epoch 10 | 2 | 937 / 173 | 0.21% / 0.00% | 9.270 / 9.879 | 4.72 / 4.04 |
| BCE epoch 10 | ≥3 | 647 / 93 | 0.00% / 0.00% | 9.087 / 9.978 | 4.24 / 4.37 |
| NLL epoch 9 | 1 | 4,459 / 608 | 24.27% / 20.56% | 7.394 / 8.206 | 2.98 / 2.92 |
| NLL epoch 9 | 2 | 937 / 173 | 0.32% / 0.00% | 9.420 / 9.717 | 4.01 / 3.35 |
| NLL epoch 9 | ≥3 | 647 / 93 | 0.00% / 0.00% | 9.301 / 10.011 | 3.62 / 3.39 |
| NLL epoch 10 | 1 | 4,459 / 608 | 24.13% / 20.07% | 7.391 / 8.255 | 3.23 / 3.14 |
| NLL epoch 10 | 2 | 937 / 173 | 0.32% / 0.00% | 9.398 / 9.751 | 4.32 / 3.66 |
| NLL epoch 10 | ≥3 | 647 / 93 | 0.00% / 0.00% | 9.253 / 10.065 | 3.96 / 3.81 |

The multi-route results support a **residual multimodal-objective/factorization
failure**: neither BCE nor NLL selects a cached mode. But singleton results
also fail badly. A singleton presents one unambiguous complete target, yet
roughly three quarters of train samples still miss it. Multimodality therefore
cannot be the primary remaining explanation.

## FULL-status groups

Group A is FULL-wrong with a correcting route. Group B is FULL-correct with a
cheaper route. Group C is FULL-correct with no cheaper route, so FULL should be
retained. The validation C group contains only two examples and is descriptive
only.

| Objective/checkpoint | Group | n train/val | Hit@1 T/V | Hamming T/V | Mean ON T/V | ALL-ON T/V |
|---|---|---:|---:|---:|---:|---:|
| BCE epoch 4 | A | 2,506 / 366 | 6.74% / 6.28% | 10.090 / 10.191 | 0.99 / 0.98 | 0% / 0% |
| BCE epoch 4 | B | 3,501 / 506 | 26.71% / 22.53% | 6.984 / 7.437 | 0.81 / 0.80 | 0% / 0% |
| BCE epoch 4 | C | 36 / 2 | 0% / 0% | 25.500 / 27.000 | 2.50 / 1.00 | 0% / 0% |
| BCE epoch 10 | A | 2,506 / 366 | 6.38% / 5.19% | 9.097 / 10.292 | 4.83 / 4.63 | 0% / 0% |
| BCE epoch 10 | B | 3,501 / 506 | 25.96% / 18.38% | 6.430 / 7.785 | 3.07 / 3.13 | 0% / 0% |
| BCE epoch 10 | C | 36 / 2 | 0% / 0% | 14.222 / 20.500 | 13.78 / 7.50 | 0% / 0% |
| NLL epoch 9 | A | 2,506 / 366 | 6.54% / 5.46% | 9.552 / 10.210 | 3.94 / 3.73 | 0% / 0% |
| NLL epoch 9 | B | 3,501 / 506 | 26.31% / 20.75% | 6.642 / 7.542 | 2.61 / 2.56 | 0% / 0% |
| NLL epoch 9 | C | 36 / 2 | 0% / 0% | 17.278 / 24.000 | 10.72 / 4.00 | 0% / 0% |
| NLL epoch 10 | A | 2,506 / 366 | 6.38% / 5.46% | 9.557 / 10.249 | 4.29 / 4.08 | 0% / 0% |
| NLL epoch 10 | B | 3,501 / 506 | 26.25% / 20.16% | 6.627 / 7.595 | 2.82 / 2.76 | 0% / 0% |
| NLL epoch 10 | C | 36 / 2 | 0% / 0% | 16.611 / 24.000 | 11.39 / 4.00 | 0% / 0% |

Neither objective fits correction programs for Group A. Both fit only about a
quarter of the cheaper correctness-preserving routes in Group B. Neither keeps
Group C at FULL: ALL-ON is zero at every reported checkpoint.

## Probability diagnostics

### BCE sigmoid probabilities

| Checkpoint | Mean margin from 0.5 | Bits in [0.45,0.55] | p > 0.9 | p < 0.1 | Singleton bit accuracy | Singleton target p | Singleton confidently correct bits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epoch 4 | 0.2295 | 5.27% | 0.00% | 10.28% | 72.20% | 0.6377 | 11.85% |
| Epoch 10 | 0.2157 | 12.32% | 0.14% | 15.74% | 74.97% | 0.6639 | 20.09% |

Even on singleton targets, the network learns many individual bits but not the
complete 28-bit mask. Epoch-10 singleton bit accuracy is 74.97%, yet singleton
exact-mask Hit@1 is only 23.93%. This is consistent with compound complete-mask
failure, not a complete absence of bit-level signal.

### NLL Pareto-set probability mass

| Checkpoint | Mean best-route probability | Mean total Pareto-set probability | Mean weighted Pareto mass | Singleton bit accuracy | Singleton target p |
|---|---:|---:|---:|---:|---:|
| Epoch 4 (best validation) | 0.00855 | 0.00855 | 0.00855 | 72.06% | 0.6339 |
| Epoch 9 (best train Hit) | 0.05889 | 0.05889 | 0.05888 | 73.59% | 0.6527 |
| Epoch 10 | 0.06373 | 0.06373 | 0.06372 | 73.60% | 0.6543 |

NLL does increase train Pareto-set mass while loss falls, but the thresholded
top-1 mask remains outside the Pareto set for about 82% of training inputs.
Validation probability-mass diagnostics were not logged originally and were
not reconstructed after cross-hardware continuous-loss parity proved
unavailable; the frozen validation route metrics above remain authoritative.

## Collapse and diversity

- **ALL-ON collapse does not occur on train or validation.** NLL is exactly 0%
  ALL-ON at every epoch. BCE reaches only 0.03% at epoch 6 and 0.02% at epoch 7.
- The models do not “escape” ALL-ON because they never enter it after Pareto
  filtering.
- There is instead a strong **ALL-OFF concentration**: 34–60% for BCE and
  45–55% for NLL over the trajectory.
- Later training partially escapes ALL-OFF and increases unique masks and
  entropy. BCE goes from 53 to 2,250 train masks; NLL goes from 21 to 1,412.
- This diversity is not successful learning. Hit@1 changes only 15.99% →
  17.69% for BCE and 15.99% → 17.86% for NLL.

Thus Pareto filtering removes the earlier ALL-ON shortcut but does not produce
coherent training-route fitting. It replaces that visible collapse with a
mixture of ALL-OFF concentration and diverse mostly non-Pareto masks.

## Epoch selection and overfitting

| Objective | Best train Hit | Best val Hit | Lowest online train loss | Lowest val loss | Final |
|---|---:|---:|---:|---:|---:|
| BCE | epoch 4 (18.27%) | epoch 4 (15.68%) | epoch 10 (0.5311) | epoch 2 (~0.5952) | epoch 10 |
| NLL | epoch 9 (17.95%) | epoch 4 (15.56%) | epoch 10 (15.1801) | epoch 3 (~16.6697) | epoch 10 |

BCE validation loss begins worsening after epoch 2 and NLL after epoch 3,
while train losses continue falling through epoch 10. Validation Hit@1 peaks
at epoch 4 for both. The late train-validation Hit gap grows to 4.88 points for
BCE and 3.90 points for NLL. This is real secondary overfitting, but train
performance never becomes strong, so it is not Case B generalization failure.

## Final diagnosis

### Duplicated BCE

- **Training-fit failure: strongly supported.** Best train Hit@1 is 18.27%;
  final is 17.69%; nearest-route Hamming remains 7.58–8.38.
- **Generalization failure: secondary.** The Hit gap grows from 1.91 to 4.88
  points, but validation is not the main bottleneck because train fit is poor.
- **Residual multimodal-objective failure: supported as an additional issue.**
  Doubleton/three-plus train Hit is effectively zero, but singleton Hit is also
  only ~24%.
- **Successful fit and generalization: rejected.**

### Exact valid-set NLL

- **Training-fit failure: strongly supported.** Best train Hit@1 is 17.95%;
  final is 17.86%; nearest-route Hamming remains 7.90.
- **Generalization failure: secondary.** The final Hit gap is 3.90 points, but
  train fit is already poor.
- **Residual multimodal-objective/factorization failure: supported as an
  additional issue.** Complete-set likelihood does not recover any meaningful
  doubleton/three-plus Hit@1, and singleton fitting remains weak.
- **Successful fit and generalization: rejected.**

## Direct answers

1. **Did Pareto BCE fit the training labels?** No. Its best train Pareto Hit@1
   is 18.27%, versus the 73.92% BCE label oracle; 81.73% of best-checkpoint
   predictions miss every cached Pareto route.
2. **Did Pareto NLL fit the training labels?** No. Its best train Hit@1 is
   17.95%; exact set likelihood raises probability mass but does not yield
   coherent thresholded masks.
3. **Which objective fits train better?** Neither meaningfully. BCE has the
   slightly higher best Hit@1 (18.27% versus 17.95%) and better final nearest
   Hamming (7.58 versus 7.90); NLL has no practical fitting advantage.
4. **What is the train→validation gap?** At the best validation checkpoint
   (epoch 4), BCE is 18.27% versus 15.68% (+2.59 points) and NLL is 17.52%
   versus 15.56% (+1.96 points). At epoch 10, the gaps are +4.88 and +3.90
   points respectively.
5. **What is the remaining bottleneck?** Primarily underfitting of coherent
   complete masks under the current predictor/optimization/input pipeline.
   Residual multimodality/factorization is severe for multi-route sets, and
   late generalization degradation exists, but neither explains the poor
   singleton training fit.
6. **Does this justify architecture, data, or objective changes?** It does not
   justify adding more data: more examples do not address failure to fit the
   existing 6,043. It does not justify another loss-only change: matched BCE
   and exact set NLL fail similarly. It motivates locating the current
   capacity/optimization/input-representation fitting bottleneck before any
   held-out scaling, but these results alone do not distinguish predictor
   architecture from optimization well enough to authorize a specific redesign.

## Artifacts

- `outputs/binary_pareto_v1/training_fit_analysis_v1/bce_training_fit_v1.json`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/nll_training_fit_v1.json`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/bce_epoch_trajectory_v1.csv`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/nll_epoch_trajectory_v1.csv`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/multiplicity_metrics_v1.csv`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/supervision_group_metrics_v1.csv`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/probability_metrics_v1.csv`
- `outputs/binary_pareto_v1/training_fit_analysis_v1/training_fit_analysis_manifest.json`
