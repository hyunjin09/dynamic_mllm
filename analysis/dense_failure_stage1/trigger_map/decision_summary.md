# Stage-1 Trigger Map Decision Summary

Frozen audit contract: `d9dd591abcc3d56fb1caf56f57815e8a5749ebe631896f095a8118e3ea5653a7`. The gate is Shared Random-4 with strict `score > 0.92538392806010306` over layers 0-27.

| Split | Dense-C retained | Dense-W detected | Trigger precision | C trigger | W trigger |
|---|---:|---:|---:|---:|---:|
| train | 0.9878 | 0.5878 | 0.9797 | 39 | 1881 |
| val | 0.9425 | 0.5300 | 0.9021 | 23 | 212 |
| test | 0.9400 | 0.5100 | 0.8947 | 24 | 204 |

## Required answers

1. Dense-C retention without triggering is train 98.78%, val 94.25%, test 94.00%.
2. Dense-W detection is train 58.78%, val 53.00%, test 51.00%.
3. The triggered population is Dense-W at rates train 97.97%, val 90.21%, test 89.47%.
4. Modal Dense-W first-trigger layers are train [0] (981 samples each), val [0] (117 samples each), test [0] (126 samples each).
5. Modal false-admission Dense-C first-trigger layers are train [25] (12 samples each), val [9] (3 samples each), test [0] (4 samples each).
6. The cumulative curves separate as follows: train maximum W-minus-C gap 0.5756 at layer 27 (final gap 0.5756); val maximum W-minus-C gap 0.4725 at layer 26 (final gap 0.4725); test maximum W-minus-C gap 0.4500 at layer 27 (final gap 0.4500).
7. Dataset behavior is materially nonuniform: train W recall gqa=0.268/chartqa=0.901/textvqa=0.915; val W recall gqa=0.180/chartqa=0.880/textvqa=0.880; test W recall gqa=0.185/chartqa=0.800/textvqa=0.870. No per-dataset calibration was applied.
8. Future train corrective suffix search contains exactly **1881** triggered Dense-W samples.
9. Future train preservation supervision contains exactly **39** triggered Dense-C samples, whose default suffix is FULL.
10. Yes. Validation and test counts, preservation, recall, and precision match all 18 checked Phase-52 Shared Random-4 aggregate fields exactly; all 1,600 Phase-53 trigger rows also match exactly.

## Provenance and boundaries

The deterministic 24-record stored-feature check passed with maximum absolute probability error `2.81e-07` at tolerance `1e-06`; 292/672 layer scores were bit-exact, and every first-trigger decision matched.
Validation/test trajectories came from saved Phase-51 scores. Train trajectories were generated only by the frozen lightweight gate from hash-verified stored dense features on four direct GPUs. No Qwen or four-action execution occurred.
The map describes who triggers and when. It does not show that early triggers are better, that any triggered failure is fixable, or that Stage 2 improves final accuracy.
Stopped at the plan boundary: no corrective suffix search, Stage-2 labels/training, threshold changes, persistence/EMA, W-to-C repair, or external evaluation ran.
