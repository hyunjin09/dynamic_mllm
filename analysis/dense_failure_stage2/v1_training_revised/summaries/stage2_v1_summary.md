# Stage-2 V1 revised summary

The frozen 698-first Stage-2 V1 completed one final-update-selected training run and one 800-sample validation rollout.

| Metric | Value |
|---|---:|
| Dense validation accuracy | 0.500000 |
| Stage-1 + Stage-2 accuracy | 0.506250 |
| Accuracy change | +0.006250 |
| W→C | 5 |
| C→W | 0 |
| Net corrections | 5 |
| C→C preservation | 1.000000 |
| Triggered any non-FULL | 0.229787 |

Diagnosis: **positive but narrow correction with near-FULL and immediate-intervention collapse**. The five rescues incur no preservation harm, but 98.397% of post-trigger actions remain FULL, 96.30% of intervened samples act at the trigger, and no GQA validation sample receives a non-FULL action. No V1.5, test-set evaluation, or added complexity was executed.
