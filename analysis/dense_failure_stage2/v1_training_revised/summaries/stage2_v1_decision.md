# Stage-2 V1 decision

1. Implementation smoke passed: **True**.
2. Small-overfit smoke passed: **True**.
3. Avoided the prospectively defined strict FULL-collapse rule: **True**, but the endpoint is still **near-FULL** (98.397% of post-trigger actions are FULL).
4. Learned all supported non-FULL actions at the frozen overfit gate: **True**.
5. Validation W→C / C→W: **5 / 0**.
6. C→C preservation: **1.000000** (400 preserved).
7. Net validation accuracy change: **+0.006250** (+5 samples).
8. It did **not** learn the intended wait behavior reliably: mean trigger-to-first-non-FULL delay is **0.3148**, and 96.30% of samples that intervene do so immediately at the trigger. Post-trigger FULL fraction is **0.983970**.
9. Teacher-forced-to-rollout transfer: final train non-FULL recall **0.031519** versus triggered any-non-FULL **0.229787**. This is descriptive because no clean held-out action-label set exists.
10. Dominant diagnosis: **positive but narrow correction with near-FULL and immediate-intervention collapse**. No validation GQA sample received a non-FULL action; all five rescues are ChartQA/TextVQA.
11. Is the 698-sample V1 sufficient to continue the method direction? **Yes only as proof of direction** (net +5, zero C harm), not as evidence of broad corrective coverage.
12. Is V1.5 with 209 MCTS-only W automatically justified? **No**; it was not run. The next plan must distinguish data diversity from sampling/loss/timing and exposure-shift limitations.
