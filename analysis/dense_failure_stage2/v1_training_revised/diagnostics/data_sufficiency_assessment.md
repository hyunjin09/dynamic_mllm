# V1 data-sufficiency assessment

- Overfit smoke passed: **True**.
- Final-epoch teacher-forced non-FULL recall (train diagnostic): 0.0315.
- Free rollout W→C / C→W / net: 5 / 0 / 5.
- Triggered samples taking any non-FULL action: 0.2298.
- Diagnosis: **positive but narrow correction with near-FULL and immediate-intervention collapse**. The router makes 98.397% FULL post-trigger decisions, 96.30% of intervened samples act immediately at the trigger, and GQA receives no non-FULL decisions.
- Is adding the 209 MCTS-only W samples justified as the next candidate? **False**.

This action does not authorize V1.5. The +5 net corrections establish proof of direction without preservation harm, but not broad coverage. The training diagnostic is not a held-out action-label evaluation, so data diversity, sampler/loss imbalance, timing collapse, and exposure shift remain confounded.
