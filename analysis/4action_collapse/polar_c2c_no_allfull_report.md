# POLAR C2C Exact-All-FULL Removal Ablation

- Config: `analysis/4action_collapse/polar_c2c_no_allfull_config.yaml`
- Config SHA-256: `3b210ae321ba3ca384a5a9b9511d1a3753d543467b9905a0d586daf9885bcbc4`
- Output: `outputs/four_action_collapse/polar_c2c_no_allfull_v1`
- Selected epoch: 4
- Objective: exact-set NLL
- Removed exact all-FULL routes from training C2C only: 3,501
- Excluded route-empty training C2C samples: 35
- Validation labels changed: 0/866

## Route prediction

- Overall top-1 valid-route coverage: 0.585450
- Overall top-5 valid-route coverage: 0.616628
- Nearest-valid Hamming distance: 1.466513
- Predicted exact all-FULL fraction: 1.000000
- Unique predicted routes: 1
- W2C top-1 valid-route coverage: 0.000000
- C2C top-1 valid-route coverage: 0.994118

Actual unified-executor routed accuracy is reported separately after
executing the validation-selected checkpoint.

## Actual unified-executor validation

- Records: 866
- W2C correct-route execution rate: 0.000000
- C2C correct-route execution rate: 1.000000
- Overall routed accuracy: 0.588915
- Predicted exact all-FULL fraction: 1.000000
- Unique executed routes: 1
- Mean IGNORE/READ_ONLY/WRITE_ONLY/FULL layers: 0.000/0.000/0.000/28.000
