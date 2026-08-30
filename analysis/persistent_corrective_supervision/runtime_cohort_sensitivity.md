# Runtime Cohort Sensitivity

## Direct observation

- POLAR epoch 1 executed all-FULL for all 256 frozen validation records.
- Current-runtime membership mismatches: 1.
- `textvqa:textvqa_train_11748` (C2C): frozen membership expected correct=true, while current all-FULL execution produced `no` and correct=false.

The frozen 256-record analysis remains primary. The following sensitivity
excludes only the mismatched UID from its frozen cohort and reapplies the
unchanged C2C >= 95% checkpoint-selection rule across all 20 epochs.

| Architecture | Selected epoch | W2C rescue | C2C preservation |
|---|---:|---:|---:|
| POLAR | 15 | 7/128 (0.054688) | 124/127 (0.976378) |
| Online | 14 | 6/128 (0.046875) | 122/127 (0.960630) |

- W2C paired population changed: false.
- Selected checkpoints and matched architecture decision invariant: true.
- This check establishes robustness to the observed cohort mismatch; it
  does not diagnose why the current runtime differs from the frozen record.
