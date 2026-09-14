# Step-D full-internal refit summary

- Contract: `efa342f0a8c6dd5b3d38de5642de87d4b405208a70affac7d1242332e3e571cf`
- Completed refits: 15/15
- Training population: all internal states; no external labels or early stopping.
- Epochs: median-low selected epoch from the five corresponding Step-B folds, frozen before external scoring.
- Stage-1 states: 291,172; Stage-2 Dense states: 15,185.
- Internal-fit thresholds: numerical strict-greater values are in `stage1_internal_calibration_thresholds.csv`; Step-B OOF remains the unbiased ID reference.
