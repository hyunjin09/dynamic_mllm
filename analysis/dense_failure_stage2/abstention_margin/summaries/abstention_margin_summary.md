# Stage-2 abstention-margin calibration

1. The grid was frozen before development outcomes from positive raw margins on the final Experiment-A training epoch: `0`, q10/q25/q40/q50/q60/q70/q80/q90/q95 after deduplication, and `+inf`.
2. At δ=0, W→C/C→W/Net were **4 / 1 / +3**. At the selected margin they were **4 / 1 / +3**.
3. C→W did not fall faster than W→C over the selected move; selected development C→C preservation was **99.7500%** against a 99.5000% floor.
4. The best eligible development Net was **+3** and the selected margin is **0**.
5. Cross-fit chose the global margin in **5/5** folds; all folds identical: **True**.
6. Intervention rate remained **2.6250%** because the selected margin is δ=0. Across the positive grid it declined monotonically, reaching zero at q90/q95.
7. Median maximum executed non-FULL margins for W→C and C→W were **0.6747255325317383** and **0.7209988832473755** respectively. This is descriptive only.
8. Action-specific raw-margin distributions are in `diagnostics/margin_by_action.csv`; no action-conditioned rule was tuned. The largest executed-action count identifies whether one action dominates.
9. A nonzero global margin is **not supported** on Historical-800.
10. External status: **not_run_delta_zero_selected**.
11. A negative result would reject this one-dimensional post-hoc confidence gate; it would not prove that Stage-2 correction is impossible or identify a unique representation/training defect.

The reviewer caveat remains: final-epoch sampler weighting can make this fixed grid coarse. Development outcomes were not used to add thresholds.
