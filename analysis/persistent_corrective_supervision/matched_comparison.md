# Matched Persistent-Corrective Comparison

| Metric | POLAR | Online |
|---|---:|---:|
| train_boundary_valid_action_at_1 | 0.546875 | 1.000000 |
| val_boundary_valid_action_at_1 | 0.015625 | 0.078125 |
| val_boundary_nonfull_recall | 0.054688 | 0.148438 |
| w2c_rescue | 0.054688 | 0.046875 |
| c2c_preservation | 0.968750 | 0.953125 |
| net_accuracy_change | 0.011719 | 0.000000 |
| exact_first_deviation | 0.031250 | 0.046875 |
| within_1_first_deviation | 0.085938 | 0.093750 |
| within_2_first_deviation | 0.109375 | 0.171875 |
| early_deviation_fraction | 0.218750 | 0.531250 |
| late_or_no_deviation_fraction | 0.750000 | 0.421875 |
| no_deviation_fraction | 0.539062 | 0.195312 |
| teacher_forced_minus_free_rollout_leave_full_gap | -0.406250 | -0.656250 |
| full_fraction | 0.974888 | 0.940709 |
| read_only_fraction | 0.006975 | 0.013811 |
| write_only_fraction | 0.011440 | 0.023856 |
| ignore_fraction | 0.006696 | 0.021624 |

- POLAR selected epoch: 15; viable: True
- Online selected epoch: 14; viable: True
- POLAR Pareto frontier: [{'c2c_preservation_rate': 0.9921875, 'epoch': 11, 'w2c_rescue_rate': 0.0234375}, {'c2c_preservation_rate': 0.9765625, 'epoch': 12, 'w2c_rescue_rate': 0.0390625}, {'c2c_preservation_rate': 0.96875, 'epoch': 15, 'w2c_rescue_rate': 0.0546875}]
- Online Pareto frontier: [{'c2c_preservation_rate': 0.65625, 'epoch': 4, 'w2c_rescue_rate': 0.1640625}, {'c2c_preservation_rate': 0.921875, 'epoch': 12, 'w2c_rescue_rate': 0.1015625}, {'c2c_preservation_rate': 0.953125, 'epoch': 14, 'w2c_rescue_rate': 0.046875}, {'c2c_preservation_rate': 0.9765625, 'epoch': 19, 'w2c_rescue_rate': 0.0390625}, {'c2c_preservation_rate': 0.9765625, 'epoch': 20, 'w2c_rescue_rate': 0.0390625}]
- Paired bootstrap (online minus POLAR): {'records': 128, 'draws': 10000, 'seed': 20260830, 'right_minus_left': -0.0078125, 'ci_low': -0.0625, 'ci_high': 0.0390625}
- Decision: no supported architecture advantage; operationally prefer POLAR.
- Runtime-cohort sensitivity: one current all-FULL C2C mismatch; selection/decision invariant: True.
