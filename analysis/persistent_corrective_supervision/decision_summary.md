# Persistent Corrective Supervision Decision

## Decision

**no supported architecture advantage; operationally prefer POLAR.** both are viable but the paired W2C difference interval includes zero.

## Confirmed observations

- POLAR selected epoch: 15; metrics: {'epoch': 15, 'train_boundary_valid_action_at_1': 0.546875, 'val_boundary_valid_action_at_1': 0.015625, 'val_boundary_nonfull_recall': 0.0546875, 'w2c_rescue': 0.0546875, 'c2c_preservation': 0.96875, 'rescues': 7, 'regressions': 4, 'net_accuracy_change': 0.01171875, 'exact_first_deviation': 0.03125, 'within_1_first_deviation': 0.0859375, 'within_2_first_deviation': 0.109375, 'early_deviation_fraction': 0.21875, 'late_or_no_deviation_fraction': 0.75, 'no_deviation_fraction': 0.5390625, 'teacher_forced_minus_free_rollout_leave_full_gap': -0.40625, 'ignore_fraction': 0.006696428571428571, 'read_only_fraction': 0.006975446428571429, 'write_only_fraction': 0.011439732142857142, 'full_fraction': 0.9748883928571429}.
- Online selected epoch: 14; metrics: {'epoch': 14, 'train_boundary_valid_action_at_1': 1.0, 'val_boundary_valid_action_at_1': 0.078125, 'val_boundary_nonfull_recall': 0.1484375, 'w2c_rescue': 0.046875, 'c2c_preservation': 0.953125, 'rescues': 6, 'regressions': 6, 'net_accuracy_change': 0.0, 'exact_first_deviation': 0.046875, 'within_1_first_deviation': 0.09375, 'within_2_first_deviation': 0.171875, 'early_deviation_fraction': 0.53125, 'late_or_no_deviation_fraction': 0.421875, 'no_deviation_fraction': 0.1953125, 'teacher_forced_minus_free_rollout_leave_full_gap': -0.65625, 'ignore_fraction': 0.021623883928571428, 'read_only_fraction': 0.013811383928571428, 'write_only_fraction': 0.023856026785714284, 'full_fraction': 0.9407087053571429}.
- Paired W2C bootstrap: {'records': 128, 'draws': 10000, 'seed': 20260830, 'right_minus_left': -0.0078125, 'ci_low': -0.0625, 'ci_high': 0.0390625}.
- POLAR Pareto frontier: [{'c2c_preservation_rate': 0.9921875, 'epoch': 11, 'w2c_rescue_rate': 0.0234375}, {'c2c_preservation_rate': 0.9765625, 'epoch': 12, 'w2c_rescue_rate': 0.0390625}, {'c2c_preservation_rate': 0.96875, 'epoch': 15, 'w2c_rescue_rate': 0.0546875}].
- Online Pareto frontier: [{'c2c_preservation_rate': 0.65625, 'epoch': 4, 'w2c_rescue_rate': 0.1640625}, {'c2c_preservation_rate': 0.921875, 'epoch': 12, 'w2c_rescue_rate': 0.1015625}, {'c2c_preservation_rate': 0.953125, 'epoch': 14, 'w2c_rescue_rate': 0.046875}, {'c2c_preservation_rate': 0.9765625, 'epoch': 19, 'w2c_rescue_rate': 0.0390625}, {'c2c_preservation_rate': 0.9765625, 'epoch': 20, 'w2c_rescue_rate': 0.0390625}].
- All 20 checkpoints for both substrates were evaluated on the same 256
  held-out records. No external evaluation ran.
- A direct all-FULL audit found one current-runtime mismatch in the frozen
  C2C cohort. Excluding that UID changes the C2C denominators to 127 but
  leaves selected epochs, W2C comparison, and the decision unchanged; see
  `runtime_cohort_sensitivity.md`.

## Interpretation boundary

The result compares these two fixed recipes under matched persistent
corrective supervision. It does not prove an architecture impossibility or
identify the underlying cause of any remaining generalization failure.

## Stop

The authorized matched action is complete. No scale-up, objective change,
external evaluation, or follow-up diagnostic is authorized.
