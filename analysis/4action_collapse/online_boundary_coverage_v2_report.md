# Online Four-Action Guaranteed-Boundary Training

- Config: `analysis/4action_collapse/online_boundary_coverage_v2_config.yaml`
- Config SHA-256: `567d5acc530f6a4b30a818febbed5bc76202ea52d0973c2be8f61dec1b4abe03`
- Output: `outputs/four_action_collapse/online_boundary_coverage_v2`
- Epochs completed: 10
- Selected epoch: 2

## Primary routed behavior

- W2C free-rollout rescue: 0.000000
- C2C preservation: 1.000000
- Overall routed accuracy: 0.588915
- Mean FULL layers: 27.995381

## Mandatory-boundary behavior

- Validation boundary Valid-Action@1: 0.0
- Validation boundary non-FULL recall: 0.0
- Validation free rollout left all-FULL: 0.0

Every training W2C sample received exactly one scheduled visit to its
latest all-FULL-prefix mandatory-deviation boundary. All remaining
visits retained the original deterministic valid-route sampler.
