# Mandatory-Boundary Overfit Pilot

- Config: `analysis/4action_collapse/mandatory_boundary_overfit_config.yaml`
- Config SHA-256: `0ccf117c902283714156aa01976ef26b64521eeeb4a50423989dc0df9d98ff5b`
- Output: `outputs/four_action_collapse/mandatory_boundary_overfit_v1`
- Completed epoch: 30
- Best evaluated epoch: 30
- Prospective gate passed: **True**

## Question 1

Can the unchanged online router recognize a mandatory deviation state when explicitly trained on it?

Answer: **YES**

## Best-checkpoint evidence

- Boundary Valid-Action@1: 0.958333
- Boundary non-FULL recall: 0.958333
- Singleton action recall: `{"IGNORE": 0.9583333333333334, "READ_ONLY": 1.0, "WRITE_ONLY": 0.9166666666666666}`
- Free rollout left all-FULL: 1.000000
- W2C rescue rate: 0.895833
- C2C preservation rate: 0.916667

The full online A2 retrain is authorized only when every frozen gate passes.
