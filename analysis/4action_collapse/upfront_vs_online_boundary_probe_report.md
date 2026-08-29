# Upfront vs Online Mandatory-Boundary Probe

## Frozen comparison

- Config SHA-256: `5bae5406c45ea21a7e4963b0f44e4ad3ec2db57e620d433fcb301130e321c80a`
- Probe records: 5168 (4504 train, 664 validation)
- Unique feature UIDs: 2746
- Matching: `exact_split_dataset_target_layer_without_replacement`
- Model parameters per probe: 955,777
- Upfront state: unified-FULL pre-layer-0 final text/control row plus mean visual row, with target-layer identity.
- Online state: unified-FULL pre-target-layer final text/control row plus mean visual row, with target-layer identity.

## Validation results

| Representation | Best epoch | AUROC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Upfront | 10 | 0.576372 | 0.566265 | 0.539936 |
| Online | 20 | 0.575097 | 0.563253 | 0.615385 |

## Paired primary comparison

- Online-minus-upfront AUROC: -0.001275
- UID-group bootstrap 95% CI: [-0.054773, 0.053437]
- Valid bootstrap draws: 2000/2000
- Frozen decision rule: lower 95% CI > 0.
- Are mandatory deviations more predictable from current routed state? **NO**

This probe is representational evidence. It does not by itself establish that a free-running router will use the signal correctly.
