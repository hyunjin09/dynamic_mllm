# Shared Stage-1 Global Risk Gate Decision Summary

Frozen contract: `264a9407e5acb35e19bd4b53436ae8bf1175ad989f8c974cf7d8e859565e095b`.
Validation-designated primary: `state_layer_random4` with `all_0_27`. Test results did not change this selection.

| Gate | Target | Test preservation | Wrong recall | Precision | Median trigger |
|---|---:|---:|---:|---:|---:|
| Independent sequential | 99% | 0.9575 | 0.4175 | 0.9076 | 3.0 |
| Shared All-28 | 99% | 0.9925 | 0.3925 | 0.9812 | 11.5 |
| Shared Random-4 | 99% | 0.9875 | 0.4050 | 0.9701 | 5.0 |
| Independent sequential | 98% | 0.9575 | 0.4175 | 0.9076 | 3.0 |
| Shared All-28 | 98% | 0.9825 | 0.4275 | 0.9607 | 4.0 |
| Shared Random-4 | 98% | 0.9775 | 0.4300 | 0.9503 | 2.0 |
| Independent sequential | 95% | 0.9550 | 0.4775 | 0.9139 | 2.0 |
| Shared All-28 | 95% | 0.9625 | 0.4800 | 0.9275 | 1.0 |
| Shared Random-4 | 95% | 0.9525 | 0.4575 | 0.9059 | 0.0 |

## Q1. Can the shared predictor match the independent probes?

Mean test layer AUROC is `0.8769` for the primary shared predictor versus `0.8737` for the 28 independent probes. Validation state+layer mean layer AUROC is `0.8803`, state-only is `0.8804`, and layer-only is `0.5000`.

## Q2. All-28 versus Random-4

Validation selected `state_layer_random4` under the frozen 99%-risk rule. Both schemes were scored once on test as predeclared confirmatory arms; the selection was not switched post hoc.

## Q3. Global threshold stability

At the primary 99% point, validation/test preservation is `0.9900` / `0.9875` and validation/test wrong recall is `0.3950` / `0.4050`.

## Q4. Validation-to-test preservation drift

Primary absolute preservation drift at 99% is `0.0025`, versus Phase-50 drift `0.0425`.

## Q5. Dataset calibration spread

Primary shared 99% test preservation/wrong-recall spreads are `0.0300` / `0.7600`; Phase-50 values are `0.0250` / `0.7250`.

## Q6. Sequential versus a shared fixed strong layer

Validation selected shared fixed layer `27` among L14/L21/L27. On test its preservation/recall is `0.9825` / `0.4200`, versus shared sequential `0.9875` / `0.4050`.

## Q7. Gate window

Validation selected `all_0_27` for All-28 and `all_0_27` for Random-4. The late window was a validation sensitivity only and was not separately opened on test.

## Q8. Ready for Stage-1 admission?

`NO` under the frozen 99%-risk readiness checks: test_preservation=pass, wrong_recall=pass, preservation_drift=pass, dataset_preservation_spread=fail.

## Scope limits

- Test was scored in one aggregate pass after all checkpoints, windows, and thresholds were frozen.
- State-only and layer-only remained validation controls and received no new test scores.
- No OOD follow-up, treatment, W-to-C repair, four-action routing, MCTS, or external evaluation ran.
