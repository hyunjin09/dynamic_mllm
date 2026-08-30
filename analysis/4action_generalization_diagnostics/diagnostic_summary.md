# Four-Action Generalization Diagnostic Summary

## Main failure decomposition

| Metric | POLAR Train | POLAR Val | Online Train | Online Val |
|---|---:|---:|---:|---:|
| KEEP vs DEVIATE AUROC | 0.915585 | 0.542877 | 0.994514 | 0.507751 |
| DEVIATE recall | 0.660156 | 0.054688 | 1.000000 | 0.148438 |
| Conditional Valid-Action@1 | 0.979290 | 0.285714 | 1.000000 | 0.526316 |
| READ_OFF AUROC | 0.930602 | 0.527860 | 0.995946 | 0.536952 |
| WRITE_OFF AUROC | 0.946790 | 0.587545 | 0.999415 | 0.469554 |
| READ_ONLY-only recall | 0.697674 | 0.060606 | 1.000000 | 0.000000 |
| WRITE_ONLY-only recall | 0.687500 | 0.000000 | 1.000000 | 0.156250 |
| IGNORE-only recall | 0.550388 | 0.000000 | 1.000000 | 0.000000 |

## Timing

| Metric | POLAR | Online |
|---|---:|---:|
| exact | 0.031250 | 0.046875 |
| within_1 | 0.085938 | 0.093750 |
| within_2 | 0.109375 | 0.171875 |
| too_early | 0.195312 | 0.476562 |
| too_late | 0.156250 | 0.156250 |
| never | 0.539062 | 0.195312 |
| rescue_given_within_2 | 0.000000 | 0.045455 |

## Representation usage

- Layer-only validation WHEN AUROC: 0.500000.
- Online joint-shuffle prediction unchanged: 0.835938.
- Online joint-shuffle WHEN AUROC drop: -0.029541.
- Online joint-shuffle READ_OFF AUROC drop: -0.013142.
- Online joint-shuffle WRITE_OFF AUROC drop: -0.079286.

## Diagnostic probes and label smoothness

- upfront: MLP WHEN AUROC 0.563141; k=10 WHEN purity 0.505859.
- online: MLP WHEN AUROC 0.666199; k=10 WHEN purity 0.555469.
- z_R: MLP WHEN AUROC 0.697815; k=10 WHEN purity 0.568359.
- z_W: MLP WHEN AUROC 0.682800; k=10 WHEN purity 0.556250.

## Bounded label-incompleteness audit

- Frozen cached-invalid validation states: 14; known-suffix route executions: 19.
- At least one supposedly invalid action executed correctly for 6/14 states
  (42.8571%): 3/5 POLAR and 3/9 online.
- By predicted action, bounded rescue was 3/5 IGNORE, 0/3 READ_ONLY, and 3/6
  WRITE_ONLY. This is positive evidence that the discovered valid-action cache
  is incomplete for this selected conditional subset.
- The other 8/14 states are `no_bounded_rescue`, not proof of global action
  invalidity. Full records are in `label_incompleteness_results.json`.

## Interpretation boundary

These measurements apply to the frozen Phase-39 subset, selected
checkpoints, exact state construction, and fixed probes. Probe success
does not authorize a new objective/head, and bounded audit failures do
not prove global action invalidity.
