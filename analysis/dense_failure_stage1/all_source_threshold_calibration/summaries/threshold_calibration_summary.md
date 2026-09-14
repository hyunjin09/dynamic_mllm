# ALL-source robust threshold calibration summary

Calibration contract: `30288d8a23d2fac0a2d468d40d1c9c0dd5ef0a4797ebf6a19c66d4b454262a38`. The selection population is Historical validation (800) plus Canonical OOF (4,000); Historical test is untouched by selection.

## Decision

**A — Freeze robust Stage-1 gate.** The primary threshold is `0.97113474103143993` under strict `score > tau`.

- Worst-source C preservation: 0.9808 (Historical 0.9875; Canonical 0.9808).
- W recall: Historical 0.0650; Canonical 0.1240; pooled 0.1054.
- Trigger precision: 0.6734; median first-trigger layer: 23.0.
- Untouched Historical test: C preservation 0.9975, W recall 0.0650, trigger precision 0.9630.

## Reference operating points

| Constraint | Threshold | Hist C preserve | Canon C preserve | Hist W recall | Canon W recall | Pooled W recall | Precision |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 99% | 0.983709 | 0.9975 | 0.9901 | 0.0250 | 0.0597 | 0.0488 | 0.6596 |
| 98% | 0.971039 | 0.9875 | 0.9802 | 0.0650 | 0.1240 | 0.1054 | 0.6667 |
| 95% | 0.944802 | 0.9775 | 0.9501 | 0.1525 | 0.2480 | 0.2179 | 0.6267 |


## Stability and cell safety

- Cross-fit thresholds: mean 0.970227, median 0.970619, IQR 0.002545, range [0.968590, 0.972199]; fixed stability result: **True**.
- Minimum held-out Canonical-fold C preservation: 0.9728.
- Catastrophic adequately supported cells: 0.
- Canonical ChartQA: C preservation 0.9842, W recall 0.1034, precision 0.4615.
- Canonical TextVQA (N_W=19): W recall 0.0000, Wilson 95% CI [0.0000, 0.1682].

## Old-gate context and trigger timing

- The old threshold `0.925384` and the new threshold are evaluated on identical ALL-head trajectories in `metrics/old_vs_new_gate_comparison.csv`.
- Canonical Dense-C first-layer-0 false triggers under these ALL-head scores: old threshold 2; new threshold 0. This comparison concerns threshold behavior on the repaired ALL head, not reuse of the stale Historical-only head.
- Full early/middle/late distributions for true-trigger W and false-trigger C are in `metrics/trigger_depth_breakdown.csv`; threshold timing was not optimized.

## Answers to the plan questions

1. The complete empirical Pareto sweep is `metrics/full_threshold_sweep.csv` and the main preservation-vs-recall view is `figures/preservation_vs_wrong_recall.png`.
2. The 99/98/95 thresholds and metrics are listed above and frozen in `metrics/reference_operating_points.csv`.
3. Historical and Canonical W recall at each reference point are reported above.
4. The primary 98%-constraint threshold is `0.97113474103143993`.
5. Cross-fit stability is **True** under the prospective criteria in `protocol.md`.
6. Source-specific preservation/recall differences and group-bootstrap intervals are in `metrics/bootstrap_intervals.csv`.
7. 0 adequately supported dataset/source cell(s) are catastrophic at the primary point.
8. Canonical ChartQA safety is acceptable under the >=90% cell rule.
9. The ALL-head layer-0 canonical false-trigger count changes from 2 at the old threshold to 0 at the new threshold.
10. The old/new same-population comparison is frozen in `metrics/old_vs_new_gate_comparison.csv`.
