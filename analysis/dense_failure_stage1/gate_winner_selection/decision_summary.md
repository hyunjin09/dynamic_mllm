# Stage-1 gate winner and threshold selection

Contract: `f065d3728ebce2c95667a566e09ab49d5a4ead901b0334e5e7566df0f2288135`

## Validation selection

| Candidate | Correct preservation | Wrong recall | Failure precision | Utility rate | Median trigger |
|---|---:|---:|---:|---:|---:|
| Shared Fixed-L27 | 0.9400 | 0.5400 | 0.9000 | 0.2400 | 27.0000 |
| Shared Random-4 Sequential | 0.9425 | 0.5300 | 0.9021 | 0.2362 | 0.0000 |
| Independent Sequential | 0.9475 | 0.5150 | 0.9075 | 0.2313 | 4.0000 |
| Shared All-28 Sequential | 0.9625 | 0.4900 | 0.9289 | 0.2263 | 2.0000 |

The frozen validation winner is **shared_fixed_l27**, using `{"control_type": "global_raw_threshold", "layers": [27], "threshold": 0.8497647428417646}`. It maximized validation utility rate subject to failure precision >= 90%. The frozen runner-up is **shared_random4**.

## Selection-held-out test confirmation

| Candidate | Correct preservation | Wrong recall | Failure precision | Utility rate | Median trigger |
|---|---:|---:|---:|---:|---:|
| independent_sequential | 0.9450 | 0.5325 | 0.9064 | 0.2387 | 4.0000 |
| shared_all28 | 0.9650 | 0.4700 | 0.9307 | 0.2175 | 1.0000 |
| shared_random4 | 0.9400 | 0.5100 | 0.8947 | 0.2250 | 0.0000 |
| shared_fixed_l27 | 0.9500 | 0.5275 | 0.9134 | 0.2387 | 27.0000 |

The winner's test preservation was 0.9500, wrong recall 0.5275, precision 0.9134, and utility rate 0.2387. Its validation-to-test utility drift was -0.0013.

Against the validation runner-up, the paired 95% bootstrap interval for test utility-rate difference was [-0.0037, 0.0312] (observed 0.0138); for wrong-recall difference it was [-0.0125, 0.0475] (observed 0.0175).

The strongest sequential result on test was the independent gate at utility rate 0.2387, exactly matching fixed L27; shared Random-4 and shared All-28 reached 0.2250 and 0.2175. Sequential depth therefore provides no aggregate utility advantage under the frozen validation-selected controls. Independent sequential triggers much earlier (median layer 4 versus layer 27), but the current utility does not reward latency and supplies no basis to override the simpler fixed gate.

## Winner by dataset on test

| Dataset | Correct preservation | Wrong recall | Failure precision | Utility rate |
|---|---:|---:|---:|---:|
| gqa | 0.9200 | 0.2350 | 0.7460 | 0.0775 |
| chartqa | 0.9900 | 0.7800 | 0.9873 | 0.3850 |
| textvqa | 0.9700 | 0.8600 | 0.9663 | 0.4150 |

## Final questions

1. **Winner and threshold:** `shared_fixed_l27` with the frozen control above.
2. **Test confirmation:** preservation 0.9500, wrong recall 0.5275, precision 0.9134, utility 0.2387.
3. **Calibration stability:** validation-to-test utility drift was -0.0013; the complete comparison is in `test_comparison.csv`.
4. **Sequential versus fixed:** winner utility 0.2387; fixed-L27 utility 0.2387.
5. **Dataset dominance:** yes, performance is strongly task-dependent. Fixed L27 test utility is 0.0775 on GQA versus 0.3850 on ChartQA and 0.4150 on TextVQA; GQA failure precision is only 0.7460. The aggregate winner's validation advantage over shared Random-4 is also driven by a GQA gain that is partly offset on ChartQA.
6. **Bootstrap uncertainty:** utility and recall intervals are reported above and frozen in `bootstrap_comparison.json`.
7. **Carry-forward:** only `shared_fixed_l27` and its frozen validation-selected control should be used in the next separately authorized treatment experiment. No treatment experiment was run here.

## Qualification

This is selection-held-out within Phase 52, not historically unopened: Phase-50/51 test trajectories already existed. Additionally, five leading Phase-51 test rows were printed during format inspection before the winner freeze, without computing any aggregate or using them for selection. This limits the strictness of the holdout claim but does not alter the deterministic validation-only winner computation.

An independent read-only review reproduced the frozen ranking and found no blocking implementation or selection flaw. It noted that fixed L27 had already been selected over fixed L14/L21 on the same validation split in Phase 51. The result therefore supports the predeclared Phase-52 carry-forward decision, but it is not an unbiased general comparison of all possible fixed layers. The three-sample validation lead and bootstrap intervals spanning zero also make the architectural conclusion “sequential is not justified here,” not “fixed L27 is universally superior.”
