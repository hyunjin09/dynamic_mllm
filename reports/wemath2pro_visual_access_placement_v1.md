# WeMath2.0-Pro Visual-Access Placement Analysis

## Executive result

This read-only analysis passed every frozen-cache integrity gate and used all
raw valid routes rather than the capped predictor view. The primary cohort is
exactly **428 V+ samples** (FULL correct and exact
ALL-OFF wrong). Each sample contributes equal weight across all discovered
minimum-budget routes; `min+2` and `min+4` sets are frozen sensitivities.

Current classification: **Outcome D — route placement varies, but not with difficulty**.
Minimum-budget visual-access schedules are heterogeneous across V+ samples, but no stable aggregate-difficulty or difficulty-axis relationship survives minimum-ON adjustment, family-paired analysis, same-image control, or the exact-min/min+2/min+4 sensitivities. The only repeated transition-level centroid shift is base-to-x with eight families, below the predeclared non-tiny multi-transition axis standard. Near-min route expansion changes exact layer identities but leaves aggregate placement summaries and the null difficulty conclusion stable.

## Why this analysis was run

Phase 28 showed that difficulty does not robustly predict the *amount* of
positive direct visual access after separating ALL-OFF-solvable records. This
analysis asks the orthogonal question: with comparable ON budgets, do harder or
specific difficulty variants move direct visual access earlier, later, or into
separated re-access episodes?

## Sources, integrity, and semantics

- 4,544/4,544 authoritative raw records, 1,658,485 evaluated routes, and
  107,671 valid routes were checksum-verified and contract-validated.
- Exact FULL and ALL-OFF anchors reproduce phases 27/28; V+=428 and A+=1,263.
- `x = contextual complexity`, `y = visual complexity`, and `z = step
  complexity`, from the authors' released `DynamicScheduler` documentation.
- The source provides complete knowledge-point lists, but no independent
  categorical reasoning-type field. Knowledge-point count is descriptive only;
  categorical reasoning-type analysis is unavailable.

## Exact-minimum V+ placement by stratum

| stratum | n | min ON | centroid | last ON | late frac | segments | late reentry |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 104 | 14.42 | 12.82 | 25.65 | 0.300 | 5.80 | 0.812 |
| x | 23 | 13.74 | 13.32 | 25.85 | 0.330 | 5.92 | 0.826 |
| y | 84 | 13.39 | 12.67 | 25.72 | 0.288 | 6.34 | 0.881 |
| z | 82 | 14.40 | 12.82 | 25.31 | 0.301 | 6.16 | 0.833 |
| xy | 24 | 11.88 | 12.38 | 25.26 | 0.281 | 6.52 | 0.917 |
| xz | 20 | 12.30 | 13.16 | 25.66 | 0.328 | 6.28 | 0.950 |
| yz | 70 | 14.17 | 12.97 | 25.47 | 0.308 | 5.95 | 0.829 |
| xyz | 21 | 13.76 | 13.23 | 25.52 | 0.325 | 5.48 | 0.810 |

The profiles and summaries are sample-balanced: a sample with many discovered
minimum routes does not outweigh a sample with one route. `late` means layers
19--27; a late re-entry is a new ON segment beginning at layer 19 or later.

## Coarse degree summary and amount control

| degree | n | centroid | centroid CI | last ON | late frac | late reentry |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 104 | 0.475 | [0.461, 0.489] | 25.65 | 0.300 | 0.812 |
| 1 | 189 | 0.475 | [0.464, 0.485] | 25.56 | 0.299 | 0.853 |
| 2 | 114 | 0.477 | [0.463, 0.492] | 25.46 | 0.306 | 0.868 |
| 3 | 21 | 0.490 | [0.449, 0.531] | 25.52 | 0.325 | 0.810 |

The amount-adjusted tables fit normalized centroid, late fraction, and late
re-entry on categorical difficulty (and separately degree or each axis) plus
minimum positive ON. These are descriptive family-clustered models, not causal
effects. Predeclared ON-budget-bin tables retain counts and mark cells with
fewer than ten observations as sparse.

## Minimum versus near-minimum sensitivity

| route set | centroid | last ON | late frac | segments | late reentry |
| --- | --- | --- | --- | --- | --- |
| min+0 | 0.476 | 25.55 | 0.302 | 6.05 | 0.845 |
| min+2 | 0.481 | 25.72 | 0.303 | 6.23 | 0.856 |
| min+4 | 0.481 | 25.89 | 0.303 | 6.41 | 0.855 |

Patterns used for the scientific classification must survive all three route-set
definitions. Exact-min-only spikes are treated as fragile finite-search
structure.

## Same-family and same-image evidence

Across all supported V+ family transitions at exact minimum, the paired mean
normalized-centroid delta is **0.005**
(95% CI **[-0.009,
0.019]**), the paired late-fraction
delta is **0.007** (CI
**[-0.013, 0.027]**),
and the late-reentry delta is **0.018**
(CI **[-0.032,
0.067]**). Transition-specific and
same-image results are preserved in the CSVs rather than pooled as independent
sample-layer observations.

## Axis-specific results

| axis | meaning | metric | paired N | paired delta | paired CI | multi-transition agreement |
| --- | --- | --- | --- | --- | --- | --- |
| x | contextual complexity | normalized_centroid | 26 | 0.021 | [-0.012, 0.056] | False |
| x | contextual complexity | late_fraction | 26 | 0.042 | [-0.005, 0.088] | False |
| x | contextual complexity | has_late_reentry | 26 | 0.077 | [-0.074, 0.227] | False |
| y | visual complexity | normalized_centroid | 128 | 0.004 | [-0.016, 0.024] | False |
| y | visual complexity | late_fraction | 128 | -0.003 | [-0.029, 0.023] | False |
| y | visual complexity | has_late_reentry | 128 | 0.002 | [-0.068, 0.065] | False |
| z | step complexity | normalized_centroid | 120 | 0.004 | [-0.016, 0.023] | False |
| z | step complexity | late_fraction | 120 | 0.011 | [-0.019, 0.040] | False |
| z | step complexity | has_late_reentry | 120 | 0.023 | [-0.052, 0.095] | False |

An axis is called stable only if multiple non-tiny transitions agree, the
family-clustered paired aggregate supports the same direction, and the pattern
persists under min/min+2/min+4.

## Secondary A+ corrections

The A+ cohort contains 1263 FULL-wrong,
ALL-OFF-wrong samples with at least one positive correcting route. Its
minimum-budget correcting placement and per-layer profiles remain separate
because its search budget and estimand differ from V+ correctness preservation.

## Direct answers required by the plan

1. **Does difficulty change placement?** No stable global or axis-specific difficulty relationship survives the primary controls.
2. **Later/re-access heavy or earlier/front-loaded?** Schedules are heterogeneous across samples, but neither consistently later nor earlier with difficulty.
3. **After controlling minimum ON?** See the categorical/degree/axis models adjusted for minimum ON and the predeclared ON-budget bins; they do not rescue a stable relationship.
4. **Across min/min+2/min+4?** The median sample profile L1 change from exact-min to min+4 is 7.652. The classification is stable across all three sets.
5. **Same-family paired?** The aggregate family-paired normalized-centroid delta is 0.005 with CI [-0.009, 0.019].
6. **Same-image control?** The same-image aggregate uses 135 transitions; normalized-centroid delta is 0.004 with CI [-0.014, 0.021].
7. **Reproducible axes?** No axis satisfies all predeclared multi-transition, paired, amount-controlled, and sensitivity requirements.
8. **Reasoning-type metadata?** The official axes are contextual/visual/step complexity and knowledge-point lists are complete, but no independent categorical reasoning-type field exists; reasoning-type analysis is therefore unavailable.
9. **A+ placement?** A+ remains secondary: degree-3 minus degree-0 exact-min normalized centroid is 0.021, and late-fraction difference is 0.020; see the stratum/profile tables for nonmonotonicity.
10. **Schedule-router motivation?** No schedule router is motivated by difficulty alone; any future predictor would need other input properties and independent execution validation.

## Claim boundary

These are discovered finite-MCTS route schedules. They do not prove that any
specific layer is causally necessary. Late ON is direct visual access late in
the decoder; it does not by itself establish semantic re-grounding,
verification, or backtracking.

Outcome D — route placement varies, but not with difficulty
