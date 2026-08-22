# WeMath2.0-Pro Visual-Dependence Reanalysis

## Executive conclusion

The previous negative difficulty–minimum-ON result was primarily a change in
the prevalence of **ALL-OFF-solvable samples**, not a stable difference in the
number of positive visual-access layers required among samples for which
ALL-OFF fails.

All **4,544** authoritative raw records passed checksum and exact-anchor
verification. Every record contained one exact 28-zero ALL-OFF route and one
exact 28-one FULL route. The raw anchor results matched the checksum-bound
per-sample index with zero discrepancies.

Among the **841 FULL-correct** records:

- **413 (49.1%)** are V0: FULL correct and ALL-OFF correct;
- **428 (50.9%)** are V+: FULL correct and ALL-OFF wrong.

V0 prevalence rises strongly with coarse difficulty, from **32.5%** at degree
0 to **73.4%** at degree 3. Once V0 is removed, mean minimum positive ON is
nearly flat: **14.42, 13.87, 13.36, and 13.76** for degrees 0–3, with medians
**12, 12, 11.5, and 12**. The original Spearman correlation of **-0.225**
(95% family-cluster CI **[-0.289, -0.159]**) becomes **-0.057** (95% CI
**[-0.154, 0.037]**) in V+.

A symmetric two-component decomposition attributes **83.7%, 85.4%, and
94.9%** of the degree 1, 2, and 3 mean declines relative to degree 0 to the
changing V0/V+ composition. The family-paired V+ aggregate is also null: mean
delta **-0.04 layers**, median **0**, 95% CI **[-0.63, 0.57]** across 274
transition occurrences.

The result is therefore **Outcome A — mostly ALL-OFF composition**.

> The apparent minimum-ON trend was primarily caused by changing prevalence
> of samples solvable without direct visual access.

The original `x` association remains defensible as a **visual-dependence
composition** observation: 70.9% of FULL-correct `x`-containing records are V0,
versus 36.9% without `x`. Evidence for an additional x-specific positive
visual-budget effect is weak and inconsistent after conditioning.

## 1. Corrected executor interpretation

The frozen binary action means:

- VISUAL_ON: visual rows execute the layer and text/control rows can attend to
  visual K/V;
- VISUAL_OFF: visual rows bypass unchanged, while text/control rows execute
  without visual tokens and cannot attend to visual K/V at that layer.

Thus ALL-OFF is qualitatively distinct from an ordinary sparse visual route:
visual states never execute a decoder layer, and no decoder layer lets text
directly read encoded visual K/V. Structural side channels such as placeholder
tokens, retained positions, or visual-token count/layout may remain, so this
report uses “no direct visual access,” not “no image whatsoever.”

For FULL-correct samples:

```text
V0 = FULL correct and ALL-OFF correct
V+ = FULL correct and ALL-OFF wrong
```

For V+, at least one VISUAL_ON layer is behaviorally necessary under the
tested executor and exact cached routes. The minimum positive ON count remains
finite-search-dependent and is not a causal requirement outside this route
space.

## 2. Frozen-cache integrity and exact anchors

This was a read-only analysis. It used the same completed hard-cap-400 raw
MCTS cache, manifest, scorer, and thresholds as the previous report. It did not
execute Qwen, generate a route, train a predictor, apply max-50 truncation, or
change correctness.

Verified evidence:

| Integrity item | Result |
|---|---:|
| Eligible raw records | 4,544 |
| Evaluated raw routes | 1,658,485 |
| Exact ALL-OFF anchors | 4,544 |
| Exact FULL anchors | 4,544 |
| Exact ALL-OFF correct | 575 |
| Exact FULL correct | 841 |
| Anchor/index mismatches | 0 |
| V0 iff FULL-correct minimum ON = 0 | Exact |

The authoritative raw-record index SHA-256 is
`7e4ed586a58efcb5150db939964dce695e1f6df3315e81059be629b6ad54dd8a11`.
The ordered raw-record hash rollup is
`8951a5db2f7a170a1f58aee30ecd50421d9f09e5534fa82e38669e94a2623bbf`.
All generated-output and source hashes are frozen in
`outputs/wemath2pro_visual_dependence_reanalysis_v1/analysis_manifest.json`.

## 3. V0/V+ decomposition across all eight strata

The eight official difficulty strata are shown before any axis aggregation.

| Stratum | Eligible | FULL correct | V0 | V0 % (95% family CI) | V+ | V+ % |
|---|---:|---:|---:|---:|---:|---:|
| base | 568 | 154 | 50 | 32.5% [25.3, 40.3] | 104 | 67.5% |
| x | 571 | 78 | 55 | 70.5% [60.3, 80.8] | 23 | 29.5% |
| y | 571 | 129 | 45 | 34.9% [27.1, 43.4] | 84 | 65.1% |
| z | 569 | 135 | 53 | 39.3% [31.1, 47.4] | 82 | 60.7% |
| xy | 569 | 71 | 47 | 66.2% [54.9, 77.5] | 24 | 33.8% |
| xz | 567 | 74 | 54 | 73.0% [63.0, 82.7] | 20 | 27.0% |
| yz | 563 | 121 | 51 | 42.1% [33.1, 51.2] | 70 | 57.9% |
| xyz | 566 | 79 | 58 | 73.4% [63.3, 82.3] | 21 | 26.6% |

The coarse degree aggregation is:

| Degree | FULL correct | V0 | V0 % | V+ | V+ % |
|---:|---:|---:|---:|---:|---:|
| 0 | 154 | 50 | 32.5% | 104 | 67.5% |
| 1 | 342 | 153 | 44.7% | 189 | 55.3% |
| 2 | 266 | 152 | 57.1% | 114 | 42.9% |
| 3 | 79 | 58 | 73.4% | 21 | 26.6% |

The pooled V0 percentage is not a trivial constant. It tracks both coarse
degree and difficulty type. In particular, `x`, `xy`, `xz`, and `xyz` have
roughly two-thirds to three-quarters V0, while `base`, `y`, `z`, and `yz` have
roughly one-third to two-fifths V0.

## 4. Exact decomposition of the previous mean

For every group, the following identity was verified to numerical precision:

```text
E[min ON | FULL correct]
= P(V+ | FULL correct) × E[min positive ON | V+].
```

Maximum reconstruction error across all strata and degrees was below
`1e-15`.

| Stratum | Original mean | V0 % | V+ mean positive ON | Reconstructed mean | Composition share of change vs base |
|---|---:|---:|---:|---:|---:|
| base | 9.74 | 32.5% | 14.42 | 9.74 | reference |
| x | 4.05 | 70.5% | 13.74 | 4.05 | 94.2% |
| y | 8.72 | 34.9% | 13.39 | 8.72 | 33.0% |
| z | 8.75 | 39.3% | 14.40 | 8.75 | 98.7% |
| xy | 4.01 | 66.2% | 11.88 | 4.01 | 77.5% |
| xz | 3.32 | 73.0% | 12.30 | 3.32 | 84.4% |
| yz | 8.20 | 42.1% | 14.17 | 8.20 | 89.8% |
| xyz | 3.66 | 73.4% | 13.76 | 3.66 | 94.9% |

The reported shares use a symmetric two-factor decomposition of the difference
from `base` into:

```text
change in V+ prevalence × average conditional mean
+
change in V+ conditional mean × average prevalence.
```

At coarse degree level:

| Degree | Original mean | V0 % | V+ mean | Difference vs degree 0 | V0-composition component | Conditional-budget component | Composition share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.74 | 32.5% | 14.42 | 0.00 | 0.00 | 0.00 | reference |
| 1 | 7.67 | 44.7% | 13.87 | -2.07 | -1.74 | -0.34 | 83.7% |
| 2 | 5.73 | 57.1% | 13.36 | -4.01 | -3.43 | -0.59 | 85.4% |
| 3 | 3.66 | 73.4% | 13.76 | -6.08 | -5.77 | -0.31 | 94.9% |

Therefore the original monotonic decline is mathematically dominated by the
growing zero mass. The `y` stratum is the main exception at the individual
level: its small one-layer difference from base is more conditional than
compositional. That exception does not create a coherent global or paired
difficulty trend.

## 5. V+-only minimum positive VISUAL_ON budget

This is the corrected primary visual-budget analysis.

| Stratum | V+ N | Mean (95% family CI) | Median | SD | P10 | Q25 | Q75 | P90 | Cheaper than FULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 104 | 14.42 [13.16, 15.77] | 12 | 6.84 | 8.3 | 9.75 | 16.25 | 28.0 | 82.7% |
| x | 23 | 13.74 [11.04, 16.74] | 11 | 7.25 | 8.0 | 9.0 | 15.0 | 28.0 | 82.6% |
| y | 84 | 13.39 [12.20, 14.71] | 11.5 | 5.79 | 8.0 | 10.0 | 15.0 | 25.0 | 89.3% |
| z | 82 | 14.40 [13.06, 15.79] | 13 | 6.47 | 8.0 | 9.25 | 17.0 | 28.0 | 86.6% |
| xy | 24 | 11.88 [10.08, 14.04] | 10 | 5.05 | 7.3 | 8.0 | 14.25 | 16.7 | 95.8% |
| xz | 20 | 12.30 [10.45, 14.50] | 11 | 4.78 | 7.9 | 9.0 | 14.0 | 16.0 | 95.0% |
| yz | 70 | 14.17 [12.70, 15.70] | 12 | 6.58 | 8.0 | 9.0 | 17.0 | 28.0 | 85.7% |
| xyz | 21 | 13.76 [11.05, 16.86] | 12 | 7.10 | 7.0 | 9.0 | 16.0 | 28.0 | 85.7% |

Coarse-degree summaries are much more stable than in the mixed cohort:

| Degree | V+ N | Mean (95% CI) | Median | Q25–Q75 | Mean removable direct-visual layers |
|---:|---:|---:|---:|---:|---:|
| 0 | 104 | 14.42 [13.16, 15.77] | 12 | 9.75–16.25 | 13.58 |
| 1 | 189 | 13.87 [12.86, 14.95] | 12 | 9–16 | 14.13 |
| 2 | 114 | 13.36 [12.31, 14.50] | 11.5 | 9–15 | 14.64 |
| 3 | 21 | 13.76 [10.95, 16.81] | 12 | 9–16 | 14.24 |

Even when direct vision is behaviorally necessary, most V+ records retain
substantial discovered redundancy: 82.7–89.5% by degree have a valid positive
route cheaper than FULL. This is a redundancy observation within the frozen
search, not a statement that the removed layers are universally unnecessary.

## 6. V+-only budget-feasibility curves

The mixed-cohort degree curves previously separated dramatically. Conditional
on V+, they largely overlap.

| Degree | C=4 | C=8 | C=12 | C=16 | C=20 | C=24 | C=28 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 10.6% | 54.8% | 75.0% | 82.7% | 82.7% | 100% |
| 1 | 0.0% | 14.3% | 52.4% | 78.3% | 85.7% | 87.3% | 100% |
| 2 | 0.0% | 15.8% | 55.3% | 79.8% | 88.6% | 89.5% | 100% |
| 3 | 0.0% | 23.8% | 52.4% | 76.2% | 85.7% | 85.7% | 100% |

The degree-3 C=8 estimate is based on only 21 V+ records and has a wide 95%
CI of 4.8–42.9%. At C=12 the four estimates are essentially identical
(52.4–55.3%), and at C=16 they span only 75.0–79.8%.

The eight-stratum curves retain some descriptive variation, especially for
the small `xy` and `xz` V+ subsets, but no monotonic ordering exists and their
uncertainty is substantial.

## 7. Original versus conditional trend

| Cohort | N | Spearman rho | Family-clustered 95% CI | Degree means | Degree medians |
|---|---:|---:|---:|---|---|
| All FULL-correct | 841 | -0.225 | [-0.289, -0.159] | 9.74, 7.67, 5.73, 3.66 | 10, 8, 0, 0 |
| V+ only | 428 | -0.057 | [-0.154, 0.037] | 14.42, 13.87, 13.36, 13.76 | 12, 12, 11.5, 12 |

This is Pattern 1 from the plan: the relationship largely disappears after V0
removal. It neither remains substantially negative nor reverses positive.

## 8. Same-family and same-image reanalysis

The previous mixed-cohort paired audit reproduced exactly: 677 transition
occurrences, mean delta **-0.821**, and median **0**.

After requiring both endpoints to be V+:

- usable families: **117**;
- transition occurrences: **274**;
- increase: **39.4%**;
- equal: **15.0%**;
- decrease: **45.6%**;
- mean delta: **-0.04 layers**;
- median delta: **0**;
- family-bootstrap 95% CI: **[-0.63, 0.57]**.

The aggregate conditional paired result is null. Most individual transition
intervals cross zero. Two small transitions are negative with intervals below
zero: `x→xz` (N=8, mean -3.63) and `y→xy` (N=7, mean -5.14); their same-image
subsets have N=5 and N=7. These are not a stable axis-wide pattern:
`base→x` has only eight pairs and crosses zero, `xy→xyz` crosses zero, and
`xz→xyz` points positive with a wide interval. Treating the two small cells as
a confirmed axis effect would be post-hoc overinterpretation.

## 9. Full FULL/ALL-OFF behavioral context

Across all 4,544 eligible records, the exact 2×2 table is:

| FULL | ALL-OFF | N | Eligible % | Interpretation |
|---|---|---:|---:|---|
| correct | correct | 413 | 9.1% | V0 |
| correct | wrong | 428 | 9.4% | V+ |
| wrong | correct | 162 | 3.6% | A0 |
| wrong | wrong | 3,541 | 77.9% | A+ candidate or no correction |

V+ population prevalence, rather than only its proportion among FULL-correct
survivors, also falls in `x`-containing strata. As a percentage of every
eligible stratum, FULL-correct/ALL-OFF-wrong is 18.3% for `base`, 14.7% for
`y`, 14.4% for `z`, and 12.4% for `yz`, but only 3.5–4.2% for `x`, `xy`, `xz`,
and `xyz`. This reinforces the survivorship warning: the conditional V+
samples are a small subset of the hardest/x-related populations.

## 10. FULL-wrong A0/A+ decomposition

Among the 3,703 FULL-wrong records:

- **A0:** 162 (4.4%) are corrected by ALL-OFF itself;
- **A+:** 1,263 (34.1%) remain wrong under ALL-OFF but have a correcting route
  with at least one ON layer;
- **no correction:** 2,278 (61.5%) have no correcting route in the frozen
  search.

| Stratum | FULL wrong | A0 N (%) | A+ N (%) | No correction N (%) | A+ mean min positive ON | A+ median |
|---|---:|---:|---:|---:|---:|---:|
| base | 414 | 25 (6.0%) | 182 (44.0%) | 207 (50.0%) | 10.55 | 10 |
| x | 493 | 19 (3.9%) | 134 (27.2%) | 340 (69.0%) | 10.34 | 10 |
| y | 442 | 24 (5.4%) | 201 (45.5%) | 217 (49.1%) | 10.65 | 10 |
| z | 434 | 33 (7.6%) | 175 (40.3%) | 226 (52.1%) | 10.70 | 10 |
| xy | 498 | 14 (2.8%) | 128 (25.7%) | 356 (71.5%) | 10.52 | 10 |
| xz | 493 | 18 (3.7%) | 127 (25.8%) | 348 (70.6%) | 10.68 | 10 |
| yz | 442 | 22 (5.0%) | 193 (43.7%) | 227 (51.4%) | 10.65 | 10 |
| xyz | 487 | 7 (1.4%) | 123 (25.3%) | 357 (73.3%) | 10.49 | 10 |

Most discovered corrections require nonzero visual access, but 162 of the
previous 1,425 Group-A corrections—**11.4%**—are actually ALL-OFF corrections.
The A+ minimum positive budget is strikingly stable across strata, with means
10.34–10.70 and median 10 everywhere. What changes strongly is whether a
correction is found at all, not the positive ON budget conditional on finding
one.

## 11. What remains of the previous x-associated result?

The robust observation is compositional:

- x absent: V0 is **36.9%** of 539 FULL-correct records;
- x present: V0 is **70.9%** of 302 FULL-correct records.

By comparison, y presence changes V0 only from 48.1% to 50.3%, and z presence
from 45.6% to 52.8%.

Among V+ records, x-present mean positive ON is 12.91 versus 14.11 when x is
absent, a much smaller 1.20-layer descriptive difference. It is not supported
as a coherent family-paired axis effect: the aggregate paired result is null,
and individual x-related transitions are sparse and inconsistent. Therefore:

> The previous x pattern primarily says that FULL-correct x-related samples
> are unusually likely to remain correct without direct visual K/V access.

It does not currently establish that x-related problems which genuinely need
vision require systematically fewer positive visual-access layers.

## 12. Direct answers

1. **How often is ALL-OFF also correct among FULL-correct samples?** Between
   32.5% and 73.4% by stratum; exact counts and intervals are in Section 3.
2. **How much of the previous low mean is explained by V0?** At coarse degree,
   83.7–94.9% of each decline from degree 0 is attributed to composition by the
   symmetric exact decomposition.
3. **How does positive ON vary in V+?** Means span 11.88–14.42 across strata,
   but most intervals overlap; degree means are 13.36–14.42 and medians are
   essentially constant.
4. **Does the negative relationship persist or reverse?** It largely
   disappears: rho moves from -0.225 to -0.057 with a CI crossing zero.
5. **Do V+ budget curves differ?** Only modestly and non-monotonically; the
   degree curves nearly coincide at the main budgets.
6. **Does a stable axis-specific relationship remain?** A strong axis-specific
   V0-prevalence pattern remains for x. A stable conditional positive-budget
   pattern does not.
7. **Does it survive paired control?** No globally. The V+ paired mean is
   -0.04 with CI [-0.63, 0.57]; isolated small transitions are insufficient for
   an axis claim.
8. **How many FULL-wrong corrections are ALL-OFF?** 162 A0 versus 1,263 A+;
   A0 is 11.4% of discovered corrections.
9. **What x interpretation remains?** x-related FULL-correct survivors are much
   more often ALL-OFF-solvable, not demonstrably lower-budget when vision is
   actually needed.
10. **Is there a genuine difficulty-dependent positive visual-access budget?**
    No stable relationship is supported. The original trend is mainly a
    visual-dependence and survivorship composition effect.

## 13. Figures and machine-readable evidence

The required figures are under
`outputs/wemath2pro_visual_dependence_reanalysis_v1/figures/`:

1. `01_all_off_rate_by_stratum.svg`
2. `02_full_correct_v0_vplus_composition.svg`
3. `03_original_vs_vplus_min_on_by_stratum.svg`
4. `04_vplus_min_on_by_degree.svg`
5. `05_vplus_budget_feasibility_by_degree.svg`
6. `06_vplus_budget_feasibility_by_stratum.svg`
7. `07_original_vs_conditional_difficulty_trend.svg`
8. `08_full_vs_alloff_2x2_by_stratum.svg`
9. `09_paired_vplus_transition_deltas.svg`

All tables, bootstrap settings, source hashes, and exact-anchor evidence are
stored beside them. No follow-up experiment was launched.

## Final classification

**Outcome A — mostly ALL-OFF composition**

The apparent min-ON trend was primarily caused by changing prevalence of
samples solvable without direct visual access.
