# WeMath2.0-Pro Visual-Compute Difficulty Analysis

## Executive conclusion

The proposed monotonic hypothesis was **not supported**. Among the 841 samples
that the current FULL model answered correctly, greater coarse difficulty was
associated with a *lower*, not higher, minimum discovered VISUAL_ON depth:

- degree 0 (`base`): mean minimum ON = **9.74**, median = **10**;
- degree 1: mean = **7.67**, median = **8**;
- degree 2: mean = **5.73**, median = **0**;
- degree 3 (`xyz`): mean = **3.66**, median = **0**.

The family-clustered Spearman correlation was **-0.225** (95% bootstrap CI
**[-0.291, -0.159]**). Budget-feasibility curves consequently shifted left,
not right, with coarse difficulty. At an eight-layer budget, feasibility rose
from **39.6%** at degree 0 to **79.7%** at degree 3.

This aggregate result is not a defensible claim that difficult problems
intrinsically need less visual computation. The eight official strata reveal
an axis-specific pattern: the low minimum-ON values are concentrated in
`x`-containing strata (`x`, `xy`, `xz`, `xyz`), whereas `y`, `z`, and `yz`
remain much closer to `base`. At the same time, the probability of finding a
correcting route for a FULL-wrong sample fell from **50.0%** at degree 0 to
**26.7%** at degree 3. The high-difficulty FULL-correct cohort is therefore an
increasingly selected subset of all difficult records.

The best classification is **Outcome E — axis-specific effect**. Aggregate
difficulty degree is too coarse to characterize visual-computation demand.
The result does not justify claiming visual-depth scaling, and it does not by
itself justify adding REPEAT or computation beyond FULL.

## 1. Scope and frozen evidence

This was a read-only analysis of the completed hard-cap-400 WeMath2.0-Pro MCTS
cache. No route was regenerated or executed, no correctness threshold was
changed, and no predictor was trained. The analysis used checksum-bound
per-sample statistics derived from **all raw evaluated routes**, rather than
the later max-50 predictor-supervision view.

The frozen binary action has 28 bits:

- `1 = VISUAL_ON`: visual rows execute the decoder layer normally;
- `0 = TEXT_ONLY`: visual rows bypass the layer while text/control rows execute;
- all 28 bits ON is `FULL`;
- there is no REPEAT action in this cache.

Source integrity passed. The analyzed cache contains **4,544 records** and
**1,658,485 evaluated routes**. The raw-derived source index checksum is
`e362d8c8f5c194503ffc12c55a921ccc79c615a41cd037cde5a8ce6a24b164e7`.
The complete source hashes and generated-output hashes are recorded in
`outputs/wemath2pro_visual_compute_difficulty_v1/analysis_manifest.json`.

The verified sample groups are:

| Group | Definition | N |
|---|---|---:|
| A | FULL wrong, correcting route discovered | 1,425 |
| B | FULL correct, cheaper valid route discovered | 784 |
| C | FULL correct, only FULL discovered | 57 |
| D | FULL wrong, no correcting route discovered | 2,278 |
| **Total** | Prospectively eligible records | **4,544** |

Thus, the primary FULL-correct cohort has **841** records, the FULL-wrong
cohort has **3,703**, and **2,266** records have at least one valid route.

## 2. Difficulty metadata audit

The source metadata contains exactly the expected eight strata. It does not
define a total ordering among `x`, `y`, and `z`, so the primary report preserves
all eight. The coarse degree is only the number of active difficulty axes:
`base=0`, `x/y/z=1`, `xy/xz/yz=2`, and `xyz=3`.

| Stratum | Degree | Eligible | FULL correct | FULL wrong | Any positive route | Zero positive |
|---|---:|---:|---:|---:|---:|---:|
| base | 0 | 568 | 154 | 414 | 361 | 207 |
| x | 1 | 571 | 78 | 493 | 231 | 340 |
| y | 1 | 571 | 129 | 442 | 354 | 217 |
| z | 1 | 569 | 135 | 434 | 343 | 226 |
| xy | 2 | 569 | 71 | 498 | 213 | 356 |
| xz | 2 | 567 | 74 | 493 | 219 | 348 |
| yz | 2 | 563 | 121 | 442 | 336 | 227 |
| xyz | 3 | 566 | 79 | 487 | 209 | 357 |

FULL accuracy itself is axis-dependent. It is 27.1% for `base`, 22.6–23.7%
for `y` and `z`, but only 12.5–14.0% for the `x`-containing strata. This is one
reason a depth comparison restricted to FULL-correct samples needs a strong
survivorship caveat.

## 3. Primary FULL-correct analysis

For every FULL-correct record, the primary quantity is

```text
min_valid_on = minimum VISUAL_ON count among all discovered valid routes.
```

`removable_visual_layers = 28 - min_valid_on`. These are discovered-route
statistics, not identified physical requirements.

### Coarse difficulty degree

| Degree | N | Mean min ON (95% cluster CI) | Median | SD | P10 | Q25 | Q75 | P90 | Mean removable | Median removable |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 154 | 9.74 [8.35, 11.14] | 10 | 8.80 | 0 | 0 | 13.75 | 28.0 | 18.26 | 18 |
| 1 | 342 | 7.67 [6.64, 8.73] | 8 | 8.33 | 0 | 0 | 13.00 | 17.9 | 20.33 | 20 |
| 2 | 266 | 5.73 [4.77, 6.77] | 0 | 7.71 | 0 | 0 | 10.75 | 16.0 | 22.27 | 28 |
| 3 | 79 | 3.66 [2.23, 5.33] | 0 | 7.10 | 0 | 0 | 6.50 | 14.4 | 24.34 | 28 |

This is the opposite of the planned direction. The degree–minimum-ON Spearman
correlation is **-0.225** with a family-clustered 95% CI of **[-0.291,
-0.159]**. The corresponding removable-layer correlation is **+0.225**
(95% CI **[0.161, 0.289]**).

### Eight individual strata

| Stratum | N | Mean min ON | Median | Q25–Q75 | Mean removable | Cheaper than FULL |
|---|---:|---:|---:|---:|---:|---:|
| base | 154 | 9.74 | 10 | 0–13.75 | 18.26 | 88.3% |
| x | 78 | 4.05 | 0 | 0–8 | 23.95 | 94.9% |
| y | 129 | 8.72 | 9 | 0–13 | 19.28 | 93.0% |
| z | 135 | 8.75 | 9 | 0–14 | 19.25 | 91.9% |
| xy | 71 | 4.01 | 0 | 0–8 | 23.99 | 98.6% |
| xz | 74 | 3.32 | 0 | 0–6.75 | 24.68 | 98.6% |
| yz | 121 | 8.20 | 9 | 0–13 | 19.80 | 91.7% |
| xyz | 79 | 3.66 | 0 | 0–6.5 | 24.34 | 96.2% |

The contrast between `x`-containing and non-`x` strata is too large to treat
degree as a single homogeneous difficulty scale. The result is not “all hard
questions use less visual depth”; it is “the observed surviving route geometry
differs by difficulty axis.”

## 4. Visual-budget feasibility

For a budget `C`, feasibility is the fraction of FULL-correct samples with a
discovered valid route using at most `C` VISUAL_ON layers.

| Degree | C=8 | C=12 | C=16 | C=18 | C=20 | C=22 | C=24 | C=28 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.6% | 69.5% | 83.1% | 87.7% | 88.3% | 88.3% | 88.3% | 100% |
| 1 | 52.6% | 73.7% | 88.0% | 91.2% | 92.1% | 93.0% | 93.0% | 100% |
| 2 | 63.9% | 80.8% | 91.4% | 93.6% | 95.1% | 95.5% | 95.5% | 100% |
| 3 | 79.7% | 87.3% | 93.7% | 94.9% | 96.2% | 96.2% | 96.2% | 100% |

The curves rise *earlier* with coarse difficulty, contradicting the proposed
right-shift. The eight-stratum curves show that this behavior is driven mainly
by `x`, `xy`, `xz`, and `xyz`.

Figures:

- `figures/01_full_correct_min_on_by_degree.svg`
- `figures/02_full_correct_removable_by_degree.svg`
- `figures/03_full_correct_budget_feasibility_degree.svg`
- `figures/04_full_correct_budget_feasibility_stratum.svg`

## 5. Cheaper-route availability among FULL-correct samples

Cheaper-route availability also does not decrease with coarse difficulty.

| Degree | FULL-correct N | Cheaper route N | Cheaper route | FULL-only N |
|---:|---:|---:|---:|---:|
| 0 | 154 | 136 | 88.3% | 18 |
| 1 | 342 | 318 | 93.0% | 24 |
| 2 | 266 | 254 | 95.5% | 12 |
| 3 | 79 | 76 | 96.2% | 3 |

This cannot be generalized to all samples because FULL correctness drops and
zero-positive coverage rises in the same difficult strata.

## 6. FULL-wrong correction discovery

The separate FULL-wrong analysis shows a clear degradation in route-search
success with coarse difficulty.

| Degree | FULL wrong | Correction found | Found rate (95% cluster CI) | No correction |
|---:|---:|---:|---:|---:|
| 0 | 414 | 207 | 50.0% [44.9, 54.9] | 207 |
| 1 | 1,369 | 586 | 42.8% [39.5, 46.1] | 783 |
| 2 | 1,433 | 502 | 35.0% [32.0, 38.1] | 931 |
| 3 | 487 | 130 | 26.7% [22.8, 30.7] | 357 |

The individual-stratum found rates are `base` 50.0%, `x` 31.0%, `y` 50.9%,
`z` 47.9%, `xy` 28.5%, `xz` 29.4%, `yz` 48.6%, and `xyz` 26.7%. Again, the
largest failure increase follows the `x` axis rather than degree alone.

### Minimum correcting budget within Group A

Conditional on a correction having been found, the minimum correcting ON
budget is comparatively stable:

| Degree | N | Mean min correcting ON (95% CI) | Median | Q25–Q75 | P90 |
|---:|---:|---:|---:|---:|---:|
| 0 | 207 | 9.28 [8.66, 9.88] | 10 | 8–12 | 14.0 |
| 1 | 586 | 9.21 [8.80, 9.61] | 10 | 8–12 | 14.0 |
| 2 | 502 | 9.48 [9.09, 9.86] | 10 | 8–12 | 14.0 |
| 3 | 130 | 9.92 [9.30, 10.51] | 10 | 8–12 | 13.1 |

This quantity means minimum discovered visual budget among alternative
correcting programs. It is not the ordinary visual-depth requirement of a
correct FULL computation.

## 7. Zero-positive population

The **2,278** zero-positive records are all FULL-wrong. Their distribution is:

| Stratum | Zero-positive N | Share of all zero-positive | Failure within FULL-wrong |
|---|---:|---:|---:|
| base | 207 | 9.1% | 50.0% |
| x | 340 | 14.9% | 69.0% |
| y | 217 | 9.5% | 49.1% |
| z | 226 | 9.9% | 52.1% |
| xy | 356 | 15.6% | 71.5% |
| xz | 348 | 15.3% | 70.6% |
| yz | 227 | 10.0% | 51.4% |
| xyz | 357 | 15.7% | 73.3% |

`xyz`, `xy`, `xz`, and `x` dominate the zero-positive population. This does
not mean those samples require more than 28 visual layers. It means no
correcting route was discovered under the frozen finite MCTS budget and
VISUAL_ON/TEXT_ONLY action space. Search incompleteness, base-model reasoning
failure, and missing actions remain alternative explanations.

## 8. Same-family and same-image analysis

The official source `question_id` was verified as the seed-family key. There
are 569 families; 550 contain one unique eligible record for every stratum.
Eight are incomplete and 13 family/difficulty cells are duplicated. Paired
transitions require exactly one FULL-correct record at each endpoint, so
ambiguous cells are excluded.

Across 203 usable families and 677 supported transition occurrences:

- minimum ON increased in **22.2%**;
- stayed equal in **47.4%**;
- decreased in **30.4%**;
- mean paired delta was **-0.82 layers** (95% family-bootstrap CI
  **[-1.37, -0.28]**);
- median paired delta was **0**.

This does not support a general within-family increase. The strongest negative
steps are those adding `x`: `base→x` mean delta **-3.63** (CI **[-6.05,
-1.34]**), `y→xy` **-4.52** (CI **[-7.55, -1.52]**), and `z→xz` **-4.76**
(CI **[-8.93, -0.69]**). The same-image subsets for those transitions retain
the negative direction: **-3.73**, **-3.90**, and **-5.35** layers,
respectively, with intervals below zero.

Most transitions involving only `y` or `z` have median zero and confidence
intervals crossing zero. This paired evidence reinforces the axis-specific
interpretation and reduces, but does not eliminate, the concern that the
global pattern is merely different image composition.

## 9. Visual-token-count control

Visual-token counts rise with coarse degree in the eligible pool: medians are
727, 812, 1,024, and 1,254 for degrees 0–3. Among FULL-correct records, medians
are 649, 783, 1,202, and 1,800. Therefore exact computation cost is not captured
by ON-layer count alone.

In a descriptive family-cluster bootstrap OLS on the 841 FULL-correct records,

```text
min_valid_on ~ difficulty_degree + log1p(visual_token_count)
```

the degree coefficient remains negative at **-1.84 layers per degree** (95%
CI **[-2.49, -1.19]**). The log-token coefficient is **-0.51** (95% CI
**[-1.08, 0.06]**) and the model R² is only **0.051**. Stratified token-quartile
tables give the same broad negative degree pattern, although some cells—most
notably degree 3 in the smallest-token quartile—are small.

Thus, larger visual inputs do not explain away the observed negative depth
association. This remains descriptive adjustment, not causal inference.

## 10. Secondary route geometry

| Degree | Eligible | Positive | Mean valid routes, all | Median, all | Mean valid routes, positive | Median, positive | Mean route ON | Mean pairwise Hamming | Mean bit entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 568 | 361 | 27.18 | 3 | 42.77 | 17 | 15.45 | 13.22 | 0.525 |
| 1 | 1,711 | 928 | 25.60 | 1 | 47.20 | 17 | 14.84 | 13.27 | 0.526 |
| 2 | 1,699 | 768 | 22.08 | 0 | 48.86 | 17 | 14.54 | 13.29 | 0.518 |
| 3 | 566 | 209 | 19.27 | 0 | 52.18 | 12 | 14.16 | 13.17 | 0.499 |

The all-eligible number of valid routes declines because zero-positive samples
become more common. Conditional on a positive sample, mean valid-route count
does not decline monotonically, illustrating why average route geometry is
not a substitute for the primary per-sample minimum.

## 11. Direct answers to the planned questions

1. **Does minimum correctness-preserving VISUAL_ON depth increase with
   difficulty among FULL-correct samples?** No. It decreases with coarse
   degree, with a family-clustered Spearman rho of -0.225. The change is mainly
   associated with the `x` axis.
2. **Does small-budget solvability decrease with difficulty?** No among the
   selected FULL-correct cohort. At C=8 it increases from 39.6% to 79.7%
   across degrees 0→3.
3. **Does cheaper-route availability decrease with difficulty?** No among
   FULL-correct records; it rises from 88.3% to 96.2% by coarse degree.
4. **Does correcting-route discovery decline among FULL-wrong samples?** Yes,
   from 50.0% at degree 0 to 26.7% at degree 3, driven mainly by `x`-related
   strata.
5. **Which strata dominate zero positives?** `xyz`, `xy`, `xz`, and `x`; their
   within-FULL-wrong failure rates are 69.0–73.3%.
6. **Does a positive difficulty–depth relationship survive paired analysis?**
   No. The aggregate paired mean is -0.82 layers, median zero, and several
   `x`-adding transitions are significantly negative.
7. **Is the result simply visual-token count?** No. The negative degree
   coefficient remains after descriptive log-token adjustment.
8. **Which Outcome A–E fits?** **Outcome E — axis-specific effect**, with
   strong survivorship/search-coverage caveats. Outcome B captures the
   correction-coverage collapse but incorrectly says difficulty has no
   relationship with minimum depth; the relationship exists and is
   axis-specific in the opposite direction to the original hypothesis.
9. **Does this justify expanding to REPEAT?** Not from this analysis alone.
   The binary route space cannot test >28 visual executions, but the observed
   evidence does not demonstrate that harder positive cases need even the full
   existing depth. A future expansion would need a separate hypothesis and
   approval, not be inferred as a rescue of the rejected monotonic claim.

## Scientific conclusion

Within the frozen WeMath2.0-Pro SKIP/KEEP search, difficult problems do not
show a monotonic increase in minimum discovered correctness-preserving visual
decoder depth. Instead, `x`-related difficulty substantially changes both
model solvability and discovered route geometry: far fewer FULL-wrong cases
are corrected, yet the selected FULL-correct survivors often retain correct
answers under very aggressive visual-layer removal. The defensible conclusion
is that difficulty axes have different relationships with route discovery and
visual-depth redundancy, while aggregate degree conflates these effects.
No new search or experiment is launched by this result.

