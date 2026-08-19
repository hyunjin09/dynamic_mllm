# Binary MCTS Label Geometry and Duplicated-BCE Oracle Analysis

## Executive conclusion

The frozen MCTS cache is not geometrically poor, and the deterministic max-50
selector does not collapse its diversity. The primary failure is instead a
combination of **Outcome C + Outcome E**:

- **Outcome C:** the exact complete mask implied by per-bit duplicated BCE is
  usually an invalid hybrid. Across 6,917 positive inputs, the exact
  training-weighted BCE label oracle has only **5.93% selected-valid Hit@1**,
  with mean nearest-valid Hamming distance **5.68/28**. It is not simply an
  ALL-ON oracle: it uses mean **17.21 ON layers**, produces **6,855 unique
  masks**, and is ALL-ON only **0.59%** of the time. The defect is complete-mask
  incoherence caused by averaging many distinct valid modes bitwise.
- **Outcome E:** dominated supervision strongly amplifies that mismatch.
  **227,897/237,802 selected route occurrences (95.83%)** are Pareto-dominated
  under stored score and ON-count compute. ALL-ON is dominated for
  **4,000/4,045 (98.89%)** samples where it is present. Diagnostic Pareto
  filtering reduces the BCE oracle from **17.21 to 9.78 ON layers** and raises
  Hit@1 from **5.93% to 73.41%**.

This does not support Outcome A: raw valid masks have mean pairwise Hamming
**13.36/28** and mean effective mode count **75.14** even at Hamming radius 4.
It does not support Outcome B: max-50 selection preserves rather than destroys
the measured Hamming diversity and entropy. Outcome D cannot be the primary
diagnosis because the perfect per-sample BCE target is already poor, although
the trained predictors introduce an additional dense/global generalization
effect.

No training, MCTS regeneration, Qwen inference, new route execution, or cache
mutation was performed.

## 1. Sources, integrity, and semantics

The analysis used these frozen sources:

- raw cache index: `outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl`;
- raw records referenced by that index, with every record SHA-256 rechecked;
- selected max-50 supervision:
  `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl`;
- frozen image-group split:
  `outputs/label_regeneration/v1/post_generation/predictor_split_manifest_v1.jsonl`;
- P4/P8 integrity audits;
- actual BCE weighting and decoder code in `binary_policy/dataset.py` and
  `binary_policy/decode.py`;
- full10 duplicated-BCE aggregate histories under
  `outputs/binary_polar/full10_bce/`.

All three frozen manifests contain the same 8,000 unique UIDs. The split is
exactly 7,000 train / 1,000 validation, and the benchmark population is 4,000
GQA, 2,000 TextVQA, and 2,000 ChartQA. Every raw-record checksum, selected-route
subset relationship, split identity, 28-bit mask, ON count, and stored
validity-threshold equality passed.

Route semantics are unchanged:

- `1 = VISUAL_ON`: visual and text/control rows execute the decoder layer;
- `0 = TEXT_ONLY`: text/control rows execute while visual hidden states bypass
  the layer unchanged;
- `111...111` is FULL/ALL-ON;
- validity is exactly `stored benchmark score >= stored correctness threshold`:
  GQA 1.0, TextVQA 0.5, and ChartQA 1.0.

The actual full10 duplicated-BCE objective normalizes route weights inside each
input. ALL-ON receives raw weight 0.3 only when it is selected and a cheaper
selected valid route coexists; all other routes receive raw weight 1.0. The
deployed decoder uses `sigmoid(logit) >= 0.5`, so exact 0.5 ties resolve ON.

Machine-readable provenance and output checksums are in
`outputs/binary_mcts_label_geometry_v1/analysis_manifest.json` (SHA-256
`0fd58601b811d5bfdd4785dc5bc804e90c1e90463fe228554ebee1d02257b36c`).

## 2. Population taxonomy and current composition

| Group | Definition | Overall | GQA | TextVQA | ChartQA |
|---|---|---:|---:|---:|---:|
| A | current FULL wrong, correcting route exists | 2,872 | 1,386 | 712 | 774 |
| B | current FULL correct, cheaper valid route exists | 4,007 | 1,996 | 1,028 | 983 |
| C | current FULL correct, no cheaper route found | 38 | 4 | 6 | 28 |
| D | current FULL wrong, no correcting route found | 1,083 | 614 | 254 | 215 |

The current executor is FULL-correct on 4,045/8,000 and FULL-wrong on
3,955/8,000. There are 6,917 positive inputs and 1,083 zero-positive inputs.
Training has 6,043 positives; validation has 874. Among positive inputs, Group
A contributes 2,872 and FULL-correct Groups B+C contribute 4,045, for an
observed A:(B+C) ratio of 0.710:1.

This taxonomy uses current regenerated outcomes, not historical easy/hard
bucket metadata.

## 3. Route-count geometry

The raw cache has 2,642,998 evaluated routes: both positives and negatives.
There are 528,047 valid route occurrences, all exact-mask unique within their
sample. The training selector retains 237,802 valid masks. It applies the cap
to 3,616 samples and never creates an exact duplicate.

| Quantity over all 8,000 samples | Mean | Median | P90 | Maximum |
|---|---:|---:|---:|---:|
| evaluated routes | 330.37 | 202 | 602 | 602 |
| raw valid routes | 66.01 | 39 | 174 | 357 |
| selected valid routes | 29.73 | 39 | 50 | 50 |

Conditioned on the 6,917 positive samples, mean raw and selected counts are
76.34 and 34.38. A nominally large route count is not treated as evidence of
useful diversity; the mode analyses below test that separately.

## 4. ON-count and distance-to-FULL geometry

Across positive samples, the minimum valid route has mean/median ON count
8.79/10; the median route within each raw set has mean/median 15.13/15.
Selected max-50 values are essentially unchanged at 8.79/10 and 15.14/15.

The raw valid-route mean ON count is 15.26, versus 15.28 after selection. Mean
distance from ALL-ON is therefore 12.74 and 12.72 layers, respectively. The
labels are not concentrated close to ALL-ON.

Dataset-wise raw mean ON counts are:

- GQA: 14.37;
- TextVQA: 15.91;
- ChartQA: 16.29.

Group A correction routes average 14.85 ON layers. Group B routes average
15.43. Group C contains only ALL-ON and therefore has 28 ON layers.

## 5. ALL-ON and ALL-OFF geometry

ALL-ON occurs in 4,045/6,917 positive selected sets (58.48%), exactly the
current FULL-correct population. It is one occurrence per such sample and only
4,045/237,802 = 1.70% of selected route occurrences. ALL-OFF appears in
1,324/6,917 positive sets (19.14%).

The distinction matters: ALL-ON has broad **sample coverage**, but it does not
dominate the route-occurrence count. It can still be a strong globally
available shortcut for a predictor.

## 6. Within-sample diversity and effective modes

Raw valid masks have sample-balanced mean pairwise Hamming distance 13.36/28;
selected masks have 13.44/28. Bit entropy is 0.5989 nats raw and 0.5986 nats
selected. Selection therefore preserves the measured diversity.

Connected components use an edge when Hamming distance is at most the fixed
radius. The effective count is inverse Simpson over component sizes.

| Source | Radius | Mean clusters | Mean effective modes | Mean largest-cluster fraction |
|---|---:|---:|---:|---:|
| raw | 1 | 76.34 | 76.34 | 0.1120 |
| raw | 2 | 76.33 | 76.32 | 0.1121 |
| raw | 4 | 75.73 | 75.14 | 0.1159 |
| selected | 1 | 34.38 | 34.38 | 0.1177 |
| selected | 2 | 34.38 | 34.38 | 0.1177 |
| selected | 4 | 34.31 | 34.24 | 0.1197 |

Cluster counts fall mainly because the selector caps the number of retained
routes, not because retained masks merge into a small number of near-duplicate
components.

## 7. Per-layer marginals and entropy

The table gives sample-balanced selected marginals under the exact training
weights. The last three columns isolate Groups A/B/C.

| Layer | Overall | Group A | Group B | Group C |
|---:|---:|---:|---:|---:|
| 0 | .6746 | .6536 | .6865 | 1.0000 |
| 1 | .5818 | .5663 | .5890 | 1.0000 |
| 2 | .5638 | .5620 | .5609 | 1.0000 |
| 3 | .4993 | .4844 | .5052 | 1.0000 |
| 4 | .5217 | .5180 | .5198 | 1.0000 |
| 5 | .5252 | .5227 | .5224 | 1.0000 |
| 6 | .5233 | .5086 | .5294 | 1.0000 |
| 7 | .5483 | .5388 | .5508 | 1.0000 |
| 8 | .5400 | .5339 | .5400 | 1.0000 |
| 9 | .5571 | .5629 | .5487 | 1.0000 |
| 10 | .5249 | .5127 | .5292 | 1.0000 |
| 11 | .5290 | .5193 | .5316 | 1.0000 |
| 12 | .5391 | .5319 | .5399 | 1.0000 |
| 13 | .5261 | .5132 | .5308 | 1.0000 |
| 14 | .5717 | .5642 | .5731 | 1.0000 |
| 15 | .5395 | .5232 | .5468 | 1.0000 |
| 16 | .5356 | .5248 | .5390 | 1.0000 |
| 17 | .5188 | .5080 | .5220 | 1.0000 |
| 18 | .5106 | .5007 | .5131 | 1.0000 |
| 19 | .5601 | .5590 | .5567 | 1.0000 |
| 20 | .5168 | .5060 | .5200 | 1.0000 |
| 21 | .5221 | .5230 | .5170 | 1.0000 |
| 22 | .5607 | .5562 | .5598 | 1.0000 |
| 23 | .5436 | .5252 | .5524 | 1.0000 |
| 24 | .5190 | .4976 | .5297 | 1.0000 |
| 25 | .5057 | .4896 | .5126 | 1.0000 |
| 26 | .5065 | .4854 | .5170 | 1.0000 |
| 27 | .5392 | .5424 | .5326 | 1.0000 |

The global thresholded marginal mask is ON at 27/28 layers; only layer 3 is
below 0.5. This is a population-pressure diagnostic, not the per-input BCE
oracle.

At the sample/layer level, the exact weighted target has mean entropy 0.5987
nats. Fully 38.28% of bits have `q` in `[0.45,0.55]`, while only 6.37% are at
least 0.9 and 3.58% are at most 0.1. This is high intrinsic bitwise ambiguity.

## 8. Exact duplicated-BCE label oracle

For each positive input and layer, the analysis computes

```text
q_i,l = sum_m alpha_i,m m_l / sum_m alpha_i,m
m_i,l^BCE = 1[q_i,l >= 0.5]
```

using the exact training weights. This assumes unlimited capacity and perfect
per-sample fitting; it is not an MCTS oracle or learned predictor.

| Stratum | n | Mean ON | ALL-ON | Unique masks | Selected-valid Hit@1 | Mean nearest Hamming |
|---|---:|---:|---:|---:|---:|---:|
| overall | 6,917 | 17.21 | 0.59% | 6,855 | 5.93% | 5.68 |
| train | 6,043 | 17.18 | 0.65% | 5,986 | 6.02% | 5.67 |
| validation | 874 | 17.38 | 0.23% | 873 | 5.26% | 5.76 |
| GQA | 3,386 | 15.24 | 0.12% | 3,383 | 4.61% | 6.06 |
| TextVQA | 1,746 | 19.19 | 0.40% | 1,738 | 5.96% | 5.48 |
| ChartQA | 1,785 | 19.00 | 1.68% | 1,745 | 8.40% | 5.15 |
| Group A | 2,872 | 16.94 | 0.03% | 2,867 | 11.80% | 5.10 |
| Group B | 4,007 | 17.30 | 0.05% | 3,998 | 0.82% | 6.15 |
| Group C | 38 | 28.00 | 100% | 1 | 100% | 0.00 |

The central result is not merely density. The per-input oracle is exceptionally
diverse yet usually absent from its own valid route set.

## 9. Weighted versus unweighted BCE oracle

Without POLAR's 0.3 ALL-ON weight, the oracle has mean 18.98 ON layers, 1.16%
ALL-ON rate, 6.04% Hit@1, and mean nearest Hamming 5.36. Weighting changes the
complete oracle mask for 47.67% of samples, by mean Hamming 1.77, and reduces
mean ON by 1.77 layers. However, Hit@1 changes only from 6.04% to 5.93%.

Thus the downweight has a material sparsifying effect but does not solve the
complete-mask incoherence.

## 10. Invalid-hybrid analysis

The weighted oracle is selected-valid for 410 inputs and invalid for 6,507.

| Property | Oracle valid | Oracle invalid |
|---|---:|---:|
| records | 410 | 6,507 |
| mean selected routes | 1.25 | 36.47 |
| mean effective modes, radius 2 | 1.25 | 36.46 |
| mean bit entropy | 0.0185 | 0.6353 |
| mean near-tie-bit fraction | 0.24% | 40.68% |
| mean nearest-valid Hamming | 0.00 | 6.04 |

Invalid hybrids are therefore concentrated exactly where the valid set has
many genuinely distinct modes and high bitwise ambiguity. Calling every
uncached learned prediction invalid would be improper, but here validity is
known because this diagnostic mask is checked against the complete selected
training set for its own input.

## 11. Raw MCTS versus selected max-50

| Metric | Raw | Selected |
|---|---:|---:|
| mean routes per positive input | 76.34 | 34.38 |
| mean route ON count | 15.26 | 15.28 |
| mean pairwise Hamming | 13.36 | 13.44 |
| mean bit entropy | 0.5989 | 0.5986 |
| ALL-ON sample presence | 58.48% | 58.48% |

The max-50 selector substantially reduces count but preserves minimum/median ON
geometry, marginal entropy, Hamming diversity, and both anchors. This rejects
Outcome B under the approved diagnostics.

## 12. Deduplication

There are zero raw within-sample exact-mask duplicates and zero selected
duplicates. Duplicated BCE duplicates an input across distinct valid paths; it
does not implicitly overweight masks through accidental identical-mask
multiplicity.

## 13. Pareto dominance and Pareto-filtered oracle

Route `b` dominates route `a` when stored score(b) is at least score(a) and
ON-count(b) is strictly smaller. This uses the stored benchmark score, not an
invented continuous utility.

- 9,905/237,802 selected route occurrences are efficient;
- 227,897/237,802 (95.83%) are dominated;
- the sample-balanced mean dominated fraction is 86.31%;
- median efficient routes per positive sample is 1; mean is 1.43;
- ALL-ON is dominated for 4,000/4,045 samples where present.

Diagnostic Pareto filtering—without training—changes the exact weighted oracle:

| Label set | Mean routes | Mean oracle ON | ALL-ON | Hit@1 | Nearest Hamming |
|---|---:|---:|---:|---:|---:|
| original max-50 | 34.38 | 17.21 | 0.59% | 5.93% | 5.68 |
| Pareto-efficient selected | 1.43 | 9.78 | 0.56% | 73.41% | 1.44 |

This is strong Outcome E evidence, but not a training result. The remaining
26.59% invalidity also shows that Pareto filtering does not eliminate Outcome
C whenever multiple efficient modes remain.

## 14. Diversity-balanced counterfactual

Starting from the lowest-ON raw route and greedily maximizing minimum Hamming
does not repair duplicated BCE:

| Raw representatives | Mean retained | Mean oracle ON | Hit@1 | Nearest Hamming |
|---|---:|---:|---:|---:|
| K=4 | 3.75 | 18.08 | 5.90% | 6.57 |
| K=8 | 7.13 | 16.93 | 5.90% | 6.67 |
| K=16 | 13.31 | 15.98 | 5.90% | 6.36 |

Merely reducing route count while preserving separated modes still induces a
bitwise hybrid. The issue is not “too many routes” in isolation.

## 15. Correct/wrong balance counterfactual

Sample-level reweighting cannot change a per-sample BCE oracle, but it changes
global ON pressure:

| Group A : Groups B+C mass | Mean global ON marginal | Thresholded global ON layers |
|---|---:|---:|
| observed 0.710:1 | 0.5394 | 27 |
| 1:1 | 0.5380 | 27 |
| 2:1 | 0.5353 | 25 |
| 3:1 | 0.5339 | 25 |

Oversampling current-FULL-wrong corrections weakens the ON prior modestly but
does not plausibly resolve within-input complete-mask incoherence by itself.

## 16. Cross-sample route diversity

Selected supervision contains 232,278 unique masks among 237,802 occurrences
(97.68% unique). ALL-ON is the most common route but only 1.70% of occurrences.
The top 5 masks cover only 2.26% of route occurrences.

At sample level, ALL-ON is available for 58.48% of positives; the top 5 global
masks intersect 61.69% of selected sets and the top 50 intersect 61.80%. For
the single minimum-ON representative, 5,555/6,917 masks are unique. The most
common representative—ALL-OFF—covers 19.14%.

The cache therefore contains genuinely sample-specific route identities, not
only a handful of global templates. Whether a route transfers to a different
sample is unresolved: existing caches evaluate routes only on their own input,
so exact interchangeability would require forbidden new Qwen executions.

## 17. BCE label oracle versus trained full10 BCE predictor

The frozen full10 histories retain aggregate validation mask counts, not
UID-level predictions. Therefore exact per-record oracle agreement, Hamming,
and per-layer agreement cannot be reconstructed without new predictor
inference; none was run.

The available matched aggregate comparison on 874 positive validation inputs is:

| System | Mean ON | ALL-ON | Unique masks | Hit@1 | Nearest Hamming |
|---|---:|---:|---:|---:|---:|
| exact weighted label oracle | 17.38 | 0.23% | 873 | 5.26% | 5.76 |
| Question BCE, frozen epoch 2 | 25.33 | 44.85% | 168 | 28.03% | 5.01 |
| Image+Question BCE, frozen epoch 2 | 22.88 | 46.22% | 147 | 27.80% | 5.92 |

The trained predictors are much denser and less diverse than the per-input
oracle, showing an additional fitting/generalization/global-shortcut effect.
Their higher cached Hit@1 comes largely from selecting a commonly available
route such as ALL-ON; it does not make the exact BCE label oracle good. Because
the oracle itself has only 5.26% validation Hit@1, Outcome D is not the primary
classification.

## 18. Figures and detailed tables

Reproducible SVG figures are under
`outputs/binary_mcts_label_geometry_v1/figures/`, including route-count,
minimum/median ON, pairwise Hamming, layer marginals, group marginals,
BCE-oracle ON, actual-vs-oracle, nearest-valid Hamming, raw-vs-selected, and
Pareto comparisons.

Detailed CSV/JSONL artifacts include:

- `population_taxonomy.csv`;
- `route_geometry_summary.csv`;
- `layer_marginals.csv`;
- `weighted_bce_oracle_summary.csv` and `unweighted_bce_oracle_summary.csv`;
- `per_sample_geometry.jsonl`;
- `invalid_hybrid_summary.csv` and `cluster_summary.csv`;
- `raw_selected_summary.csv` and `pareto_summary.csv`;
- `counterfactual_oracle_summary.csv` and `counterfactual_oracles.jsonl`;
- `balance_pressure.csv`;
- `cross_sample_route_diversity.csv`;
- `actual_predictor_aggregate_comparison.csv`.

## 19. Decision matrix and recommendation

| Outcome | Decision | Evidence |
|---|---|---|
| A — poor MCTS geometry | Rejected | raw Hamming 13.36; radius-4 effective modes 75.14; mean ON 15.26 |
| B — poor max-50 selection | Rejected | selected Hamming/entropy/ON geometry closely match raw |
| C — duplicated BCE creates incoherent hybrids | **Supported, primary** | exact oracle Hit@1 5.93%; 94.07% invalid; invalid sets have high entropy/modes |
| D — good oracle, poor predictor | Rejected as primary | predictor is denser, but label oracle is already poor |
| E — dominated FULL/routes are central | **Supported, major contributor** | 95.83% route-occurrence dominance; Pareto oracle Hit@1 73.41% |

### Concrete next recommendation

Do not regenerate MCTS merely to obtain more routes, and do not treat the
current max-50 selector as the main defect. The smallest scientifically
justified next comparison—requiring separate approval—is a matched training
study that first removes Pareto-dominated routes and then contrasts:

1. the current factorized duplicated BCE on the frozen Pareto-efficient sets;
2. a complete-route-coherent supervision objective on the identical
   Pareto-efficient sets.

This ordering separates the two supported causes: dominated-label pressure
(Outcome E) and residual multimodal hybridization (Outcome C). It should retain
the raw cache, split, architecture, initialization, optimizer, compute budget,
and execution evaluator unchanged. No such training is authorized or executed
by this analysis.

The strongest objection is that Pareto filtering often leaves nearly one route
per input, so its improved label oracle is partly a mathematical consequence
of making the target nearly unimodal. That is precisely why Pareto-filtered BCE
alone must not be taken as sufficient evidence: it should be compared against
a coherent complete-route objective on the same filtered sets, and both must be
judged by held-out actual mask execution rather than label Hit@1 alone.
