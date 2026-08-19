# WeMath2.0-Pro MCTS Label Training-Suitability Analysis

## Executive decision

The hard-cap-400 WeMath2.0-Pro extraction is complete and internally valid for
all **4,544 prospectively eligible records**. The source pool contained 4,552
rows; the other eight were excluded before MCTS because their question or
answer field was empty. The cache contains no missing sample, temporary/error
record, scoring timeout, above-cap search, duplicate UID, or source-binding
failure.

The labels are **conditionally suitable**, not universally suitable:

- **Suitable for exact valid-set NLL:** 2,266 samples have at least one complete
  valid mask. These can supply one-of-valid-set supervision after an
  image-group-disjoint split and a separately frozen route-cap/selection view.
- **Suitable for positive/negative route ranking:** all 4,544 records retain
  every evaluated mask, including 1,550,814 negative and 107,671 positive
  route executions. The severe class imbalance must be handled explicitly.
- **Not suitable for unfiltered POLAR-style duplicated-route BCE as a coherent
  complete-mask target:** even the perfect per-sample duplicated-BCE label
  oracle lands in its own selected valid set only **13.72%** of the time. The
  remaining **86.28%** are bitwise hybrids outside the known positive set,
  with mean nearest-valid Hamming distance **5.10/28**.
- **Not usable as positive-only supervision for every record:** 2,278 samples
  have no correcting/valid route in the evaluated cache. They must not be
  silently dropped from population accounting or given an empty set-NLL; they
  remain useful as negative/ranking examples.

This is a label-only finding. It does not show predictor generalization, and no
predictor training or route execution was performed.

## 1. What was analyzed and why

The purpose was to decide whether the completed WeMath2.0-Pro MCTS cache can be
used in later binary-router training, and under which supervision objective.
The analysis was deliberately matched to the earlier GQA/TextVQA/ChartQA label
analysis because that work showed that many individually valid masks can still
produce an incoherent bitwise-BCE target.

The audited source is:

- dataset: `We-Math/We-Math2.0-Pro`, source revision
  `c1d9f3ccea7361069f0442362e781d1ae7a28e94`;
- frozen eligible manifest:
  `outputs/label_regeneration/wemath2pro_cap400_v2/manifest/wemath2pro_valid_mcts_v1.jsonl`;
- raw cache: `outputs/label_regeneration/wemath2pro_cap400_v2/raw_route_cache/`;
- active execution contract:
  `outputs/label_regeneration/wemath2pro_cap400_v2/frozen_execution_contract_cap400_v5.json`;
- analysis outputs: `outputs/wemath2pro_mcts_label_analysis_v1/`.

The route is the same unrestricted 28-bit binary visual mask used elsewhere:
ON executes visual and text/control rows normally at that layer; OFF lets the
visual rows bypass the layer while text/control rows execute it. Validity is
the frozen WeMath MathRuler decision `score >= 1.0`.

## 2. Extraction completion and integrity

| Check | Result |
|---|---:|
| Source records | 4,552 |
| Prospectively technical-invalid records | 8 |
| Eligible manifest records | 4,544 |
| Terminal raw sample records | 4,544 |
| Unique UIDs | 4,544 |
| Unique image groups | 1,629 |
| Repeated image groups | 1,193 |
| Maximum questions for one image | 32 |
| Evaluated unique route records | 1,658,485 |
| Temporary/error records | 0 |
| Scoring-timeout records/route occurrences | 0 / 0 |
| Integrity decision | PASS |

The current executor, not historical metadata, determined each root result.
It assigned 841 records the 200-simulation current-FULL-correct budget and
3,703 records the 400-simulation current-FULL-wrong budget. Every record
completed its assigned budget. No record used the superseded 600-simulation
extension.

Three records contain one fewer *unique* candidate mask than
`simulations + 2`. This is valid MCTS evaluation-cache reuse: a rollout
revisited an already evaluated anchor or mask. The full simulation trace is
present, and the union of the two anchors and every simulated mask exactly
equals the candidate/evaluated-mask index. The analyzer initially rejected
this legitimate behavior, was repaired to use the repository's authoritative
trace-linkage invariant, and then passed all 4,544 records.

## 3. Current FULL outcomes and positive-route coverage

| Taxonomy | Definition | Records | Fraction of all records |
|---|---|---:|---:|
| A | FULL wrong, correcting route found | 1,425 | 31.36% |
| B | FULL correct, cheaper valid route found | 784 | 17.25% |
| C | FULL correct, only FULL found valid | 57 | 1.25% |
| D | FULL wrong, no correcting route found | 2,278 | 50.13% |

Current FULL is correct for **841/4,544 (18.51%)** and wrong for
**3,703/4,544 (81.49%)**. MCTS found at least one correcting mask for
**1,425/3,703 (38.48%)** current-FULL-wrong samples. Among current-FULL-correct
samples, **784/841 (93.22%)** have a cheaper valid mask. ALL-OFF itself is
correct for 575 samples.

Overall, **2,266/4,544 (49.87%)** records have at least one valid evaluated
route. The raw cache contains 107,671 valid routes:

| Coverage threshold | Records | Fraction of all records |
|---|---:|---:|
| at least 1 valid route | 2,266 | 49.87% |
| at least 3 | 1,815 | 39.94% |
| at least 10 | 1,306 | 28.74% |
| at least 20 | 1,064 | 23.42% |
| at least 32 | 905 | 19.92% |
| at least 50 | 752 | 16.55% |
| at least 100 | 511 | 11.25% |

The median over all records is zero valid routes. Conditioned on positive
records, the mean is **47.52** and the median is **16**. A deterministic
diagnostic max-50 selector retains 54,365 routes, mean **23.99** per positive
sample, and caps 745 samples. This derived max-50 view is checksum-bound
diagnostic evidence; it is not yet a frozen WeMath training manifest.

## 4. Route geometry

Positive route sets are genuinely multimodal rather than tightly clustered
around FULL:

| Metric over positive samples | Raw | Diagnostic max-50 |
|---|---:|---:|
| Mean route ON count | 14.78 | 14.76 |
| Mean minimum ON count | 8.52 | preserved by selector |
| Median minimum ON count | 9 | preserved by selector |
| Mean pairwise Hamming distance | 13.26/28 | 13.32/28 |
| Mean bit entropy | 0.5204 nats | 0.5205 nats |
| Mean effective modes at radius 4 | 46.77 | 23.90 |

The cap reduces count but preserves ON-count, Hamming, and entropy geometry.
Layers 0 and 1 are more often ON under the selected training weights
(0.612 and 0.596), while most remaining layer marginals sit close to 0.5.
This creates exactly the regime where marginal/BCE decoding can combine bits
from different valid modes.

Native image processing also makes this a materially heavier input population:
mean visual-token count is 2,514.94, median 900, P90 8,453, and maximum 11,342.
Any eventual training pipeline must budget encoder preprocessing accordingly;
the label analysis itself did not load the base MLLM.

## 5. Exact duplicated-BCE label oracle

The diagnostic max-50 set uses the same POLAR-compatible route weighting used
by the current duplicated-BCE pipeline: ALL-ON has raw weight 0.3 only when a
cheaper selected valid route coexists; other routes have weight 1.0. The
perfect per-sample BCE target is decoded with `q >= 0.5 -> ON`.

| Oracle | Positive records | Mean ON | ALL-ON | Unique masks | Valid-set Hit@1 | Mean nearest-valid Hamming |
|---|---:|---:|---:|---:|---:|---:|
| Weighted duplicated BCE | 2,266 | 15.64 | 2.56% | 2,209 | 13.72% | 5.10 |
| Equal-weight duplicated BCE | 2,266 | 16.81 | 3.88% | 2,176 | 13.77% | 5.01 |

Only **311/2,266** weighted-oracle masks belong to the selected valid set. The
problem remains when weights are removed, so the 0.3 FULL weighting is not the
cause. Group A Hit@1 is 15.79%; Group B Hit@1 is only 3.70%; Group C is 100%
because its valid set contains only FULL.

This is an ideal label oracle, not a learned model. Therefore, a learned
duplicated-BCE predictor cannot solve the complete-mask coherence problem by
better optimization alone. Uncached predictions still require execution in a
behavioral evaluation, but this label-level diagnostic knows whether the
per-sample target is one of its own selected positives.

## 6. Pareto redundancy

Using stored MathRuler score and visual-ON count, 51,610 of 54,365 selected
route occurrences are dominated. Thus **94.93%** of selected occurrences are
no better in score and strictly more expensive than another selected route.
The sample-balanced mean dominated fraction is 74.97%, and only 1.22 efficient
routes remain per positive sample on average. FULL is dominated for
**93.22%** of the 841 positive sets where it appears.

A diagnostic Pareto-only target changes the weighted BCE oracle as follows:

| Label set | Mean oracle ON | Valid-set Hit@1 | Mean nearest-valid Hamming |
|---|---:|---:|---:|
| Original selected max-50 | 15.64 | 13.72% | 5.10 |
| Pareto-efficient diagnostic | 9.18 | 84.64% | 0.86 |

This supports the same two recurring findings as the earlier three-benchmark
analysis: duplicated-BCE hybridization and extensive dominated supervision.
Pareto filtering is only a diagnostic here. It was not substituted into an
approved training objective, and the remaining 15.36% oracle invalidity shows
that filtering does not fully solve multimodality.

## 7. Difficulty and scoring reliability

All eight released difficulty strata are represented (563–571 records each).
Positive-route coverage ranges from 36.93% (`xyz`) to 63.56% (`base`). The
current FULL-correct fraction ranges from 12.48% (`xy`) to 27.11% (`base`).
No stratum is absent, but easier/base and single-axis `y`/`z` strata contribute
more positive supervision than the combined-axis strata. Later splits should
stratify difficulty while grouping by image.

There were **zero MathRuler scoring timeouts** across all 1,658,485 retained
route executions. Hence this cache has no observed timeout-induced false
negative contamination. This does not prove the evaluator is semantically
perfect; it establishes deterministic completion under the frozen scorer.

## 8. Comparison with GQA, TextVQA, and ChartQA labels

| Diagnostic | Earlier 8K labels | WeMath2.0-Pro |
|---|---:|---:|
| Records | 8,000 | 4,544 |
| Positive-record coverage | 86.46% | 49.87% |
| Correcting route among current-FULL-wrong | 72.62% | 38.48% |
| Raw valid routes per positive sample | 76.34 | 47.52 |
| Raw mean pairwise Hamming | 13.36 | 13.26 |
| Weighted BCE-oracle Hit@1 | 5.93% | 13.72% |
| Weighted BCE-oracle nearest Hamming | 5.68 | 5.10 |
| Selected routes Pareto-dominated | 95.83% | 94.93% |
| Pareto-oracle Hit@1 | 73.41% | 84.64% |

WeMath labels are less abundant and much less likely to contain a positive
route, consistent with the lower current FULL accuracy and harder exact math
scoring. Their complete-mask diversity is nevertheless almost identical to the
earlier cache. The duplicated-BCE oracle is better in absolute Hit@1 but still
fails on more than six of every seven positive inputs. The supervision-form
problem therefore replicates rather than disappears.

## 9. Training suitability by objective

### Exact valid-set NLL

**Yes, conditionally.** Use only the 2,266 nonempty positive sets for the set
likelihood. Keep complete masks grouped per image-question, use one frozen
deterministic cap/selection rule matched to the comparator, and make the split
image-group-disjoint across the 1,629 image groups. The exact-set objective
avoids treating the bitwise marginal hybrid as the desired complete route. It
does not make the factorized binary head expressive enough to model arbitrary
cross-layer dependence, so held-out executed-mask evaluation remains the real
gate.

### POLAR-style duplicated valid-route BCE

**No, not in its current unfiltered form if coherent complete-mask prediction
is the goal.** The label oracle's 13.72% Hit@1 shows that the supervision itself
targets an often-uncached hybrid. Another full training run would conflate
predictor generalization with a known label-objective mismatch.

### Single-route or Pareto-efficient supervision

**Potentially viable, but not yet approved or frozen.** Minimum-budget or
Pareto-efficient routes remove large amounts of redundant supervision, but
they change the supervision policy and must be compared prospectively against
the already approved exact-set formulation. The present analysis does not
authorize that experiment.

### Positive/negative route ranking

**Suitable with class-imbalance controls.** All evaluated masks are retained,
and their stored threshold validity is internally consistent. Positives are
only **6.49%** of route executions, so batching/weighting must be declared
without changing the meaning of correctness.

### Standalone WeMath predictor corpus

**Scientifically limited.** Only 2,266 positive questions and 1,629 unique
image groups exist, with repeated questions per image and difficulty-dependent
coverage. The cache is better treated as a math-domain supplement or a
separately reported domain experiment, not silently pooled as though it had
the same positive coverage as GQA/TextVQA/ChartQA.

## 10. Final recommendation and strongest objection

If WeMath2.0-Pro is added to later predictor work, the smallest defensible use
is a separately stratified, image-group-disjoint exact-valid-set-NLL condition
over the 2,266 positive samples, with all 2,278 zero-positive samples retained
for coverage reporting and optional frozen ranking supervision. Do not use the
raw multi-route sets as unfiltered duplicated-BCE targets.

The strongest objection is cache incompleteness: MCTS observed only 200 or 400
rollouts per sample, so a predicted mask outside the cached valid set may still
execute correctly. This makes cached Hit@1 a coherence diagnostic rather than
the final behavioral metric. It does not invalidate the duplicated-BCE oracle
finding—its own target misses the known selected positives—but it requires
actual held-out mask execution before judging a trained predictor.

No predictor training, final WeMath training split, or supervision-policy
amendment was executed in this action.
