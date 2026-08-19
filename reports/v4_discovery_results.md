# v4 GQA Same-Image Query-Conditional Discovery

## Scope and integrity

The bounded discovery completed exactly the approved experiment:

- 120 unique GQA images and exactly two natural questions per image;
- 240 image-question records;
- layers `[0,4,8,12,16,20,24]`;
- `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, and `FULL` at one layer at a time;
- unchanged dense prefix/suffix and accepted-reference answer-token scoring;
- 6,720 finite action scores and 1,680 complete question-layer Q matrices.

The frozen manifest SHA-256 is
`089b55c2e704c47d8c2cf9d516821d3f02dabf7e4dac47727f7a41153b267164`.
There was no inspected-record, inspected-image, or independent-calibration
overlap. No sample was replaced after outcomes were opened.

The 12-image common-padding entry gate passed. Same-image visual pre-states,
post-states, and WRITE residuals were bitwise identical at all seven layers;
visual attention to later question/padding keys was zero. Instrumented FULL
score parity was exact. Maximum READ/WRITE reconstruction-hook discrepancies
in the complete sweep were `5.96e-8` and `2.98e-8`. The per-token epsilon was
frozen at `1e-6` nats/token.

## Primary image-level results

All confidence intervals below are 10,000-draw percentile bootstrap intervals
with the image as the resampling unit. The joint row is the prespecified
equal-layer average, not a maximum over layers.

| Equal-layer per-token quantity | Mean (95% CI) | Median | 20% trimmed mean |
|---|---:|---:|---:|
| Robust epsilon-set action disagreement | 0.6762 [0.6405, 0.7119] | 0.7143 | 0.6905 |
| Bidirectional action-transfer regret | 0.1121 [0.0870, 0.1401] | 0.0634 | 0.0711 |
| Image+query minus image-only oracle gap | 0.0144 [0.0104, 0.0188] | 0.00344 | 0.00644 |
| FULL-relative V-vector distance | 0.6405 [0.5244, 0.7722] | 0.4244 | 0.4775 |
| Four-action variance | 0.2353 [0.1136, 0.3892] | 0.0165 | 0.0300 |

Action disagreement is therefore not a numerical-tie artifact: the epsilon-tie
ambiguity rate was 0.0262 across image-layer comparisons, and robust-set
disagreement differed only slightly from exact-argmax disagreement (0.6833).

Conditional sign reversals were common but are descriptive, not a semantic
mechanism result. Equal-layer rates were 0.4762 and 0.4583 for READ with WRITE
off/on, and 0.4440 and 0.4726 for WRITE with READ off/on.

## Layer distribution

| Layer | Robust disagreement | Transfer regret | Query-oracle gap |
|---:|---:|---:|---:|
| 0 | 0.5667 | 0.2399 | 0.0260 |
| 4 | 0.6917 | 0.0877 | 0.0106 |
| 8 | 0.7250 | 0.1221 | 0.0203 |
| 12 | 0.7333 | 0.1158 | 0.0171 |
| 16 | 0.6750 | 0.0911 | 0.0109 |
| 20 | 0.6750 | 0.0776 | 0.00921 |
| 24 | 0.6667 | 0.0502 | 0.00651 |

The pattern is not confined to one layer, although layer 0 contributes the
largest transfer regret and exhibits a heavy upper tail. No action dominates
globally: layer-specific exact-best frequencies range from 0.1375 to 0.3750
across the four actions.

## Robustness and scale

- On 108 images with no epsilon ambiguity at any layer, transfer regret had
  mean 0.1137, median 0.0679, and 20%-trimmed mean 0.0767.
- On 98 answer-length-matched images, these values were 0.1035, 0.0573, and
  0.0652.
- On the strict 40-image type/program-depth/answer-format/answer-length matched
  subset, they were 0.0952, 0.0440, and 0.0511.
- Five-percent winsorization gave transfer regret 0.1004; removing the largest
  5% gave 0.0854. The largest 5% of images contributed 27.6% of total transfer
  regret.
- The corresponding image-only oracle-gap estimates stayed small: complete-set
  median 0.00344 and 20%-trimmed mean 0.00644; strict-matched median 0.00275 and
  20%-trimmed mean 0.00659. Only 10% of images had an equal-layer gap at least
  0.05 nats/token.

Sequence-level results preserved the continuous pattern: sequence/per-token
Pearson correlations were 0.933 for transfer regret and 0.934 for the
query-oracle gap. Sequence transfer regret was 0.1504 mean and 0.0679 median;
the sequence query-oracle gap was 0.0192 mean and 0.00385 median.

## Semantic controls

The outcome-blind manifest contains 60 image pairs linked to distinct resolved
scene objects and 60 metadata-matched comparison pairs. It contains no official
GQA equivalent pair meeting the prospective same-answer, question-type, and
semantic-target rule; no generated paraphrases were added.

Different-evidence minus comparison estimates were mixed:

| Quantity | Paired difference (95% CI) | Covariate-adjusted coefficient |
|---|---:|---:|
| Robust action disagreement | +0.0667 [approximately 0, +0.1310] | +0.0494 |
| V-vector distance | +0.1192 [-0.1245, +0.3754] | +0.0226 |
| Transfer regret | +0.00335 [-0.0483, +0.0557] | -0.0353 |
| Query-oracle gap | -0.00792 [-0.0162, -0.00007] | -0.00636 |

Thus the discovery establishes question-associated four-action variation under
identical visual computation, but it does not establish the planned semantic
ordering. In particular, the fixed paraphrase-stability gate is not evaluable,
and the quantity most directly measuring insufficiency of one image-only action
is small and is not larger for the different-evidence stratum.

## Claim boundary

These are inspected GQA discovery results for one pinned model. They do not
establish a semantic causal mechanism, a deployable policy, pre-action
predictability, acceleration, accuracy improvement, cross-task/model
generality, or harmful visual participation. A negative action contrast is not
called harmful, and oracle quantities are not compute or latency results.

Machine-readable results and checksums are under
`outputs/v4_discovery/analysis_v1/`; the complete per-question Q/V table is
`question_layer_q_v_v1.parquet`, and the image-layer table is
`image_layer_query_dependence_v1.csv`.
