# Four-Action Answer-Unaligned Report

## Scope and causal contract

The matched cache supplied 1,912 candidate A+ samples. After freezing current unified-FULL correctness, the primary sweep contains 1,880 eligible samples: 1,222 GQA and 658 TextVQA; 32 candidates were excluded because current unified FULL no longer satisfied the frozen cohort's FULL-wrong condition. Every sample was evaluated at all 28 decoder layers with M00=IGNORE, M10=READ_ONLY, M01=WRITE_ONLY, and M11=unified FULL.

All factorial effects below are internal to the unified executor. Native Qwen FULL is used only for the matched-cache candidate cohort, generation/correctness sanity checks, and implementation-drift measurement. No native/unified drift value is used as a causal-effect threshold.

The eligibility freeze evaluated 4,890 total primary/control candidates and retained 4,832; the per-cohort and per-dataset counts are preserved in the eligibility summary.

## Semantic validation and implementation drift

| Comparison | Comparisons | Token-ID matches | Answer matches | Evaluator-score matches | Correctness matches |
|---|---:|---:|---:|---:|---:|
| Unified FULL vs native FULL | 72 | 72 | 72 | 72 | 72 |
| Unified IGNORE vs old binary single-OFF | 1816 | 1816 | 1816 | 1816 | 1816 |

Native-to-unified FULL margin drift is signed as unified minus native:

| Distribution | Mean | Median | Std | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| signed | -0.0000 | 0.0000 | 0.0517 | 0.0498 | 0.0899 | 0.1250 | 1.1250 |
| absolute | 0.0257 | 0.0033 | 0.0449 | 0.0899 | 0.1250 | 0.1388 | 1.1250 |

## Within-unified layerwise factorial landscape

These extrema summarize population means; the full distributions and sample- and image-group bootstrap intervals are in the aggregate tables.

| Effect | Most negative mean layer | Mean | Most positive mean layer | Mean |
|---|---:|---:|---:|---:|
| read_w1 | 19 | -0.0992 | 5 | 0.0399 |
| read_w0 | 19 | -0.0995 | 5 | 0.0404 |
| write_r1 | 0 | -0.0945 | 22 | 0.0168 |
| write_r0 | 0 | -0.0909 | 22 | 0.0206 |
| interaction | 26 | -0.0077 | 14 | 0.0068 |

## Discrete local rescues

- gqa: 380/1,222 samples had at least one local rescue; median rescue layers per sample was 0.0.
- textvqa: 456/658 samples had at least one local rescue; median rescue layers per sample was 2.0.
- joint: 836/1,880 samples had at least one local rescue; median rescue layers per sample was 0.0.

## Relationship to binary correcting routes

Positive OFF-minus-ON values for ignore gain or harmfulness mean the nearest correcting routes preferentially turn off more locally harmful layers. Negative OFF-minus-ON READ/WRITE values mean the OFF layers have more negative conditional effects.

| Local quantity | Mean nearest-route OFF minus ON | Mean within-sample OFF-frequency Spearman |
|---|---:|---:|
| ignore_gain_m00_minus_m11 | 0.1081 | 0.1934 |
| read_w1 | -0.0674 | -0.1701 |
| strongest_local_harmfulness | 0.0429 | 0.1393 |
| write_r1 | -0.0382 | -0.0872 |

## Hamming-distance stratification

| Sample-level quantity | Spearman with nearest correcting-route distance | Pearson |
|---|---:|---:|
| negative_read_layer_count | 0.0193 | 0.0163 |
| negative_write_layer_count | -0.0513 | -0.0351 |
| negative_either_layer_count | -0.0172 | -0.0155 |
| rescue_layer_count | -0.5606 | -0.3885 |
| strongest_negative_read_magnitude | -0.0421 | -0.0589 |
| strongest_negative_write_magnitude | -0.0560 | -0.0634 |
| strongest_negative_component_magnitude | -0.0662 | -0.0654 |
| mean_negative_component_magnitude | -0.0414 | -0.0515 |
| mean_absolute_interaction | 0.0345 | 0.0138 |
| maximum_absolute_interaction | -0.0151 | -0.0275 |

## Answer erosion and local causal alignment

In the primary A+ cohort, 93.7% of samples had a positive intermediate correct-vs-FULL-wrong margin. Mean peak-to-final erosion was 4.3271. The strongest harmful local operation lay within two layers of the largest trajectory drop for 16.1% of samples, versus a deterministic random-layer reference of 16.2%.

Population-level single-operation trajectory reruns:

| Culprit operation | Reruns | Mean final-margin improvement | Mean erosion reduction | Fraction final margin improved |
|---|---:|---:|---:|---:|
| JOINT | 689 | 0.4690 | 0.2938 | 0.9303 |
| READ | 4504 | 0.6384 | 0.4273 | 0.9645 |
| WRITE | 5003 | 0.6243 | 0.4100 | 0.9660 |

## Controls

The no-correction control means only that no correcting route was found under the matched binary-search budget. The full cohort-comparison table contains negative fractions and distribution-derived q75/q90 enrichment for each effect, separately by dataset and jointly.

| Cohort | Effect | Mean | Median | Negative fraction | Strong-negative q90 fraction |
|---|---|---:|---:|---:|---:|
| control_full_correct_all_off_wrong | read_w1 | 0.0170 | 0.0000 | 0.4538 | 0.0004 |
| control_full_correct_all_off_wrong | read_w0 | 0.0168 | 0.0000 | 0.4486 | 0.0008 |
| control_full_correct_all_off_wrong | write_r1 | 0.0191 | 0.0000 | 0.4254 | 0.0002 |
| control_full_correct_all_off_wrong | write_r0 | 0.0189 | 0.0000 | 0.4286 | 0.0007 |
| control_full_correct_all_off_wrong | interaction | 0.0002 | 0.0000 | 0.4847 | 0.0050 |
| control_no_correction_found | read_w1 | -0.0057 | -0.0000 | 0.5026 | 0.0427 |
| control_no_correction_found | read_w0 | -0.0063 | -0.0000 | 0.5080 | 0.0430 |
| control_no_correction_found | write_r1 | -0.0083 | 0.0000 | 0.4899 | 0.0420 |
| control_no_correction_found | write_r0 | -0.0090 | 0.0000 | 0.4925 | 0.0424 |
| control_no_correction_found | interaction | 0.0007 | 0.0000 | 0.4790 | 0.0355 |
| primary_a_plus | read_w1 | -0.0181 | -0.0000 | 0.5046 | 0.0561 |
| primary_a_plus | read_w0 | -0.0175 | -0.0000 | 0.5053 | 0.0562 |
| primary_a_plus | write_r1 | -0.0237 | 0.0000 | 0.4950 | 0.0576 |
| primary_a_plus | write_r0 | -0.0231 | 0.0000 | 0.4927 | 0.0578 |
| primary_a_plus | interaction | -0.0006 | 0.0000 | 0.4700 | 0.0501 |

## Evidence inventory and interpretation boundary

Raw per-sample/layer/action scores, generated answers, evaluator decisions, factorial effects, route metadata, trajectories, bootstrap aggregates, and plots are retained under this analysis directory with SHA-256 sidecars. Intermediate logit-lens trajectories are supporting evidence; the exact within-unified four-action interventions are the primary causal evidence.
