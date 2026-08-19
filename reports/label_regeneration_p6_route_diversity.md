# Label Regeneration P6 Route-Diversity Analysis

Status: **PASS**

This outcome-blind post-generation analysis uses every valid complete mask from the strict P4-frozen GQA, TextVQA, and ChartQA cache. WeMath2.0-Pro is excluded. No valid-route cap or later training-view selection was applied.

## Primary sample-balanced geometry

Each positive sample contributes one mean so samples with many successful routes do not dominate.

| Dataset | Positive samples | Valid masks | Mean ON | Mean transitions | Mean Hamming to min | Mean pairwise Hamming | Mean run length |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 3,386 | 353,518 | 14.37 | 13.41 | 12.64 | 13.58 | 1.99 |
| TEXTVQA | 1,746 | 98,344 | 15.91 | 13.21 | 12.18 | 13.19 | 2.07 |
| CHARTQA | 1,785 | 76,185 | 16.29 | 12.79 | 11.59 | 13.10 | 2.44 |
| **Total** | **6,917** | **528,047** | **15.26** | **13.20** | **12.25** | **13.36** | **2.13** |

## Route- and pair-weighted distributions

These summaries describe all masks or all within-sample mask pairs and therefore give more weight to high-yield samples. They are secondary to the sample-balanced table.

| Quantity | Count | Mean | Median | P10 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Visual-ON layers | 528,047 | 14.56 | 15.0 | 11.0 | 18.0 | 19.0 | 28.0 |
| ON/OFF transitions | 528,047 | 13.44 | 14.0 | 10.0 | 17.0 | 18.0 | 25.0 |
| Hamming to minimum-ON route | 528,047 | 13.88 | 14.0 | 10.0 | 17.0 | 18.0 | 28.0 |
| Within-sample pairwise Hamming | 36,163,535 | 13.88 | 14.0 | 10.0 | 17.0 | 18.0 | 28.0 |
| Maximal run length | 7,625,375 | 1.94 | 1.0 | 1.0 | 4.0 | 5.0 | 28.0 |
| ON-run length | 3,851,281 | 2.00 | 1.0 | 1.0 | 4.0 | 5.0 | 28.0 |
| OFF-run length | 3,774,094 | 1.88 | 1.0 | 1.0 | 4.0 | 4.0 | 28.0 |

## Transition and segment structure

- Valid masks with at most 3 transitions: `5,389` (`1.02%`).
- Valid masks with at least 14 transitions: `268,174` (`50.79%`).
- Valid ALL-OFF anchors: `1,324`; valid ALL-ON anchors: `4,045`.
- Segment means use maximal contiguous runs of equal ON/OFF decisions; no POLAR segmentation constraint was imposed during MCTS.

## Current ALL-ON status stratification

| Dataset/status | Positive samples | Valid masks | Mean transitions | Mean pairwise Hamming |
|---|---:|---:|---:|---:|
| GQA / correct | 2,000 | 268,810 | 13.31 | 13.82 |
| GQA / wrong | 1,386 | 84,708 | 13.55 | 13.20 |
| TEXTVQA / correct | 1,034 | 70,886 | 13.09 | 13.35 |
| TEXTVQA / wrong | 712 | 27,458 | 13.39 | 12.93 |
| CHARTQA / correct | 1,011 | 52,825 | 12.26 | 13.08 |
| CHARTQA / wrong | 774 | 23,360 | 13.49 | 13.13 |

## Interpretation boundary

The observed quantities describe the geometry of MCTS-discovered successful masks. High Hamming distance or frequent transitions indicates that the raw valid sets are not merely duplicate masks, but it does not prove that a factorized predictor can generalize, that segment prediction will fail, or that any route yields real latency reduction. A later POLAR representation remains a controlled derived baseline from these same masks.

## Integrity and scope

- P4 record-index SHA-256: `f61eb0ac6c40e0498cdfaa53c328b3de34cbb67733a8fbd44c5bd590db051ebe`.
- P5 per-sample SHA-256: `4ef2d6e1f5cbdd4506068fc2b8e0896256678bc162a8c7a3e1ebb55e37ff744d`.
- Raw records checksum-verified: `8,000`.
- Per-sample diversity rows: `8,000` at `outputs/label_regeneration/v1/post_generation/per_sample_route_diversity_v1.jsonl`.
- Aggregate JSON: `outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json`.
- P7 splitting, P8 derived supervision, P9 final freeze, and P10 predictor training were not executed.

## P6 decision

P6 passes because all valid masks were analyzed without truncation, exact within-sample pairwise distances were computed, sample-balanced and weighted summaries were both saved, and all inputs remain checksum-bound. The next bounded action is P7 image-group-disjoint predictor split freezing.
