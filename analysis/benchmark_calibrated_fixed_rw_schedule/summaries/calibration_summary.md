# Calibration complete; held-out results pending

Both sweeps are complete on the same 1,019 CAL UIDs: 28,532 READ branches and 28,532 WRITE branches. All 1,019 terminal-layer WRITE controls had identical generated tokens and correctness to Dense within the repaired executor; q also matched wherever valid. Layer 27 WRITE was excluded from selection. Dense separately achieved exact native token parity on all 20,471 CAL/TEST UIDs. No Stage1 or learned router is used.

The following are calibration measurements, not held-out improvements. Primary selectors, nested pools and random seeds were fixed prospectively. Benchmark, global and random schedules were frozen before the first TEST intervention.

## Primary selections

No bit selected NONE. Gains are proportions; net is the count of corrections minus regressions. Bootstrap resamples complete image/content groups.

| benchmark   | bit   |   selected_layer |   n |    gain |     net |   w_to_c |   c_to_w |    mean_q |   primary_selection_frequency |   selection_entropy_nats |
|:------------|:------|-----------------:|----:|--------:|--------:|---------:|---------:|----------:|------------------------------:|-------------------------:|
| chartqa     | R     |               12 | 256 | 0.01172 | 3.00000 |  4.00000 |  1.00000 |   0.00220 |                       0.28600 |                  2.06390 |
| chartqa     | W     |               25 | 256 | 0.01172 | 3.00000 |  3.00000 |  0.00000 |   0.00064 |                       0.29800 |                  1.59952 |
| textvqa     | R     |               11 | 255 | 0.01176 | 3.00000 |  3.00000 |  0.00000 | nan       |                       0.58650 |                  1.52195 |
| textvqa     | W     |               24 | 255 | 0.01176 | 3.00000 |  4.00000 |  1.00000 | nan       |                       0.55600 |                  1.56952 |
| mmmupro     | R     |               17 | 256 | 0.01562 | 4.00000 |  7.00000 |  3.00000 |   0.01640 |                       0.29250 |                  1.96474 |
| mmmupro     | W     |               14 | 256 | 0.01953 | 5.00000 | 12.00000 |  7.00000 |   0.07697 |                       0.39300 |                  2.06647 |
| pope        | R     |                7 | 252 | 0.01190 | 3.00000 |  3.00000 |  0.00000 |   0.02946 |                       0.56100 |                  1.00338 |
| pope        | W     |               14 | 252 | 0.01190 | 3.00000 |  3.00000 |  0.00000 |   0.01182 |                       0.22250 |                  1.70864 |

## Split-half structure

| benchmark   | bit   |   spearman |   half_a_layer |   half_b_layer | top1_agreement   |   top3_overlap |   sign_agreement |
|:------------|:------|-----------:|---------------:|---------------:|:-----------------|---------------:|-----------------:|
| chartqa     | R     |    0.50064 |             17 |       24.00000 | False            |              0 |          0.50000 |
| chartqa     | W     |    0.67696 |              3 |       25.00000 | False            |              1 |          0.59259 |
| textvqa     | R     |    0.09159 |             11 |       12.00000 | False            |              0 |          0.46429 |
| textvqa     | W     |    0.33411 |             24 |       15.00000 | False            |              1 |          0.55556 |
| mmmupro     | R     |    0.35027 |              2 |       16.00000 | False            |              0 |          0.42857 |
| mmmupro     | W     |    0.25281 |             14 |        7.00000 | False            |              0 |          0.40741 |
| pope        | R     |    0.38340 |              7 |      nan       | False            |              2 |          0.60714 |
| pope        | W     |    0.00000 |             14 |       12.00000 | False            |              0 |          0.55556 |

No split-half top-1 pair agrees. This observation alone does not settle held-out utility; the required frozen global/random controls are still pending. Undefined correlations and NONE are not labeled unstable.

## Nested calibration pools

| benchmark   | bit   |   target_n |   actual_n |   image_groups |   selected_layer |   selected_gain |   rank_under_full_pool |
|:------------|:------|-----------:|-----------:|---------------:|-----------------:|----------------:|-----------------------:|
| chartqa     | R     |         32 |         32 |             14 |         17.00000 |         0.03125 |                8.00000 |
| chartqa     | R     |         64 |         63 |             28 |         24.00000 |         0.01587 |                2.00000 |
| chartqa     | R     |        128 |        127 |             58 |         17.00000 |         0.01575 |                8.00000 |
| chartqa     | R     |        256 |        256 |            118 |         12.00000 |         0.01172 |                1.00000 |
| chartqa     | W     |         32 |         32 |             14 |          1.00000 |         0.03125 |               20.00000 |
| chartqa     | W     |         64 |         63 |             28 |         12.00000 |         0.03175 |               12.00000 |
| chartqa     | W     |        128 |        127 |             58 |          3.00000 |         0.02362 |                2.00000 |
| chartqa     | W     |        256 |        256 |            118 |         25.00000 |         0.01172 |                1.00000 |
| textvqa     | R     |         32 |         31 |             20 |         14.00000 |         0.03226 |               23.00000 |
| textvqa     | R     |         64 |         64 |             41 |         19.00000 |         0.03125 |               11.00000 |
| textvqa     | R     |        128 |        127 |             78 |         14.00000 |         0.00787 |               23.00000 |
| textvqa     | R     |        256 |        255 |            160 |         11.00000 |         0.01176 |                1.00000 |
| textvqa     | W     |         32 |         31 |             20 |         24.00000 |         0.03226 |                1.00000 |
| textvqa     | W     |         64 |         64 |             41 |          9.00000 |         0.01562 |                6.00000 |
| textvqa     | W     |        128 |        127 |             78 |         23.00000 |         0.00787 |                3.00000 |
| textvqa     | W     |        256 |        255 |            160 |         24.00000 |         0.01176 |                1.00000 |
| mmmupro     | R     |         32 |         30 |             15 |         16.00000 |         0.06667 |                3.00000 |
| mmmupro     | R     |         64 |         64 |             31 |         16.00000 |         0.04688 |                3.00000 |
| mmmupro     | R     |        128 |        128 |             63 |         18.00000 |         0.01562 |                4.00000 |
| mmmupro     | R     |        256 |        256 |            127 |         17.00000 |         0.01562 |                1.00000 |
| mmmupro     | W     |         32 |         30 |             15 |          9.00000 |         0.06667 |               10.00000 |
| mmmupro     | W     |         64 |         64 |             31 |          9.00000 |         0.04688 |               10.00000 |
| mmmupro     | W     |        128 |        128 |             63 |          1.00000 |         0.03906 |                3.00000 |
| mmmupro     | W     |        256 |        256 |            127 |         14.00000 |         0.01953 |                1.00000 |
| pope        | R     |         32 |         18 |              1 |        nan       |         0.00000 |              nan       |
| pope        | R     |         64 |         54 |              3 |         14.00000 |         0.01852 |               28.00000 |
| pope        | R     |        128 |        126 |              7 |        nan       |         0.00000 |              nan       |
| pope        | R     |        256 |        252 |             14 |          7.00000 |         0.01190 |                1.00000 |
| pope        | W     |         32 |         18 |              1 |        nan       |         0.00000 |              nan       |
| pope        | W     |         64 |         54 |              3 |        nan       |         0.00000 |              nan       |
| pope        | W     |        128 |        126 |              7 |         12.00000 |         0.01587 |                9.00000 |
| pope        | W     |        256 |        252 |             14 |         14.00000 |         0.01190 |                1.00000 |

## Population and validity

ChartQA uses 256 CAL UIDs / 118 groups and 2,500 TEST UIDs. TextVQA uses 255 / 160 and 5,000 TEST. Both CAL pools use locally available official train frames, with their original TEST sets intact. MMMU-Pro uses 256 / 127 and 3,204 TEST; paired Standard/Vision content stays within one split. POPE uses 252 / 14 and 8,748 TEST; all co-image questions and variants stay together. POPE therefore has only 14 independent calibration images. MMMU-Pro and POPE TEST figures cover the frozen held-out partition, not their full official evaluation populations. Raw-byte, decoded-RGB and native-identity transitive grouping found no CAL/TEST leakage.

One CAL and four TEST TextVQA rows have empty normalized q references; correctness remains included. Incomplete q pools disable q tie-breaking for the entire relevant bit/pool. The primary TextVQA and pooled global selectors therefore skip q; nested/half pools without that row may retain it.

Global selection is READ 17 / WRITE 19. Twenty seeded matched-bit random schedules per family are frozen, sampled with replacement; ChartQA has 19 unique random routes and the other families have 20. Duplicate draws retain their statistical weight.

See [schedule freeze report](../schedules/schedule_freeze_report.md), [split audit](../splits/leakage_check.md), and the complete CSVs under `calibration/`. No held-out outcome has been used to revise a schedule. The independent calibration-scope review shows that A/B/C are precluded by their frozen stability predicates, so residual qualified CAL-D is already determined. This is not evidence of no held-out efficacy; completing the required held-out/control matrix remains necessary. See [scope review](../parity/calibration_category_scope_review.md).
