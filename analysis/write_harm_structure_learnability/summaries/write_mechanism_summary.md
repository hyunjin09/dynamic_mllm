# WRITE mechanism diagnostic

Same-layer text parity passed on all 15,185 compact cached states, with full-text fresh replay on 32 UIDs /346 states. READ_ONLY visual states exactly equal the pre-WRITE states; both branch hashes match frozen parent artifacts. No target was regenerated. Features use all valid visual tokens, exact covariance spectrum, and validated image-grid coordinates. Optional attention-source F6 was omitted; this limits claims about attention-level causes.

Selected distribution summaries (FULL-minus-READ_ONLY updates):

| feature                          | cohort                |   states |      mean |    median |       std |
|:---------------------------------|:----------------------|---------:|----------:|----------:|----------:|
| f1_frobenius                     | ALL                   |    15185 | 2124.9731 | 1441.3129 | 2202.2976 |
| f1_frobenius                     | write_harmful_flip    |      448 | 1694.9119 | 1268.0876 | 1147.5348 |
| f1_frobenius                     | write_beneficial_flip |       40 | 1125.9888 |  814.8518 |  809.5373 |
| f1_frobenius                     | stable_wrong          |    13780 | 2153.3166 | 1447.4527 | 2241.8135 |
| f1_frobenius                     | stable_correct        |      917 | 1952.7291 | 1414.3386 | 1986.1170 |
| f1_over_pre                      | ALL                   |    15185 |    0.3813 |    0.3596 |    0.1360 |
| f1_over_pre                      | write_harmful_flip    |      448 |    0.3459 |    0.3320 |    0.0696 |
| f1_over_pre                      | write_beneficial_flip |       40 |    0.3204 |    0.3053 |    0.0662 |
| f1_over_pre                      | stable_wrong          |    13780 |    0.3823 |    0.3609 |    0.1371 |
| f1_over_pre                      | stable_correct        |      917 |    0.3851 |    0.3675 |    0.1420 |
| f3_top1_share                    | ALL                   |    15185 |    0.0055 |    0.0061 |    0.0033 |
| f3_top1_share                    | write_harmful_flip    |      448 |    0.0040 |    0.0029 |    0.0029 |
| f3_top1_share                    | write_beneficial_flip |       40 |    0.0066 |    0.0067 |    0.0025 |
| f3_top1_share                    | stable_wrong          |    13780 |    0.0055 |    0.0060 |    0.0033 |
| f3_top1_share                    | stable_correct        |      917 |    0.0068 |    0.0070 |    0.0029 |
| f4_full_minus_off_pair_cosine    | ALL                   |    15185 |   -0.0016 |    0.0097 |    0.0536 |
| f4_full_minus_off_pair_cosine    | write_harmful_flip    |      448 |    0.0158 |    0.0150 |    0.0297 |
| f4_full_minus_off_pair_cosine    | write_beneficial_flip |       40 |    0.0047 |    0.0095 |    0.0228 |
| f4_full_minus_off_pair_cosine    | stable_wrong          |    13780 |   -0.0019 |    0.0095 |    0.0537 |
| f4_full_minus_off_pair_cosine    | stable_correct        |      917 |   -0.0065 |    0.0087 |    0.0598 |
| f4_full_minus_off_effective_rank | ALL                   |    15185 |    0.5600 |    1.3747 |    8.8953 |
| f4_full_minus_off_effective_rank | write_harmful_flip    |      448 |    2.3504 |    1.6373 |    5.0239 |
| f4_full_minus_off_effective_rank | write_beneficial_flip |       40 |    2.1412 |    1.1178 |    3.8068 |
| f4_full_minus_off_effective_rank | stable_wrong          |    13780 |    0.4847 |    1.3837 |    9.0767 |
| f4_full_minus_off_effective_rank | stable_correct        |      917 |    0.7473 |    1.1840 |    7.5826 |
| f5_full_minus_off_mean           | ALL                   |    15185 |   -0.0072 |    0.0088 |    0.0531 |
| f5_full_minus_off_mean           | write_harmful_flip    |      448 |    0.0107 |    0.0108 |    0.0215 |
| f5_full_minus_off_mean           | write_beneficial_flip |       40 |    0.0036 |    0.0063 |    0.0152 |
| f5_full_minus_off_mean           | stable_wrong          |    13780 |   -0.0076 |    0.0087 |    0.0534 |
| f5_full_minus_off_mean           | stable_correct        |      917 |   -0.0103 |    0.0101 |    0.0581 |

Within exact nuisance-matched harmful versus beneficial correctness flips, the largest absolute standardized effect in each prespecified feature family is shown below. These are exploratory maxima across correlated features; unadjusted bootstrap intervals are not multiple-testing-corrected discovery claims. Image-group resampling uses both matched arms with shared multiplicities and separately normalized means.

| family   | description           | feature                            |   effect |   raw_difference_ci_low |   raw_difference_ci_high |   features_ci_excludes_zero |   tested_features |   pairs |
|:---------|:----------------------|:-----------------------------------|---------:|------------------------:|-------------------------:|----------------------------:|------------------:|--------:|
| f1       | magnitude             | f1_max_token_norm                  |   0.0818 |                -48.7156 |                  64.3746 |                           0 |                 8 |      22 |
| f2       | direction             | f2_cos_pre_min                     |   0.1306 |                 -0.1093 |                   0.1349 |                           0 |                24 |      22 |
| f3       | token concentration   | f3_top5_share                      |   0.1733 |                 -0.0085 |                   0.0134 |                           0 |                 5 |      22 |
| f4       | visual diversity      | f4_full_minus_pre_spectral_entropy |  -0.1652 |                 -0.0474 |                   0.0275 |                           0 |                25 |      22 |
| f5       | query alignment       | f5_pre_max                         |  -0.2480 |                 -0.0446 |                   0.0239 |                           0 |                15 |      22 |
| f7       | spatial concentration | f7_index_spread                    |   0.5222 |                 -0.0014 |                   0.0046 |                           0 |                 8 |      22 |

Full harmful-versus-stable-wrong/correct matches and all feature effects are retained in matched_mechanism. Positive effective-rank changes mean expansion, negative changes mean collapse; positive query-alignment change means stronger alignment with the fixed pre-query. Magnitude/direction/concentration contrasts are associations conditional on the exact matching support, not evidence that a geometric change caused answer harm.

Matched logistic/MLP probes:

| population   | feature_group   | model   |   spearman |   pearson |   mae |   rmse |   harmful_auroc |   harmful_auprc |   harmful_prevalence |   states |   precision_at_0.05 |   precision_at_0.1 |   precision_at_0.2 |   recall_at_precision_0.9 |   recall_at_precision_0.95 |   harmful_flip_auroc |   harmful_flip_auprc |
|:-------------|:----------------|:--------|-----------:|----------:|------:|-------:|----------------:|----------------:|---------------------:|---------:|--------------------:|-------------------:|-------------------:|--------------------------:|---------------------------:|---------------------:|---------------------:|
| matched      | F_ALL           | linear  |        nan |       nan |   nan |    nan |          0.4215 |          0.4901 |                  nan |       44 |                 nan |             0.4000 |                nan |                       nan |                        nan |                  nan |                  nan |
| matched      | F_ALL           | mlp     |        nan |       nan |   nan |    nan |          0.4132 |          0.4721 |                  nan |       44 |                 nan |             0.2000 |                nan |                       nan |                        nan |                  nan |                  nan |

No harmful-versus-beneficial feature interval excludes zero across all 85 frozen scalar features in this completed census. There is no stable feature family to designate as a robust mechanism; the largest exploratory standardized point contrast is spatial index spread, but its raw mean-difference interval includes zero. Matched probe AUROC is 0.4215 (linear) and 0.4132 (MLP), with 22 matched pairs. Only 40 beneficial flips exist before matching; strict matched support is consequently small. Probe support and unestimable folds are recorded separately. Feature effects do not by themselves justify a specific WRITE mechanism, a causal mediator, or deployment performance.
