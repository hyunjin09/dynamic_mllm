# WRITE local learnability: W-LOCAL-WEAK

Primary F_ALL/MLP reaches Spearman 0.04752 / AUROC 0.48188 and does not materially beat either baseline. Dense-W refits remain weak at 0.04248/0.48512. Full-population Recall@90% precision is 0.000461 (0.046%); harmful correctness-flip AUROC is 0.39109. Fusion lowers Spearman to 0.03718 despite AUROC 0.50614. The strongest source-cell rho is canonical TextVQA 0.24138, but it has only 68 states, AUROC 0.54047 and no high-precision subset; no niche is validated. Filtering the original OOF predictions to 13,772 nonzero-HW states yields Spearman 0.04250 / AUROC 0.52187, with no refit or gate change. This sensitivity preserves the weak local conclusion while exposing the zero-label denominator difference from READ.

Complete image-group-disjoint five-fold OOF estimates, mean of three raw-unit seed predictions per state. All models use the frozen capacity/optimizer/early-stop recipe, fold-local normalization and UID-balanced fit weights. Primary F_ALL/MLP was fixed prospectively. Generic-prestate and one-step delta baselines explicitly negate both old targets and raw predictions, then align exact states. Harmful means HW>0, with zero states included as nonharmful; legacy nonzero-only metrics are separately saved. Dense-W entries are actual refits.

| population   | feature_group   | model   |   states |   spearman |   harmful_auroc |   precision_at_0.1 |   recall_at_precision_0.9 |   harmful_flip_auroc |
|:-------------|:----------------|:--------|---------:|-----------:|----------------:|-------------------:|--------------------------:|---------------------:|
| full         | F1              | linear  |    15185 |     0.0311 |          0.4465 |             0.2877 |                    0.0000 |               0.3940 |
| full         | F1              | mlp     |    15185 |     0.0459 |          0.4474 |             0.2403 |                    0.0000 |               0.3871 |
| full         | F2              | linear  |    15185 |    -0.0028 |          0.4168 |             0.1804 |                    0.0002 |               0.4291 |
| full         | F2              | mlp     |    15185 |     0.0200 |          0.4341 |             0.2423 |                    0.0002 |               0.4445 |
| full         | F3              | linear  |    15185 |     0.0277 |          0.4603 |             0.3910 |                    0.0003 |               0.3588 |
| full         | F3              | mlp     |    15185 |     0.0252 |          0.4718 |             0.3562 |                    0.0000 |               0.3680 |
| full         | F4              | linear  |    15185 |     0.0309 |          0.4388 |             0.2403 |                    0.0000 |               0.3762 |
| full         | F4              | mlp     |    15185 |     0.0532 |          0.4652 |             0.3864 |                    0.0000 |               0.3451 |
| full         | F5              | linear  |    15185 |     0.0145 |          0.4265 |             0.0994 |                    0.0000 |               0.4802 |
| full         | F5              | mlp     |    15185 |     0.0106 |          0.4257 |             0.1567 |                    0.0000 |               0.4547 |
| full         | F7              | linear  |    15185 |    -0.0026 |          0.4954 |             0.4220 |                    0.0002 |               0.5324 |
| full         | F7              | mlp     |    15185 |     0.0074 |          0.4802 |             0.3641 |                    0.0002 |               0.5065 |
| full         | F_ALL           | linear  |    15185 |     0.0270 |          0.4561 |             0.3186 |                    0.0003 |               0.3760 |
| full         | F_ALL           | mlp     |    15185 |     0.0475 |          0.4819 |             0.4457 |                    0.0005 |               0.3911 |
| full         | NUISANCE        | linear  |    15185 |     0.0168 |          0.4505 |             0.3022 |                    0.0000 |               0.3960 |
| full         | NUISANCE        | mlp     |    15185 |     0.0395 |          0.4950 |             0.4529 |                    0.0003 |               0.3784 |
| full         | FUSION          | linear  |    15185 |     0.0247 |          0.4993 |             0.3562 |                    0.0000 |               0.4756 |
| full         | FUSION          | mlp     |    15185 |     0.0372 |          0.5061 |             0.5049 |                    0.0000 |               0.4592 |
| dense_w      | F_ALL           | mlp     |    14228 |     0.0425 |          0.4851 |             0.4800 |                    0.0002 |               0.3843 |
| dense_w      | PRE             | mlp     |    14228 |     0.0385 |          0.5018 |             0.5088 |                    0.0000 |               0.4316 |
| dense_w      | DELTA           | mlp     |    14228 |     0.0353 |          0.4967 |             0.4968 |                    0.0000 |               0.4511 |
| dense_w      | F_ALL           | linear  |    14228 |     0.0281 |          0.4744 |             0.3851 |                    0.0002 |               0.3817 |
| full         | B1_GENERIC      | mlp     |    15185 |     0.0359 |          0.4984 |             0.5003 |                    0.0002 |               0.4163 |
| full         | B2_DELTA        | mlp     |    15185 |     0.0380 |          0.4897 |             0.5043 |                    0.0000 |               0.4419 |

Paired 5,000-draw image-group differences:

| population   | model     | baseline   |   spearman_gain |   spearman_ci_low |   spearman_ci_high |   harmful_auc_gain |   harmful_auc_ci_low |   harmful_auc_ci_high |   draws | bootstrap_unit   |
|:-------------|:----------|:-----------|----------------:|------------------:|-------------------:|-------------------:|---------------------:|----------------------:|--------:|:-----------------|
| full         | F_ALL/mlp | B1_GENERIC |          0.0116 |           -0.0127 |             0.0374 |            -0.0165 |              -0.0304 |               -0.0017 |    5000 | image_group_id   |
| full         | F_ALL/mlp | B2_DELTA   |          0.0095 |           -0.0137 |             0.0332 |            -0.0078 |              -0.0210 |                0.0053 |    5000 | image_group_id   |
| dense_w      | F_ALL/mlp | PRE        |          0.0040 |           -0.0232 |             0.0323 |            -0.0167 |              -0.0317 |               -0.0018 |    5000 | image_group_id   |
| dense_w      | F_ALL/mlp | DELTA      |          0.0071 |           -0.0167 |             0.0304 |            -0.0116 |              -0.0250 |                0.0015 |    5000 | image_group_id   |

The gate requires material improvement over both baselines, Dense-W survival and useful high precision. Each component is recorded in local_decision_gate.json. A statistically nonzero but sub-threshold gain does not pass. All family ablations and fusion controls above remain secondary; no best-family substitution was made.

Early/middle/late breakdown:

| depth_bin   |   spearman |   pearson |    mae |   rmse |   harmful_auroc |   harmful_auprc |   harmful_prevalence |   states |   precision_at_0.05 |   precision_at_0.1 |   precision_at_0.2 |   recall_at_precision_0.9 |   recall_at_precision_0.95 |   harmful_flip_auroc |   harmful_flip_auprc |
|:------------|-----------:|----------:|-------:|-------:|----------------:|----------------:|---------------------:|---------:|--------------------:|-------------------:|-------------------:|--------------------------:|---------------------------:|---------------------:|---------------------:|
| Early       |    -0.0500 |   -0.0374 | 0.1896 | 0.2935 |          0.4845 |          0.4991 |               0.5124 |      603 |              0.4194 |             0.4426 |             0.4298 |                    0.0065 |                     0.0065 |               0.4076 |               0.0255 |
| Middle      |     0.0599 |    0.0299 | 0.1658 | 0.2767 |          0.5370 |          0.5041 |               0.4731 |     4669 |              0.5256 |             0.5225 |             0.5118 |                    0.0000 |                     0.0000 |               0.3442 |               0.0286 |
| Late        |     0.0321 |    0.0127 | 0.0836 | 0.1450 |          0.4421 |          0.3685 |               0.4018 |     9913 |              0.3911 |             0.3700 |             0.3399 |                    0.0003 |                     0.0003 |               0.4322 |               0.0214 |

Dataset/source niches:

| dataset   | source_regime   |   spearman |   pearson |    mae |   rmse |   harmful_auroc |   harmful_auprc |   harmful_prevalence |   states |   precision_at_0.05 |   precision_at_0.1 |   precision_at_0.2 |   recall_at_precision_0.9 |   recall_at_precision_0.95 |   harmful_flip_auroc |   harmful_flip_auprc |
|:----------|:----------------|-----------:|----------:|-------:|-------:|----------------:|----------------:|---------------------:|---------:|--------------------:|-------------------:|-------------------:|--------------------------:|---------------------------:|---------------------:|---------------------:|
| chartqa   | canonical       |     0.0405 |    0.1089 | 0.0584 | 0.1098 |          0.4039 |          0.3475 |               0.4012 |      339 |              0.1765 |             0.2059 |             0.2500 |                    0.0147 |                     0.0147 |               0.2160 |               0.0160 |
| chartqa   | historical      |     0.0673 |    0.0645 | 0.0738 | 0.1315 |          0.4465 |          0.3578 |               0.4036 |     2205 |              0.2252 |             0.2308 |             0.2948 |                    0.0000 |                     0.0000 |               0.3596 |               0.0297 |
| gqa       | canonical       |     0.0187 |    0.0045 | 0.1459 | 0.2536 |          0.4875 |          0.4557 |               0.4514 |     3857 |              0.5181 |             0.5104 |             0.4650 |                    0.0000 |                     0.0000 |               0.4893 |               0.0214 |
| gqa       | historical      |     0.0199 |    0.0046 | 0.1397 | 0.2333 |          0.4763 |          0.4278 |               0.4390 |     4902 |              0.4106 |             0.4399 |             0.4393 |                    0.0019 |                     0.0019 |               0.5012 |               0.0136 |
| textvqa   | canonical       |     0.2414 |    0.3520 | 0.0239 | 0.0376 |          0.5405 |          0.4445 |               0.3676 |       68 |              0.5000 |             0.5714 |             0.5000 |                    0.0000 |                     0.0000 |               0.9545 |               0.3929 |
| textvqa   | historical      |     0.0685 |    0.0549 | 0.0750 | 0.1263 |          0.4800 |          0.3989 |               0.4082 |     3814 |              0.4607 |             0.4110 |             0.3984 |                    0.0000 |                     0.0000 |               0.4100 |               0.0427 |

Small-cell highs are descriptive, not validated transfer. Strong correctness-flip ranking is measured separately from continuous harm and all-positive harm ranking:

| cohort                |   states |   median_predicted_H_W |   mean_predicted_H_W |
|:----------------------|---------:|-----------------------:|---------------------:|
| stable_correct        |      917 |                -0.0041 |              -0.0044 |
| stable_wrong          |    13780 |                -0.0052 |              -0.0053 |
| write_beneficial_flip |       40 |                -0.0024 |              -0.0024 |
| write_harmful_flip    |      448 |                -0.0081 |              -0.0081 |

Utility sign and correctness flips are distinct: 364 of 448 harmful correctness flips have positive HW, and 84 have negative HW; 37 of 40 beneficial flips have negative HW, and 3 have positive HW. The continuous surrogate and discrete behavior therefore require separate evaluations. This mismatch is descriptive and does not by itself explain model failures. The local study does not establish a useful router, benchmark improvement, savings, or impossibility of later/history-dependent information. The authorized W3 discriminator is now complete; see write_propagation_summary.md for qualified W-PROP-C.
