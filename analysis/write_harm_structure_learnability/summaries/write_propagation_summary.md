# WRITE propagation: W-PROP-C

The result is **qualified W-PROP-C / WRITE-S4 under the tested pooled representation and predictor family**. WRITE does influence future text: every one of the 6,044 common-support states has zero control-text difference at H0 and nonzero difference at H1. The absolute median visual difference rises from 612.48 to 1,272.82 by H8, while its mean relative norm changes from 0.2642 to 0.2456; absolute growth does not imply relative amplification.

Combined-delta Spearman on common support rises monotonically from 0.04119 to 0.09732. The H8−H0 gain is **+0.05613 [0.02670,0.08523]**; harmful AUROC improves by **+0.02255 [0.00514,0.03957]** to 0.53872. These are statistically supported but below the prospectively fixed +0.10/+0.08 materiality thresholds. Dense-W refits show the same bounded improvement: H8 Spearman 0.10377 / AUROC 0.54435, gains +0.05590/+0.02203. Dense-W's intermediate curve is not monotone.

H8 delta exceeds ON-only by 0.03806 Spearman and OFF-only by 0.04354 Spearman, below the 0.05 point-gain requirement; these intervals do not rule out larger increments. No single-branch improvement meets the material/high-precision requirements. The ordered pair remains weak (full H8 Spearman 0.03131); text-only delta reaches 0.09682 and visual-only delta 0.06031. The pre-fit secondary claim guard finds no flags and leaves the original primary category unchanged.

Full H8 delta Precision@Top10% is 0.55372 versus prevalence 0.47303 (gain 0.08069, below 0.10); Recall@90% precision is 0. Strong harmful-flip AUROC remains 0.48467. Dense-W H8 Top10 precision 0.56491 versus 0.47805 and Recall@90% precision 0.000734 also fail the useful-subset gate. Thus modest later signal exists, but it does not establish a useful harmful-WRITE detector, a mechanism-specific critic, or reliable correctness rescue. **Do not read W-PROP-C as statistical absence of all signal or proof of intrinsic nonlocality.**

WRITE H0 is immediately after intervention layer l; Hk includes k subsequent FULL layers through l+k. READ remains ON. Both branches use identical FULL continuation; no later routing/search or target regeneration. ON raw states reuse exact parent reached-layer FULL references; OFF raw compact text/all-visual states are retained and hashed. Fresh smoke verifies all ON horizons, H0 cache parity, repeated OFF, prestate identities and action traces.

Native support:

|   horizon |   states |   uids |   image_groups |   dense_w_states |   max_layer |
|----------:|---------:|-------:|---------------:|-----------------:|------------:|
|         0 |    15185 |   1413 |           1385 |            14228 |          27 |
|         1 |    13772 |   1291 |           1266 |            12921 |          26 |
|         2 |    12481 |   1221 |           1197 |            11721 |          25 |
|         4 |    10056 |   1105 |           1083 |             9457 |          23 |
|         8 |     6044 |    772 |            760 |             5696 |          19 |

Primary emergence uses the exact H8-common support (l<=19); native-support results remain descriptive. Native and common H8 are identical and share fitted predictions. Text is the last control token, visual is pooled by its mean; raw visual norms use all tokens. This is a fixed pooled-representation diagnostic, not a test of every token-level architecture.

Primary effect growth on exact H8-common support:

|   horizon | family     | cohort                |   states |   text_l2_mean |   text_l2_median |   visual_l2_mean |   visual_l2_median |   text_relative_l2_mean |   visual_relative_l2_mean |   nonzero_text_fraction |
|----------:|:-----------|:----------------------|---------:|---------------:|-----------------:|-----------------:|-------------------:|------------------------:|--------------------------:|------------------------:|
|         0 | write_sign | beneficial            |     3185 |         0.0000 |           0.0000 |         768.5559 |           633.0754 |                  0.0000 |                    0.2654 |                  0.0000 |
|         0 | write_sign | harmful               |     2859 |         0.0000 |           0.0000 |         730.0161 |           594.7005 |                  0.0000 |                    0.2630 |                  0.0000 |
|         1 | write_sign | beneficial            |     3185 |         4.4149 |           3.7162 |         800.4733 |           648.6078 |                  0.0415 |                    0.2617 |                  1.0000 |
|         1 | write_sign | harmful               |     2859 |         4.2101 |           3.4754 |         760.4639 |           619.3633 |                  0.0400 |                    0.2594 |                  1.0000 |
|         2 | write_sign | beneficial            |     3185 |         6.5060 |           5.8727 |         832.5456 |           696.6888 |                  0.0544 |                    0.2553 |                  1.0000 |
|         2 | write_sign | harmful               |     2859 |         6.2364 |           5.4853 |         789.7065 |           643.9182 |                  0.0526 |                    0.2530 |                  1.0000 |
|         4 | write_sign | beneficial            |     3185 |        10.9193 |           9.6288 |         947.8024 |           847.9021 |                  0.0678 |                    0.2442 |                  1.0000 |
|         4 | write_sign | harmful               |     2859 |        10.4090 |           9.0628 |         895.4261 |           728.7939 |                  0.0652 |                    0.2421 |                  1.0000 |
|         8 | write_sign | beneficial            |     3185 |        25.8311 |          21.9475 |        1644.1910 |          1315.8364 |                  0.0743 |                    0.2472 |                  1.0000 |
|         8 | write_sign | harmful               |     2859 |        24.4024 |          20.9239 |        1540.8344 |          1207.3177 |                  0.0710 |                    0.2439 |                  1.0000 |
|         0 | cohort     | stable_correct        |      325 |         0.0000 |           0.0000 |         590.7546 |           530.1721 |                  0.0000 |                    0.2514 |                  0.0000 |
|         0 | cohort     | stable_wrong          |     5473 |         0.0000 |           0.0000 |         754.9669 |           618.2109 |                  0.0000 |                    0.2646 |                  0.0000 |
|         0 | cohort     | write_beneficial_flip |       23 |         0.0000 |           0.0000 |         606.9808 |           501.7799 |                  0.0000 |                    0.2589 |                  0.0000 |
|         0 | cohort     | write_harmful_flip    |      223 |         0.0000 |           0.0000 |         883.7518 |           872.9846 |                  0.0000 |                    0.2743 |                  0.0000 |
|         1 | cohort     | stable_correct        |      325 |         4.1160 |           3.3326 |         611.0028 |           552.0054 |                  0.0382 |                    0.2472 |                  1.0000 |
|         1 | cohort     | stable_wrong          |     5473 |         4.3053 |           3.5870 |         786.6517 |           637.5829 |                  0.0407 |                    0.2610 |                  1.0000 |
|         1 | cohort     | write_beneficial_flip |       23 |         3.9976 |           3.1973 |         622.5851 |           519.3692 |                  0.0378 |                    0.2521 |                  1.0000 |
|         1 | cohort     | write_harmful_flip    |      223 |         4.9578 |           4.1156 |         921.2273 |           903.3599 |                  0.0469 |                    0.2709 |                  1.0000 |
|         2 | cohort     | stable_correct        |      325 |         6.0909 |           5.5839 |         635.7567 |           560.5132 |                  0.0496 |                    0.2415 |                  1.0000 |
|         2 | cohort     | stable_wrong          |     5473 |         6.3654 |           5.6762 |         817.5912 |           677.1880 |                  0.0535 |                    0.2545 |                  1.0000 |
|         2 | cohort     | write_beneficial_flip |       23 |         5.6502 |           4.9450 |         638.1521 |           532.6482 |                  0.0479 |                    0.2459 |                  1.0000 |
|         2 | cohort     | write_harmful_flip    |      223 |         7.1923 |           6.7990 |         957.1918 |           921.0106 |                  0.0613 |                    0.2645 |                  1.0000 |
|         4 | cohort     | stable_correct        |      325 |        10.0195 |           9.2390 |         725.0738 |           610.5338 |                  0.0595 |                    0.2321 |                  1.0000 |
|         4 | cohort     | stable_wrong          |     5473 |        10.6644 |           9.2709 |         928.7295 |           802.0402 |                  0.0666 |                    0.2435 |                  1.0000 |
|         4 | cohort     | write_beneficial_flip |       23 |        10.0612 |           7.1573 |         719.4205 |           573.7949 |                  0.0653 |                    0.2363 |                  1.0000 |
|         4 | cohort     | write_harmful_flip    |      223 |        12.0337 |          11.0968 |        1092.5631 |          1013.1987 |                  0.0768 |                    0.2524 |                  1.0000 |
|         8 | cohort     | stable_correct        |      325 |        22.4274 |          21.1135 |        1250.5034 |           962.6614 |                  0.0611 |                    0.2345 |                  1.0000 |
|         8 | cohort     | stable_wrong          |     5473 |        25.1521 |          21.3751 |        1606.2288 |          1286.7307 |                  0.0729 |                    0.2461 |                  1.0000 |
|         8 | cohort     | write_beneficial_flip |       23 |        27.1456 |          21.4569 |        1199.9790 |           789.1748 |                  0.0827 |                    0.2414 |                  1.0000 |
|         8 | cohort     | write_harmful_flip    |      223 |        29.0038 |          24.8483 |        1870.3602 |          1540.1401 |                  0.0851 |                    0.2508 |                  1.0000 |

First measured text-divergence horizon over these 6,044 states: {1: 6044} (-1 means none through H8).

Descriptive native-support effect growth:

|   horizon | family     | cohort                |   states |   text_l2_mean |   text_l2_median |   visual_l2_mean |   visual_l2_median |   text_relative_l2_mean |   visual_relative_l2_mean |   nonzero_text_fraction |
|----------:|:-----------|:----------------------|---------:|---------------:|-----------------:|-----------------:|-------------------:|------------------------:|--------------------------:|------------------------:|
|         0 | write_sign | beneficial            |     7271 |         0.0000 |           0.0000 |        1586.8160 |          1270.1998 |                  0.0000 |                    0.3077 |                  0.0000 |
|         0 | write_sign | harmful               |     6501 |         0.0000 |           0.0000 |        1585.2255 |          1236.6404 |                  0.0000 |                    0.3063 |                  0.0000 |
|         0 | write_sign | zero                  |     1413 |         0.0000 |           0.0000 |        7377.7432 |          5746.9238 |                  0.0000 |                    0.6781 |                  0.0000 |
|         1 | write_sign | beneficial            |     7271 |        18.4322 |           9.4808 |        2311.0779 |          1385.8519 |                  0.0510 |                    0.3538 |                  1.0000 |
|         1 | write_sign | harmful               |     6501 |        18.1837 |           9.4941 |        2362.3539 |          1357.8701 |                  0.0498 |                    0.3559 |                  1.0000 |
|         2 | write_sign | beneficial            |     6608 |        20.0748 |          12.2125 |        1911.8470 |          1398.8508 |                  0.0586 |                    0.3060 |                  1.0000 |
|         2 | write_sign | harmful               |     5873 |        19.4052 |          12.1328 |        1889.0158 |          1314.4700 |                  0.0569 |                    0.3049 |                  1.0000 |
|         4 | write_sign | beneficial            |     5323 |        22.8180 |          16.0464 |        1903.8572 |          1327.5898 |                  0.0671 |                    0.2865 |                  1.0000 |
|         4 | write_sign | harmful               |     4733 |        21.7373 |          15.3353 |        1820.3482 |          1278.9044 |                  0.0650 |                    0.2834 |                  1.0000 |
|         8 | write_sign | beneficial            |     3185 |        25.8311 |          21.9475 |        1644.1910 |          1315.8364 |                  0.0743 |                    0.2472 |                  1.0000 |
|         8 | write_sign | harmful               |     2859 |        24.4024 |          20.9239 |        1540.8344 |          1207.3177 |                  0.0710 |                    0.2439 |                  1.0000 |
|         0 | cohort     | stable_correct        |      917 |         0.0000 |           0.0000 |        1952.7527 |          1414.3464 |                  0.0000 |                    0.3466 |                  0.0000 |
|         0 | cohort     | stable_wrong          |    13780 |         0.0000 |           0.0000 |        2153.3366 |          1447.5663 |                  0.0000 |                    0.3424 |                  0.0000 |
|         0 | cohort     | write_beneficial_flip |       40 |         0.0000 |           0.0000 |        1126.0271 |           814.8688 |                  0.0000 |                    0.2918 |                  0.0000 |
|         0 | cohort     | write_harmful_flip    |      448 |         0.0000 |           0.0000 |        1694.9736 |          1268.1855 |                  0.0000 |                    0.3089 |                  0.0000 |
|         1 | cohort     | stable_correct        |      811 |        18.1332 |          10.2011 |        2110.7574 |          1270.0088 |                  0.0459 |                    0.3613 |                  1.0000 |
|         1 | cohort     | stable_wrong          |    12473 |        18.3598 |           9.4703 |        2354.5473 |          1379.2230 |                  0.0505 |                    0.3552 |                  1.0000 |
|         1 | cohort     | write_beneficial_flip |       40 |        14.0874 |           5.9300 |        1392.6727 |           893.4009 |                  0.0461 |                    0.3065 |                  1.0000 |
|         1 | cohort     | write_harmful_flip    |      448 |        17.7703 |           9.0767 |        2289.5334 |          1373.6888 |                  0.0564 |                    0.3365 |                  1.0000 |
|         2 | cohort     | stable_correct        |      721 |        19.0847 |          12.6018 |        1672.8812 |          1208.5430 |                  0.0521 |                    0.3048 |                  1.0000 |
|         2 | cohort     | stable_wrong          |    11301 |        19.7862 |          12.1234 |        1911.1049 |          1369.0045 |                  0.0579 |                    0.3056 |                  1.0000 |
|         2 | cohort     | write_beneficial_flip |       39 |        18.4187 |           8.5861 |        1463.4611 |           809.1592 |                  0.0541 |                    0.2963 |                  1.0000 |
|         2 | cohort     | write_harmful_flip    |      420 |        20.3313 |          12.5849 |        2064.4190 |          1465.8690 |                  0.0660 |                    0.3026 |                  1.0000 |
|         4 | cohort     | stable_correct        |      567 |        21.8141 |          15.8444 |        1642.1116 |           983.3303 |                  0.0592 |                    0.2847 |                  1.0000 |
|         4 | cohort     | stable_wrong          |     9100 |        22.2565 |          15.6405 |        1870.4141 |          1307.1055 |                  0.0661 |                    0.2850 |                  1.0000 |
|         4 | cohort     | write_beneficial_flip |       32 |        16.4482 |          11.4812 |        1262.0082 |           672.0795 |                  0.0642 |                    0.2681 |                  1.0000 |
|         4 | cohort     | write_harmful_flip    |      357 |        24.9684 |          17.7420 |        2122.4360 |          1468.1764 |                  0.0764 |                    0.2882 |                  1.0000 |
|         8 | cohort     | stable_correct        |      325 |        22.4274 |          21.1135 |        1250.5034 |           962.6614 |                  0.0611 |                    0.2345 |                  1.0000 |
|         8 | cohort     | stable_wrong          |     5473 |        25.1521 |          21.3751 |        1606.2288 |          1286.7307 |                  0.0729 |                    0.2461 |                  1.0000 |
|         8 | cohort     | write_beneficial_flip |       23 |        27.1456 |          21.4569 |        1199.9790 |           789.1748 |                  0.0827 |                    0.2414 |                  1.0000 |
|         8 | cohort     | write_harmful_flip    |      223 |        29.0038 |          24.8483 |        1870.3602 |          1540.1401 |                  0.0851 |                    0.2508 |                  1.0000 |

Exact common-support OOF metrics, with Dense-W models independently refitted:

|   horizon | support   | population   | condition    |   states |   spearman |   harmful_auroc |   precision_at_0.1 |   recall_at_precision_0.9 |   harmful_flip_auroc |
|----------:|:----------|:-------------|:-------------|---------:|-----------:|----------------:|-------------------:|--------------------------:|---------------------:|
|         0 | common    | full         | on           |     6044 |     0.0426 |          0.5202 |             0.5107 |                    0.0000 |               0.4515 |
|         0 | common    | full         | off          |     6044 |     0.0563 |          0.5258 |             0.5124 |                    0.0000 |               0.4450 |
|         0 | common    | full         | visual_delta |     6044 |     0.0338 |          0.5113 |             0.4909 |                    0.0014 |               0.4578 |
|         0 | common    | full         | text_delta   |     6044 |     0.0088 |          0.5035 |             0.4843 |                    0.0000 |               0.5125 |
|         0 | common    | full         | delta        |     6044 |     0.0412 |          0.5162 |             0.5058 |                    0.0000 |               0.4537 |
|         0 | common    | full         | pair         |     6044 |     0.0571 |          0.5234 |             0.5355 |                    0.0007 |               0.4515 |
|         0 | common    | dense_w      | on           |     5696 |     0.0738 |          0.5378 |             0.5509 |                    0.0007 |               0.4362 |
|         0 | common    | dense_w      | off          |     5696 |     0.0628 |          0.5304 |             0.5140 |                    0.0007 |               0.4448 |
|         0 | common    | dense_w      | visual_delta |     5696 |     0.0395 |          0.5178 |             0.5193 |                    0.0000 |               0.4345 |
|         0 | common    | dense_w      | text_delta   |     5696 |     0.0109 |          0.5019 |             0.4807 |                    0.0000 |               0.5123 |
|         0 | common    | dense_w      | delta        |     5696 |     0.0479 |          0.5223 |             0.5000 |                    0.0004 |               0.4556 |
|         0 | common    | dense_w      | pair         |     5696 |     0.0599 |          0.5305 |             0.5158 |                    0.0000 |               0.4564 |
|         1 | common    | full         | on           |     6044 |     0.0566 |          0.5263 |             0.4975 |                    0.0000 |               0.4495 |
|         1 | common    | full         | off          |     6044 |     0.0438 |          0.5179 |             0.5058 |                    0.0007 |               0.4552 |
|         1 | common    | full         | visual_delta |     6044 |     0.0426 |          0.5195 |             0.4975 |                    0.0000 |               0.4486 |
|         1 | common    | full         | text_delta   |     6044 |     0.0371 |          0.5185 |             0.4711 |                    0.0000 |               0.4459 |
|         1 | common    | full         | delta        |     6044 |     0.0490 |          0.5224 |             0.5091 |                    0.0007 |               0.4628 |
|         1 | common    | full         | pair         |     6044 |     0.0378 |          0.5174 |             0.5273 |                    0.0000 |               0.4686 |
|         1 | common    | dense_w      | on           |     5696 |     0.0654 |          0.5325 |             0.5333 |                    0.0007 |               0.4385 |
|         1 | common    | dense_w      | off          |     5696 |     0.0573 |          0.5299 |             0.5263 |                    0.0000 |               0.4197 |
|         1 | common    | dense_w      | visual_delta |     5696 |     0.0454 |          0.5205 |             0.5211 |                    0.0007 |               0.4354 |
|         1 | common    | dense_w      | text_delta   |     5696 |     0.0453 |          0.5260 |             0.5158 |                    0.0000 |               0.4567 |
|         1 | common    | dense_w      | delta        |     5696 |     0.0616 |          0.5300 |             0.5193 |                    0.0004 |               0.4614 |
|         1 | common    | dense_w      | pair         |     5696 |     0.0842 |          0.5396 |             0.5351 |                    0.0000 |               0.4714 |
|         2 | common    | full         | on           |     6044 |     0.0716 |          0.5325 |             0.5306 |                    0.0003 |               0.4619 |
|         2 | common    | full         | off          |     6044 |     0.0554 |          0.5274 |             0.5190 |                    0.0007 |               0.4862 |
|         2 | common    | full         | visual_delta |     6044 |     0.0408 |          0.5188 |             0.5058 |                    0.0000 |               0.4404 |
|         2 | common    | full         | text_delta   |     6044 |     0.0406 |          0.5163 |             0.4942 |                    0.0003 |               0.4586 |
|         2 | common    | full         | delta        |     6044 |     0.0549 |          0.5234 |             0.5207 |                    0.0000 |               0.4464 |
|         2 | common    | full         | pair         |     6044 |     0.0337 |          0.5143 |             0.5157 |                    0.0007 |               0.4690 |
|         2 | common    | dense_w      | on           |     5696 |     0.0609 |          0.5317 |             0.5298 |                    0.0000 |               0.4496 |
|         2 | common    | dense_w      | off          |     5696 |     0.0632 |          0.5303 |             0.5439 |                    0.0000 |               0.4650 |
|         2 | common    | dense_w      | visual_delta |     5696 |     0.0524 |          0.5257 |             0.5228 |                    0.0004 |               0.4369 |
|         2 | common    | dense_w      | text_delta   |     5696 |     0.0531 |          0.5236 |             0.5140 |                    0.0011 |               0.4573 |
|         2 | common    | dense_w      | delta        |     5696 |     0.0489 |          0.5199 |             0.5456 |                    0.0000 |               0.4750 |
|         2 | common    | dense_w      | pair         |     5696 |     0.0528 |          0.5286 |             0.5456 |                    0.0007 |               0.4816 |
|         4 | common    | full         | on           |     6044 |     0.0590 |          0.5314 |             0.5438 |                    0.0017 |               0.4420 |
|         4 | common    | full         | off          |     6044 |     0.0591 |          0.5310 |             0.5339 |                    0.0000 |               0.4420 |
|         4 | common    | full         | visual_delta |     6044 |     0.0305 |          0.5145 |             0.4992 |                    0.0000 |               0.4358 |
|         4 | common    | full         | text_delta   |     6044 |     0.0580 |          0.5281 |             0.5074 |                    0.0000 |               0.4293 |
|         4 | common    | full         | delta        |     6044 |     0.0648 |          0.5300 |             0.5405 |                    0.0010 |               0.4392 |
|         4 | common    | full         | pair         |     6044 |     0.0554 |          0.5251 |             0.5355 |                    0.0000 |               0.4565 |
|         4 | common    | dense_w      | on           |     5696 |     0.0676 |          0.5311 |             0.5456 |                    0.0000 |               0.4698 |
|         4 | common    | dense_w      | off          |     5696 |     0.0678 |          0.5311 |             0.5737 |                    0.0000 |               0.4514 |
|         4 | common    | dense_w      | visual_delta |     5696 |     0.0480 |          0.5245 |             0.5211 |                    0.0007 |               0.4167 |
|         4 | common    | dense_w      | text_delta   |     5696 |     0.0664 |          0.5334 |             0.5351 |                    0.0000 |               0.4255 |
|         4 | common    | dense_w      | delta        |     5696 |     0.0628 |          0.5277 |             0.5351 |                    0.0004 |               0.4453 |
|         4 | common    | dense_w      | pair         |     5696 |     0.0562 |          0.5238 |             0.5140 |                    0.0007 |               0.4970 |
|         8 | common    | full         | on           |     6044 |     0.0593 |          0.5304 |             0.5438 |                    0.0000 |               0.4165 |
|         8 | common    | full         | off          |     6044 |     0.0538 |          0.5268 |             0.5207 |                    0.0000 |               0.4405 |
|         8 | common    | full         | visual_delta |     6044 |     0.0603 |          0.5264 |             0.5388 |                    0.0000 |               0.4491 |
|         8 | common    | full         | text_delta   |     6044 |     0.0968 |          0.5390 |             0.5058 |                    0.0003 |               0.4926 |
|         8 | common    | full         | delta        |     6044 |     0.0973 |          0.5387 |             0.5537 |                    0.0000 |               0.4847 |
|         8 | common    | full         | pair         |     6044 |     0.0313 |          0.5136 |             0.5074 |                    0.0000 |               0.4366 |
|         8 | common    | dense_w      | on           |     5696 |     0.0643 |          0.5314 |             0.5333 |                    0.0004 |               0.4485 |
|         8 | common    | dense_w      | off          |     5696 |     0.0638 |          0.5321 |             0.5561 |                    0.0000 |               0.4457 |
|         8 | common    | dense_w      | visual_delta |     5696 |     0.0565 |          0.5297 |             0.5298 |                    0.0000 |               0.4709 |
|         8 | common    | dense_w      | text_delta   |     5696 |     0.1039 |          0.5461 |             0.5246 |                    0.0000 |               0.4984 |
|         8 | common    | dense_w      | delta        |     5696 |     0.1038 |          0.5443 |             0.5649 |                    0.0007 |               0.4815 |
|         8 | common    | dense_w      | pair         |     5696 |     0.0409 |          0.5163 |             0.5053 |                    0.0000 |               0.4440 |

Native-support metrics:

|   horizon | support   | population   | condition    |   states |   spearman |   harmful_auroc |   precision_at_0.1 |   recall_at_precision_0.9 |   harmful_flip_auroc |
|----------:|:----------|:-------------|:-------------|---------:|-----------:|----------------:|-------------------:|--------------------------:|---------------------:|
|         0 | native    | full         | on           |    15185 |     0.0459 |          0.5010 |             0.4852 |                    0.0000 |               0.4227 |
|         0 | native    | full         | off          |    15185 |     0.0403 |          0.4952 |             0.5036 |                    0.0000 |               0.4426 |
|         0 | native    | full         | visual_delta |    15185 |     0.0373 |          0.4881 |             0.5089 |                    0.0000 |               0.4297 |
|         0 | native    | full         | text_delta   |    15185 |     0.0028 |          0.5064 |             0.4529 |                    0.0000 |               0.5023 |
|         0 | native    | full         | delta        |    15185 |     0.0348 |          0.4981 |             0.4931 |                    0.0000 |               0.4533 |
|         0 | native    | full         | pair         |    15185 |     0.0430 |          0.4946 |             0.4878 |                    0.0000 |               0.4314 |
|         0 | native    | dense_w      | on           |    14228 |     0.0413 |          0.5072 |             0.5137 |                    0.0005 |               0.4362 |
|         0 | native    | dense_w      | off          |    14228 |     0.0361 |          0.5088 |             0.5067 |                    0.0000 |               0.4478 |
|         0 | native    | dense_w      | visual_delta |    14228 |     0.0364 |          0.4872 |             0.4905 |                    0.0000 |               0.4488 |
|         0 | native    | dense_w      | text_delta   |    14228 |     0.0045 |          0.5038 |             0.4512 |                    0.0000 |               0.4894 |
|         0 | native    | dense_w      | delta        |    14228 |     0.0353 |          0.4967 |             0.4968 |                    0.0000 |               0.4511 |
|         0 | native    | dense_w      | pair         |    14228 |     0.0443 |          0.5127 |             0.5081 |                    0.0003 |               0.4382 |
|         1 | native    | full         | on           |    13772 |     0.0249 |          0.5092 |             0.4971 |                    0.0000 |               0.4413 |
|         1 | native    | full         | off          |    13772 |     0.0342 |          0.5153 |             0.5073 |                    0.0000 |               0.4251 |
|         1 | native    | full         | visual_delta |    13772 |     0.0389 |          0.5177 |             0.5102 |                    0.0000 |               0.4471 |
|         1 | native    | full         | text_delta   |    13772 |     0.0186 |          0.5104 |             0.4695 |                    0.0002 |               0.4565 |
|         1 | native    | full         | delta        |    13772 |     0.0375 |          0.5183 |             0.4848 |                    0.0005 |               0.4484 |
|         1 | native    | full         | pair         |    13772 |     0.0261 |          0.5100 |             0.5029 |                    0.0002 |               0.4504 |
|         1 | native    | dense_w      | on           |    12921 |     0.0492 |          0.5233 |             0.5104 |                    0.0003 |               0.4241 |
|         1 | native    | dense_w      | off          |    12921 |     0.0443 |          0.5230 |             0.5251 |                    0.0000 |               0.4439 |
|         1 | native    | dense_w      | visual_delta |    12921 |     0.0434 |          0.5195 |             0.5259 |                    0.0000 |               0.4562 |
|         1 | native    | dense_w      | text_delta   |    12921 |     0.0241 |          0.5121 |             0.4872 |                    0.0002 |               0.4580 |
|         1 | native    | dense_w      | delta        |    12921 |     0.0352 |          0.5219 |             0.5081 |                    0.0002 |               0.4669 |
|         1 | native    | dense_w      | pair         |    12921 |     0.0437 |          0.5190 |             0.5290 |                    0.0000 |               0.4505 |
|         2 | native    | full         | on           |    12481 |     0.0485 |          0.5194 |             0.5116 |                    0.0005 |               0.4529 |
|         2 | native    | full         | off          |    12481 |     0.0486 |          0.5221 |             0.5260 |                    0.0003 |               0.4583 |
|         2 | native    | full         | visual_delta |    12481 |     0.0395 |          0.5184 |             0.4956 |                    0.0000 |               0.4514 |
|         2 | native    | full         | text_delta   |    12481 |     0.0316 |          0.5139 |             0.4804 |                    0.0002 |               0.4728 |
|         2 | native    | full         | delta        |    12481 |     0.0449 |          0.5212 |             0.5140 |                    0.0002 |               0.4637 |
|         2 | native    | full         | pair         |    12481 |     0.0405 |          0.5186 |             0.5180 |                    0.0000 |               0.4680 |
|         2 | native    | dense_w      | on           |    11721 |     0.0310 |          0.5155 |             0.5311 |                    0.0002 |               0.4550 |
|         2 | native    | dense_w      | off          |    11721 |     0.0314 |          0.5174 |             0.5064 |                    0.0000 |               0.4524 |
|         2 | native    | dense_w      | visual_delta |    11721 |     0.0308 |          0.5113 |             0.5090 |                    0.0000 |               0.4498 |
|         2 | native    | dense_w      | text_delta   |    11721 |     0.0338 |          0.5134 |             0.4876 |                    0.0000 |               0.4679 |
|         2 | native    | dense_w      | delta        |    11721 |     0.0455 |          0.5222 |             0.4987 |                    0.0000 |               0.4524 |
|         2 | native    | dense_w      | pair         |    11721 |     0.0483 |          0.5242 |             0.5107 |                    0.0004 |               0.4563 |
|         4 | native    | full         | on           |    10056 |     0.0386 |          0.5129 |             0.4970 |                    0.0000 |               0.4590 |
|         4 | native    | full         | off          |    10056 |     0.0299 |          0.5116 |             0.4980 |                    0.0006 |               0.4525 |
|         4 | native    | full         | visual_delta |    10056 |     0.0375 |          0.5149 |             0.4990 |                    0.0002 |               0.4450 |
|         4 | native    | full         | text_delta   |    10056 |     0.0317 |          0.5128 |             0.4920 |                    0.0000 |               0.4947 |
|         4 | native    | full         | delta        |    10056 |     0.0476 |          0.5191 |             0.5050 |                    0.0000 |               0.4643 |
|         4 | native    | full         | pair         |    10056 |     0.0289 |          0.5144 |             0.4911 |                    0.0002 |               0.4620 |
|         4 | native    | dense_w      | on           |     9457 |     0.0499 |          0.5254 |             0.5074 |                    0.0000 |               0.4529 |
|         4 | native    | dense_w      | off          |     9457 |     0.0435 |          0.5225 |             0.4947 |                    0.0000 |               0.4421 |
|         4 | native    | dense_w      | visual_delta |     9457 |     0.0353 |          0.5156 |             0.5127 |                    0.0000 |               0.4477 |
|         4 | native    | dense_w      | text_delta   |     9457 |     0.0442 |          0.5164 |             0.5032 |                    0.0002 |               0.4967 |
|         4 | native    | dense_w      | delta        |     9457 |     0.0511 |          0.5232 |             0.5222 |                    0.0007 |               0.4773 |
|         4 | native    | dense_w      | pair         |     9457 |     0.0376 |          0.5208 |             0.5148 |                    0.0004 |               0.4651 |
|         8 | native    | full         | on           |     6044 |     0.0593 |          0.5304 |             0.5438 |                    0.0000 |               0.4165 |
|         8 | native    | full         | off          |     6044 |     0.0538 |          0.5268 |             0.5207 |                    0.0000 |               0.4405 |
|         8 | native    | full         | visual_delta |     6044 |     0.0603 |          0.5264 |             0.5388 |                    0.0000 |               0.4491 |
|         8 | native    | full         | text_delta   |     6044 |     0.0968 |          0.5390 |             0.5058 |                    0.0003 |               0.4926 |
|         8 | native    | full         | delta        |     6044 |     0.0973 |          0.5387 |             0.5537 |                    0.0000 |               0.4847 |
|         8 | native    | full         | pair         |     6044 |     0.0313 |          0.5136 |             0.5074 |                    0.0000 |               0.4366 |
|         8 | native    | dense_w      | on           |     5696 |     0.0643 |          0.5314 |             0.5333 |                    0.0004 |               0.4485 |
|         8 | native    | dense_w      | off          |     5696 |     0.0638 |          0.5321 |             0.5561 |                    0.0000 |               0.4457 |
|         8 | native    | dense_w      | visual_delta |     5696 |     0.0565 |          0.5297 |             0.5298 |                    0.0000 |               0.4709 |
|         8 | native    | dense_w      | text_delta   |     5696 |     0.1039 |          0.5461 |             0.5246 |                    0.0000 |               0.4984 |
|         8 | native    | dense_w      | delta        |     5696 |     0.1038 |          0.5443 |             0.5649 |                    0.0007 |               0.4815 |
|         8 | native    | dense_w      | pair         |     5696 |     0.0409 |          0.5163 |             0.5053 |                    0.0000 |               0.4440 |

Common-support paired bootstrap differences:

| population   |   horizon | left   |   reference_horizon | right   |   spearman_gain |   spearman_ci_low |   spearman_ci_high |   harmful_auc_gain |   harmful_auc_ci_low |   harmful_auc_ci_high |   draws | bootstrap_unit   |
|:-------------|----------:|:-------|--------------------:|:--------|----------------:|------------------:|-------------------:|-------------------:|---------------------:|----------------------:|--------:|:-----------------|
| full         |         1 | delta  |                   0 | delta   |          0.0078 |           -0.0162 |             0.0322 |             0.0063 |              -0.0082 |                0.0215 |    5000 | image_group_id   |
| full         |         1 | delta  |                   1 | on      |         -0.0076 |           -0.0406 |             0.0264 |            -0.0039 |              -0.0220 |                0.0151 |    5000 | image_group_id   |
| full         |         1 | delta  |                   1 | off     |          0.0051 |           -0.0303 |             0.0410 |             0.0046 |              -0.0147 |                0.0237 |    5000 | image_group_id   |
| full         |         1 | on     |                   0 | on      |          0.0139 |           -0.0033 |             0.0307 |             0.0061 |              -0.0037 |                0.0155 |    5000 | image_group_id   |
| full         |         1 | off    |                   0 | off     |         -0.0124 |           -0.0372 |             0.0119 |            -0.0080 |              -0.0213 |                0.0052 |    5000 | image_group_id   |
| dense_w      |         1 | delta  |                   0 | delta   |          0.0137 |           -0.0111 |             0.0391 |             0.0077 |              -0.0065 |                0.0223 |    5000 | image_group_id   |
| dense_w      |         1 | delta  |                   1 | on      |         -0.0039 |           -0.0404 |             0.0323 |            -0.0026 |              -0.0231 |                0.0170 |    5000 | image_group_id   |
| dense_w      |         1 | delta  |                   1 | off     |          0.0042 |           -0.0344 |             0.0409 |             0.0001 |              -0.0200 |                0.0196 |    5000 | image_group_id   |
| dense_w      |         1 | on     |                   0 | on      |         -0.0083 |           -0.0282 |             0.0117 |            -0.0052 |              -0.0168 |                0.0072 |    5000 | image_group_id   |
| dense_w      |         1 | off    |                   0 | off     |         -0.0055 |           -0.0266 |             0.0160 |            -0.0004 |              -0.0130 |                0.0126 |    5000 | image_group_id   |
| full         |         2 | delta  |                   0 | delta   |          0.0137 |           -0.0134 |             0.0410 |             0.0073 |              -0.0081 |                0.0224 |    5000 | image_group_id   |
| full         |         2 | delta  |                   2 | on      |         -0.0166 |           -0.0517 |             0.0190 |            -0.0091 |              -0.0274 |                0.0105 |    5000 | image_group_id   |
| full         |         2 | delta  |                   2 | off     |         -0.0005 |           -0.0356 |             0.0343 |            -0.0039 |              -0.0223 |                0.0153 |    5000 | image_group_id   |
| full         |         2 | on     |                   0 | on      |          0.0289 |            0.0075 |             0.0501 |             0.0123 |               0.0002 |                0.0243 |    5000 | image_group_id   |
| full         |         2 | off    |                   0 | off     |         -0.0009 |           -0.0241 |             0.0217 |             0.0015 |              -0.0115 |                0.0143 |    5000 | image_group_id   |
| dense_w      |         2 | delta  |                   0 | delta   |          0.0010 |           -0.0254 |             0.0291 |            -0.0024 |              -0.0170 |                0.0127 |    5000 | image_group_id   |
| dense_w      |         2 | delta  |                   2 | on      |         -0.0120 |           -0.0448 |             0.0209 |            -0.0118 |              -0.0310 |                0.0065 |    5000 | image_group_id   |
| dense_w      |         2 | delta  |                   2 | off     |         -0.0143 |           -0.0463 |             0.0176 |            -0.0104 |              -0.0286 |                0.0076 |    5000 | image_group_id   |
| dense_w      |         2 | on     |                   0 | on      |         -0.0129 |           -0.0371 |             0.0112 |            -0.0060 |              -0.0194 |                0.0073 |    5000 | image_group_id   |
| dense_w      |         2 | off    |                   0 | off     |          0.0004 |           -0.0215 |             0.0235 |            -0.0001 |              -0.0124 |                0.0129 |    5000 | image_group_id   |
| full         |         4 | delta  |                   0 | delta   |          0.0236 |           -0.0038 |             0.0531 |             0.0138 |              -0.0018 |                0.0298 |    5000 | image_group_id   |
| full         |         4 | delta  |                   4 | on      |          0.0059 |           -0.0269 |             0.0399 |            -0.0014 |              -0.0191 |                0.0171 |    5000 | image_group_id   |
| full         |         4 | delta  |                   4 | off     |          0.0058 |           -0.0298 |             0.0413 |            -0.0010 |              -0.0191 |                0.0181 |    5000 | image_group_id   |
| full         |         4 | on     |                   0 | on      |          0.0163 |           -0.0125 |             0.0442 |             0.0112 |              -0.0045 |                0.0263 |    5000 | image_group_id   |
| full         |         4 | off    |                   0 | off     |          0.0028 |           -0.0210 |             0.0273 |             0.0052 |              -0.0084 |                0.0187 |    5000 | image_group_id   |
| dense_w      |         4 | delta  |                   0 | delta   |          0.0149 |           -0.0134 |             0.0427 |             0.0054 |              -0.0102 |                0.0210 |    5000 | image_group_id   |
| dense_w      |         4 | delta  |                   4 | on      |         -0.0047 |           -0.0373 |             0.0283 |            -0.0034 |              -0.0214 |                0.0148 |    5000 | image_group_id   |
| dense_w      |         4 | delta  |                   4 | off     |         -0.0050 |           -0.0375 |             0.0278 |            -0.0034 |              -0.0221 |                0.0155 |    5000 | image_group_id   |
| dense_w      |         4 | on     |                   0 | on      |         -0.0062 |           -0.0318 |             0.0192 |            -0.0067 |              -0.0214 |                0.0085 |    5000 | image_group_id   |
| dense_w      |         4 | off    |                   0 | off     |          0.0049 |           -0.0224 |             0.0323 |             0.0008 |              -0.0148 |                0.0164 |    5000 | image_group_id   |
| full         |         8 | delta  |                   0 | delta   |          0.0561 |            0.0267 |             0.0852 |             0.0226 |               0.0051 |                0.0396 |    5000 | image_group_id   |
| full         |         8 | delta  |                   8 | on      |          0.0381 |            0.0016 |             0.0752 |             0.0083 |              -0.0106 |                0.0281 |    5000 | image_group_id   |
| full         |         8 | delta  |                   8 | off     |          0.0435 |            0.0073 |             0.0806 |             0.0119 |              -0.0069 |                0.0314 |    5000 | image_group_id   |
| full         |         8 | on     |                   0 | on      |          0.0166 |           -0.0152 |             0.0474 |             0.0102 |              -0.0070 |                0.0271 |    5000 | image_group_id   |
| full         |         8 | off    |                   0 | off     |         -0.0025 |           -0.0308 |             0.0262 |             0.0010 |              -0.0142 |                0.0161 |    5000 | image_group_id   |
| dense_w      |         8 | delta  |                   0 | delta   |          0.0559 |            0.0261 |             0.0864 |             0.0220 |               0.0051 |                0.0391 |    5000 | image_group_id   |
| dense_w      |         8 | delta  |                   8 | on      |          0.0394 |            0.0026 |             0.0756 |             0.0130 |              -0.0074 |                0.0328 |    5000 | image_group_id   |
| dense_w      |         8 | delta  |                   8 | off     |          0.0400 |            0.0027 |             0.0757 |             0.0122 |              -0.0078 |                0.0316 |    5000 | image_group_id   |
| dense_w      |         8 | on     |                   0 | on      |         -0.0094 |           -0.0373 |             0.0194 |            -0.0064 |              -0.0227 |                0.0106 |    5000 | image_group_id   |
| dense_w      |         8 | off    |                   0 | off     |          0.0009 |           -0.0288 |             0.0325 |             0.0017 |              -0.0150 |                0.0186 |    5000 | image_group_id   |

The prospective primary is combined delta/MLP. Material emergence, advantage over both single branches, Dense-W survival and high-precision checks are recorded in propagation_decision.json. Secondary ordered-pair and text/visual ablations do not replace the primary after outcomes. Positive text divergence establishes downstream influence of the WRITE intervention under this fixed continuation; it does not establish predictable answer harm or identify a sufficient causal mediator. A weak result is bounded by the target, support, horizons and representation/capacity family.
