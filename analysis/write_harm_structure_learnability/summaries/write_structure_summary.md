# WRITE structure: complete dense census

All 15,185 states / 1,413 UIDs / 1,385 image groups are included. Harmful/beneficial/zero counts: {'beneficial': 7271, 'harmful': 6501, 'zero': 1413}. Behavioral cohorts: {'stable_correct': 917, 'stable_wrong': 13780, 'write_beneficial_flip': 40, 'write_harmful_flip': 448}. Every UID contributes layer 27; the 1413 terminal WRITE interventions have zero downstream target effect. Zeros remain in the prospective primary denominator.

Harmful adjacent persistence is 0.4975 versus UID-shuffled mean 0.4561, null 95% interval [0.4473, 0.4648], enrichment 1.091x, empirical one-sided p=0.00020. This is modest population-level organization under the specified within-UID layer shuffle, not a global predictable harmful regime. Source cells have unequal support and do not all exceed their null intervals.

| sign       |   spans |   mean_length |   median_length |   max_length |   fraction_states_in_ge2 |   fraction_states_in_ge3 |
|:-----------|--------:|--------------:|----------------:|-------------:|-------------------------:|-------------------------:|
| beneficial |    3339 |        2.1776 |          2.0000 |           19 |                   0.7826 |                   0.5485 |
| harmful    |    3267 |        1.9899 |          1.0000 |           13 |                   0.7436 |                   0.4867 |

Harmful span proportions by length: {1: 0.5103, 2: 0.2556, 3: 0.1169, 4: 0.0514, 5: 0.0303, 6: 0.0153, 7: 0.0092, 8: 0.0049, 9: 0.0021, 10: 0.0006, 11: 0.0015, 12: 0.0009, 13: 0.0009}. The complete span records preserve longer tails. Strong-flip neighborhoods (offset 0 is selected on correctness, so its enrichment is not independent evidence):

|   offset |   states |   mean_H_W |   median_H_W |   harmful_prevalence |
|---------:|---------:|-----------:|-------------:|---------------------:|
|  -3.0000 | 312.0000 |    -0.0252 |      -0.0094 |               0.4519 |
|  -2.0000 | 355.0000 |    -0.0139 |      -0.0022 |               0.4845 |
|  -1.0000 | 398.0000 |     0.0001 |       0.0027 |               0.5251 |
|   0.0000 | 448.0000 |     0.1013 |       0.0632 |               0.8125 |
|   1.0000 | 448.0000 |     0.0130 |       0.0023 |               0.5268 |
|   2.0000 | 420.0000 |    -0.0147 |      -0.0002 |               0.4357 |
|   3.0000 | 394.0000 |    -0.0179 |       0.0000 |               0.4239 |

Trigger-relative structure:

| relative_bin   |   states |   uids |   mean_H_W |   median_H_W |   harmful_prevalence |   harmful_flip_prevalence |
|:---------------|---------:|-------:|-----------:|-------------:|---------------------:|--------------------------:|
| d0             |     1413 |   1413 |    -0.0130 |       0.0000 |               0.4183 |                    0.0354 |
| d1_2           |     2512 |   1291 |    -0.0073 |      -0.0004 |               0.4594 |                    0.0342 |
| d3_5           |     3377 |   1204 |    -0.0132 |       0.0000 |               0.4368 |                    0.0341 |
| d6_plus        |     7883 |    967 |    -0.0065 |       0.0000 |               0.4162 |                    0.0250 |

UID harmful-layer fraction median/90th percentile/maximum: 0.400/0.625/0.929; 36.4% of UIDs have at least half their layers harmful. This measures observed sample burden; it does not establish future-sample predictability.

Dataset/source prevalence:

| dataset   | source_regime   |   states |   uids |   mean_H_W |   median_H_W |   harmful_prevalence |   harmful_flip_prevalence |
|:----------|:----------------|---------:|-------:|-----------:|-------------:|---------------------:|--------------------------:|
| chartqa   | canonical       |      339 |     42 |    -0.0150 |       0.0000 |               0.4012 |                    0.0236 |
| chartqa   | historical      |     2205 |    245 |    -0.0164 |       0.0000 |               0.4036 |                    0.0417 |
| gqa       | canonical       |     3857 |    321 |     0.0001 |       0.0000 |               0.4514 |                    0.0202 |
| gqa       | historical      |     4902 |    449 |    -0.0101 |       0.0000 |               0.4390 |                    0.0139 |
| textvqa   | canonical       |       68 |     11 |    -0.0012 |       0.0000 |               0.3676 |                    0.0294 |
| textvqa   | historical      |     3814 |    345 |    -0.0111 |      -0.0002 |               0.4082 |                    0.0524 |

WRITE harm shows modest adjacency and multi-layer spans, with many isolated signs and substantial source/sample variation. Structural organization alone does not establish a mechanism or learnable router.
