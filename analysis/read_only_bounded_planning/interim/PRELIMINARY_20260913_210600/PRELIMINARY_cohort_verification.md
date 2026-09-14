# PRELIMINARY: completed-versus-pending cohort verification

Snapshot 2026-09-13T21:06:00.071160+09:00. Common completed W=612; pending W=695. Completion is schedule-selected, not a random sample.

| group   | feature        | value              |   completed_n |   completed_denominator |   completed_fraction |   pending_n |   pending_denominator |   pending_fraction |   difference_pp |
|:--------|:---------------|:-------------------|--------------:|------------------------:|---------------------:|------------:|----------------------:|-------------------:|----------------:|
| W       | dataset_source | canonical/chartqa  |            11 |                     612 |           0.0179739  |          17 |                   695 |          0.0244604 |       -0.648658 |
| W       | dataset_source | canonical/gqa      |            87 |                     612 |           0.142157   |         196 |                   695 |          0.282014  |      -13.9858   |
| W       | dataset_source | canonical/textvqa  |             3 |                     612 |           0.00490196 |           0 |                   695 |          0         |        0.490196 |
| W       | dataset_source | historical/chartqa |           148 |                     612 |           0.24183    |          96 |                   695 |          0.138129  |       10.3701   |
| W       | dataset_source | historical/gqa     |           101 |                     612 |           0.165033   |         305 |                   695 |          0.438849  |      -27.3816   |
| W       | dataset_source | historical/textvqa |           262 |                     612 |           0.428105   |          81 |                   695 |          0.116547  |       31.1558   |

| group   | feature          | cohort    |   n |      mean |   median |   p25 |   p75 |   total |   standardized_mean_difference |
|:--------|:-----------------|:----------|----:|----------:|---------:|------:|------:|--------:|-------------------------------:|
| W       | trigger          | completed | 612 | 13.719    |       14 |     9 |    18 |    8396 |                            nan |
| W       | trigger          | pending   | 695 | 20.1036   |       22 |    18 |    25 |   13972 |                            nan |
| W       | T                | completed | 612 | 14.281    |       14 |    10 |    19 |    8740 |                            nan |
| W       | T                | pending   | 695 |  7.8964   |        6 |     3 |    10 |    5488 |                            nan |
| W       | hamming1_rescued | completed | 612 |  0.240196 |        0 |     0 |     0 |     147 |                            nan |
| W       | hamming1_rescued | pending   | 695 |  0.109353 |        0 |     0 |     0 |      76 |                            nan |

Trigger and suffix length are deterministically related: T=28-trigger. No hypothesis test is needed to establish observed cohort composition differences.
