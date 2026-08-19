# Stage C Pre-Run Power Analysis

Status: frozen before any held-out intervention result was inspected.

## Input and Endpoint

The calculation uses the 200 Stage B TextVQA records and only the frozen
layer-0 conditional READ effect with WRITE enabled:

```text
mean_answer_logprob(FULL) - mean_answer_logprob(WRITE_ONLY)
```

The input is `outputs/stage_b/stage_b_results_v1.jsonl` (SHA-256
`411cb24899973ad19d7c3186bfa9f88ae59f0a1ca09366b889b57053c32ba4d4`).
Those 200 discovery records have unique effective image assets.

## Observed Planning Values

| Quantity | Stage B value |
|---|---:|
| Mean | -0.052953 nats/token |
| Sample SD | 0.407473 |
| Standard error at n=200 | 0.028813 |
| Median | -0.000004 |
| 5th / 95th percentile | -0.406769 / 0.125852 |
| 5% trimmed mean | -0.012113 |
| Leave-one-out mean range | -0.058114 to -0.027584 |

The distribution is heavy-tailed and the mean is materially influenced by the
most negative record. That limits the transportability of any discovery-based
power estimate; it does not change the frozen mean endpoint.

## Calculation and Decision

Using the Stage B sample SD and a normal approximation for the probability that
a two-sided 95% mean CI lies below zero gives:

| Planning magnitude | 80% power | 90% power | 95% power | Power at n=800 |
|---|---:|---:|---:|---:|
| Observed `0.052953` | 465 | 623 | 770 | 95.7% |
| Conservative `0.05` | 522 | 698 | 864 | 93.5% |

The `0.05` row is a sizing sensitivity, not a replacement primary success
threshold. The Stage C primary analysis remains the image-clustered bootstrap
CI relative to zero.

**Decision:** retain the target of **800 eligible unique-image TextVQA
records**. The analysis does not justify reducing the user-proposed target.
Outcome-dependent sample-size adaptation is prohibited.

Strongest objection: the calculation assumes the Stage B mean and variance are
transportable despite a near-zero median and heavy tails. The 800-record target
is therefore a defensible planning size, not a guarantee of replication.

