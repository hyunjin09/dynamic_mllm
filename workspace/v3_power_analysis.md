# v3 Held-Out Power and Precision Analysis

Status: prospective and outcome-blind. No held-out action value was loaded or
computed.

## Planning quantities

The proposed primary statistic is the per-record maximum FULL-relative gain
over seven nonterminal layers and three non-FULL actions. The relevant inspected
Stage B distribution is therefore the sample-level maximum, not the raw
sample-layer oracle mean `0.0976`.

| Dataset | n | Mean | SD | Median | 20% trimmed mean | Fraction > 0.05 |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 200 | 0.3795 | 0.7206 | 0.1341 | 0.1787 | 0.655 |
| TextVQA | 200 | 0.2580 | 0.6069 | 0.0773 | 0.0939 | 0.590 |
| Equal-weight joint | 400 | 0.3188 | 0.6681 | 0.0972 | — | 0.6225 |

These are discovery descriptions only. Their large mean-to-median gaps require
tail-robust gates. The old single-contrast Stage C paired real-minus-null SDs
were about `0.21–0.29`, but the variance of the new maximum-over-21-cells
real-minus-null statistic is unknown. Planning therefore uses paired-difference
SD scenarios `0.30`, `0.50`, and `0.75`.

## Candidate sizes

Normal approximations below use a two-sided 95% lower-bound criterion and 80%
power. They are precision guides, not substitutes for the frozen clustered
bootstrap.

| Unique-image records | Paired SD | 95% half-width | 80% MDE |
|---:|---:|---:|---:|
| 800 | 0.30 | 0.0208 | 0.0297 |
| 800 | 0.50 | 0.0346 | 0.0495 |
| 800 | 0.75 | 0.0520 | 0.0743 |
| 1,200 | 0.30 | 0.0170 | 0.0243 |
| 1,200 | 0.50 | 0.0283 | 0.0404 |
| 1,200 | 0.75 | 0.0424 | 0.0607 |
| 1,600 | 0.30 | 0.0147 | 0.0210 |
| 1,600 | 0.50 | 0.0245 | 0.0350 |
| 1,600 | 0.75 | 0.0367 | 0.0525 |

## Frozen size proposal

Use `1,600` unique images, balanced as `800 GQA + 800 TextVQA`, with an
equal-dataset-weight joint primary estimator and dataset-specific secondary
estimates. This choice is driven by precision under the conservative full-max
paired-SD envelope, not by the discovery oracle mean. Unique images make the
primary cluster count equal to the record count.

The final manifest is intentionally not frozen in this preflight because the
joint-path donor geometry and final processor/token validation have not yet
passed. The deterministic construction rule and a 1,600-record identity
preview are recorded in `outputs/v3_preflight/candidate_pool_audit.json`.

## Remaining uncertainty

One structured-null draw per sample and cell would devote compute to independent
images and is statistically defensible only if an inspected-calibration
seed-stability check shows negligible seed contribution to the full-max mean.
That check has not been run. The old narrow-endpoint variance alone is not
sufficient to freeze the draw count.
