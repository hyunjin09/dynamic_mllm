# Stage C Real-Residual Caliper Amendment v1

Status: **approved and frozen before resumed outcome execution on 2026-08-05**

## Authorized change

The same-layer cross-sample real-residual matching caliper changes from `1.5`
to the exact outcome-blind coverage value `1.5833333333333333` (`19/12`). The
800-target geometry audit established this as the smallest common cap that
admits eight eligible donors for every frozen target.

No other protocol element changes. The manifest, primary endpoint, scoring,
layer and READ hook, 200-record donor pool, exclusions, matching distance and
covariates, eight-donor count, nearest-donor ranking, exact norm matching,
covariance null, seeds, bootstrap, comparison statistic, and scientific gates
remain frozen.

## Frozen amended donor index

- Index: `outputs/stage_c/nulls/stage_c_real_residual_match_index_v2.jsonl`
- Target records: 800
- Donors per target: 8
- Original-caliper coverage: 798/800
- Amended-caliper coverage: 800/800
- Original-supported targets whose nearest-eight selection changed: 0
- Targets requiring the amendment:
  `textvqa:textvqa_validation_39543` and
  `textvqa:textvqa_validation_36174`

Target `textvqa:textvqa_validation_36174` is explicitly flagged as having
weaker coverage: only three donors were available at 1.5 and five additional
selected donors enter at the amended `19/12` boundary.

The index was derived only from the frozen geometry audit, donor metadata, and
tie seeds. No likelihood, generated answer, correctness, intervention effect,
or partial Stage C result was loaded or used.

## Execution and analysis rules

The prior 93 partial records remain scientifically unopened and are not reused.
The resumed sweep recomputes all 800 records in fresh shards, including every
real-residual-null score under the amended index.

The primary real-residual comparison uses all 800 targets. A secondary,
prespecified sensitivity repeats that comparison on the 798 targets that
already had eight donors under 1.5. It cannot replace the all-800 gate.

No retuning is permitted after execution begins. Stage D remains unauthorized.

