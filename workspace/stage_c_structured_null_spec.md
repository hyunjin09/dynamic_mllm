# Frozen Stage C Structured READ-Residual Null Specification

Status: **frozen outcome-blind on 2026-08-05**. All estimator, matching,
draw, seed, and comparison choices below were fixed without computing or
inspecting the Stage C primary likelihood endpoint.

## Shared Estimand and Hook

The real sample effect is

```text
U_i = mean_positive_i(FULL) - mean_positive_i(WRITE_ONLY).
```

Both nulls operate at Qwen2.5-VL decoder layer 0, immediately after
`self_attn.o_proj` and before the attention residual addition. The READ
residual is the representable `FULL - READ_OFF` attention-output delta on
post-image nonvisual prompt rows. Visual rows and every nonvisual attention
path remain the actual FULL path. Each null replaces only that READ delta,
starts from the identical cached pre-layer state, keeps WRITE enabled, and runs
the unchanged suffix.

For draw `d`, the null effect has the same orientation as the real effect:

```text
U_i,null,d = mean_positive_i(NULL_READ_d + WRITE) - mean_positive_i(WRITE_ONLY).
```

No likelihood or greedy behavior is used to construct either null.

## Calibration Data

- Pool: all 200 frozen Stage B TextVQA records in
  `data_manifests/stage_b_discovery_candidates_400.jsonl`.
- Effective images: 200, using the frozen Stage B `selection_asset_key` when a
  source image ID is absent.
- Layer/hook/task: layer 0, exact READ hook above, TextVQA only.
- Extracted object: post-image nonvisual-row `FULL - READ_OFF` residual; the
  forward is stopped immediately after the hook, before the suffix or any
  answer scoring.
- Stage C inputs, effects, scores, and correctness are absent from fitting.

The frozen residual tensor and metadata index are
`outputs/stage_c/nulls/stage_b_read_residuals_v1.pt` and
`outputs/stage_c/nulls/real_residual_donor_index_v1.jsonl`.

## Family 1: Covariance/Subspace-Matched Random Residual

1. Map every variable-length residual to 32 rows by channel-wise linear
   interpolation with aligned endpoints; flatten the `32 x 3584` tensor.
2. Center by the equal-sample mean over the 200 calibration vectors.
3. Estimate the unbiased sample covariance with denominator `n-1`, using the
   exact sample Gram eigendecomposition rather than a feature-space
   approximation.
4. Retain the smallest leading rank reaching at least 90% total positive
   eigenvalue variance. This freezes rank **66**, explaining
   `0.9017906189`; leave-one-effective-image-out ranks range from 65 to 66
   (median 65).
5. Shrink each retained eigenvalue 5% toward the mean retained eigenvalue:
   `lambda'_j = 0.95 lambda_j + 0.05 mean(lambda)`.
6. Draw independent standard-normal coefficients in the retained basis, add
   the fitted mean, map from 32 rows to the target post-image row count, and
   rescale to the target record's actual layer-0 READ-residual Frobenius norm.

There are eight draws per Stage C record. Base seed `2026080511` is converted
to a record/draw seed by the first eight bytes of
`SHA256("2026080511:{record_id}:{draw_index}")`, reduced modulo `2^63-1`.
Native-grid affine-subspace relative error must be at most `0.05`; row-mapping
shape and exact norm matching are tested separately with relative norm error at
most `1e-5`.

Zero target norm deterministically yields a zero null. Empty residuals,
nonfinite norms, no positive covariance eigenvalue, rank below one, or a
nonzero target paired with a degenerate sampled vector fail closed. No
held-out-driven rank or regularization fallback is permitted.

Fitted parameters are frozen at
`outputs/stage_c/nulls/covariance_subspace_parameters_v1.pt`.

## Family 2: Same-Layer Cross-Sample Real Residual

- Donor pool: the same 200 Stage B TextVQA layer-0 residuals and frozen donor
  metadata above.
- Exclusions: donor sample ID must differ from the target sample ID and donor
  image ID must differ from the target image ID. The held-out manifest already
  has no Stage B image overlap.
- Matching covariates: pre-rescale residual Frobenius norm, post-image
  nonvisual row count, and image-token count. Define each multiplicative ratio
  as `max(donor/target, target/donor)` and their composite distance as the
  maximum of the three ratios.
- Frozen caliper rule: on Stage B only, find each target's eighth-nearest
  eligible donor and take the maximum such distance. The resulting common cap
  is **1.5**. Thus every matched donor differs by at most 1.5-fold on every
  matching covariate. Leave-one-image-out calibration coverage is 15–197
  eligible donors per target (median 186), with zero failures.
- Selection: sort eligible donors by composite distance, then norm ratio, row
  ratio, image-token ratio, and a seeded SHA-256 tie break. Take the first
  eight without replacement. No likelihood effect enters ranking.
- Generation: linearly map each real donor residual to the target row count
  and rescale it to the exact target residual Frobenius norm.

Answer length is not a matching covariate because the answer is absent at the
prompt hook. Prompt length is recorded but not separately matched: its two
relevant geometry components are represented by post-image row count and
image-token count. The base tie seed is `2026080521`; each target receives the
precomputed SHA-derived seed recorded in
`outputs/stage_c/nulls/deterministic_null_seeds_v1.jsonl`.

Fewer than eight donors within the 1.5 cap, any same-sample/image donor,
invalid row geometry, or a degenerate nonzero-norm donor fails closed. Donors
are never replaced using downstream behavior. The donor pool, residual tensor
indices, norms, row/image/prompt lengths, task, layer, hook, and selection
seeds are frozen before Stage C scoring.

## Frozen Comparison and Success Rule

For each family, average its eight null effects within sample and form

```text
D_i,family = U_i,real - mean_d(U_i,null,family,d).
```

Negative `D` means the actual READ effect is more negative than that structured
null. Aggregate questions to equal-weight effective-image cluster means, then
use a two-sided 95% percentile bootstrap with 10,000 image-cluster resamples.
Seeds are `2026080532` for the covariance family and `2026080533` for the real
residual family.

The Stage C primary replication gate is separate and passes only when the
image-clustered 95% CI for mean real `U` lies wholly below zero. The phrase
**confirmed answer-misaligned READ effect** additionally requires the upper
95% CI endpoint of `D` to be below zero for **both** null families. This is an
intersection-union conjunction: each family-specific two-sided 95% interval
must pass, with no alpha split and no pooled-null rescue. Report means and
paired differences for both families. The secondary `-0.05` threshold cannot
replace any gate.

The frozen analysis implementation is
`tools/research_analysis/v2/stage_c_null_comparison.py`. All null artifacts are covered by
`outputs/stage_c/nulls/null_artifacts_v1.sha256`.
