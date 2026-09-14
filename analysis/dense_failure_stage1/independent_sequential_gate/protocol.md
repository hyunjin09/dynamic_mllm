# Independent Layer-Wise Sequential Gate Protocol

- Frozen contract: `6b1e4812a0a1e51f3dec822964b848a3a0410451dfe7a289d25c4b378653831d`
- Reused Phase-48 contract: `3cf49a46d0a47a0ae1955f8a518f5a74bef65b7de8736980a6e3d26e170234cd`
- No predictor is retrained. The 28 frozen Phase-48 linear probabilities are rescored on its exact 800 validation and 800 test UIDs.
- Test scores remain unopened until validation freezes every sequential and fixed-layer threshold.
- At each layer, the threshold is the empirical higher `(1-alpha)` quantile of validation-correct scores. The gate uses strict `p_l > tau_l` and stops at the first crossing.
- Shared-alpha grid: `41` values from `0.00000000` through `0.10000000`, covering every attainable 400-correct empirical tail breakpoint up to 10% plus exact 10%.
- For each 99%/98%/95% target, select the largest alpha whose full sequential validation trajectory retains at least the target fraction of correct samples.
- Fixed L14/L21/L27 thresholds are tie-safe strict-crossing thresholds calibrated independently on validation at the same preservation targets.
- The same global sequential threshold vector is used across GQA, ChartQA, and TextVQA. No dataset-specific calibration is allowed.
- A trigger is admission for possible intervention only; no treatment is executed.

Finite-sample note: empirical higher quantiles may produce coarse steps or a zero-trigger conservative point. This limitation is reported directly rather than hidden with interpolated tail resolution.
