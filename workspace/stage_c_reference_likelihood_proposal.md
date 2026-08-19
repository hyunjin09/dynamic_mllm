# Frozen Stage C Reference-Likelihood Protocol

Status: **approved and frozen at the endpoint level on 2026-08-05** by the
user's “Stage C Amendment Decision.” This is an explicit Stage C operational
amendment. The approved source-plan file remains unchanged.

This approval does not authorize Stage D, broaden the primary analysis, or
support a harmful-mechanism or accuracy-improvement claim.

## Frozen Primary Analysis

- Model/runtime: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, unchanged stock-eager path.
- Dataset: a new held-out TextVQA set with no Stage B record or effective-image
  overlap.
- Target: 800 eligible unique-image records. The manifest must be frozen before
  any held-out intervention score is inspected.
- Layer: `0` only.
- Operation/conditioning: conditional READ effect with WRITE enabled only.
- Primary state contrast:

  ```text
  U_read_l0_w1(i) = M_i(FULL) - M_i(WRITE_ONLY)
  ```

  where `M_i` is the accepted-reference per-token mean log-likelihood score.
- No other layer, dataset, operation, factorial effect, interaction,
  correctness stratum, or greedy behavior may replace or be searched for in
  the primary analysis.

For each normalized accepted answer `a` with annotation-frequency weight
`w_a`, the inherited Stage B scoring rule is:

```text
m_i(a, state) = sum_t log p(y_t | image, prompt, y_<t, state) / T_a
M_i(state)    = logsumexp_a(log(w_a) + m_i(a, state))
```

Weights sum to one and remain identical across states. Only answer tokens are
scored. The primary prompt, answer prefix, normalization, tokenizer, processor,
and chat template are identical across `FULL` and `WRITE_ONLY` and are frozen
before held-out scoring. Per-answer scores remain auditable.

## Primary Success Gate

Cluster at effective image ID. The primary replication criterion passes only
if the two-sided image-clustered 95% bootstrap confidence interval for the
held-out mean `U_read_l0_w1` lies entirely below zero.

Sequence-sum accepted-reference likelihood is secondary. It cannot rescue a
failed primary endpoint.

## Secondary Practical-Effect Analysis

The `-0.05` nats/token threshold is not a primary success gate. Report:

- whether the held-out mean is at or below `-0.05` nats/token;
- the image/sample-level fraction with `U_read_l0_w1 <= -0.05`;
- an image-clustered 95% confidence interval for that fraction.

Retain the Stage B numerical floors (`1e-6` nats/token and `1e-5` sequence
nats) as noise descriptors, not as practical or confirmatory thresholds.

## Required Replacement Controls

Before opening the primary result, the Stage C implementation must pass:

1. exact unmodified/FULL/no-op parity plus READ and WRITE reconstruction at
   the validated hook and through the unchanged suffix;
2. covariance/subspace-matched random READ-residual controls at layer 0;
3. same-layer cross-sample real READ-residual controls with norm matching;
4. a frozen FULL-wrong-answer contrast: for records whose original frozen FULL
   greedy answer is wrong, compare accepted-reference support against that
   exact frozen wrong answer under `FULL` and `WRITE_ONLY`;
5. accepted-answer aggregation and answer-prefix robustness checks;
6. image-level clustered inference whenever questions share an image.

The actual READ-removal effect must outperform both required structured
residual families before the result may be called a **confirmed
answer-misaligned READ effect**. The exact covariance estimator, subspace rank,
matching caliper, number of null draws, and paired null comparison statistic
remain a pre-execution implementation specification. They must be frozen and
validated without held-out outcome inspection; no convenient fallback is
allowed after outcomes are visible.

## Secondary Outcomes

The other factorial states may be collected for audit efficiency. Their
effects, interaction, correctness strata, and deterministic greedy behavior
are secondary and cannot change the primary decision. Sequence-sum likelihood
and accepted-answer/prefix robustness are also secondary.

Top-1 changes do not establish a causal correction. Stage C is not a search
over secondary outcomes.

## Interpretation Boundary

If the primary CI is entirely below zero and the actual effect outperforms both
structured residual nulls, Stage C supports a held-out answer-misaligned
reference-support effect. It does not establish a harmful mechanism or an
accuracy improvement.

Exact add-back, dose response, and grounded mediation remain required in Stage
D before any harmful visual participation claim. Stage D requires separate
authorization.

## Preconditions Still Required Before Execution

- audit the eligible TextVQA source and freeze the 800-record manifest with
  record- and effective-image-overlap checks;
- freeze the structured-null construction and paired comparison details listed
  above;
- freeze primary and robustness prompt prefixes without inspecting held-out
  intervention outcomes;
- implement and pass the Stage C validity/control probe;
- log the pinned runtime and the final analysis seed/bootstrap configuration.

