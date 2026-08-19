# Frozen Stage C Outcome B Closure

Status: **closed under the frozen protocol on 2026-08-05**

## Closed hypothesis

The project no longer pursues the hypothesis that Qwen2.5-VL-7B-Instruct has a
confirmed harmful or answer-misaligned visual READ contribution at TextVQA
layer 0 under the approved `FULL - WRITE_ONLY` reference-likelihood protocol.

The previously planned Stage D is cancelled for this path. Do not run exact
add-back, dose-response, grounded mediation, attention/FFN decomposition, or a
new READ-specific harmful-mechanism search as a continuation of this protocol.

## Supported conclusion

1. The held-out TextVQA layer-0 reference-support effect replicated: mean
   `-0.07294332` nats/token with image-clustered 95% CI
   `[-0.14127645, -0.01710262]`.
2. The effect was heavy-tailed and prompt-sensitive. The median and trimmed
   means were near zero, and the contextual `Answer:` prefix contrast crossed
   zero with sign agreement `0.5725`.
3. The actual READ-removal intervention did not outperform either frozen
   structured residual-null family. Both paired 95% confidence intervals
   crossed zero, including the secondary 798-target original-caliper
   sensitivity.
4. Therefore, the evidence does not support a confirmed answer-misaligned
   visual READ effect or a harmful visual participation claim.

The frozen classification is **Outcome B**:

> The reference-support effect replicated, but it was not distinguishable from
> the frozen structured intervention nulls.

## Preserved descriptive secondary findings

- The frozen reference-versus-original-wrong-answer margin shifted positively:
  mean `Delta C = 0.41465425`, clustered 95% CI
  `[0.17302903, 0.69316965]` over 189 eligible FULL-wrong records.
- Deterministic greedy outcomes contained 22 FULL-wrong to WRITE_ONLY-correct
  transitions and 12 FULL-correct to WRITE_ONLY-wrong transitions.
- The descriptive net is `+10` strictly correct answers over 800 records.

These are secondary observations only. They are not an accuracy-improvement
claim, a causal correction claim, a READ-specific mechanism claim, or evidence
of harmful visual participation. They do not override the failed structured-
null conjunction.

## Scope of closure

The harmful layer-0 READ hypothesis is closed for this model revision, TextVQA
population, frozen manifest, intervention, reference scoring, null families,
and analysis protocol. No post-hoc endpoint search or Stage D execution is
authorized. A materially different future direction would be a new strategic
proposal requiring explicit user approval; it cannot be described as
continuation or rescue of this frozen confirmation.

## Evidence and archive

- `outputs/stage_c/stage_c_results_v1.jsonl`
- `outputs/stage_c/analysis_v1/analysis_manifest.json`
- `reports/stage_c_results.md`
- `reports/stage_c_conclusion.md`
- `outputs/stage_c/stage_c_completion_v1.sha256`
- `archives/stage_b_stage_c_frozen_outcome_b_v1/`

