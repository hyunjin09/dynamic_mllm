# Binary POLAR Loss-Only Comparison: Implementation and Sanity Report

Date: 2026-08-12

Status: **implementation sanity passed; no predictor training was run**.

## Controlled question

The implemented comparison changes only how multiple complete valid 28-bit
routes supervise the direct binary head:

- `duplicated_bce`: one predictor/loss row per selected valid route, ordinary
  mean-over-bit BCE, with the selected routes for each input carrying total
  weight one;
- `exact_set_nll`: one grouped input with all selected valid routes, complete
  Bernoulli-mask log probabilities, normalized route log-weights, and a masked
  `logsumexp` over the valid-route dimension.

The exact objective is

```text
-log sum_m w_m exp(
    sum_l [m_l logsigmoid(z_l) + (1-m_l) logsigmoid(-z_l)]
)
```

with padded routes set to negative infinity before `logsumexp`. For the primary
comparison, `w_m = 1/K` for the `K` selected routes. Relative to an unnormalized
set-mass sum, this adds `log(K)` to a sample's reported loss but leaves its
parameter gradients unchanged.

## Existing POLAR-style pipeline audit

### Input and predictor

The current predictor remains question-only, matching the released POLAR input
contract:

1. the question is left-padded and tokenized to at most 512 tokens;
2. Qwen3-Embedding-0.6B produces frozen token representations;
3. a learned projection maps token features to width 256;
4. 28 learned layer queries cross-attend to the question tokens with four
   heads;
5. a two-block Transformer encoder exchanges information across the 28 layer
   representations;
6. a shared linear binary head produces one `VISUAL_ON` logit per layer.

The base embedding encoder remains frozen and is always called under
`torch.no_grad()`. The direct head is factorized across layers; the set loss
scores complete masks but does not introduce a dependency model between bits.

Two pre-existing deviations from literal released POLAR remain common to both
runs and were not changed: the output is a direct binary head instead of
POLAR's segmentation/three-operation heads, and the current layer encoder adds
the learned layer query as a residual before cross-layer encoding.

### Route representation and weighting

The regenerated-label plan now freezes at most 50 diverse, deduplicated
successful masks per input. The new comparison configuration refuses to load a manifest
that exceeds that cap; it never subsamples inside the training loader.

The duplicated baseline presents every selected route as a separate predictor
and BCE row. Because the question encoder is frozen and deterministic, its
output is computed once per unique input and indexed for the duplicated route
rows. This is an exact encoding reuse, not an averaged target: predictor
dropout and BCE are still evaluated independently for each route row. Route
microbatches are accumulated into one optimizer step for the same 32 unique
inputs used by grouped set-NLL.

Both primary runs use equal within-input route weights. The historical
all-ON-downweighting rule is not active. If tested later, it must be a separate
matched configuration applied identically to both objectives.

### Optimization and inference held fixed

The comparison configuration freezes:

- AdamW, learning rate `3e-4`, weight decay `0.01`;
- no scheduler;
- 10 epochs;
- 32 unique inputs per optimizer step;
- BF16 autocast, gradient clipping at `1.0`;
- seed `20260809`;
- dropout `0.1`;
- top-1 decoding by the mode of the 28 factorized Bernoulli decisions;
- exact factorized top-k enumeration only for the already-supported offline
  diagnostics.

Each future run records a SHA-256 over the initial predictor tensors. The two
training runs must have identical initialization hashes before their results
are compared.

## Minimal code changes

| File | Change |
|---|---|
| `binary_policy/losses.py` | Complete-mask probability now uses explicit stable `logsigmoid`; added per-route BCE needed for route microbatch accumulation. |
| `binary_policy/dataset.py` | Added grouped and duplicated collators with equal/manifest weighting, duplicate-mask rejection, route-cap enforcement, and image-group split-leakage checks. |
| `binary_policy/training.py` | Added the two-objective training switch, exact duplicated-route gradient accumulation, frozen-encoding reuse, and initialization hashing. |
| `experiments/train_binary_polar.py` | Added an explicit objective selector while leaving the predictor, optimizer, validation decoding, and encoder unchanged. |
| `configs/binary_polar_loss_comparison_v1.yaml` | Added the matched comparison configuration. Its missing regenerated-label P9 gates intentionally prevent training. |
| `tests/test_binary_policy_objective_comparison.py` | Added deterministic objective/data/gradient contracts. |
| `experiments/audit_binary_objective_comparison.py` | Added a CPU-only, no-model sanity artifact generator. |

The older `configs/binary_polar_qwen2_5_vl_7b_v1.yaml` remains on its historical
old-label configuration. The regenerated-label comparison now independently
uses the same maximum of 50 routes while preserving its own deterministic
diversity selection and equal-weight matched-objective contract.

## Deterministic sanity results

Evidence:
`outputs/binary_polar/preflight/loss_comparison_sanity_v1.json` and its SHA-256
sidecar.

| Check | Result |
|---|---:|
| Single-route exact set-NLL vs complete Bernoulli-mask NLL | absolute error `0.0` |
| Padded-route exclusion | absolute error `0.0` |
| Finite synthetic gradients | pass for both objectives |
| Frozen encoder parameter tensors receiving gradients | `0` for both objectives |
| Predictor tensors with finite gradients | `21/21` for both objectives |

For the contradictory valid set `{1100, 0011}`:

- duplicated BCE converged to the per-bit marginal solution: loss
  `0.6931471806`, probabilities within `3.06e-9` mean absolute distance of
  `0.5`;
- exact set-NLL fell from `2.7722637880` to `0.7028308300` and concentrated on
  `1100`, with probabilities approximately
  `[0.997575, 0.997589, 0.002425, 0.002411]`.

This is an implementation check only. Free per-example logits do not establish
shared-predictor optimization or held-out generalization.

Ten new deterministic objective tests pass. Sixteen existing binary-policy and
executor behavior tests pass, the existing label-regeneration unit suite
passes, and Python byte-compilation passes for every changed Python module.

## Review findings and boundaries

- No predictor architecture, MCTS route, MLLM executor, tokenizer, inference
  decoder, optimizer family, learning rate, scheduler, epoch count, or model
  weight was changed by this implementation.
- The exact likelihood sums 28 bit log probabilities while POLAR BCE averages
  28 bit losses. Their raw loss magnitudes are therefore not directly
  comparable. This is inherent to the frozen objectives; it was not rescaled.
- Duplicate masks are rejected at collation because multiplicity has no frozen
  semantic meaning. They must be deduplicated in the derived P8 supervision
  view.
- Samples with zero cached valid masks remain excluded from positive training
  but must remain in later execution evaluation.
- Cached Hit@1 remains an offline diagnostic only. Actual execution of every
  predicted top-1 mask, including uncached masks, is still the behavioral gate.

## Gate status

The loss/data/gradient implementation gate passes. Real smoke training remains
blocked because this task did not complete or approve the regenerated-label
P4-P9 audits, freeze the image-group-disjoint predictor manifest, load the
embedding model, train a predictor, or execute a predicted route.

The next bounded action is to finish/freeze the regenerated-label derived
manifest and then run only the matched duplicated-BCE versus exact-set-NLL
smoke training. Full training remains unauthorized until that smoke satisfies
the user's finite-loss, gradient, leakage, execution, and plausible-improvement
gate.
