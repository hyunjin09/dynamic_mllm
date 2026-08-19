# v4 Discovery Decision

## Decision target

Decide whether the inspected 120-image GQA discovery supports immediate
held-out confirmation, requires one missing semantic discriminator, or closes
the v4 direction. The held-out stage, probe training, routing, TextVQA
substitution, and any claim expansion remain unauthorized.

## Confirmed observations

- The architecture, common-padding identity, intervention, FULL parity, target
  alignment, and completion gates passed.
- Epsilon-aware action disagreement is frequent (equal-layer mean 0.676, median
  0.714), and conditional sign reversals occur in roughly 44%--48% of
  image-layer comparisons.
- Bidirectional transfer regret is robust: mean 0.112, median 0.0634, and
  20%-trimmed mean 0.0711 nats/token. The trimmed mean remains 0.0767 after
  removing any image with an epsilon tie and 0.0511 on the strict 40-image
  difficulty/type/answer-format/answer-length matched subset.
- The image+query versus best image-only action gap is much smaller: mean
  0.0144, median 0.00344, and 20%-trimmed mean 0.00644. Only 10% of images reach
  0.05 on the equal-layer gap.
- The effect is heavy-tailed but not confined to one layer. Removing the top
  5% leaves transfer regret 0.0854.
- The semantic ordering is mixed. Different-evidence pairs have more robust
  disagreement, but not more transfer regret after adjustment, and their
  query-oracle gap is smaller than the comparison stratum.
- No natural official paraphrase passed the prospective metadata rule, so the
  mandatory paraphrase-stability comparison is not evaluable.

## Interpretation

The discovery supports query-associated variation in exact four-action value
patterns while visual state and WRITE are held fixed. It does not yet support
the stronger semantic interpretation or the claim that a fixed image-only
action is substantially insufficient: the direct oracle gap is practically
small, and the planned semantic coherence test is missing.

This is not an accuracy, acceleration, routing, harmfulness, or semantic causal
mechanism result.

## Candidate comparison

1. **Prospective paraphrase-only amendment.** Freeze verified paraphrases for a
   deterministic discovery subset without consulting action outcomes, then
   test whether within-question paraphrases are more stable than the already
   frozen different-evidence pairs. This resolves the one mandatory missing
   discriminator at moderate cost. Its main weakness is that even a favorable
   result would not enlarge the currently small image-only oracle gap.
2. **Close v4 now.** This is defensible from the small oracle gap and mixed
   different-evidence ordering. Its main weakness is that robust transfer
   regret clears the prespecified 0.05 criterion across the main sensitivities,
   so closure would occur before the plan's specified paraphrase repair is
   attempted.
3. **Proceed directly to held-out confirmation.** Rejected: the frozen semantic
   gate is unevaluable, and discovery cannot be used as confirmation.

## Challenge and independent review

The provisional choice was immediate closure because the direct image-only
oracle gap is small. The strongest counterargument was the stable transfer
regret and the plan's explicit prospective-amendment route when official
equivalents are insufficient. One independent read-only reviewer ranked a
paraphrase-only amendment first with high confidence. That correction is
accepted because it follows the frozen plan and addresses a specific missing
discriminator rather than searching a new layer, task, action, or metric.

## Required user decision

No further experiment is authorized. The user must choose whether to approve a
prospective paraphrase-only discovery amendment. If it is not approved, the
scientifically defensible alternative is to close v4 without confirmation.

The amendment, if approved, must freeze its image IDs, literal questions,
semantic/answer equivalence validation, scoring, comparisons, and failure rule
before any new action score is computed. It may not reopen v3 harmfulness,
change the existing Q values, or act as held-out evidence.

`STOP_BEFORE_V4_CONFIRMATION`
