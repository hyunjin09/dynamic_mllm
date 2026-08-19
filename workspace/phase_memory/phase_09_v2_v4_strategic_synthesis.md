# Phase 09: v2–v4 Strategic Synthesis Memory

## Current Objective

Synthesize the closed v2–v4 local READ/WRITE program and recommend one
genuinely new, bounded direction without executing an experiment.

## Active Constraints

- Preserve v2 Outcome B, v3 `STOP_V3_CONFIRMATION`, and v4
  `STOP_DYNAMIC_POLICY_DIRECTION`.
- No experiment, model/router/probe training, null-gate relaxation, new
  READ/WRITE-route search, or continuation of local layer skipping.
- Separate descriptive action-value observations from causal and practical
  policy claims.
- Recommend exactly one next experiment, but do not execute it.

## Current State

- Done: reviewed the v2, v3, and v4 decision chain and active workflow state.
- Done: evidence classification, failure diagnosis, new-direction comparison,
  and independent decision review.
- Blocked: no execution blocker; any proposed strategic pivot requires later
  explicit approval.
- Most recent useful observation: v4 found frequent local action disagreement
  but only `0.02370` maximum pooled matched-compute mean utility gain, while
  the unconstrained query-conditioned oracle used more compute.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Early layer-0 WRITE has large positive reference-support effects in both tasks. | `reports/stage_b_conclusion.md`; `reports/v3_stage_b_reanalysis.md` | Supports a robust functional READ/WRITE asymmetry, not harmfulness. | confirmed discovery |
| The held-out layer-0 READ contrast replicated but failed both structured null comparisons. | `reports/stage_c_conclusion.md` | Closes the causal answer-misaligned READ claim and warns that likelihood variation can be nonspecific. | confirmed Outcome B |
| Complete four-action values are heterogeneous, interactive, and poorly fit by fixed schedules. | `reports/v3_stage_b_reanalysis.md` | Supports descriptive local heterogeneity while leaving specificity unresolved. | confirmed discovery |
| Both v3 search-matched null families failed unchanged geometry gates after independent calibration. | `reports/v3_null_redesign_v2.md` | Prevents causal confirmation; does not establish a positive or negative action mechanism. | confirmed technical failure |
| Same-image visual state/WRITE is exactly query-invariant under common padding, yet action ranking varies by question. | `reports/v4_common_padding_preflight.md`; `reports/v4_discovery_results.md` | Shows query-associated downstream valuation of a query-blind visual computation. | confirmed discovery |
| Query-conditioned local action choice has little sustained practical cost–utility headroom. | `reports/v4_cost_utility_reanalysis.md` | Rejects continued local routing as the next direction. | confirmed discovery |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| v2 harmful layer-0 READ confirmation | Real removal did not beat either structured residual null; median/trimmed effects were near zero and prefix-sensitive. | supported | `reports/stage_c_results.md` | Reference-likelihood movement is not sufficient for causal harmfulness. | READ-specific Stage D or endpoint relabeling |
| v3 maximum-over-21 confirmation | Valid search-matched covariance and donor nulls could not be frozen under unchanged gates. | supported | `reports/v3_null_redesign_v2.md` | Preserve discovery only; do not weaken specificity controls. | Wider calipers, relaxed fidelity, or another residual-route search |
| v4 query-conditioned local routing | Frequent disagreement produced small pooled oracle and cost-frontier gains. | supported | `reports/v4_cost_utility_reanalysis.md` | Selecting among query-blind keep/drop actions lacks practical separation. | Paraphrase/held-out work intended to rescue the same local policy |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Query-conditioned visual refinement/revisitation | The frozen prefix architecture cannot write question information into visual state; a post-question visual update adds the missing capability. | Whether genuinely query-specific visual computation creates headroom. | low for frozen existing-token replay; medium for adapter | selected proposal |
| Explicit query-writable visual memory | READ/WRITE asymmetry suggests a controlled memory interface may be more meaningful than dense residual suppression. | Whether separable semantic memory operations improve evidence use. | medium-high | promising but second |
| Sparse query-triggered raw-image evidence acquisition | Heavy tails may reflect a few questions needing additional localized evidence rather than layer skipping. | Whether revisiting selected pixels/resolution improves robust utility. | low-medium with oracle regions | overlaps first; candidate component |

## Next-Step Decision

- Deliberation mode: deep.
- Active objective and bottleneck: choose one new capability whose value can be
  falsified cheaply; avoid another local-routing variant.
- Relevant memory item used: exact common-padding query-invariance of visual
  WRITE plus the small v4 cost–utility frontier.
- Confirmed observation: existing actions only retain/remove query-blind visual
  operations and have insufficient practical pooled separation.
- Unverified interpretation: allowing the question to actively refine or
  revisit visual evidence will create robust behavioral headroom.
- Diagnosis: suspected for query-blind action-space limitation as a cause;
  supported for the architectural query-invariance fact.
- Evidence path if diagnosis is not unknown: `reports/v4_discovery_results.md`;
  `reports/v4_cost_utility_reanalysis.md`.
- Viable alternatives considered: query-conditioned refinement/revisitation,
  explicit visual memory, or project closure; multi-layer suppression remains
  technically unresolved but is not a genuinely new capability.
- Chosen action: recommend a frozen-model, oracle-grounded existing-token
  replay/refinement falsification test; do not execute it.
- Strongest objection: privileged target boxes and extra visual tokens may
  create gains unrelated to a learnable query-conditioned refinement.
- How this differs from failed attempts: it creates new post-question visual
  evidence/state instead of selecting among the same four local keep/drop
  actions.
- Automatic execution authorized: no.
- Authorization basis: this task authorizes synthesis and design only.
- Stop condition: save the two requested planning documents with one final
  decision and stop before any experiment or training.

## Latest Research-Action Result

- Action taken: synthesized v2–v4, ranked five failure explanations, compared
  three new capability directions, and reconciled one independent review.
- Result: recommend `TEST_QUERY_CONDITIONED_VISUAL_REFINEMENT` using fixed-
  budget post-question replay of already encoded target-region tokens.
- Evidence saved: `reports/dynamic_mllm_v2_v4_synthesis.md` and
  `workspace/dynamic_mllm_next_direction.md`.
- Failure or issue: the initial crop-reencoding proposal had a privileged-
  evidence confound; review-supported revision reuses existing tokens and adds
  matched target/swap/non-target/random/whole-image controls.
- Lesson learned: the smallest clean discriminator must change the causal
  capability without simultaneously adding new pixels, learned capacity, or a
  second vision pass.
- Next implication: wait for explicit approval; no feasibility freeze,
  implementation, inference, or training is authorized.
