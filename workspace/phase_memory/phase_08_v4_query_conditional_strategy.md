# Phase 08: v4 Query-Conditional Strategy

## Current Objective

Determine whether the completed same-image GQA discovery supplies a meaningful
query-conditioned compute-allocation advantage, then close or justify the one
remaining semantic control.

## Active Constraints

- Preserve v2 Outcome B and v3 `STOP_V3_CONFIRMATION` artifacts and gates.
- The authorized discovery is complete; no further action values, held-out
  outcomes, or training are authorized.
- Use exact validated actions, dense suffix, accepted-reference likelihood, and
  common right-padding.
- No harmfulness, semantic-mechanism, routing, or acceleration claim.

## Current State

- Done: v3 closure evidence, reusable implementation, query-invariance proof,
  common-padding diagnostic, pool capacity, and GQA semantic metadata audited.
- Done: v4 plan, frozen 120-image manifest, 12-image exact common-padding gate,
  6,720 action scores, and image-clustered discovery analysis.
- Closed: the dynamic-policy direction under this protocol; the cost–utility
  frontier does not justify the proposed paraphrase control or confirmation.
- Most recent useful observation: transfer regret is robust (mean 0.1121,
  median 0.0634), while the image+query oracle gap is small (mean 0.0144,
  median 0.00344) and semantic-stratum ordering is mixed.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Both v3 specificity-null families fail unchanged gates. | `reports/v3_null_redesign_v2.md` | Closes harmfulness confirmation, but not the distinct within-image question. | confirmed |
| Common padding restored bitwise visual-state/WRITE equality in the existing diagnostic. | `workspace/v3_query_invariance_validation.md` | Supports the v4 causal premise while requiring a broader prospective gate. | confirmed, one image |
| GQA has ample new multi-question groups; Stage B has none. | `outputs/v3_preflight/stage_c2_reserved_pool_audit.json`; `outputs/v3_discovery/same_image_feasibility_v1.json` | A new discovery is necessary and feasible. | confirmed |
| GQA instruction data expose semantic programs, object annotations, types, equivalents, and entailment. | `datasets/datasets/lmms-lab___gqa/train_balanced_instructions/.../dataset_info.json` | Enables outcome-blind semantic controls. | confirmed schema; pair counts unverified |
| The 12-image common-padding gate passes exactly. | `outputs/v4_discovery/preflight/v4_common_padding_preflight_v1.json` | Establishes identical visual state/WRITE for the discovery comparison. | confirmed |
| Action transfer differs across questions, but fixed-image oracle loss is small. | `outputs/v4_discovery/analysis_v1/image_clustered_summaries_v1.csv` | Supports action-pattern variation while weakening the fixed-policy-insufficiency claim. | confirmed discovery |
| No prospective official paraphrase pair is available; semantic ordering is mixed. | `outputs/v4_discovery/analysis_v1/semantic_control_sensitivity_v1.json` | Blocks the frozen semantic confirmation gate. | confirmed |
| Query conditioning does not provide a sustained material pooled frontier advantage. | `outputs/v4_discovery/cost_utility_frontier_summary_v1.json`; `reports/v4_cost_utility_reanalysis.md` | Closes the dynamic-policy direction despite frequent local action disagreement. | confirmed discovery |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| v3 structured-null confirmation | Donor caliper `3.09375`; covariance native-fidelity gates fail. | supported | `reports/v3_null_redesign_v2.md` | Ask a distinct within-image question; do not weaken specificity gates. | More caliper/rank/pool retuning or harmfulness relabeling. |
| Unequal-shape same-image comparison | BF16 visual/WRITE divergence despite zero future attention. | supported | `outputs/v3_preflight/query_invariance_equal_length_diagnostic.json` | Common right-padding is mandatory before query comparisons. | Unpadded cross-question visual-state comparisons. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Prospective paraphrase-only discovery amendment | Robust transfer survives existing controls, but semantic stability is untested. | The one missing semantic discriminator. | low-medium | rejected after cost–utility reanalysis |
| Close v4 | The pooled frontier advantage is small and not explained by image-only over-computation. | Ends the direction without more discovery compute. | none | selected |
| Proceed to held-out confirmation | Core dependence is measurable. | Would test replication. | high | rejected; semantic gate not met |

## Next-Step Decision

- Deliberation mode: deep.
- Active objective and bottleneck: resolved by the deterministic local-FLOP
  frontier reanalysis.
- Confirmed observation: the maximum pooled matched-compute mean utility gain
  is `0.02370`; mean matched-utility saving is `1.40%` FULL and applies above
  10% FULL on only `3.40%` of utility targets. Query-conditioned unconstrained
  choices use `2.62%` more FULL-equivalent compute.
- Unverified interpretation: none required for closure; wall-clock benefit and
  pre-action predictability were not tested.
- Viable alternatives considered: run the paraphrase-only control or close the
  dynamic-policy direction.
- Chosen action: close the direction and preserve the query-associated action
  variation as inspected discovery evidence.
- Strongest objection: median frontiers save at least 10% FULL across 26.97%
  of utility targets, but the median utility range is narrow and the maximum
  matched-compute median gain is only `0.01674`.
- Authorization and stop condition: the authorized reanalysis is complete; do
  not execute another experiment under v4.

## Latest Research-Action Result

- Action taken: computed exact operation-level action costs and the image-only
  versus image+query oracle frontier from the existing 1,680 Q matrices.
- Result: local pairwise frontier expansion is common, but pooled practical
  gains are small and the conservative-FULL hypothesis is rejected.
- Evidence saved: `outputs/v4_discovery/cost_utility_frontier_v1.csv`,
  `outputs/v4_discovery/oracle_action_costs_v1.csv`, and
  `reports/v4_cost_utility_reanalysis.md`.
- Failure or issue: none. An isolated matched-utility saving and layer-0 mean
  excursion are retained as sensitivity evidence rather than hidden.
- Lesson learned: frequent query-specific oracle disagreement does not imply a
  material compute-allocation frontier advantage.
- Next implication: stop the dynamic-policy direction under the frozen v4
  protocol; do not run the paraphrase control.
