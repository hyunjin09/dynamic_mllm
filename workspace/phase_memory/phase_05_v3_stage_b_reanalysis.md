# Phase 05: v3 Stage B Reanalysis Memory

## Current Objective

Deterministically reanalyze the preserved 400-record Stage B result set as
complete v3 dense-suffix four-action values and decide whether a bounded v3
preflight is scientifically justified.

## Active Constraints

- Read existing outcomes only; do not load the model or collect interventions.
- Treat per-token accepted-reference likelihood as the primary cross-sample
  metric and sequence likelihood as secondary.
- Treat all Stage B/C outcomes as inspected discovery evidence.
- Preserve v2 Stage C as Outcome B and do not reuse its 800 records as v3
  confirmation.
- No training, old Stage D, new large sweep, or confirmatory-manifest freeze.

## Current State

- Done: v3 migration audit and reuse decision.
- Done: deterministic Stage B reanalysis and metadata-only same-image
  feasibility audit.
- Blocked: None at action start.
- Most recent useful observation: v2 Stage B contains 3,200 complete
  sample-layer matrices and 12,800 finite action cells with exact FULL parity.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Existing branches satisfy the v3 dense-prefix/action/dense-suffix semantics. | `reports/v3_migration_audit.md` | Authorizes deterministic reuse without a new sweep. | confirmed |
| The old narrow READ endpoint failed both structured-null superiority gates. | `reports/stage_c_conclusion.md` | Any broader v3 endpoint must face a search-budget-matched structured null. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| v2 narrow confirmation | Primary likelihood replicated but specificity to actual READ removal did not. | supported | `reports/stage_c_conclusion.md` | Do not promote discovery gains without an all-action, all-layer matched null. | Reusing v2 Stage C as v3 confirmation. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Proceed to a bounded v3 preflight | Complete four-action heterogeneity survives fixed schedules and requires a prospectively matched null. | Whether a new held-out confirmation can be validly frozen. | medium | selected, not authorized for execution |
| Pivot to interaction analysis | Main effects are unstable while conditional sign changes are frequent. | A narrower non-harmfulness scientific claim. | medium | runner-up |
| Pivot to answer-silent redundancy or stop | Gains are heavy-tailed and many are practically small. | Prevents an unjustified new confirmation. | low | not selected |

## Next-Step Decision

- Deliberation mode: deep; the deterministic analysis is specified, but its
  interpretation selects among materially different scientific continuations.
- Active objective and bottleneck: Determine whether a prospectively frozen
  four-action statistic exceeds an equally searched structured null; this is
  unknown because the current evidence is inspected discovery only.
- Relevant memory item used: Outcome B shows that an observed suppression
  effect must beat a structured null with the same search budget.
- Confirmed observation: All 3,200 matrices are valid; heterogeneity is not
  explained entirely by exact ties or fixed global/layer/task schedules.
- Unverified interpretation: Whether the apparent oracle gains exceed
  best-of-many effects under a matched structured null.
- Diagnosis: unknown until a prospectively frozen preflight exercises the
  equally searched null.
- Viable alternatives considered: Proceed to preflight; pivot to interaction;
  pivot to redundancy; stop the causal direction.
- Chosen next action: If explicitly approved, run only the bounded v3 preflight
  specified in `reports/v3_stage_b_decision.md`.
- Strongest objection: The existing Stage B data lack a search-adjusted null,
  so the analysis can justify only a preflight, never confirmation.
- How this differs from failed attempts: It evaluates the full four-cell object
  and fixed schedules while preserving the old null failure as a warning.
- Automatic execution authorized: no; the authorized reanalysis is complete.
- Authorization basis: Preflight requires a new explicit user approval.
- Stop condition: Satisfied. Required artifacts and one Stage B decision are
  written; no preflight or new outcomes followed.

## Latest Research-Action Result

- Action taken: Reconstructed and validated 3,200 complete four-action matrices,
  computed all frozen v3 discovery summaries and fixed-policy comparisons, and
  audited same-image feasibility without collecting new outcomes.
- Result: Heterogeneity exceeds exact numerical ties and is poorly captured by
  fixed schedules, but is heavy-tailed and untested against a full-search-budget
  null. Decision: `PROCEED_TO_V3_PREFLIGHT` with medium confidence.
- Evidence saved: `outputs/v3_discovery/analysis_manifest.json`,
  `reports/v3_stage_b_reanalysis.md`, and `reports/v3_stage_b_decision.md`.
- Failure or issue: One first-run column-name bug was corrected with a regression
  test. A metadata integrity check then found 17 TextVQA records without
  official image IDs; frozen `selection_asset_key` preserves the 400-asset
  discovery grouping, but future image-disjointness needs hash resolution.
- Lesson learned: A fixed schedule recovers little of the inspected oracle gap,
  yet Outcome B makes a search-budget-matched structured null the decisive next
  gate rather than evidence that the oracle represents useful suppression.
- Next implication: Stop. If explicitly approved, execute only the bounded v3
  preflight described in the decision report; do not open confirmation data.
