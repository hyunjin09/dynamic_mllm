# Phase 06: v3 Confirmation Preflight Memory

## Current Objective

Freeze and technically validate an outcome-blind v3 held-out confirmation
protocol without executing or inspecting any held-out intervention outcome.

## Active Constraints

- Preserve the active v3 four-action estimand and all v2 artifacts.
- Use new image- and record-disjoint confirmation data; reserve separate
  image-disjoint multi-question pools for Stage C2.
- Match the real layer/action maximum exactly in every structured-null family.
- Do not freeze a final manifest until the prospective sample-size and
  construction rule are approved by this preflight.
- No model/probe/router training, held-out terminal scoring, or v2 Stage D.

## Current State

- Done: candidate-pool, power/precision, branch/null mechanics, and
  query-invariance preflight.
- In progress: None; the authorized action is complete.
- Blocked: exact joint-path donor calipers/index, null-draw seed stability,
  joint covariance smoke, and a sufficient visual-grounding control are not
  frozen within the no-large-sweep boundary.
- Most recent useful observation: equal right-padding restores exact
  same-image visual-state/WRITE equality, while unequal prompt lengths create
  shape-dependent BF16 divergence despite a causally valid mask.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| The complete nonterminal sparse grid is `[0,4,8,12,16,20,24]`; layer 27 is structurally WRITE-silent. | `reports/v3_stage_b_reanalysis.md` | A narrower raw-mean-selected grid would be post-selection; layer 27 adds only structural ties. | confirmed |
| Fixed per-layer schedules recover little of the sample-layer oracle gain. | `outputs/v3_discovery/fixed_policy_regret_v1.csv` | Justifies testing a sample-level max rather than a fixed schedule, subject to matched nulls. | confirmed |
| v2 Stage C Outcome B failed both structured-null superiority gates. | `reports/stage_c_conclusion.md` | The v3 nulls must receive the same layer/action search budget as the real statistic. | confirmed |
| Disjoint 800-image previews and separate 800-image-group Stage C2 reserves exist for both datasets. | `outputs/v3_preflight/candidate_pool_audit.json`; `outputs/v3_preflight/stage_c2_reserved_pool_audit.json` | Data availability is not the current blocker. | confirmed |
| Every real/null smoke branch was finite and FULL parity was exact. | `outputs/v3_preflight/null_preflight_manifest.json` | The action analogues are technically executable. | confirmed |
| Unequal lengths break bitwise WRITE equality; equal padding restores it. | `outputs/v3_preflight/query_invariance_equal_length_diagnostic.json` | C2 needs a prospective fixed-shape execution rule. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| v2 single-contrast specificity test | A held-out reference-support effect replicated but did not outperform either residual null. | supported | `reports/stage_c_conclusion.md` | Replication of a likelihood shift is insufficient; require paired superiority to all prospectively frozen nulls. | Reuse a single-endpoint v2 null for the broader v3 maximum. |
| Unpadded same-image query check | Visual WRITE diverged numerically from layer 0 onward for lengths 281 vs 273. | supported: shape-dependent finite-precision execution | `outputs/v3_preflight/null_preflight_manifest.json`; equal-length diagnostic | Structural causality does not imply bitwise invariance across different execution shapes; freeze common shape before C2. | Claiming exact runtime invariance from the mask alone. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Freeze the complete nonterminal grid | It is the full validated grid after removing the structurally degenerate terminal layer. | Avoids discovery-mean layer selection while preserving the intended search. | high | selected |
| Calibration-only joint-path repair | Fits joint covariance, draw stability, exact calipers, and donor coverage without held-out terminal scores. | Closes the remaining Stage C entry gate. | high | requires approval |
| Fixed-shape C2 amendment | Equal padding restored exact visual-state equality. | Preserves the structural query-invariance premise numerically. | low | requires approval |

## Next-Step Decision

- Deliberation mode: deep.
- Active objective and bottleneck: Close joint-path null geometry and
  fixed-shape query-invariance rules before any held-out terminal outcome.
- Relevant memory item used: Outcome B requires specificity against structured
  intervention nulls, not merely a positive held-out maximum.
- Confirmed observation: Discovery supports heterogeneity but has a near-zero
  median at many cells and a heavy-tailed sample-level maximum.
- Unverified interpretation: Whether one null draw per sample is seed-stable
  for the full 21-cell maximum and what exact donor caliper the paired geometry
  supports.
- Diagnosis: supported for the numerical query-invariance failure; unknown for
  null seed stability and full donor coverage.
- Viable alternatives considered: 1,600 images/one provisional draw; 800
  images/multiple draws; narrower grid; stop.
- Chosen action: Stop at `REPAIR_V3_NULL_DESIGN` and request approval for one
  calibration-only geometry action; do not freeze the final manifest or open
  held-out terminal scores.
- Strongest objection: The repair requires a 400-record, seven-layer activation
  extraction and exceeds the current no-large-sweep boundary.
- How this differs from failed attempts: Every null receives the exact same
  layer/action maximum as the real statistic.
- Automatic execution authorized: yes, for this bounded preflight only.
- Authorization basis: Explicit user task to freeze and validate the v3
  confirmatory preflight.
- Stop condition: Satisfied because a required C2 execution rule and core null
  hyperparameters remain unfrozen.

## Latest Research-Action Result

- Action taken: Completed outcome-blind pool, power, null-action, and
  query-invariance preflight and reconciled an independent design review.
- Result: Data and branch mechanics pass, but the confirmation entry gate does
  not; decision `REPAIR_V3_NULL_DESIGN`.
- Evidence saved: `reports/v3_preflight_report.md` and all required
  `outputs/v3_preflight/` and `workspace/v3_*` artifacts.
- Failure or issue: Joint-path caliper/draw count and the grounding gate are
  not frozen; unpadded numerical query invariance failed.
- Lesson learned: Exact causal masking is insufficient for bitwise invariance
  when BF16 eager kernels run different total sequence shapes.
- Next implication: Stop. Do not run confirmation without a separately approved
  calibration-only repair and Stage C2 padding amendment.
