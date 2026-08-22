# Phase 27: WeMath Visual-Compute Difficulty Memory

## Current Objective

Use the completed hard-cap-400 WeMath2.0-Pro MCTS cache to test whether the
minimum discovered correctness-preserving visual decoder depth increases with
the dataset's three-axis difficulty metadata.

## Active Constraints

- Read-only analysis of the 4,544 prospectively eligible frozen records.
- Use raw evaluated routes and the frozen MathRuler threshold; do not use the
  max-50 predictor-training view for the primary endpoint.
- Treat FULL-correct samples as the primary cohort and FULL-wrong samples as a
  separate correction-discovery analysis.
- Preserve all eight difficulty strata; only `base`, one-axis, two-axis, and
  three-axis degree is ordinal.
- No MCTS, route execution, Qwen inference, training, Pareto filtering, route
  cap, repeat action, threshold change, or cache modification.

## Current State

- Done: source cache completion/integrity was previously established for all
  4,544 records and 1,658,485 evaluated routes.
- Done: the frozen-cache difficulty analysis authorized by
  `plans/motivation.md` passed and is classified as Outcome E.
- Blocked: none.
- Most recent useful observation: minimum discovered ON decreases with coarse
  difficulty among FULL-correct survivors (Spearman -0.225, clustered 95% CI
  [-0.291, -0.159]), driven mainly by x-containing strata, while FULL-wrong
  correction discovery falls from 50.0% at degree 0 to 26.7% at degree 3.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Cache has 4,544 records, 841 FULL-correct, 3,703 FULL-wrong, and no errors/timeouts | `outputs/wemath2pro_mcts_label_analysis_v1/completion_audit_v1.json` | Establishes the analysis population and execution validity | confirmed |
| Raw-derived per-sample index is checksum-bound to the frozen cache analysis | `outputs/wemath2pro_mcts_label_analysis_v1/analysis_manifest.json` | Avoids reparsing a derived max-50 view while preserving raw-route metrics | confirmed |
| Difficulty has eight labels across three independent axes | frozen manifest and official WeMath2.0-Pro metadata | Prevents an invalid total order over x/y/z | confirmed |
| Positive-route coverage varies materially by difficulty | `reports/wemath2pro_mcts_training_suitability.md` | Requires explicit survivorship and correction-discovery analyses | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat zero-positive as insufficient visual depth | Finite MCTS and the binary SKIP/KEEP action space cannot establish that interpretation | supported claim-boundary issue | `plans/motivation.md` | Report search failure separately from depth among discovered positives | Do not claim a need for >28 visual layers |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Frozen raw-route difficulty analysis | Directly measures the approved diagnostic without new inference | Whether minimum discovered ON depth scales with difficulty | low | selected |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: estimate difficulty-versus-minimum-ON
  relationships without conflating route-search coverage with visual-depth
  requirement.
- Relevant memory item used: the cache is complete, but only 2,266/4,544
  records have a discovered valid route and coverage is difficulty-dependent.
- Confirmed observation: all required primary fields are available from a
  checksum-bound raw-route-derived index.
- Unverified interpretation: higher difficulty requires more standard visual
  decoder depth.
- Diagnosis: unknown; this is the hypothesis being tested.
- Viable alternatives considered: none; the user specified a single read-only
  analysis action.
- Chosen action: compute the complete stratified, feasibility, paired-family,
  visual-token-controlled, route-geometry, and uncertainty analysis in
  `plans/motivation.md`.
- Strongest objection: minimum discovered ON is search-dependent and the
  FULL-correct cohort may be an increasingly selected subset at high
  difficulty.
- How this differs from failed attempts: it does not add search budget or pool
  FULL-wrong correction routes with correctness-preserving FULL-correct routes.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request to read and perform
  `plans/motivation.md`.
- Stop condition: stop if source checksums/counts fail, raw-route metrics cannot
  be reconstructed, grouping metadata is not defensible, or a new experiment
  would be required.

## Latest Research-Action Result

- Action taken: completed the integrity-bound raw-route difficulty,
  budget-feasibility, correction-coverage, paired-family, visual-token, and
  route-geometry analyses without new inference or search.
- Result: Outcome E — axis-specific effect. The planned monotonic
  difficulty-to-higher-visual-depth hypothesis is unsupported; x-containing
  FULL-correct strata have much lower minimum ON, y/z-only strata remain near
  base, and higher-difficulty correction coverage collapses.
- Evidence saved:
  `outputs/wemath2pro_visual_compute_difficulty_v1/`,
  `reports/wemath2pro_visual_compute_difficulty_v1.md`, and
  `runs/wemath2pro_visual_compute_difficulty_final_v1.log`.
- Failure or issue: none.
- Lesson learned: WeMath difficulty degree conflates distinct axes and cannot
  be used as a simple monotonic proxy for required visual decoder depth. A
  zero-positive sample remains search/action-space failure evidence, not proof
  of needing more than FULL computation.
- Next implication: stop. The result alone does not authorize or motivate a
  REPEAT search; any follow-up requires a separately approved hypothesis.
