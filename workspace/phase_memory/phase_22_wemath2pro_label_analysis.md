# Phase 22: WeMath2.0-Pro Label Training-Suitability Analysis

## Current Objective

Verify terminal completeness of the 4,544-record hard-cap-400 WeMath2.0-Pro
MCTS cache, then determine whether its unrestricted 28-bit route labels are
suitable supervision for later binary-router training.

## Active Constraints

- Use only `outputs/label_regeneration/wemath2pro_cap400_v2/` and the frozen
  WeMath manifest/contract; preserve the eight technical-invalid inventory
  records separately.
- No predictor training, Qwen inference, label regeneration, scorer change,
  route re-execution, or extension beyond 400 simulations.
- Retain exact MathRuler validity (`score >= 1.0`) and report every scoring
  timeout rather than silently treating the cache as clean.
- Derive a diagnostic deterministic max-50 supervision view using the existing
  selector and actual duplicated-BCE weighting; do not mutate the raw cache.
- Run the full checksum/geometry pass as a CPU-only Slurm job on node05; never
  use node04.

## Current State

- Done: all seven shard summaries exist under contract
  `80c7ea4ca2ca9df091696290dc644a4092508337f89cf85ecc5b849a0f4092c7`.
- Done: coarse reconciliation finds exactly 4,544 manifest rows and 4,544
  sample JSON files, zero temporary/error paths, and zero shard errors.
- Done: strict per-record audit and training-suitability analysis passed for
  all 4,544 eligible records.
- Blocked: none.
- Most recent useful observation: only 2,266/4,544 samples have a positive
  route; the exact weighted duplicated-BCE label oracle has 13.72% selected-
  valid Hit@1, while the diagnostic Pareto oracle has 84.64% Hit@1.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Seven terminal summaries cover every selected sample with zero errors | `outputs/label_regeneration/wemath2pro_cap400_v2/raw_route_cache/shard_*/summary.json` | Strong initial evidence the run completed | confirmed |
| Manifest/sample-file counts are both 4,544 | frozen manifest and raw cache | Permits strict UID-level completeness audit | confirmed |
| Exact route records retain score, validity, timeout, geometry, and current FULL result | raw record schema | Supports label-only suitability analysis | confirmed |
| Eight source records were prospectively technical-invalid | `outputs/label_regeneration/wemath2pro_v1/preflight/data_validity_failure_v1.json` | Defines why analysis population is 4,544 rather than 4,552 | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Earlier unbounded WeMath scoring | Two ranks stopped publishing | unknown; timeout repair bounded one recurrence class | phase 13 logs and timeout amendment | Audit timeout counts explicitly | Do not infer the old stall was definitively MathRuler |
| 600-simulation extension | Only 25/528 snapshots found post-400 corrections | supported low extension yield | `extension_yield_snapshot_v1.json` | Analyze only rerun 200/400 records | Do not mix superseded 600-simulation labels |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Full matched geometry/BCE-oracle analysis | Directly comparable to prior 8K diagnosis | Whether labels are usable and under which objective | medium | selected |
| Route-count/coverage summary only | Cheap | Basic quality only, not objective suitability | low | rejected as insufficient |
| Start training immediately | Cache appears complete | Predictor result | high | rejected; supervision suitability is unresolved |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: completed; the remaining boundary is whether
  the user later authorizes a frozen WeMath split/training comparison.
- Relevant memory item used: the previous 8K analysis showed that high route
  count/diversity can still induce an invalid duplicated-BCE oracle.
- Confirmed observation: strict cache integrity passes 4,544/4,544; 2,266 have
  positives, and unfiltered duplicated BCE has only 13.72% label-oracle Hit@1.
- Unverified interpretation: whether a factorized exact-set predictor can
  generalize to useful executed masks on image-group-disjoint WeMath data.
- Diagnosis: the completed labels are objective-dependent—coherent for grouped
  complete-mask likelihood, incoherent as unfiltered bitwise-marginal targets.
- Viable alternatives considered: declare all labels unusable; use duplicated
  BCE; or retain exact-set/ranking uses while preserving zero-positive rows.
- Chosen action: close this analysis with conditional exact-set/ranking
  suitability and reject unfiltered duplicated BCE for complete-mask training.
- Strongest objection: MCTS valid sets are incomplete, so cached Hit@1 is not a
  substitute for held-out executed-mask behavior.
- How this differs from failed attempts: the conclusion uses terminal cap-400
  cache evidence and separates label coherence from predictor generalization.
- Automatic execution authorized: no further action; user approval is required
  before freezing a WeMath training split or training any predictor.
- Authorization basis: the requested cache completion and suitability analysis
  is complete.
- Stop condition: reached—analysis, report, checksums, state, and verification
  are complete.

## Latest Research-Action Result

- Action taken: strict cache audit plus matched route-geometry, BCE-oracle,
  Pareto, timeout, and difficulty analysis on node05 (Slurm `101423`).
- Result: PASS for all 4,544 eligible records. Exact-set/ranking supervision is
  conditionally usable; unfiltered duplicated BCE remains label-incoherent.
- Evidence saved: `outputs/wemath2pro_mcts_label_analysis_v1/` and
  `reports/wemath2pro_mcts_training_suitability.md`.
- Failure or issue: the first analyzer version incorrectly required exactly
  `simulations + 2` unique masks. One focused diagnostic showed legitimate
  evaluation-cache reuse; the invariant was repaired to exact trace linkage,
  regression-tested, and the full rerun passed.
- Lesson learned: completed simulation count and unique evaluated-mask count
  differ when an MCTS rollout revisits an anchor or cached mask; audit their
  exact set linkage instead of assuming one new mask per simulation.
- Next implication: no training is authorized. If WeMath supervision is later
  approved, freeze an image-group-disjoint split and matched max-route view,
  using exact-set NLL as the coherent primary objective.
