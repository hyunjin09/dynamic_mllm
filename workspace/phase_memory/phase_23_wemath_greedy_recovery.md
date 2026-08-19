# Phase 23: WeMath Greedy Recovery Memory

## Current Objective

Execute conditional greedy Phase-1/Phase-2 label recovery for the exact 2,278
WeMath2.0-Pro samples with no valid cap-400 MCTS route.

## Active Constraints

- Preserve the completed WeMath MCTS cache and all v2/v3 label artifacts.
- Recovery population is conditional Group D only: current FULL wrong and zero
  valid routes in the completed cache.
- Preserve snapshot, Transformers 5.3.0 project environment, verified binary
  executor, native image processing, 96-token greedy generation, MathRuler
  threshold 1.0, and scorer timeout.
- Do not run the old four-benchmark package unchanged, edit its frozen core, or
  start GPU search without explicit approval.
- Do not start predictor training.
- Preserve every evaluated route in raw evidence and cap only the derived
  valid-route training view at 50 masks per sample.
- Use four deterministic shards: two GPU workers on node06 and two on node07.

## Current State

- Done: reproduction README, configuration, Phase-1/Phase-2 collectors,
  aggregator, launcher, runtime record, and checksums inspected.
- Done: all frozen package checksums pass.
- Done: current Group-D population confirmed as 2,278 records across 1,104
  image groups.
- Done: G0 manifest PASS (2,278 UIDs, 1,104 image groups, all linked cache
  checksums); G1 deterministic adapter tests PASS; G2 execution parity PASS.
- In progress: G3 Phase 1, Slurm 101708 on node06 (shards 0--1) and 101709 on
  node07 (shards 2--3), one process per GPU.
- Interim at 2026-08-19 14:36 KST: 1,028/2,278 records complete with zero
  errors. Phase 1 has recovered 89 records and found 694 unique valid masks;
  this is 0.675 valid masks per completed record or 7.80 per recovered record.
- Done: the active recovery plan is frozen in
  `plans/dynamic_mllm_wemath2pro_greedy_recovery_plan.md`.
- Pending gate: all 2,278 Phase-1 records and 22,780 finals must reconcile
  before the global Phase-2 request manifest is frozen.
- Most recent useful observation: valid routes are being found for 8.66% of
  completed recovery records so far, but 939/1,028 completed records still have
  zero valid route; final yield remains incomplete.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase 1 uses ten 28-step greedy removal orders and retains accepted/rejected candidates | `search/greedy_phase1_phase2_reproduction/scripts/core/collect_phase1_candidates.py` | Defines complementary search geometry | confirmed |
| Phase 2 uses random budgets, local neighbors, and successful-base recombinations | `search/greedy_phase1_phase2_reproduction/scripts/core/collect_phase2_candidates.py` | Defines expansion and its no-base limitation | confirmed |
| Frozen package expects old 10K counts, modules, runtime, and image policy | README, preflight, config, runtime versions | Prevents direct execution for WeMath | confirmed |
| WeMath Group D contains 2,278 records / 1,104 image groups | `outputs/wemath2pro_mcts_label_analysis_v1/per_sample_training_suitability_v1.jsonl` | Freezes intended recovery scope | confirmed |
| Prior MCTS extension beyond 400 had low yield | `outputs/label_regeneration/wemath2pro_v1/extension_yield_snapshot_v1.json` | Supports a different search geometry rather than more MCTS | confirmed |
| Interim Phase-1 recovery is 89/1,028 with 694 valid masks and zero errors | `outputs/label_regeneration/wemath2pro_greedy_recovery_v1/phase1/` | Confirms the search is productive while preventing premature final prevalence claims | confirmed interim |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Continue MCTS beyond 400 | Only 25/528 monitored samples first found a correction after 400 | supported low marginal yield in monitored snapshot | extension-yield snapshot | Keep MCTS capped and test a different route geometry if approved | Do not re-enable 600 simulations |
| Treat old portable package as drop-in current code | Required modules/runtime/manifest contract differ | supported incompatibility | package README/preflight and current runtime | Port algorithm semantics onto current executor | Do not disable old preflight and call it equivalent |
| First G2 deterministic launch | CuBLAS stopped before any scientific output because `CUBLAS_WORKSPACE_CONFIG` was absent | supported launch-contract omission | `runs/wemath_greedy_recovery/preflight.log` | Freeze `:4096:8` before CUDA use and rerun unchanged fixtures | Do not relax determinism or tolerances |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Current-runtime Phase 1 only | Nested removals complement graph MCTS | Basic recovery yield | high | viable first execution stage |
| Current-runtime Phase 1+2 with Phase-1 gate | Faithfully covers the documented algorithm | Full complementary recovery yield | high | recommended protocol |
| Frozen package unchanged | Literal reproduction | Old experiment only | high | rejected as contract-invalid |
| More MCTS | Existing implementation available | Marginal same-family yield | high | rejected by prior low extension yield |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: preserve the algorithm while replacing only
  the incompatible execution/data interfaces.
- Relevant memory item used: executor-contract mismatch previously invalidated
  old routing labels, so route-search semantics cannot override runtime parity.
- Confirmed observation: the old core package is intact but not directly
  compatible; the target recovery set is exactly 2,278 records.
- Unverified interpretation: greedy nested/local search will recover a
  practically useful fraction of MCTS-zero-positive samples.
- Diagnosis: supported implementation-contract mismatch, not a search-method
  failure.
- Evidence path if diagnosis is not unknown:
  `reports/wemath2pro_greedy_recovery_package_audit.md`.
- Viable alternatives considered: unchanged package, Phase 1 only, faithful
  gated Phase 1+2, and further MCTS.
- Chosen action: execute G0--G5 under the frozen current-runtime contract. Run
  the five-record G2 gate, then Phase 1 on four global shards split between
  node06 and node07; freeze the global Phase-2 expansion before executing it.
- Strongest objection: binary zero-score acceptance may drive traces toward
  ALL-OFF, while Phase 2 supplies only six random masks without a successful
  Phase-1 base.
- How this differs from failed attempts: it changes search geometry while
  preserving the verified current executor and 400-cap MCTS evidence.
- Automatic execution authorized: yes, for this bounded search only.
- Authorization basis: user explicitly requested the search, max 50 derived
  valid routes per pair, and all four specified GPUs in two separate jobs.
- Stop condition: stop on a parity failure, unresolved execution error,
  incomplete Phase-1 population, or undefined global Phase-2 budget center.
- Interim decision: continue the already authorized Phase 1 unchanged. The
  strongest objection is that recovery is only 8.66% so far, but stopping now
  would bias the conditional yield and prevent the frozen Phase-2 expansion.

## Latest Research-Action Result

- Action taken: froze G0, implemented and tested the current-runtime adapter,
  passed the repaired unchanged G2 gate, and launched four Phase-1 workers.
- Result: G0--G2 PASS; G3 runs on node06/node07 under Slurm 101708/101709.
- Evidence saved: `reports/wemath2pro_greedy_recovery_launch.md`, the frozen
  manifest/contract, and the preflight report.
- Failure or issue: the first gate launch omitted the deterministic CuBLAS
  workspace setting and stopped before results; the root launch contract was
  repaired and the unchanged gate passed.
- Lesson learned: deterministic CUDA launches must freeze the CuBLAS workspace
  setting; no executor-contract mismatch remained in the five-record gate.
- Next implication: monitor G3, reconcile all shards, freeze the single global
  Phase-2 request manifest, then submit Phase 2 on the same four-way layout.
