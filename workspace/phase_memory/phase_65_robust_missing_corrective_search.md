# Phase 65: Robust Missing Corrective Search Memory

## Current Objective
Rebuild threshold-specific Stage-2 corrective supervision for the frozen P98,
P95, and P90 robust Stage-1 gates by sharing route discovery across thresholds
and exact-replaying every retained route under each applicable handoff.

## Active Constraints
- Follow `plans/robust_stage1_missing_corrective_search_plan.md` as one bounded research action.
- Freeze the Phase-64 head, thresholds, trigger maps, existing route replay results, model/runtime, LMMS evaluator, four-action executor, and MCTS@200 semantics.
- Search exactly the frozen 1,104-UID union; do not use Stage-2 outcomes to alter thresholds or populations.
- Run exhaustive singles once from each UID's earliest needed trigger, then launch MCTS only for remaining unresolved pair-level roots.
- Retain all successful singles and up to eight distinct MCTS successes per launched root; require exact replay validity before corpus admission.
- Build independent P98/P95/P90 preservation, single, MCTS, and unresolved views while storing routes once globally.
- Use all four local GPUs after a passing smoke. Do not train Stage 2, change Stage 1, run test deployment, or run external evaluation.

## Current State
- Done: Read the plan, Phase-64 memory, promoted decision, compute policy, and prior validated search implementation.
- Done: Confirmed 1,906 missing pairs and 1,104 unique UIDs with no trigger-order violations.
- Done: Focused implementation/integration suite passes 47/47.
- Done: Frozen execution contract `767284f1e8ed76d5d1c797562a3aaea4ea59183703d62b1521545cacc7c57ab4`.
- Done: Four-GPU 12-UID smoke passed with 23 retained routes and 191 routed-state rows.
- Done: Four-GPU full search completed all 1,413 work UIDs, including all 1,104 search UIDs, with zero failed or duplicate records.
- Done: Aggregation and the 611-file artifact/hash audit passed; all required threshold corpora are frozen.
- Blocked: none.
- Most recent useful observation: new search resolved 333/1,906 missing threshold pairs (55 single, 278 MCTS), covering at least one missing point for 235/1,104 UIDs; 1,573 pairs remain unresolved at budget.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| P98/P95/P90 require search for 277/581/1,048 W, totaling 1,104 unique UIDs | `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/compatibility/per_sample_compatibility.jsonl` | Freezes the Phase-65 population | confirmed |
| All 2,618 previously reusable route×threshold replays are correct with exact token parity | `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/compatibility/replay_results.jsonl` | Existing supervision can be preserved | confirmed |
| All missing-population trigger layers obey P90 <= P95 <= P98 | Phase-65 preflight audit over frozen trigger maps | Validates earliest-root sharing | confirmed |
| Prior cap-200 search produced 306,564 terminal executions in 28.26 GPU-hours | `analysis/dense_failure_stage2/full_corrective_labels/metrics/compute_summary.csv` | Bounds expected cost and motivates four-GPU sharing | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Apply old Historical-only gate to Canonical rows | Excess Canonical C admission and incompatible handoffs | supported source-boundary failure | Phase 59-64 evidence | Search only the new robust-gate pair population | Do not reuse old trigger membership as admission |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Shared earliest-root single plus conditional shared MCTS | Exact authorized plan; reduces duplicate executions | Complete robust-gate corrective corpora | high | completed |
| Three independent per-threshold searches | Mechanically simple | Same labels without sharing | very high | rejected by plan and avoidable cost |
| Single-only execution then stop | Lower immediate cost | Partial labels only | medium | rejected because it leaves the authorized MCTS phase incomplete |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: interpret the completed corpus reconstruction without prematurely selecting a threshold or launching Stage-2 training.
- Confirmed observation: P98/P95/P90 known-corrective coverage is 108/344, 234/727, and 463/1,307; the new search adds 41/88/204 pair outcomes, predominantly through MCTS.
- Unverified interpretation: the lower late-trigger and GQA coverage may limit learned treatment, but corpus-label availability alone does not establish rollout performance.
- Diagnosis: supported task/depth-dependent bounded correctability; evidence is in `metrics/dataset_source_breakdown.csv` and `metrics/trigger_depth_breakdown.csv`.
- Viable alternatives considered: select a threshold from label yield, immediately train all three, or first design a matched threshold-specific training comparison. Label yield cannot select deployment, and training is not authorized in this phase.
- Chosen action and strongest objection: close Phase 65 with the verified corpora and recommend a separately authorized matched training design; P98 has only 7 preservation and 65 single-corpus bases, so a naive equal-recipe comparison could conflate operating point with support size.
- How this differs from failed attempts: all routes and state slices are current robust-trigger exact replays, not stale trigger labels.
- Authorization and stop condition: Phase-65 search was authorized and is complete; stop before Stage-2 training or threshold selection.

## Latest Research-Action Result
- Action taken: completed shared exhaustive-single plus conditional MCTS@200 search, exact replay/state capture, threshold corpus reconstruction, and final artifact verification.
- Result: 1,104/1,104 search UIDs and 1,413/1,413 total work UIDs completed. Pair outcomes are 423 existing-single, 49 existing-MCTS, 55 new-single, 278 new-MCTS, and 1,573 unresolved-at-budget. P98/P95/P90 corrective coverage is 0.3140/0.3219/0.3542.
- Evidence saved: `analysis/dense_failure_stage2/robust_gate_corrective_search/artifact_manifest.json`, `run_summary.json`, and `summaries/robust_gate_corrective_search_summary.md`.
- Failure or issue: the first smoke contract failed closed on two record-adapter defects; its full evidence was preserved at `analysis/dense_failure_stage2/robust_gate_corrective_search_failed_smoke_16769938/`. Both defects have regression tests and the failed contract is not reused.
- Lesson learned: route records from the terminal cache and compact existing-route entries require explicit schema adapters before replay/state capture.
- Next implication: any next phase should define a matched P98/P95/P90 Stage-2 training comparison and account explicitly for corpus-size imbalance; no threshold is selected from these search results alone.
