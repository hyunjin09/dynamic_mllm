# Phase 67: Stage-2 MCTS Failure Diagnosis Memory

## Current Objective
Diagnose why adding frozen replay-valid MCTS supervision suppressed the small single-route Stage-2 rollout gain, without retraining, searching, changing Stage 1, or opening the held-out test.

## Active Constraints
- Follow `plans/stage2_mcts_failure_diagnosis_plan.md` as one bounded research action.
- Freeze Phase-66 A/B checkpoints, architecture, corpora, P98/P95/P90 triggers, four-action executor, model/runtime, and LMMS scoring.
- Use P90 as the primary controlled-rollout point; P98/P95 are secondary oracle/deviation strata.
- Existing Phase-65 shards contain only pooled diagnostic features, not the full token sequences consumed by the Stage-2 attention router. Exact oracle predictions therefore require deterministic route replay; pooled-state substitution is forbidden.
- Stop after diagnosis and one written recommendation. Do not execute the recommendation.

## Current State
- Done: Read the active plan, research-control skill, Phase-66 memory/result, promoted Phase-66 lesson, compute/environment policy, route/state schemas, and A/B rollout behavior.
- Done: Confirmed 2,519 exact union routes: 106 preservation, 1,688 single, and 725 MCTS; all MCTS union routes are P90-valid.
- Done: Froze authoritative contract `ffc2b02b...23bbcf`; the implementation smoke passed exact oracle token parity and finite A/B router outputs for preservation, single, and MCTS routes.
- Done: Four GPUs completed all 2,519 route replays, 68,506 A/B oracle-state predictions, and all 725 P90 MCTS controlled-prefix sweeps (3,863 rollout rows, 6,653 drift rows) with no missing rank or failed oracle control.
- Done: Aggregated all required metrics, case studies, six figures, UID bootstraps, hypothesis table, and one unexecuted next-experiment recommendation. All 50 artifact hashes verify.
- Blocked: No.
- Most recent useful observation: B's exact MCTS non-FULL recall is only 0.0845; it also reduces exact single corrective recall from A's 0.1078 to 0.0136. Exact-prefix ambiguity recovers 0.2163 of B's multi-valid rows, while prefix release reproduces at most 0.0835 of remaining C1-C3 corrective actions.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| A/P90 gives W-to-C/C-to-W/net 4/1/+3; B gives 0/0/0 at all points | `analysis/dense_failure_stage2/shared_union_training/` | Defines the failure to diagnose | confirmed |
| Both implementation and overfit gates passed | Phase-66 experiment smoke artifacts | Rules out a failed launch or wholly incapable small-cohort optimizer | confirmed |
| Phase-65 routes and routed pooled states passed exact replay/provenance | `analysis/dense_failure_stage2/robust_gate_corrective_search/` | Provides frozen successful trajectories and correctness controls | confirmed |
| Saved Phase-65 tensors are pooled summaries only | `states/feature_schema.json` and routed-state shard schema | Cannot query the token-attention Stage-2 router exactly; route replay is required | confirmed |
| A/B exact oracle-state and prefix-forcing diagnostics | `analysis/dense_failure_stage2/mcts_failure_diagnosis/` | Separates poor action recognition, transfer, ambiguity, and downstream drift | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Add MCTS supervision at a 1:1 corrective-W mixture | B emitted fewer non-FULL actions and lost A/P90's +3 net | unknown | Phase-66 A/B rollout and teacher-forced metrics | Diagnose oracle learning, exposure, transfer, ambiguity, and confidence before changing training | Do not retrain the same B recipe or add more MCTS labels |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Oracle action-learning weakness | B recognizes only 8.45% of oracle MCTS corrective actions | H1 versus exposure | complete | supported |
| Exposure/state drift | Drift appears one layer after a mistake, but oracle recognition is already weak and release rarely reproduces later actions | H2 | complete | secondary/not dominant |
| Negative transfer | B loses A's single corrective behavior with a strictly negative UID-bootstrap interval | H3 | complete | supported |
| Valid-action ambiguity | Multi-valid nominal errors greatly exceed single-valid errors and set membership recovers 21.63% of rows | H4 | complete | supported |
| Representation limitation | First and later MCTS recall are both weak without a large later-only gap | H5 after alternatives | complete | not isolated |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: distinguish action learning, exposure shift, negative transfer, label ambiguity, and representation limitation under the exact frozen A/B systems.
- Relevant memory item used: Phase 66 established a valid matched negative result but left its cause unknown.
- Confirmed observation: B underperforms A in free non-FULL activation despite an overfit-capable implementation and much richer oracle supervision.
- Unverified interpretation: exposure, conflicting labels, or negative transfer may suppress MCTS behavior; none is yet diagnosed.
- Diagnosis: unknown
- Viable alternatives considered: exact route replay; invalid pooled-state proxy; immediate retraining. Only exact replay preserves the diagnosed router and current authorization.
- Chosen action: execute the plan's frozen, exact-replay oracle/deviation/prefix/drift/transfer/ambiguity diagnostics on all union routes, with P90 MCTS prefix forcing and UID-level bootstrap.
- Strongest objection: replaying 2,519 routes plus controlled P90 prefixes duplicates model compute, but it is required because the saved pooled tensors cannot reproduce the router's attention inputs.
- How this differs from failed attempts: no parameters or labels change; the same checkpoints are queried on controlled oracle and free trajectories to isolate mechanism.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform `plans/stage2_mcts_failure_diagnosis_plan.md`.
- Stop condition: required artifacts, hypothesis assessment, and one unexecuted next-experiment recommendation; no retraining/search/test.

## Latest Research-Action Result
- Action taken: completed the frozen A/B oracle-state, first-deviation, controlled-prefix, state-drift, case-study, negative-transfer, ambiguity, and confidence diagnosis.
- Result: H1, H3, and H4 are jointly supported. B has only 0.0845 exact MCTS non-FULL recall and first disagrees at the first intervention on 88.83% of MCTS routes. B-minus-A single corrective recall is -0.0942 state-weighted and -0.2174 UID-weighted (95% CI [-0.2556, -0.1801]). B multi-valid nominal error is 30.96% versus 5.65% on single-valid states; accepting any observed successful action recovers 21.63% of multi-valid rows.
- Evidence saved: `analysis/dense_failure_stage2/mcts_failure_diagnosis/`; authoritative contract `ffc2b02b...23bbcf`; 50-file artifact manifest verified.
- Failure or issue: the first preliminary aggregation overinterpreted a one-UID C5 forcing cell. It is quarantined as `mcts_failure_diagnosis_preliminary_invalid_20260904/`; the authoritative rerun requires an unforced later corrective action for exposure comparisons. C1/C2/C3 matched UID rescue gains are +0.0325/+0.1814/+0.4340, but later non-FULL reproduction remains only 8.35%/6.75%/8.33%.
- Lesson learned: adding MCTS data did not primarily create a later-only exposure failure. It weakly learned oracle corrective actions, interfered with single corrective behavior, and encountered substantial exact-prefix valid-action conflict.
- Next implication: the one recommended separately authorized experiment is an unchanged-router, fixed-data observed-valid-set loss `-log(sum p(valid actions))`, evaluated for both oracle corrective recall and preservation of A-like single behavior. Do not execute automatically.
