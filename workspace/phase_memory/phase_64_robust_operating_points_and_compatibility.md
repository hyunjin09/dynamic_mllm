# Phase 64: Robust Operating Points and Stage-2 Label Compatibility Memory

## Current Objective
Characterize conservative operating points for the frozen Phase-63 ALL-source
Stage-1 system, build robust train trigger maps, and measure exact compatibility
of existing Historical and Canonical corrective labels.

## Active Constraints
- Follow `plans/robust_stage1_operating_points_trigger_label_compatibility_plan.md` as one bounded action.
- Preserve the five Phase-62 ALL-source checkpoints, old normalization, 28-layer score definition, and strict `score > tau` rule frozen by Phase 63.
- Treat the five-checkpoint per-layer probability mean as the one frozen deployment scoring system; do not invent or train a singleton full-data checkpoint.
- Use only Phase-63 held-out/cross-fit scores for operating-point selection and stored dense features for train-map scoring.
- Retain at most three downstream operating points and do not use Stage-2 outcomes to choose a final threshold.
- Replay only structurally compatible existing routes under the unchanged current Qwen/LMMS contract.
- Do not run new single/MCTS search, train Stage 2, retune Stage 1, or run external evaluation.

## Current State
- Done: Read the plan and Phase-54/55/56/59/63 state.
- Done: Confirmed the frozen Phase-63 gate binds five ALL-source fold checkpoints rather than a separately trained full-data singleton.
- Done: Frozen protocol `ab254e98...546674`, reconstructed exact five-head-mean scores for all 10,399 train rows, and reproduced every one of the 12,000 fold-held-out score checks with maximum absolute error 0.0.
- Done: Retained P98/P95/P90 under the prospective stability/distinctness/recall-spacing rule and created complete Historical/Canonical trigger maps.
- Done: Exact-replayed all 2,607 unique structurally compatible route/handoff jobs (2,618 route×operating-point rows); every replay remained LMMS-correct with exact stored-token parity.
- Done: Completed transition, preservation, route-reuse, lower-bound, workload, figure, summary, and 31-artifact integrity audits.
- Blocked: none.
- Most recent useful observation: P98/P95/P90 trigger 344/727/1,307 train Dense-W and retain exact-replay-compatible correction routes for 67/146/259 of them; the remaining new-search workloads are 277/581/1,048.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-63 P98 reference is strict `tau=0.9711347410314399` and gate hash `d3b019b3...cae4f` | `analysis/dense_failure_stage1/all_source_threshold_calibration/frozen/robust_stage1_gate.json` | Freezes the head and conservative reference | confirmed |
| Historical train has 6,399 stored feature rows; Canonical train has 4,000 stored feature rows | Phase-48 and Phase-59 feature manifests used by Phase 62 | Enables trigger-map scoring without Qwen | confirmed |
| Historical labels contain 698 single-fixable, 209 MCTS-only, and 974 unresolved W | `analysis/dense_failure_stage2/full_corrective_labels/` | Defines prior Historical correction coverage | confirmed |
| Canonical labels contain 75 single-fixable, 33 MCTS-only, and 149 unresolved W | `analysis/dense_failure_stage2/data_scale_search/` | Defines prior Canonical correction coverage | confirmed |
| Held-out P98/P95/P90 trade-offs are C preservation 0.9808/0.9501/0.9003 worst-source and pooled W recall 0.1054/0.2179/0.3572 | `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/thresholds/named_operating_points.csv` | Defines the retained conservative/middle/permissive frontier | confirmed |
| All 2,618 expanded compatible route replays are correct and exact-token-parity | `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/compatibility/replay_results.jsonl` | Validates existing Stage-2 label reuse under new handoffs | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Apply the Historical-only gate to Canonical data | 54.04% of Canonical Dense-C triggered | supported source-boundary failure | Phase-59/60 diagnostics | Recompute all train trajectories with the frozen ALL-source system | Do not threshold stale old-head scores |
| First Phase-64 score launch | Four workers stopped before scoring because `CUBLAS_WORKSPACE_CONFIG` was absent | confirmed launch-environment omission | fail-closed worker traces; no score artifacts written | Relaunch unchanged jobs with required `:4096:8` setting | Do not launch the frozen scorer without its deterministic environment |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| P98 | Frozen conservative reference | High-preservation treatment opportunity | low | retained; not final |
| P95 | Candidate middle point | Moderate admission/coverage trade-off | low | retained; not final |
| P90 | Candidate permissive point | Larger treatment opportunity | low | retained; not final |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: completed; the remaining scientific question is which treatment-dependent operating point wins after missing corrective labels are generated and Stage-2 behavior is measured.
- Relevant memory item used: Phase 63 froze a source-robust conservative gate, while Phase 54/59 trigger and label artifacts were generated with the source-sensitive old head.
- Confirmed observation: all required dense features and successful full-route action vectors already exist; only train scoring and compatibility replay are required.
- Confirmed interpretation: P98/P95/P90 are distinct, fold-stable conservative/middle/permissive candidates under the frozen prospective rule.
- Diagnosis: supported old/new handoff-contract mismatch; exact route compatibility is now measured rather than assumed.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage1/all_source_threshold_calibration/`, `analysis/dense_failure_stage1/trigger_map/`, and `analysis/dense_failure_stage2/data_scale_search/`.
- Viable alternatives considered: P98/P95/P90 when stable and materially separated; substitute P97 if the middle/permissive spacing or stability gate fails.
- Chosen action: stop at the completed compatibility audit. A future separately authorized plan may reuse verified routes and search only the missing triggered-W union.
- Strongest objection: in-sample five-head ensemble train admission rates remain descriptive and cannot replace the held-out frontier when selecting a final threshold.
- How this differs from failed attempts: no old-head score is reused for new admission and no structural route check is promoted without exact Qwen/LMMS replay.
- Automatic execution authorized: yes
- Authorization basis: the user explicitly requested this plan.
- Stop condition: met; no new search or training executed.

## Latest Research-Action Result
- Action taken: built robust five-head-mean train trigger maps at P98/P95/P90 and exact-replayed every structurally compatible prior single/MCTS route from its new handoff.
- Result: P98/P95/P90 trigger 344/727/1,307 W and 7/30/106 C. Existing routes cover 67/146/259 W, yielding descriptive known-repairable W-recall lower bounds 0.0165/0.0359/0.0636 and residual search workloads 277/581/1,048. All 2,618 expanded replays are correct with exact token parity.
- Evidence saved: `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/` with contract `ab254e98...546674` and 31 verified final artifacts.
- Failure or issue: none in accepted outputs; the initial scorer launch omitted the required deterministic cuBLAS environment and failed before writing rows.
- Lesson learned: route validity across a later/earlier Stage-1 handoff cannot be inferred from the old label alone, but every route that passed the fixed structural pre-filter also passed exact current-runtime replay in this population.
- Next implication: preserve all three operating points for future treatment-dependent evaluation; if authorized, search only the 1,104-UID union of triggered-W rows lacking reusable routes, with operating-point-specific handoff tags.
