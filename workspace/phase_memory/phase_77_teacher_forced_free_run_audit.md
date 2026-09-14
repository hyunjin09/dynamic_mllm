# Phase 77: Teacher-Forced vs Free-Run Divergence Audit Memory

## Current Objective
Determine whether the frozen Phase-76 router primarily fails because it does not choose sparse corrective actions even on exact successful-route states, because free rollout leaves known successful-route support, or because the state/action mapping fails to generalize.

## Active Constraints
- Freeze the Phase-76 full-refit and internal-dev checkpoints, 569-UID/4,948-route corpus, 35,565-state cache, Qwen snapshot, P90 Stage-1 gate, four actions, executor, and LMMS training-side evaluator.
- Audit every training UID and route-state occurrence; use the original image-group-disjoint dev split and checkpoint for held-out analysis.
- Do not retrain, change objectives/thresholds/representations, search for routes, collect labels, or use the 19,960 external labels.
- Freeze one highest-responsibility route per Dense-W UID before R0/R1/R2 execution.
- Treat off-support only as leaving the observed successful-route corpus, not as proof of invalidity.

## Current State
- Done: completed the full seen/held-out teacher-forced audit, all training-UID R0/R1/R2 and Dense-C rollouts, support tracking, fixed decision rule, summaries, figures, and global artifact verification under contract `3771fd97...a69be0f`.
- In progress: none; stopped at the authorized research-action boundary.
- Blocked: none.
- Most recent useful observation: seen first-nonFULL recall is 2.54%; exact-state R1 changes rescue only 9.07% to 9.29%, while forcing that first non-FULL action in R2 raises rescue to 57.02%. The earliest dominant failure is corrective-action learning, not pre-intervention exposure.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-76 full replay cache covers 569 UIDs, 4,948 routes, 35,565 states, and 69,178 occurrences with zero quarantines | `analysis/dense_failure_stage2/closed_loop_trajectory_set/work/replay_completion.json` | Supplies exact successful-route states for the primary audit | confirmed |
| Full-refit checkpoint is available and hash-bound | `analysis/dense_failure_stage2/closed_loop_trajectory_set/training/full_refit_checkpoint_manifest.json` | Enables seen-UID fit measurement | confirmed |
| Epoch-1 internal checkpoint and original 454/115 group-disjoint split are available | `analysis/dense_failure_stage2/closed_loop_trajectory_set/training/frozen_epoch_selection.json` | Makes held-out generalization estimable without reconstructing a split | confirmed |
| Closed-loop external result was 0 W-to-C and 8 C-to-W | `analysis/dense_failure_stage2/closed_loop_trajectory_set/metrics/full_benchmark_summary.csv` | Motivates but is not used as audit supervision | confirmed |
| Seen FULL/non-FULL/first-nonFULL recall is 99.47%/1.83%/2.54%; highest-responsibility first-nonFULL recall is 11.02% | `analysis/dense_failure_stage2/teacher_forced_free_run_audit/teacher_forced/` | Directly shows the trajectory marginal fits dominant FULL positions but not corrective actions even on exact training states | confirmed |
| R0/R1/R2 success is 42/43/264 of 463 W UIDs | `analysis/dense_failure_stage2/teacher_forced_free_run_audit/release/release_success_summary.csv` | Removing pre-intervention drift has no material effect; executing the first correction causes a 47.73-point jump | confirmed |
| Held-out first-nonFULL recall is 1.79%, only 0.75 points below seen; held-out W success is 8/93 | `analysis/dense_failure_stage2/teacher_forced_free_run_audit/generalization/seen_vs_heldout_summary.csv` | Generalization is also weak, but there is no strong seen fit to generalize from | confirmed |
| 429/463 W free rollouts leave known support; only 4 do so before the selected first intervention, while 186 do so at it | `analysis/dense_failure_stage2/teacher_forced_free_run_audit/free_run/first_off_support.csv` | Most divergence is concurrent with or downstream of the unlearned corrective action | confirmed |
| Every required UID/route/state/release row and artifact hash passes | `analysis/dense_failure_stage2/teacher_forced_free_run_audit/artifact_manifest.json` | Rules out partial-rank completion or mixed provenance as the explanation | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-76 offline closed-loop trajectory-set training | 0 external rescues, 8 regressions | corrective-action learning failure supported; exposure-only diagnosis rejected | Phase-77 exact-state and R0/R1/R2 audit | Change the training objective before collecting on-policy labels | infer exposure from geometric route probability alone |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Teacher-forced versus free-run audit | Directly measures exact-state action fit, support departure, hybrid release, and held-out drop | Objective vs exposure vs generalization | medium | completed |
| One fixed-weight first-nonFULL auxiliary CE term while retaining the trajectory marginal | Directly targets the earliest demonstrated failure without changing architecture/data | Whether sparse corrective-action dilution is causal | medium | recommended, not authorized |
| Immediate on-policy relabeling | Could address later rollout exposure only after exact-state correction is learned | Exposure after action-learning repair | high | rejected as the first response |
| Immediate representation change | Could address held-out action mapping | Generalization | high | deferred because seen fit is already weak |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: identify the earliest dominant failure in the frozen Phase-76 rollout before changing training.
- Relevant memory item used: Phase-76 zero-rescue result despite apparently strong sequence-level route probability.
- Confirmed observation: all exact states, both checkpoints, the frozen split, evaluator inputs, and four GPUs are available.
- Unverified interpretation: the fixed-weight first-nonFULL auxiliary term will be sufficient to retain corrective choices during later free rollout.
- Diagnosis: corrective-action learning failure supported; later exposure remains secondary/unresolved.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage2/teacher_forced_free_run_audit/summaries/teacher_forced_free_run_audit_summary.md`.
- Viable alternatives considered: requested audit; immediate on-policy relabeling; immediate representation change.
- Chosen action: execute the complete teacher-forced/free-run divergence audit in the user plan.
- Strongest objection: divergence from a bounded observed route set is not causal proof of exposure and may reflect incomplete route support.
- How this differs from failed attempts: it does not train another router; it directly separates exact expert-state action selection, free rollout, hybrid forced-prefix release, and held-out generalization.
- Automatic execution authorized: yes
- Authorization basis: the user explicitly requested execution of `plans/teacher_forced_vs_free_run_divergence_audit_plan.md`.
- Stop condition: all cached occurrences are scored for seen and frozen held-out views; all 569 training UIDs complete required free/hybrid rollouts; fixed decision logic, required artifacts, and one unexecuted recommendation are saved and verified, or a validity gate fails.

## Latest Research-Action Result
- Action taken: froze the Phase-77 contract; scored all exact expert route states with the full-refit and frozen internal-dev checkpoints; ran full training-side R0/R1/R2, Dense-C, support, and held-out controls on four GPUs; applied the prospectively frozen decision rule.
- Result: full-refit seen FULL/non-FULL/first-nonFULL recall is 99.47%/1.83%/2.54%; highest-responsibility first correction recall is 11.02%. R0/R1/R2 W success is 42/43/264 of 463. Dense-C preservation is 106/106. Held-out first correction recall is 1.79%. Decision: `objective_action_learning`.
- Evidence saved: `analysis/dense_failure_stage2/teacher_forced_free_run_audit/`; all required artifacts and global counts are hash-verified in `artifact_manifest.json`.
- Failure or issue: no execution failure. Scientific failure is severe non-FULL action underlearning despite excellent FULL recall; later free rollout also remains imperfect after the forced first correction.
- Lesson learned: sequence-level trajectory probability can mask near-zero sparse corrective-action top-1 recall. Exact expert-state action recall and hybrid release must precede an exposure-bias conclusion.
- Next implication: recommend exactly one bounded objective test—a fixed-weight auxiliary CE term at first non-FULL expert states while retaining the existing trajectory marginal. Do not execute it without separate authorization.
