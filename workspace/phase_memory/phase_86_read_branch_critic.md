# Phase 86: Frozen Stage-1 READ branch critic

## Current Objective
Execute `plans/read_counterfactual_stage1_branch_critic_diagnostic_plan.md`: test the exact frozen Stage-1 predictor on one-layer ON/OFF branch states and the frozen first-trigger offline policy.

## Active Constraints
- All 1,413 P90 UIDs / 1,385 image groups / 15,185 dense states; exact Step-A q and correctness labels.
- Same five-head ensemble, normalization, representation and P90 tau=0.9061332901863008. No training, calibration fit, threshold tuning, search, branch suffix generation or controller.
- User explicitly replaces Phase85 and authorizes its safe termination. Preserve all old evidence; do not restart it.
- Four GPUs remain explicitly authorized even occupied; terminate no unrelated workload. Large caches under /mnt/hyemin; existing project .venv unchanged.

## Current State
- Done: safely stopped Phase85 supervisor and workers; preserved status/process/artifact snapshot and old interim evidence.
- Done: audited original Stage1 feature/indexing contract and Phase82 cache coverage; independent protocol review stable.
- Done: contract `293a36fa...8b5e1a` frozen; fresh 32-UID / 308-state smoke passed all parity checks; four focused metric/pooling/policy tests pass; all 30,370 frozen ON/OFF label records align.
- Done: all 15,185 states / 1,413 UIDs scored, with all layer27 states eligible; full parity and original ON score reproduction pass (max difference 1.37e-7).
- Done: all requested metrics, subgroup reports, six figures, 5,000 image-group bootstrap draws and the one post-result fixed-target cross-score diagnostic.
- Done: independent interpretation review stable, medium confidence; BC-C qualified. All 850 repository artifacts and 1,413 external feature caches verify under manifest `9c033a70...427941c`.
- In progress: none; phase complete and stopped.
- Blocked: none.
- Most recent useful observation: decoder output of action layer l is Stage1 position l, including layer27. Compact Stage2 caches lack both required user-text summaries.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Original post-layer Stage1 hooks and feature pooling | dense_failure_stage1/runtime.py | Correct index l and valid final layer27 | confirmed |
| Compact cache retains last control token only | dense_failure_stage2/closed_loop_trajectory_set.py | Must reconstruct missing OFF user-text features, not substitute compact token | confirmed |
| Prior one-step utility regression weak | workspace/phase_memory/phase_82_counterfactual_effect_identifiability.md | New question is frozen branch failure prediction, not another utility fit | confirmed |
| Old phase termination | analysis/read_counterfactual_stage1_branch_critic/old_phase_stop_report.md | New priority explicitly supersedes unfinished search | confirmed |

## Failed Attempts and Lessons
No research failure. Initial safe-stop script met an already-exited worker; second identity-checked SIGTERM pass verified zero live old-phase processes.

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Exact one-layer feature reconstruction + frozen critic | User selected and missing features reconstructible | Relative branch preference under absolute failure critic | moderate | selected |
| Exclude unsupported cached representations | Avoid replay | Would discard the scientific population unnecessarily | low | rejected |

## Next-Step Decision
- Deliberation mode: deep for indexing/representation validity; fully specified execution afterward.
- Active objective and bottleneck: branch states exist only in an insufficient compact representation.
- Relevant memory item used: exact Phase82 branch hashes and native Stage1 cache provide parity references.
- Confirmed observation: Stage1 features are post-output indexed l; original trigger l already uses that output.
- Unverified interpretation: zero-shot failure-score differences may rank branch outcomes.
- Diagnosis: supported cache representation gap was repaired by exact one-layer reconstruction. Cause of weak paired preference remains unknown.
- Chosen action: exact plan with authorized missing one-layer regeneration; independent reviewer stable/high confidence.
- Strongest objection: historical post-layer trigger and pre-layer StepA action imply a retrospective rollback diagnostic; do not claim a forward-only online controller.
- How this differs from failed attempts: no utility training; frozen failure critic scores actually executed alternative states.
- Automatic execution authorized: completed for this named plan; no follow-on action authorized.
- Stop condition: satisfied. BC-C qualified and exactly one unexecuted paired branch-risk calibration/ranking recommendation. No controller, training, search restart or new phase.

## Latest Research-Action Result
- Action taken: complete frozen Stage1 READ branch-critic diagnostic, after safely stopping Phase85.
- Result: BC-C, qualified. ON/OFF AUROC 0.6728 [0.6200,0.7247] / 0.5949 [0.5557,0.6334]. Delta-p Spearman -0.0187; harmful/beneficial flip preference 54.08%/41.46%, balanced 47.77% [38.29%,57.11%], discordant AUROC 0.4684 [0.3574,0.5799]. Q3 enrichment +2.36pp CI crosses zero.
- Policy: W chooses OFF 591/1307, with 30 W→C; C chooses OFF 57/106, with 6 C→W. Net+24 [12,36]; treated-W success 5.08%, treated-C regression 10.53%.
- One decision-changing check: on fixed ON labels, ON/OFF score AUROC 0.6728/0.6710; on fixed OFF labels 0.5948/0.5949. The cross-branch AUROC gap is already present with scores held fixed, so it does not establish OFF-score collapse. This changes the provisional interpretation from BC-B toward qualified BC-C; no cause of relative-choice failure is established.
- Evidence saved: `analysis/read_counterfactual_stage1_branch_critic/`, especially `summaries/branch_critic_summary.md`, `scores/fixed_target_cross_score_diagnostic.csv`, `parity/independent_result_verification.json`, manifest `9c033a70...427941c`.
- Failure or issue: scientific execution valid. Independent verification initially used pandas default JSON parsing, perturbing near-ties below 1e-15; lossless json.loads exactly reproduces the unchanged analysis. No experiment or metric was altered.
- Lesson learned: Stage1 decoder output indexing and user-text pooling must be preserved; branch-to-branch AUROC changes confound target changes with score changes. Positive one-decision Net alone does not establish relative ranking or controller readiness.
- Next implication: stop. Exactly one unexecuted paired branch-risk calibration/ranking recommendation; no assertion that calibration will succeed. Current first-trigger policy is retrospective because historical trigger l was post-layer while StepA intervened pre-layer.
