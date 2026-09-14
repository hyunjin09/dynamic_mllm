# Phase 85: READ-only bounded planning/search Memory

## Current Objective
Execute `plans/read_only_bounded_planning_search_plan.md` once: characterize binary READ trajectory search difficulty, empirical correction coverage, and oracle-q ranking while WRITE stays ON.

## Active Constraints
- Unchanged robust P90, exact Step-A triggered population, FULL before trigger; FULL/WRITE_ONLY only afterward.
- Reuse exact q and LMMS generation contracts. Gold-answer q is an analysis oracle, not a deployable score.
- All four GPUs authorized by the user's current instruction, explicitly including occupied devices; this supersedes the earlier occupancy restriction. Do not terminate other workloads.
- Primary B<=128, adaptive 256/512, optional exhaustive Hamming-2 only <=250,000 new candidates; freeze operational rules before new outcomes.
- No learned planner, WRITE study, external deployment, or follow-on execution.
- Large artifacts under `/mnt/hyemin`; repository filesystem has only about 2.2 GiB free.

## Current State
- Done: read access policy, requested plan, local research-control skill, current phase and relevant promoted cache/coverage lessons.
- Stopped by explicit user-selected Phase86 plan on 2026-09-13. Supervisor and all four workers terminated with SIGTERM; no live bounded-planning job remains. Preserve all results and do not restart. See `analysis/read_counterfactual_stage1_branch_critic/old_phase_stop_report.md`.
- Blocked: none.
- Most recent useful observation: four GPUs each have about 16–17 GiB occupied, leaving about 32 GiB per GPU.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| H-READ-D | `workspace/phase_memory/phase_84_read_short_horizon_propagation.md` | Local/H<=8 family did not establish practical identifiability | confirmed |
| Exact dense single-toggle measurements | `analysis/predictability_generalization/stepA_measurement/` | Authoritative Hamming-1 cache and target | pending current census |
| Prior 463/1307 P90 corrective coverage | `workspace/decision_log.md`, 2026-09-04 threshold-specific corpus lesson | Compare only after UID/trigger alignment | pending alignment |

## Failed Attempts and Lessons
None in this action yet.

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Execute approved bounded binary planning audit | Explicit user-selected discriminator after H-READ-D | Search difficulty and q ranking versus empirical opportunity | high | selected |

## Next-Step Decision
- Deliberation mode: deep (expensive search; remaining prospective accounting definitions).
- Active objective and bottleneck: characterize trajectory-level correction after local and short-horizon prediction remained weak.
- Confirmed observation: H8 common-support rho 0.1097 and failed materiality gates.
- Unverified interpretation: binary trajectory search may expose useful structure.
- Diagnosis: unknown beyond the tested representation family.
- Chosen action: only the requested bounded planning study, with independent protocol review before execution.
- Strongest objection: oracle search existence, adaptive cohorts, and unequal finite search supports can overstate planning ease or ceiling.
- How this differs from failed attempts: multiple coordinated READ decisions; direct candidate-discovery versus q-ranking measurements.
- Automatic execution authorization superseded by the user-selected Phase86 plan. Phase85 is stopped; do not restart or submit extensions.
- Stop condition: complete plan outputs, final characterization, and one unexecuted next-direction recommendation.

## Latest Research-Action Result
- Action taken: completed the user-requested PRELIMINARY analysis at the 2026-09-13 21:06 KST snapshot; the full study remains in progress.
- Result: all 13 sessions complete on the same 612 W UIDs. Completed versus 695 pending W differ materially in source/dataset, trigger/suffix and Hamming-1 rescue (24.02% versus 10.94%). Beam8 ANY/SELECTED at maximum B128 is 50.49%/39.71%; B64→128 adds 9.80/8.50 percentage points. Greedy/beam2 flatten through natural termination. No overall saturation established.
- Evidence saved: `analysis/read_only_bounded_planning/interim/PRELIMINARY_20260913_210600/PRELIMINARY_report.md`; immutable snapshot hashes, all per-UID metrics and figures accompany it. Direct reconstruction passes all 47,736 budget cells and 468 complete Hamming-2 censuses (70 rescued UIDs; 53,305 routes).
- Next implication: independent review agrees the remaining population and frozen B256 audit remain warranted; B512 remains conditional on its original gain gate. No population extrapolation, final planning category, execution changes or follow-on experiment.

## Preparation update
- Exact UID/trigger alignment passes:1307W/106C, prior four-action463W. Cached Hamming1 rescues223W; Hamming2 new evaluations81058 (<250000).
- Independent review reconciled: include enumeration baseline, bind initial-cache hashes/identities, measure B512 continuation gain only on B128-unresolved members, replay fixed seeds in smoke.
- Preparation-only source mismatch: Step-A executor hash predates Phase82's added API. Current executor exactly matches Phase82 and Phase84; frozen lineage explicitly uses Phase82 hash and requires fresh Step-A q/token parity. No executor edits.
- Contract `67d2c6a1aab1f4964e2b4836ba9d9e29448f9ff3829d0c31617b0bcfcffb0861`;40 tests pass. GPU smoke pending.

## Smoke and full-launch boundary
- Passed40/40 infrastructure UIDs (32W/8C): exact generated tokens/LMMS, q tolerance1e-6, multi-OFF full-versus-prefix state equality, repeats, seed replay and accounting.
- Original execution contract `67d2c6a1...fb0861` remains unchanged. Supplemental pre-outcome analysis contract `8d143e37...5e9e39` binds shared-actual-budget comparisons, seed/censoring summaries, enumeration-qualified categories and the corrected finalizer.
-13 planning/accounting tests pass, in addition to29 executor tests (42 distinct focused tests). Existing project AUROC implementation reused; no dependency installed.
- Smoke mean new-route cost0.614s gives Hamming2 estimate13.82 GPU-hours (3.46 ideal four-GPU wall hours); full suite and adaptive extensions add substantial work.
- Full study released on all four explicitly authorized occupied GPUs. Durable per-route caches permit resumption. Supervisor proceeds through base128, declared256/512 cohorts and supplemental finalizer, then stops.
- Supervisor/logs: `/mnt/hyemin/qwen_train_eval/outputs/read_only_bounded_planning_v1/`; status `supervisor_status.json`.

## Authorized interim-analysis boundary
- User requests immediate PRELIMINARY analysis using only fully completed UID cohorts; do not wait for remaining jobs. Running jobs remain unchanged.
- Selected action: freeze a completion snapshot, verify completed/pending selection differences, then analyze common completed W UIDs with all13 algorithm/seed sessions complete. Hamming2 gets its own exact completed-census denominator.
- Prior constraint used: initial scheduling prioritizes prompt-length/suffix work, so completion is not random; finite algorithm exhaustion cannot be called population/search saturation.
- Strongest objection: apparent late-budget flattening on a selected cohort may not transfer to pending UIDs or establish a binary ceiling.
- Stop after interim report and bounded necessity assessment; no new experiment or automatic cancellation is authorized by this request.
