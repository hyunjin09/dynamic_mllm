# Phase 88: Benchmark-calibrated fixed READ/WRITE schedules

## Current Objective
Complete `plans/benchmark_calibrated_fixed_read_write_schedule_plan.md`, explicitly authorized2026-09-14. Test calibration-only benchmark schedules on held-out correctness without Stage1 or learning. The named plan is the only authorized research action; no automatic top2 or follow-up.

## Active Constraints
- Four families; exact native evaluators/generation; no triggers, routers, fitting, or search. Phase85 remains stopped.
- Whole image/content groups disjoint across CAL/TEST. Primary/nested pools, independent bit selectors, seeds,20matched random controls and category rules frozen prospectively.
- Four GPUs authorized, including occupied GPUs; direct execution. Project `.venv` only. Large artifacts under `/mnt/hyemin/qwen_train_eval/outputs/benchmark_calibrated_fixed_rw_schedule_v1`, linked as `analysis/benchmark_calibrated_fixed_rw_schedule/work`.
- No schedule or budget changes based on TEST outcomes. Report exact19,452 TEST UIDs, not the full official MMMU/POPE populations.

## Current State
- Complete splits: CAL256/255/256/252 and TEST2500/5000/3204/8748 for ChartQA/TextVQA/MMMU-Pro/POPE. CAL groups118/160/127/14. Zero transitive raw/RGB/native-identity leakage; five overlapping ChartQA train annotations excluded. Available official train frames for ChartQA/TextVQA; group-disjoint evaluation partitions for MMMU/POPE.
- Complete Dense:20,471/20,471 exact native token matches, zero dual-output fallbacks;20,466 valid q values. One CAL/four TEST TextVQA empty-q rows remain in correctness.
- Complete seven CAL action/repeat/direct-versus-suffix/terminal-WRITE checks. Complete R and W sweeps:1,019UIDs and28,532branches each (57,064total). All terminal-WRITE token/correctness controls pass; q matches where valid. WRITE27 excluded from selection.
- Complete calibration analysis and schedule freeze before TEST launch. ChartQA R12/W25; TextVQA R11/W24; MMMU-Pro R17/W14; POPE R7/W14. Global R17/W19. Twenty random draws/family; ChartQA19unique, others20; duplicate weights retained.
- Complete M1/M2/M3 and global on all19,452TEST UIDs. Active: four-GPU random controls (15,703/19,452 complete at 2026-09-14T19:42:52+09:00), then aggregation. Supervisor2361209. Durable status: `analysis/benchmark_calibrated_fixed_rw_schedule/work/continuation_status.json`. Do not launch duplicates. Supervisor stops at `analysis_ready_for_interpretation`; main agent must finish reports/review/hash verification/state.
- Fourteen focused checks passed:7bound execution tests and7reporting/rubric tests, including integer-count random ties despite CSV float rounding.

## Evidence That Matters
| Evidence | Path relative to analysis/benchmark_calibrated_fixed_rw_schedule | Implication |
|---|---|---|
| Full native Dense parity | parity/dense_population_parity.csv | Repaired executor matches every selected Dense UID |
| Complete CAL censuses | calibration/R_complete.json; W_complete.json | Selection uses complete identical UID populations |
| Frozen schedules and manifests | schedules/schedule_freeze.json | TEST launch occurred strictly after freeze |
| CAL-only summary | summaries/calibration_summary.md | Selected bit gains are only3–5net correct UIDs; no split-half top1 agrees |
| Category scope review | parity/calibration_category_scope_review.md | A/B/C are precluded by CAL stability; qualifiedD is not a no-efficacy finding |
| POPE effective N | splits/benchmark_support.csv |252questions represent only14independent CAL images |

## Failed Attempts and Lessons
- Empty TextVQA q references stopped first Dense pass. Implemented the already-frozen invalid-q rule without dropping correctness. Old attempt preserved under work/failed_attempt1_q_validity. Primary TextVQA/global q tie-breaking is disabled for incomplete pools; eligible nested/half pools can retain q.
- Native Music53 punctuation mismatch initially received a reviewed scorer-equivalent dual-output repair, retaining hash-bound parent rows. A later Psychology128 scorer mismatch kept the hard gate closed. That provisional reuse is superseded: all old partial Dense rows were archived and none entered current analysis.
- Supported cause: every prefill layer and initial logits matched, but native continuation used final prompt position194 while inherited decoder used maximum-derived323. Phase88-only `align_native_decode_positions` sets effective delta=last_position+1-full_length for both generation and q. It restores exact native output. Shared historical executor files unchanged; no retrospective prior-phase re-audit. Evidence: parity/dense_scorer_mismatch_diagnostic.json, decode_position_repair.md; old records/logs in work/failed_attempt3_decode_position.
- Schedule selection import failed on missing SciPy after both CAL sweeps. Installed scipy1.18.1 with uv in `.venv`; numpy2.5.1 and model packages unchanged. Import/Spearman smoke passed; resumed selection without repeating valid GPU work. Evidence: parity/analysis_environment.json and workspace/env_state.md.

## Open Candidates
| Candidate | Status | Reason |
|---|---|---|
| Finish frozen held-out/control matrix | selected | Measures efficacy, interaction and benchmark specificity still unknown from CAL |
| Stop after CAL | rejected | Predetermined category does not answer held-out efficacy |
| Revise category thresholds after CAL | rejected | Would change prospective interpretation after viewing calibration outcomes |
| Top2/new research direction | unexecuted | Separate user authorization required |

## Next-Step Decision
- Deliberation mode: DEEP for high-impact interpretation and remaining expensive comparisons; existing read-only reviewer reconciled stable/high.
- Confirmed: only TextVQA WRITE and POPE READ meet both frozen stability predicates. No family has all active bits stable, neither named bit is stable in two families, and no family has both bits unstable.
- Logical consequence: A/B/C cannot pass; residual qualified CAL-D is already determined by CAL. This is NOT evidence that held-out schedules cannot improve Dense.
- Unknown: held-out efficacy, corrections/regressions, interaction, global/random contrasts and grouped intervals.
- Selected action: finish the same authorized M1/M2/M3/global/20random matrix, with no schedule/rubric/budget change. Strongest objection is remaining cost despite categorical foreclosure; response is that the central efficacy question remains unanswered and is explicitly required by the plan.
- Stop condition: complete all requested reports/figures, separate efficacy evidence from qualifiedD, reconcile final interpretation, verify every artifact, update compact state, give exactly one unexecuted recommendation. No new experiment follows.

## Latest Research-Action Result
CAL, TEST methods and global complete; random controls active. Split SHA256 `21dc39f7ca4ac02a47fafda2dd583ed6fb37515a059f8eb74b5547fe42caa512`. Current contract SHA256 `b995483d75b300ef0170db9f06eff2a68801ba4f78fcb922e17951e96ee7492e`. No held-out performance conclusion has been made.

Final recommendation must be reviewed against actual efficacy/control evidence, not mechanically inferred as absence of benefit from the precluded category. The reporter D-direction text is provisional until full-result interpretation; exactly one unexecuted recommendation is still required.

## User-requested pause
The user asked the main agent to stop monitoring for now and will announce when computation finishes. Leave supervisor2361209 and its existing pipeline running; do not terminate or duplicate jobs. At pause, stage was methods. Pipeline continues through methods/global/random/aggregation and stops at analysis_ready_for_interpretation (or failed). Resume main-agent reports, scientific interpretation/review, final artifact verification and state updates only after the user returns. No final held-out conclusion has been made. See analysis/benchmark_calibrated_fixed_rw_schedule/main_agent_pause.json.

## Cross-server Git snapshot
The user requested committing/pushing necessary research state and updating the handoff. See `handoff/phase88_server_transfer/README.md` for the exact environment, hashed/compressed frozen metadata and separately transferred raw payloads. Source jobs remain active; packaging is not a migration of execution ownership or authorization for a new experiment. Final interpretation/reporting remains pending. The timestamped progress snapshot is historical on the destination.
