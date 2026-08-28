# Three-Action Answer-Aligned Label Conversion Experiment Log

## 2026-08-25: Approved method replacement

- Approved specification: `plans/4way_labeling_fix.md`.
- Research-control mode: standard. A required independent read-only review returned `stable` with high confidence for a clean new-root implementation.
- Rejected alternatives: continue obsolete job 1605 or retrofit old C2C/mechanical schemas. Job 1605 was canceled at zero progress.
- Source inventory remains 12,278 positive samples / 545,531 positive binary routes; no MCTS or source reselection is performed.

## 2026-08-25: Implementation and prospective gates

- Implemented semantic aliases READ_OFF=`WRITE_ONLY`, WRITE_OFF=`READ_ONLY`, BOTH_OFF=`IGNORE`; FULL remains a cached reference.
- Implemented route-conditioned W2C hard/soft/redundant screening, C2C correctness-plus-support-gain acceptance, three-suppression coordinate beam, Pareto routes, max-score routes, wrong W2C partial candidates, deduplication, and canonical selection.
- Added uncached identical-route execution only for repeatability calibration. Epsilon rule is prospectively frozen as max(1e-6 floor, empirical absolute repeat-difference p99); native/unified drift is excluded.
- Added clean calibration/pilot/full execution contracts, atomic records/checksums, exact resume, 16-worker dynamic full queue, calibration finalizer, and pilot audit.
- Code review found and repaired: canonical W2C fallback after cumulative tolerated margin losses; missing-contract audit handling; evaluator-target metadata gate; explicit decomposition cache accounting.
- Test evidence: 17 new focused tests, 42 combined focused tests, and 395/395 complete active project tests pass.
- Frozen pilot: 56 samples / 4,026 routes across all five datasets, manifest SHA-256 `890ddbf...`.
- Initial pre-orchestration calibration contract SHA-256: `3ff379cc6c42b61d0df9be95f393f143e2440245a5173fd4382e0e062595e69a`; it was never executed.

## 2026-08-25: Pre-start submission correction

- Initial pending job 1606 had zero runtime and produced no output.
- Review of its inline `--wrap` command found that shell `&` precedence could background the preceding environment-setup AND-list together with telemetry.
- Job 1606 was canceled before execution. Calibration/pilot orchestration was moved to the syntax-checkable `experiments/run_three_action_calibration_pilot.sh`, which activates the environment in the foreground, starts only telemetry in the background, freezes epsilon after successful calibration, and then runs the pilot.
- Because the orchestration script is contract-bound, a new calibration execution contract is required before resubmission.
- Corrected calibration execution-contract SHA-256: `ab7e9013c817d9b7dafee6829695016f72f9ef890d7e0003d1ed695356e402af`.

## 2026-08-25: Calibration and pilot submission

- Submitted corrected job 1607 (`3act-cal-pilot-v2`) with eight H100s, 64 CPUs, 180G RAM, 16 workers/two replicas per GPU, and a two-day limit.
- Frozen order inside the allocation: repeated-route calibration -> checksum-bound epsilon finalization -> new-semantics pilot. `set -euo pipefail` prevents the pilot from starting if calibration or epsilon finalization fails.
- Job-scoped five-second GPU telemetry starts only inside the allocation and is stopped by an EXIT trap.
- Initial state: pending with reason `AssocGrpGRES` behind another user's exclusive eight-GPU job 1600. Current scheduler projection is 2026-08-26 15:57:56 KST; no external job is disturbed.

## 2026-08-25: Final prelaunch correction and replacement run

- Job 1607 was canceled at zero runtime before the external allocation ended. The final audit found that the plan's warning against independently combining local actions also required an explicit jointly executed independently-best route control in the saved schema.
- Added that control, full integrity/finalization/aggregate/report/checksum tooling, resume-safe resolved-failure accounting, a pilot-derived full compute estimator, and a fail-closed dependent full-run wrapper.
- Final complete CPU gate: 404/404 tests. Final calibration contract: `2cbee2bab1da18ee3be8eb46cbd99f61253e063d298b466c4d04acc829c386f1`.
- Submitted job 1609 (`3act-cal-pilot-v3`) on all eight H100s with 16 workers/two replicas per GPU. Submitted job 1610 (`3act-full-v1`) with `afterok:1609`; it first runs the completed-pilot audit and compute estimate, and starts full inference only if the audit passes.
- Job 1609 started at 2026-08-25 18:50:57 KST. Calibration completed 56/56 samples with zero failures and froze a passing within-unified epsilon of `1e-6` from 224 signed repeat differences. The modified-label pilot then started all 16 workers.

## 2026-08-25: Prospective beam gate failed and run superseded

- At cancellation, job 1609 had produced 24/56 checksum-valid pilot records,
  covering all five datasets and 1,460 source routes with zero executor,
  parity, positive-correctness, C2C-gain, cache, checksum, or worker failures.
- The immutable partial results already contained 322/1,417 beam-8/16
  canonical mismatches and 167/1,417 positive-set Jaccards below the frozen
  0.50 floor (minimum 0.0). Completing the remaining records could not make the
  all-route stability gate pass.
- The user approved `plans/4way_labeling_3.md`, which removes beam search and
  continuous-score label selection in favor of exact sequential verified
  branching for W2C and mechanical C2C preservation.
- Jobs 1609 and 1610 were canceled at 23:13:12 KST. Job 1609 consumed 04:22:15;
  job 1610 never ran. Partial outputs remain historical provenance only.
- Detailed evidence: `early_stop_audit.md`.
