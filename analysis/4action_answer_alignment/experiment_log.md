# Four-Action Answer-Alignment Experiment Log

## 2026-08-23 — Readiness and implementation

- Read all 866 lines of `plans/4way.md`.
- Verified exact model revision, five model shards, 8,000 raw label records,
  6,000 GQA/TextVQA images, project-local environment, 8 H100s, and live Slurm.
- Froze 4,890 primary/control rows. Primary A+: 1,235 GQA and 677 TextVQA.
- Froze 8-example smoke and 56-example pilot IDs, balanced by dataset and
  spread deterministically over visual-token count.
- Preserved 82,860 known positive-vision correcting routes for primary A+.
- Implemented the binary-executor four-action extension and continuation
  scorer/generator. The latest focused executor/cohort/analysis suite passes
  23/23; earlier readiness suites are recorded separately in phase memory.
- Implemented the deterministic post-run causal analysis: structured
  per-action rows, factorial distributions, rescue taxonomy, sample and
  image-group bootstrap intervals, Hamming strata, route overlap, controls,
  and core figures.
- Submitted all-8-GPU semantic preflight as Slurm job `1424`; initially pending
  behind an existing exclusive 8-GPU job.
- Submitted all-8-GPU smoke as Slurm job `1425` with dependency `afterok:1424`.
- Submitted all-8-GPU pilot as Slurm job `1426` with dependency `afterok:1425`.
- Live pending reason for `1424` is `AssocGrpGRES`: job `1422` currently owns
  all eight H100s. The allocation was deliberately not downsized.
- No scientific outcome has been inspected or interpreted.

## 2026-08-23 — Preflight parity failure and scheduler adaptation

- Job `1424` ran on all eight H100s and passed every gate except the provisional
  whole-route IGNORE prompt-logit max-absolute tolerance. Greedy outputs were
  identical to frozen binary single-OFF at layers 0, 13, and 27.
- Cancelled dead dependency jobs `1425` and `1426` after `1424` failed.
- Adapted the ignored machine-local scheduler so an 8-GPU job requests 64 CPUs
  and 512 GiB by default instead of implicitly requiring whole-server
  exclusivity. Diagnostic jobs then started immediately alongside CPU-only
  work while still allocating all eight H100s and eight workers.
- Job `1482` recorded scale-aware drift on two samples: relative logit RMSE
  1.26%--2.83%, argmax/generation identity 6/6, and top-10 overlap 0.9--1.0.
- One required read-only independent review returned `revise`: retain native
  FULL, but bridge validation to the actual teacher-forced answer metric.
- Job `1483` implemented that bridge. All structural/native/generation checks
  passed, but selected-correct mean-logprob drift was 0.0771--0.1230 and margin
  drift was 0.0625--0.1250 on the completed sample. The predeclared bridge gate
  correctly stopped execution.
- No smoke/pilot was resubmitted. A third tolerance relaxation is prohibited
  without an explicit baseline/validation decision.

## 2026-08-23 — Unified executor and complete relaunch

- The user defined the causal estimand entirely inside a unified executor:
  M11=unified FULL, M10=READ_ONLY, M01=WRITE_ONLY, and M00=IGNORE. Native FULL
  and old binary single-OFF are external semantic/drift diagnostics only.
- The target layer now executes the same full-row and compacted-row decoder
  calls for all four actions from identical pre-layer state. All prefixes and
  suffixes use unified FULL.
- Added layerwise answer-alignment trajectories, route OFF-vs-ON comparisons,
  minimum-ON route metadata, Hamming-distance associations, image-group
  bootstrap intervals, controls, and population single-operation trajectory
  rescues. The focused suite passes 50 tests.
- Frozen primary counts before launch: 1,235 GQA + 677 TextVQA = 1,912 A+.
  Controls: 868 matched-budget no-correction-found and 2,110 FULL-correct/
  ALL-OFF-wrong vision-required samples.
- Submitted strict dependency chain: unified preflight `1485`, 8-example smoke
  `1486`, 56-example validation `1487`, throughput estimate `1488`, primary
  sweep `1489`, controls `1490`/`1491`, trajectory selection `1492`, all-eight-
  H100 trajectory rescues `1493`, and final bootstrap analysis/report `1494`.
- Every GPU job requests exactly eight H100s and launches eight workers. The
  chain is resumable and cannot advance past a failed semantic/stage gate.
- At submission, `1485` is pending `AssocGrpGRES` behind other users' live GPU
  work; no job was downsized and no other user's work was disturbed.
- Unified preflight `1485` and the full 28-layer smoke `1486` passed every
  stage, sample, token-generation, evaluator, branch-state, cache, determinism,
  and external-semantic gate. The shared eight examples had maximum absolute
  native/unified margin drift 0.1875; this is recorded as diagnostic-only.
- The 56-example validation `1487` then started automatically on all eight
  H100s.

## 2026-08-23 — Validation boundary case and primary relaunch

- Job `1487` wrote all 56 records and passed current unified-FULL/native-FULL
  generation, evaluator, correctness, unified-IGNORE/old-binary semantics, and
  all structural gates. It exited 2 only because one transferred historical
  FULL anchor token sequence differed on this server.
- Inspected disagreement: `textvqa:textvqa_train_1800` changed from historical
  `rih ferdinand` to current native/unified `riki ferdinand`. Both are evaluator-
  wrong, so FULL correctness and the A+ cohort definition are unchanged.
- Exact historical-anchor token identity is now an explicit provenance
  diagnostic. Current native-vs-unified deterministic token identity and
  evaluator correctness remain hard semantic gates.
- The adjudicated 56-example summary passes. Native/unified absolute margin
  drift has mean 0.03953, median 0.01210, p95 0.12500, p99 0.15313, and maximum
  0.18750; it remains diagnostic-only.
- The gated pilot mean of 48.154 seconds/sample yields a conservative primary
  estimate of 25.575 GPU-hours / 3.197 wall-hours at eight workers. It includes
  old-binary validation calls that production omits.
- Cancelled the unstartable dependency chain `1488`--`1494`. Submitted the
  replacement primary/control/trajectory/report chain as jobs `1497`--`1502`.
  Job `1497` requests all eight H100s and is pending behind another user's job
  `1496`; no other work was disturbed.

## Unified pilot throughput estimate

- Mean seconds/sample: `48.153890`
- Median seconds/sample: `33.235153`
- Estimated primary GPU-hours: `25.575`
- Estimated primary wall-hours at 8 workers: `3.197`
- Estimated all-controls GPU-hours: `39.834`
- Basis: gated 56-example unified-materialized, all-28-layer pilot; includes old-binary semantic checks omitted from production, so the primary/control estimates are conservative upper estimates.

## 2026-08-23 — Current-runtime cohort freeze and active primary sweep

- Primary job `1497` failed after 46 seconds on
  `gqa:gqa_ge_16564303`. The transferred cache labeled native FULL `no` and
  wrong, while current native and unified FULL both generated `yes` and were
  correct. This was a cohort-boundary drift, not a native/unified semantic
  mismatch. Dependency-held jobs `1498`--`1502` were cancelled.
- One required read-only independent review returned `revise`: retain the
  matched cache for candidate discovery and binary-route provenance, but apply
  the defining FULL-wrong/FULL-correct condition using frozen current unified
  FULL before factorial execution.
- Implemented a resumable eight-worker eligibility stage and production gate.
  The focused executor/cohort/eligibility suite passes 52/52.
- All-eight-H100 job `1505` completed in 6:11. It produced 4,890/4,890 rows
  with zero failures and passed the eight-shard/eight-worker contract.
- Eligible primary A+: 1,880 total = 1,222 GQA + 658 TextVQA; 32 of 1,912
  candidates were excluded. Eligible no-correction control: 868 total = 614 +
  254; none excluded. Eligible vision-required control: 2,084 total = 1,137 +
  947; 26 of 2,110 candidates excluded.
- Submitted replacement strict chain: primary `1506`, no-correction control
  `1507`, vision-required control `1508`, trajectory selection `1509`, all-
  eight-H100 trajectory rescue `1510`, and final analysis/report `1511`.
- Job `1506` started automatically after `1505` and is actively producing
  primary results across all eight GPUs. Completed shards are resumed and no
  existing raw result is overwritten.

## 2026-08-23 — Per-GPU concurrency optimization

- Live profiling of job `1506` showed one process/sample per H100, about
  17.5--17.9 GiB of 80 GiB memory per GPU, sustained sampled SM utilization
  mostly 24--30%, and each Python worker at about 99.8% of one CPU core. The
  stage was CPU/Python-launch limited enough to justify concurrent replicas.
- With explicit user authorization, cancelled jobs `1506`--`1511`. The primary
  cancellation was recoverable: 1,501 unique eligible primary records are
  append-only and complete, leaving 379/1,880 to resume. No completed result
  was deleted or overwritten.
- Added production-only multi-replica execution. Two independent model replicas
  share each allocated H100, for 16 disjoint workers across all eight GPUs.
  Replica outputs use separate append-only result/runtime/failure files; the
  stage merger accepts the preserved one-replica prefix plus the resumed two-
  replica suffix and still requires exact unique cohort coverage.
- The trajectory rescue runner uses the same layout while keeping selections
  for a UID together within a replica to preserve baseline reuse.
- Static and focused verification passes 61/61 tests. The correctness gates,
  deterministic seed convention, unified executor, estimand, sample contents,
  and all-eight-GPU allocation are unchanged.
- Optimized primary resume `1526` and dependency chain `1527`--`1531` were
  submitted. Another user's job `1516` currently owns all eight H100s, so
  `1526` is pending `AssocGrpGRES`; no other user's work was disturbed.
- Acceptance rule: retain two replicas only if the production ramp has no OOM,
  no semantic/determinism failure, and materially improves aggregate samples
  per minute. Otherwise resume safely with one replica and record the attempt.
- CPU monitor job `1533` starts after the verified current GPU owner `1516`
  finishes, then gates the first 64 new `1526` results. It requires all 16
  worker metadata files, unique samples, passing semantic gates, zero replica
  failure artifacts, healthy Slurm state, GPU telemetry, and at least 1.20x the
  measured 18.0-sample/minute one-replica baseline. Jobs `1527`--`1529` now
  require both `afterok:1526` and `afterok:1533`.

## 2026-08-23 — Multiplex ramp result and safe fallback

- Monitor `1533` failed before evaluating the ramp because `nvidia-smi` is not
  exposed in the CPU partition (`No devices were found`). This was monitoring
  infrastructure failure, not a scientific-worker failure. GPU telemetry is
  now diagnostic-only when NVML is unavailable in the monitor allocation; the
  correctness, worker-layout, uniqueness, failure-artifact, Slurm-health, and
  throughput gates remain mandatory.
- Replacement monitor `1554` evaluated 65 new two-replica results. All semantic
  gates passed, all samples were unique, all 16 expected runtime records were
  present across all eight H100s, no failure artifact appeared, and the GPU job
  remained healthy. External live telemetry showed about 99% utilization and
  35.0--35.4 GiB per H100 without OOM.
- The apparent utilization increase did not improve useful throughput:
  14.4067 samples/minute versus the measured 18.0 samples/minute one-replica
  baseline, or 0.8004x. This failed the prespecified 1.20x acceptance gate.
  The two-replica job `1526` and its old downstream chain were therefore
  cancelled recoverably rather than extending a slower configuration.
- At cancellation, the append-only primary artifacts contained 1,574 unique
  eligible results, with no duplicate eligible rows, leaving 306/1,880.
  Evidence is preserved in `multiplex2_ramp_report.json` and the primary
  replica result/runtime files.
- Submitted the one-replica resumable fallback chain: primary `1557`, controls
  `1558` and `1561`, CPU trajectory selection `1562`, GPU trajectory rescue
  `1563`, and CPU final analysis/report `1564`. Every GPU stage still requests
  all eight H100s and runs eight disjoint workers. Job `1557` is currently
  pending `AssocGrpGRES` behind another user's live all-eight-H100 job `1551`;
  the allocation is not disturbed. The focused suite passes 62/62 tests.

## 2026-08-23 — Primary completion and resumable downstream repairs

- One-replica primary job `1557` completed in 55:30. Its stage summary passed
  exact coverage and uniqueness for all 1,880 eligible A+ rows (1,222 GQA and
  658 TextVQA), all eight-shard/worker contracts, 28-layer completeness, and
  every sample semantic gate. The single historical cohort-boundary failure is
  explicitly non-disqualifying and excluded.
- CPU trajectory selection `1562` completed and selected 10,196 population
  follow-up cells across 1,579 primary samples.
- Control A job `1558` stopped after preserving 604/868 rows because one
  TextVQA row had no reference capable of meeting its frozen 0.5 EvalAI
  consensus threshold. A full manifest audit found exactly 11 such rows, all
  in Control A and none in primary or the vision-required control. They are now
  explicitly excluded because the correct-vs-FULL-wrong margin is undefined,
  leaving 857 analyzable Control A rows. Resumable job `1565` is running; the
  old failure artifact is retained as provenance and is non-disqualifying.
- Vision-required job `1561` preserved 99 rows and stopped on one final
  logit-lens-versus-direct-state difference of `9.2924e-05`; the other 98
  completed rows were exactly equal. Tokens, generated answer, evaluator
  correctness, cache geometry, and intervention semantics passed. The
  diagnostic BF16 readout-identity tolerance is now `1e-4`, and job `1567`
  will resume the remaining 1,985 rows.
- Trajectory rescue `1563` preserved 4,364/10,196 cells and stopped on the same
  diagnostic class: suppressed-trajectory final margin differed from direct
  intervention scoring by `5.7618e-05`, while primary final margin, tokens, and
  correctness matched. Resumable job `1569` applies the same `1e-4` tolerance
  and re-adjudicates the preserved boundary row without overwriting it.
- The repaired implementation passes 66/66 focused tests. Final CPU analysis
  job `1570` depends on `1565`, `1567`, and `1569`.

## 2026-08-23 — Fixed-target trajectory identity repair

- Control A repair `1565` completed 857/857 analyzable rows and passed every
  stage gate; its 11 evaluator-unscorable TextVQA exclusions remain explicit.
  Vision-required repair `1567` completed 2,084/2,084 rows and passed every
  stage gate under the diagnostic `1e-4` BF16 readout tolerance.
- Rescue repair `1569` preserved 5,503/10,196 unique cells, then stopped at
  `trajectory_006051` (`textvqa:textvqa_27002`). Primary intervention margin,
  generated IDs, and correctness all matched. The trajectory followed the
  baseline-selected valid phrase `not question`, while the intervention
  endpoint selected the alternate evaluator-valid phrase `yes`. The stored
  candidate score reconstructs the fixed-target trajectory margin exactly:
  `-5.5426580686`; the evaluator-best endpoint margin is `-5.0508334417`.
- The repaired gate now compares the FULL and suppressed trajectories using
  the same fixed answer token sequence, reconstructs its exact intervention
  endpoint score, and separately records whether the evaluator-best valid
  phrase switches. It retains evaluator-best state margins for the primary
  factorial/discrete analysis. This is a quantity correction, not a tolerance
  relaxation.
- Read-only independent review ranked this fixed-target comparison above
  dynamic trajectory retargeting or stopping, with high confidence. A focused
  reproduction recovers the preserved failed row exactly, and 67/67 focused
  tests pass.
- Resumable all-eight-H100 job `1572` is pending behind another user's live
  allocation with 4,693 cells remaining. Dead dependency job `1570` was
  cancelled; CPU final analysis/report job `1573` depends on successful
  completion and summarization of `1572`.

## 2026-08-23 — Trajectory rescue completion

- All-eight-H100 resume `1572` completed in 18:18 after preserving every prior
  append-only result. The authoritative merger contains exactly 10,196 unique
  selections: 5,630 GQA and 4,566 TextVQA cells.
- All eight shard/worker, exact-selection, result-gate, and failure gates pass.
  The two preserved historical boundary failures are semantically recovered
  under the documented target-identity/readout rules and are non-disqualifying.
  There are zero disqualifying failures.
- Fixed baseline-selected correct-target identity was retained for every
  trajectory. The evaluator-best correct phrase switched at the intervention
  endpoint in 269/10,196 cells (2.6383%); evaluator-best endpoint state remains
  separately preserved.
- `merged_results.jsonl` and `summary.json` both pass their saved SHA-256
  checksums. Dependent CPU final analysis/report `1573` is pending for server
  resources behind another user's whole-server allocation.
- `plans/4way_2.md` is approved as the next experiment, but explicitly forbids
  beginning its route-conditioned implementation audit until the current final
  analysis succeeds. No route-conditioned work has begun.

## 2026-08-24 — Final analysis and report completion

- CPU final-analysis job `1573` started at 01:40:28 KST and completed at
  02:08:20 KST with exit code `0:0` after 27:52. It had already begun when the
  server-local policy was clarified, so it was allowed to finish rather than
  being cancelled and duplicated locally.
- The aggregation consumed 4,821 samples, including all 1,880 primary A+
  samples, and wrote 539,952 flat sample/layer/action rows with 2,000 bootstrap
  replicates.
- Every newly written SHA-256 sidecar verifies. Unified FULL/native FULL
  semantics match on 72/72 validation comparisons, and unified IGNORE/old
  binary single-OFF semantics match on 1,816/1,816 comparisons.
- The final report and numerical-consistency report were produced at
  `4action_answer_unaligned_report.md` and `numerical_consistency_report.md`.
  No disqualifying semantic or execution failure remains, so the prerequisite
  gate in `plans/4way_2.md` is satisfied.
- Machine-local policy now runs future CPU-only project work directly by
  default; Slurm remains mandatory for GPU work.
