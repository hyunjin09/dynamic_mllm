# Exact Sequential Four-Action Label Conversion Experiment Log

## 2026-08-25: Method replacement authorized

- Approved plan: `plans/4way_labeling_3.md`.
- Old beam-based calibration/pilot job 1609 and dependent full job 1610 were cancelled after partial evidence made the prospective beam-stability gate impossible to pass.
- The replacement uses exact sequential verified branching and mechanical C→C preservation. No old beam output is used as a label.
- Frozen source inventory remains 12,278 samples / 545,531 routes; no MCTS rerun or source reselection is performed.

## Implementation and prelaunch gates

- Added an isolated exact policy core, execution contract, 8-sample smoke selector, 8/16-worker runner, semantic smoke audit, exact-resume verifier, compute estimator, finalizer, aggregate analysis, plots, and report generator.
- Fixed smoke: 8 samples, all five datasets, 4 source-status W→C and 4 source-status C→C samples, 0–28 source OFF positions, one one-route ALL-OFF W→C stress case, and six multi-route samples.
- Independent review required replacing the old conversion core rather than adapting its beam/canonical-selection policy.
- Focused exact-policy gates pass, shell/Python syntax checks pass, and the complete active test suite passes 424/424.

## 2026-08-25: Smoke and dependent full submission

- Fresh prelaunch state: Slurm node idle; all eight H100s at 0 MiB / 0% utilization; all required source/model/smoke manifests present; all 84 pinned packages compatible.
- Submitted job 1611 (`4act-seq-smoke-v1`) for the exact 8-sample smoke with eight H100s, 64 CPUs, 180G RAM, and 8 workers/one replica per GPU.
- Submitted job 1612 (`4act-seq-full-v1`) with `afterok:1611`, a seven-day limit, and 16 workers/two replicas per GPU. It audits the completed smoke and records a compute estimate before any full inference.
- Job 1611 began immediately; job 1612 is pending on the semantic/integrity dependency.

## 2026-08-25: Smoke passed and full conversion started

- Job 1611 completed `0:0` in 95 seconds. The fail-closed audit passed every check: 8/8 samples, all five datasets, both route types, 61 source routes, 56 replay-valid routes, five explicitly quarantined replay failures, 60 final correct branch occurrences, 59 unique routes, exact resume, checksum integrity, stable targets, and old-binary semantic parity.
- Real smoke path counts: 96 FULL restorations, 10 READ_ONLY-only decisions, 34 WRITE_ONLY-only decisions, seven IGNORE fallbacks, and four both-partial branch events. Maximum active branch count was two.
- Smoke-derived estimate: 30.95 wall-hours with 16 workers / 247.59 allocated GPU-hours. This remains provisional because exact branching is data-dependent.
- Job 1612 passed the smoke audit and compute-estimate gates and began full inference. All eight GPUs reached 98–99% utilization at roughly 33–34 GiB each with two replicas/GPU and no initial worker failures.

## 2026-08-26: User-requested pause

- The user requested a pause and will explicitly authorize resumption later.
- Full job 1612 was canceled cleanly at 2026-08-26 12:48:48 KST after 13:04:44 elapsed (`CANCELLED by 1003`).
- Preserved state: 262 atomic checksum-backed completed sample records and zero worker failure files. No output, claim, source label, or report was deleted.
- The 16 samples active at cancellation did not reach the per-sample atomic commit boundary and will be reclaimed from the beginning on a later launch.
- Resume requirement: keep the existing config/code execution contract unchanged, submit a new eight-GPU full job, and run the existing full wrapper with `--resume` behavior. The new Slurm job ID creates a fresh launch-scoped claim root; all 262 completed UIDs are skipped automatically.
- All eight GPUs returned to 0% utilization after cancellation.

## 2026-08-26: User-authorized resume

- The user explicitly authorized resumption.
- Before submission, the 262 completed records and zero-failure state were verified, all eight H100s were free, and the recomputed full execution contract exactly matched the preserved contract SHA-256 `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
- Submitted job 1628 (`4act-seq-full-resume-v2`) with eight H100s, 64 CPUs, 180G RAM, a seven-day limit, and the unchanged 16-worker/two-replica topology.
- The wrapper revalidated the completed smoke and compute-estimate artifacts before inference. All 16 workers created fresh launch-scoped claims; all eight GPUs reached 99–100% utilization with zero failure files. The 262 completed UIDs remain skipped and intact.

## 2026-08-26: VQA-first / WeMath-last relaunch

- At the user's request, job 1628 was canceled cleanly at 16:03:19 KST after 33:42 elapsed. The committed state was unchanged at 262 checksum-backed records (187 WeMath2.0 Standard, 75 WeMath2.0 Pro) with zero failure files; 16 active uncommitted samples will restart later.
- Added a launch-only dataset-priority helper and wrapper. They are not included in `SEQUENTIAL_CONVERSION_CODE_PATHS`, and the preserved scientific execution contract remains SHA-256 `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
- The complete active test suite passes 428/428; shell syntax and focused queue/runner/audit tests pass.
- Submitted job 1629 (`4act-seq-vqa-first`) with eight H100s, 64 CPUs, 180G RAM, seven days, and 16 workers/two replicas per GPU. Its launch-scoped manifest defers exactly 5,361 WeMath samples and activates GQA 3,386 + TextVQA 1,746 + ChartQA 1,785 = 6,917 samples.
- Submitted job 1630 (`4act-seq-wemath-last`) with `afterok:1629` and the same resources. Its fresh claim root will skip all records completed by prior launches and job 1629, process remaining WeMath last, then finalize, analyze, plot, and report the complete result.
- Startup verification: all 16 job-1629 workers claimed one sample; all 16 UIDs were GQA/TextVQA/ChartQA, all eight H100s held two model replicas, and no failure file was present.

## 2026-08-26: Three-replica performance trial rejected

- At the user's request, jobs 1629 and 1630 were canceled. Job 1629 preserved 33 new VQA records (30 GQA, one TextVQA, two ChartQA), bringing the accepted-contract output to 295 records with zero failures.
- Job 1631 tested three replicas per H100 (24 workers) in an isolated output/contract so the 16-worker records were neither overwritten nor mislabeled.
- All 24 replicas loaded successfully at approximately 50–50.5 GiB/H100 and 98–99% utilization with zero failure files.
- At the final matched 551-second window, job 1629 had committed five samples / 4,192 estimated-cost units and job 1631 had committed five samples / 4,151 units. Candidate/baseline cost throughput was 0.990x, below the prospective 1.10x keep threshold.
- Job 1631 was canceled cleanly after 9:11. Its five records remain isolated as negative performance evidence and do not enter the active label output.
- Restored the accepted 16-worker topology in VQA-first job 1632. WeMath-last job 1633 is pending on `afterok:1632` and will run finalization/analysis/reporting after inference.

## 2026-08-26: One-replica trial stopped; user-selected 16-worker fallback

- At the user's request, jobs 1632/1633 were canceled before any additional accepted record committed; the accepted output remained 295 records with zero failures.
- Isolated job 1634 tested one replica per H100 (8 workers). All workers loaded at approximately 16.7 GiB/H100, sampled utilization was 24–33%, and no failure file appeared.
- The user requested fallback before the planned 551-second decision gate. At its clean 440-second cancellation boundary, job 1634 had committed four samples / 3,382 estimated-cost units versus two / 1,682 for the matched job-1629 baseline (2.011x partial cost throughput).
- Because the prospective comparison was stopped early, this is recorded as promising but incomplete evidence, not a final selection result. The four trial records remain isolated and are not merged.
- VQA-first job 1636 restores the user-selected 16-worker topology; WeMath-last job 1637 is pending on `afterok:1636`.

## 2026-08-26: One-replica repeat rejected at the complete gate

- At the user's request, jobs 1636/1637 were canceled and isolated job 1638 repeated the 8-worker/one-replica topology from a fresh output root. Job 1636 committed one additional GQA record before cancellation, bringing the accepted output to 296 records with zero failures.
- All eight job-1638 workers loaded cleanly at approximately 16.7 GiB/H100. The matched 551-second window had zero failures and committed five samples / 4,282 estimated-cost units, versus job 1629's five / 4,192.
- Candidate/baseline cost throughput was 1.021x, below the prospective 1.10x keep threshold. The complete repeat therefore rejects one replica for the active run and shows that job 1634's 2.011x partial 440-second snapshot did not reproduce.
- Job 1638 was canceled after the gate. Seven records completed before cancellation and remain isolated; only the five committed inside the 551-second window enter the performance comparison, and none enter the accepted labels.
- Initial fallback jobs 1639/1640 were immediately replaced because the live default `gpu-normal` QOS does not permit a seven-day request. With the required machine-local `gpu-large` QOS, VQA-first job 1641 started on all eight H100s with 16 workers; WeMath-last/finalization job 1642 is pending on `afterok:1641`.

## 2026-08-27: VQA complete; WeMath paused by user

- Job 1641 completed `0:0` in 17:32:42 with exact coverage of GQA 3,386, TextVQA 1,746, and ChartQA 1,785. No worker failure file was produced.
- Dependent job 1642 started automatically on all eight H100s with 16 workers and processed the remaining WeMath queue in descending estimated-cost order.
- At the user's request, job 1642 was cleanly canceled after 10:39:05. It had committed 742 WeMath2.0 Standard and 339 WeMath2.0 Pro records, for 1,081/5,361 WeMath and 7,998/12,278 total accepted records.
- All 7,998 records have matching checksum sidecars. There are zero nonempty failure files, temporary records, or zero-byte records. The 16 samples active at cancellation remain uncommitted and will be reclaimed from their sample boundaries by a fresh resume launch.
- No resume job is submitted while paused. Resume requires explicit user authorization and the unchanged full wrapper under the machine-local `gpu-large` QOS.
