# Workflow State

- Completed phase (2026-08-29): `plans/four_action_collapse.md` (SHA-256
  `f61f7476ff9a5872f823c7df837e1a2ba21774c83e4efc88f152d2b77d5aceb9`).
  A0 froze all 2,397 W2C mandatory boundaries. A1 job `1700` established local
  online capacity on the fixed overfit pilot (boundary Valid@1 0.9583, W2C
  rescue 0.8958). The full isolated tests then completed in dependency order:
  A2 online job `1725` (`1:06:35`, `0:0`), B1 POLAR training/internal-execution
  job `1729` (`0:14:13`, `0:0`), and matched probe job `1749` (`0:02:17`,
  `0:0`). A2 activated exactly one mandatory-boundary visit for every W2C
  train sample but had zero validation boundary Valid@1 and zero W2C rescue at
  all ten epochs. B1 removed 3,501 exact all-FULL train-C2C routes, excluded 35
  newly empty train samples, and left validation unchanged; its selected epoch
  still predicted/executed all-FULL on 866/866, with zero W2C rescue. The
  2,584-pair matched probe found upfront/online AUROC 0.5764/0.5751 and online
  minus upfront -0.0013 (UID-bootstrap 95% CI [-0.0548, 0.0534]). Thus neither
  isolated fix breaks collapse and current state has no measured advantage;
  neither architecture is selected as final. If newly authorized, the smallest
  discriminator is a matched low-budget persistent targeted-W2C/non-FULL
  supervision comparison across both substrates. External evaluation and any
  additional remedy were not run. Evidence:
  `analysis/4action_collapse/decision_summary.md`; phase memory:
  `workspace/phase_memory/phase_38_four_action_collapse.md`.

- Operating context (2026-08-28): work may proceed concurrently on multiple
  servers. The shared Git branch carries portable code, frozen configs/plans,
  tests, compact reports/checksums, and current workflow/phase decisions.
  Datasets, labels, checkpoints, raw outputs, generated analysis payloads,
  machine-local environment/access files, symlinks, and live scheduler state
  are not implied by Git and must be transferred or verified separately. At
  each handoff, record the exact commit, asset hashes/counts/paths, completed
  output boundary, live-vs-historical job status, failures, and next action in
  the relevant phase memory and experiment log, then push the portable state
  without force-updating shared history. Promoted rule:
  `workspace/decision_log.md` (2026-08-28 cross-server handoff entry).

- Completed/stopped phase (2026-08-29): online state-conditioned four-action router from
  `plans/four_action_train.md`. The checksum-bound GQA/ChartQA/TextVQA
  population has 6,811 samples (5,945 train / 866 validation), 248,804 valid
  routes, and 5,112,442 exact prefix-trie nodes. The 7,621,638-parameter router
  uses actual routed text/visual states, separate READ/WRITE queries, and
  set-valued valid-next-action supervision; Qwen remains frozen. All 476 tests
  pass. The first fail-closed Slurm chain is historical: smoke 1663 failed on
  2026-08-28 because `outputs/four_action_online_router/smoke_v1` already
  existed, and the user explicitly requested cancellation of never-started
  dependent training job 1664 and evaluation job 1665 at 11:00:54 KST. The
  fresh user queue was empty immediately afterward. On 2026-08-29 the user
  authorized the main training and restricted external evaluation. A focused
  red/green regression proved and repaired the DDP smoke-directory race; all
  480 project tests pass and portable fix commit `23ed41c` is pushed. A fresh
  v2 chain was then submitted. Smoke 1684 ran and failed the unchanged loss-
  decrease gate: mean loss rose from 1.414473 to 1.457190. Direct code/runtime
  evidence showed that smoke skipped the frozen training warmup/cosine
  scheduler and instead applied the full `5e-4` learning rate on all four tiny-
  batch steps. A red/green regression now requires smoke and training to share
  the optimizer/scheduler construction; the full project suite passes 481
  tests and portable fix commit `f6a0c42` is pushed. Dead dependents 1685/1686
  were canceled. Fresh eight-H100 smoke 1690 passed every gate with mean loss
  1.414473 -> 0.980423. Training 1691 then completed nine atomic epochs and
  validations. Every epoch had zero W2C rescues; epochs 2--8 executed exactly
  all-FULL, and epoch 9 differed by one IGNORE while preserving C2C at 1.0.
  Training loss improved to 0.571311 and node Valid-Action@1 reached 0.724555,
  but neither produced useful routed behavior. At the user's explicit request,
  jobs 1691 and 1692 were canceled at 14:19:22 KST while epoch 10 was at step
  478/480. Nine checksum-valid checkpoints and 7,794 unique validation rows are
  preserved; epoch 10 left no partial checkpoint, external evaluation never
  started, and `external_v3` is absent. A deterministic follow-up label/sampler
  audit supports severe action/prefix imbalance and missing all-FULL-prefix
  boundary exposure as contributors: the exact sampler never visits the latest
  mandatory deviation boundary for 1,045/2,397 W2C samples. The exact all-FULL
  route is also present in 3,501/3,548 C2C train samples, but removing it alone
  would not repair W2C coverage. No sole cause is established and no new pilot
  is authorized. Final reports:
  `reports/four_action_online_router_early_stop_20260829.md` and
  `reports/four_action_router_collapse_label_audit_20260829.md`.
  Phase memory:
  `workspace/phase_memory/phase_37_online_four_action_router.md`.

- Completed phase (2026-08-29): four-action Image+Question POLAR training on the
  current A6000 server. Two matched ten-epoch runs are frozen: duplicated
  one-hot action BCE and exact complete-valid-set NLL. A deterministic
  machine-local path rebase reproduces 6,811 GQA/ChartQA/TextVQA records
  (5,945 train / 866 validation), 248,804 routes, 6,490 image groups, and 106
  explicit zero-valid exclusions. Both static preflights pass against the
  local Qwen2.5-VL/Qwen3 assets and exact 14,960-row ChartQA/MMMU-Pro/POPE
  population. Never-started four-GPU node07 job `105063` is cancelled. The
  supplied node03 allocation `105067` was released without project execution
  after a device-handle failure made PyTorch report zero CUDA devices.
  Replacement job `105068` finalized the fresh cache and completed both ten-
  epoch training runs (470 steps each; BCE selected epoch 8, NLL epoch 6).
  Both selected checkpoints decode the full 866-record validation set as
  all-FULL with Hit@1 0.585450. Pipeline `105068` then exited `1` before the
  first external preflight sample completed because Qwen2.5-VL position-ID
  construction received CPU `mm_token_type_ids` with GPU prompt tensors. That
  contract is now repaired with a focused regression (9/9 focused tests pass).
  BCE evaluation-only preflight passed all six native-parity and determinism
  fixtures. One monitor-only schema error stopped job `105448` after 32 rows
  had been atomically saved; no evaluator fault or partial file occurred.
  Replacement job `105451` completed both 14,960-record evaluations and both
  merged integrity manifests pass. Direct result parsing shows complete top-1
  collapse for both objectives: one unique all-FULL mask, 418,880/418,880 FULL
  layer decisions, and zero non-FULL decisions. Thus predicted accuracy equals
  unified-FULL accuracy mechanically, with zero corrections or regressions.
  This supports policy collapse but does not by itself identify why training
  produced FULL dominance.
  Other-server job `1662` independently completed its machine-local cache and
  both training processes before the same preflight failure; it is terminal
  historical evidence on this server and its payloads are not implied by Git.
  Evidence: `reports/four_action_polar_tmux2_launch_20260828.md` and
  `reports/four_action_polar_action_collapse_audit_20260829.md`; phase memory:
  `workspace/phase_memory/phase_36_four_action_polar_training.md`.

- Active phase (2026-08-25): executing `plans/4way_labeling_3.md` over the
  frozen five-dataset 12,278-sample / 545,531-route authority. The replacement
  reuses the validated unified executor/runtime/queue but implements a new
  exact early-to-late all-branch W2C converter and preserves C2C mechanically;
  it contains no score calibration or beam search. Superseded job 1609 was
  canceled after 24 pilot records proved its beam gate could not pass (322
  canonical mismatches and 167 Jaccard failures among 1,417 comparisons), and
  dependent job 1610 was canceled before execution. The isolated exact
  implementation passes 432/432 active tests. Eight-H100 smoke job 1611
  completed `0:0` and passed every semantic/integrity gate (8 samples, 61
  routes, 56 replay-valid, five quarantined replay failures, max branch count
  two). At the user's request, full job 1612 was paused by clean cancellation
  on 2026-08-26 after preserving 262 atomic completed records and zero
  failures. The user subsequently authorized resumption, then requested that
  VQA precede WeMath. Job 1628 was cleanly canceled with the committed count
  still 262 and zero failures. Contract-neutral launch-priority job 1629 is
  now running GQA (3,386), TextVQA (1,746), and ChartQA (1,785) first with 16
  workers/eight H100s and preserved 33 VQA records before a user-requested
  three-replica test. Isolated 24-worker job 1631 loaded cleanly but delivered
  only 0.990x matched estimated-cost throughput (5 samples / 4,151 units versus
  5 / 4,192), so it was rejected and canceled. A subsequent isolated
  one-replica job 1634 showed promising 2.011x partial cost throughput at 440
  seconds but was stopped at the user's request before its 551-second gate;
  its four records remain isolated. A fresh one-replica repeat in job 1638
  completed the full 551-second gate but achieved only 1.021x cost throughput
  (5 samples / 4,282 units versus 5 / 4,192), below the prospective 1.10x keep
  threshold, with zero failures. Its seven eventual records remain isolated.
  Job 1641 completed `0:0` in 17:32:42 with all 6,917 VQA records and zero
  failures. Job 1642 then processed WeMath for 10:39:05 before the user
  requested a pause. It was cleanly canceled on 2026-08-27 with 1,081/5,361
  WeMath records complete. The accepted output now contains 7,998 atomic
  checksum-backed records and zero failure, temporary, or zero-byte record
  files. Work is paused pending explicit user authorization; a fresh
  full-wrapper launch will skip completed records and reclaim the 16
  interrupted samples.
  The active scientific contract remains SHA-256
  `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
  Phase memory:
  `workspace/phase_memory/phase_35_exact_sequential_four_action_labels.md`.

- Completed phase (2026-08-24): executed the route-conditioned READ/WRITE
  decomposition in `plans/4way_2.md`. The prerequisite four-action pipeline is
  complete. Audit, the 1,880-row deterministic candidate manifest, arbitrary-
  route unified executor extension, resumable anchor/pilot/full runners,
  mergers, monitor, and aggregate-analysis preparation are implemented; 84
  focused tests pass. Initial eight-H100 anchor job `1576` failed before any
  scientific result because the deterministic CuBLAS workspace variable was
  missing. Resumable eight-H100 job `1578` completed `0:0`, and the local freeze
  retained 1,804 current-correct cached anchors (GQA 1,170; TextVQA 634),
  excluding 76 by the prespecified rule. Both matched 56-sample pilots passed
  all gates; two replicas/GPU achieved 12.183885 valid cells/s (1.414456x one
  replica) at 34,745 MiB peak VRAM/H100 and was selected. Full all-eight-H100
  job `1581` completed `0:0` in 29m13s with 16 workers. The exact local merge
  passed 1,804 samples, 17,262 anchor-OFF positions, 51,786 new cells, 69,048
  action rows, and zero failures. Of OFF positions, 45.65% are individually
  necessary; among those, READ-mediated/WRITE-mediated/either/both shares are
  20.55%/42.88%/9.94%/26.64%. READ-mediated positions are 7.82 layers later on
  average than WRITE-mediated positions (95% CI 7.31--8.31), and FULL-context
  local rescue recalls only 7.30% of route-necessary positions. Final raw-table
  audit and checksums pass; 90 focused tests pass. A bounded compositionality
  pilot is proposed but not authorized or launched. Phase memory:
  `workspace/phase_memory/phase_32_route_conditioned_four_action.md`.

- Completed phase (2026-08-24): executed `plans/4way.md` with M00/M10/M01/M11
  defined entirely inside one unified materialized-mask executor. Native FULL
  and old binary single-OFF are external semantic/drift diagnostics only.
  Readiness supplied 1,235 GQA and 677 TextVQA A+ candidates, 868 matched-budget
  no-correction-found candidates, and 2,110 FULL-correct/ALL-OFF-wrong
  candidates. A current unified-FULL eligibility freeze in completed all-eight-
  GPU job `1505` retained 1,880 primary A+ (1,222 GQA, 658 TextVQA), all 868
  no-correction controls, and 2,084 vision-required controls; 58 candidates
  were excluded because current correctness no longer matched the defining
  FULL-wrong/FULL-correct condition.
  The executor, answer-erosion readout, population trajectory rescues, route/
  Hamming/control analysis, and final report automation pass the focused test
  suite. Unified preflight, 8-example smoke, and 56-example validation passed
  all current semantic gates; one historical cached FULL token mismatch kept
  correctness unchanged and is reported as provenance-only. Job `1497` later
  exposed a current-runtime cohort-boundary mismatch and failed; its dead
  dependents were cancelled. Job `1506` then completed 1,501/1,880 eligible
  primary records before the user authorized a utilization relaunch: live
  profiling showed only 24--30% sampled SM use and 17.5--17.9 GiB memory per
  H100 with one CPU-saturated worker/GPU. The replacement runner uses two
  independent replicas per H100 (16 disjoint workers), preserves all completed
  rows, and passes 62 focused tests. The two-replica ramp passed semantic,
  worker-layout, uniqueness, failure, and Slurm-health gates but produced only
  14.4067 samples/minute, 0.8004x the 18.0 one-replica baseline despite about
  99% GPU utilization. It was recoverably rejected under the prespecified
  1.20x throughput gate after preserving 1,574/1,880 unique primary rows.
  One-replica primary `1557` subsequently passed exact 1,880-row coverage and
  every structural/semantic stage gate. Selection `1562` produced 10,196
  trajectory cells. Downstream runs exposed recoverable audit boundaries:
  11 Control A TextVQA rows have no evaluator-valid correct target and are
  explicitly excluded, leaving 857 analyzable controls; one vision row and one
  rescue row showed only `9.2924e-05` and `5.7618e-05` final readout/direct-
  score drift with tokens, correctness, caches, and intervention semantics
  intact. A `1e-4` BF16 readout-identity diagnostic tolerance and resumable
  mergers preserve those rows. Control A repair `1565` and vision repair
  `1567` are now complete and pass their exact stage gates. Rescue `1569`
  preserved 5,503/10,196 cells before revealing a TextVQA target-identity gate
  error: its fixed baseline phrase trajectory matched that phrase's endpoint
  score exactly, but the intervention state selected a different valid phrase.
  The repaired gate fixes target identity across FULL/suppressed trajectories,
  retains evaluator-best state margins, and reports phrase switching. All 67
  focused tests pass. Eight-H100 resume `1572` completed and its checksum-
  verified 10,196-cell merge passes every coverage, worker, semantic, and
  failure gate. Final CPU analysis/report `1573` completed with exit `0:0`,
  all newly written checksums verify, and the final aggregate/report covers
  all 1,880 primary A+ samples plus both controls. Phase memory:
  `workspace/phase_memory/phase_31_four_action_answer_alignment.md`.

- Completed bounded analysis (2026-08-22): `plans/motivation_check4.md` passed
  all 12,544 raw-record, checksum, anchor, trace, and route-semantics checks.
  Outcome C: matched-prefix V+ minimum ON means are 8.66 GQA, 10.74 TextVQA,
  12.47 ChartQA, and 13.86 WeMath2.0-Pro; the differences remain after native
  visual-token adjustment. Placement profiles are highly similar (exact-min
  cosine 0.982--0.996; min+4 0.994--0.999), with at most a 0.019 normalized-
  centroid gap. V+ prevalence also differs, but is descriptive for the frozen
  selected populations. No new inference, MCTS, training, or routes were run.
  Evidence: `reports/cross_dataset_visual_access_v1.md` and
  `outputs/cross_dataset_visual_access_v1/`.

- Completed bounded analysis (2026-08-22): `plans/motivation_check3.md` passed
  all 4,544 raw-record/hash/anchor/route checks and ended as Outcome D. Across
  428 V+ samples, exact-minimum schedules vary materially (normalized centroid
  0.210--0.794; 1--11 ON segments), but difficulty does not explain the
  variation. Family-paired centroid delta is 0.0053 (95% CI
  [-0.0091, 0.0190]); same-image delta is 0.0041 (CI [-0.0138, 0.0214]); every
  global, amount-adjusted, and axis aggregate crosses zero across exact-min,
  min+2, and min+4. No new inference, MCTS, training, or routes were run.
  Evidence: `reports/wemath2pro_visual_access_placement_v1.md` and
  `outputs/wemath2pro_visual_access_placement_v1/`.

- Completed bounded reanalysis (2026-08-22):
  `plans/motivation_check2.md` passed with all 4,544 exact ALL-OFF/FULL raw
  anchors and hashes verified. Outcome A: 413/841 FULL-correct records are V0;
  V0 prevalence rises from 32.5% to 73.4% across degree 0 to 3 and explains
  83.7–94.9% of degree-level mean declines. The 428-record V+-only rho is
  -0.057 (95% CI [-0.154, 0.037]); paired V+ mean delta is -0.04 (CI
  [-0.63, 0.57]). Among FULL-wrong records, 162 corrections are A0 and 1,263
  are A+. No inference, search, training, or REPEAT was run. Evidence:
  `reports/wemath2pro_visual_dependence_reanalysis_v1.md` and
  `outputs/wemath2pro_visual_dependence_reanalysis_v1/`.

- Completed bounded analysis (2026-08-22): `plans/motivation.md` passed over
  all 4,544 hard-cap-400 WeMath2.0-Pro records using raw-route-derived fields.
  Outcome E: the proposed monotonic visual-depth scaling failed. Among 841
  FULL-correct survivors, minimum ON decreases with degree (rho -0.225,
  clustered 95% CI [-0.291, -0.159]) and is lowest in x-containing strata;
  y/z-only strata remain near base. FULL-wrong correction discovery separately
  declines from 50.0% to 26.7% across degrees 0 to 3. This is axis-specific,
  search-conditioned evidence and does not justify REPEAT or >FULL claims.
  Evidence: `reports/wemath2pro_visual_compute_difficulty_v1.md` and
  `outputs/wemath2pro_visual_compute_difficulty_v1/`.

- New active bounded experiment (2026-08-20): CAP26 versus CAP24
  Image+Question exact-set-NLL, five epochs, on the identical 6,007 train / 872
  validation CAP24-eligible population. All five checkpoints receive actual
  route-conditioned validation execution; selection is accuracy first, then
  lower mean ON, lower validation NLL, and earlier epoch. The selected epoch
  receives the unchanged 22,307-record external evaluation.
- Readiness is PASS (20 focused tests; manifest/config/source checks pass).
  Slurm jobs 102961 (CAP26) and 102960 (CAP24) request one node02 GPU each and
  have ten-hour limits. At submission, two other one-GPU jobs had just occupied
  node02's previously free GPUs, so both jobs are pending without protocol
  changes. Evidence: `outputs/binary_cap_nll5_v1/audits/training_readiness_v1.json`
  and `workspace/phase_memory/phase_26_binary_cap_nll5_execval.md`.

- Active phase (2026-08-20): approved four-way binary duplicated-BCE absolute
  VISUAL_ON-cap sweep under `plans/cap_training.md` (CAP 24/22/20/18).
- The primary comparison uses one identical CAP=18-eligible GQA/TextVQA/ChartQA
  train/validation population; only the surviving max-50 route sets differ.
- All four matched models completed ten epochs. CAP18, CAP22, and CAP24 also
  completed the unchanged 22,307-record external evaluation with integrity
  PASS; CAP20 is the only remaining external run (job 102859 on node07).
- Manifest/readiness gate: PASS. Common population is 5,944 train / 857
  validation; initialization/component hashes are identical across caps.
- Interim nonoverlapping-suite pooled results: ALL-ON accuracy 75.89%; CAP22
  75.89% at mean ON 28.00 (effectively ALL-ON); CAP24 67.10% at mean ON 15.26;
  CAP18 58.23% at mean ON 9.79. CAP20 had completed 6,897/22,307 external rows
  at the latest checkpoint. Do not assign the final cap outcome until it
  completes. Evidence: `outputs/binary_cap_sweep_v1/interim_results_20260820.json`.
- Phase memory: `workspace/phase_memory/phase_25_binary_bce_cap_sweep.md`.

- Repository organization update (2026-08-19): root-level `analysis*` packages
  were migrated to versioned packages under `tools/research_analysis/`, and
  `03_experiments/` was moved to `runs/experiments/`. The scheduler default and
  all live imports/tests were updated; loose generated reports were moved into
  `reports/`. Focused verification preserved the baseline result (60 passing,
  one pre-existing Transformers 5.3 `DynamicCache` failure). Active jobs were
  unaffected. Evidence: `reports/repository_root_cleanup_20260819.md`.

- Label-storage update (2026-08-19): canonical MCTS labels were atomically
  relocated to the dataset tree. GQA/TextVQA/ChartQA now live at
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/` (8,000 records), and
  WeMath2.0-Pro lives at
  `datasets/math_labels/wemath20_pro_mcts_max400_v2/` (4,544 records). Stored
  manifest/audit SHA-256 checks pass. The former `outputs/label_regeneration/v1`
  and `outputs/label_regeneration/wemath2pro_cap400_v2` paths are compatibility
  symlinks; active recovery jobs 101708/101709 remained running. Evidence:
  `reports/mcts_label_relocation_20260819.md`.

- Active detour (2026-08-18): WeMath2.0-Pro conditional greedy route recovery
  for exactly 2,278 current-FULL-wrong records with zero valid hard-cap-400
  MCTS routes (1,104 image groups).
- Plan: `plans/dynamic_mllm_wemath2pro_greedy_recovery_plan.md`. Preserve the
  supplied Phase-1/Phase-2 search semantics while adapting only data/executor/
  scorer interfaces to the current Transformers 5.3.0 verified WeMath runtime.
- Authorization: full bounded G0--G5 recovery search, with at most 50 valid
  masks in the derived training view and no truncation of raw route evidence.
- G0 PASS: exactly 2,278 records / 1,104 image groups; every linked MCTS cache
  checksum passed. G1 PASS: deterministic search/dedup/max-50 tests pass. G2
  PASS: unchanged 5/5 native/binary/cached/new-mask gate.
- Current stage: G3 Phase 1 runs one process per GPU in four global shards:
  Slurm 101708 on node06 (0--1) and 101709 on node07 (2--3).
- First G2 launch stopped before scientific output on a missing deterministic
  CuBLAS setting. Supported repair: `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the
  unchanged rerun passed.
- Next gate: 2,278 atomic Phase-1 records and 22,780 finals with zero unresolved
  errors, then freeze one global Phase-2 budget/request manifest.
- Evidence: `reports/wemath2pro_greedy_recovery_package_audit.md` and
  `workspace/phase_memory/phase_23_wemath_greedy_recovery.md`.

- Active phase completed (2026-08-18): training-set fitting diagnosis for the
  completed Pareto-filtered Image+Question duplicated-BCE and exact-set-NLL
  runs. No new training was performed; all 20 saved checkpoints were evaluated
  read-only on the frozen 6,043 positive training inputs.
- Decision: both objectives exhibit primary training-fit failure. Best train
  Pareto Hit@1 is 18.27% for BCE and 17.95% for NLL, versus the frozen 73.92%
  train BCE-label oracle. Singleton train Hit is only ~24%; doubleton and
  three-plus Hit is approximately zero. Residual multimodality and a late
  generalization gap coexist but are not the primary bottleneck.
- Collapse diagnosis: Pareto filtering removes ALL-ON collapse (essentially
  0% throughout), but creates 34–60% ALL-OFF concentration. Later mask
  diversity increases without coherent Pareto-route learning.
- Validation source: original frozen A6000 histories. A4000 read-only train
  metrics reproduced early decoded validation Hit exactly, but small
  cross-hardware continuous/threshold differences prevented exact validation
  re-evaluation; no tolerance was tuned further.
- Compute placement: node03 remains prohibited. Node04 is allowed again by
  explicit user amendment; CPU work continues to prefer node05.
- Evidence: `reports/binary_pareto_training_fit_analysis.md` and
  `outputs/binary_pareto_v1/training_fit_analysis_v1/`.

- Proposed detour audited (2026-08-17), not executed: use the frozen greedy
  Phase-1/Phase-2 algorithm to search the 2,278 WeMath2.0-Pro Group-D samples
  (1,104 image groups) with no valid cap-400 MCTS route.
- The reproduction package is checksum-clean but not directly runnable for
  WeMath: it targets an old 10K four-benchmark manifest, Transformers 4.57.1,
  old project imports, per-row image caps, and non-WeMath preflight/auditing.
- Provisional protocol: preserve the package, port only its ten greedy orders
  and Phase-2 candidate rules onto the current Transformers 5.3.0 verified
  executor/MathRuler/native-processing contract, then pass a five-sample gate.
  No manifest was frozen, no code adapter implemented, and no GPU search run.
- Evidence: `reports/wemath2pro_greedy_recovery_package_audit.md` and
  `workspace/phase_memory/phase_23_wemath_greedy_recovery.md`.

- Active phase completed (2026-08-17): hard-cap-400 WeMath2.0-Pro MCTS cache
  audit and training-suitability analysis. Exactly 4,544/4,544 eligible records
  pass; eight source rows remain prospectively technical-invalid, and no
  timeout/temp/error record exists.
- WeMath label coverage: current FULL 841 correct / 3,703 wrong; 1,425 current-
  wrong samples have a correcting route. Overall 2,266 samples have at least
  one valid route and 2,278 have none. The raw cache holds 107,671 positive and
  1,550,814 negative evaluated routes.
- Suitability decision: conditionally usable for image-group-disjoint exact
  valid-set NLL and positive/negative ranking; not suitable for unfiltered
  duplicated-route BCE. The ideal weighted BCE label oracle has 13.72% Hit@1
  and mean nearest-valid Hamming 5.10; 94.93% of diagnostic selected routes are
  Pareto-dominated. No training was authorized or run.
- Evidence: `reports/wemath2pro_mcts_training_suitability.md` and
  `outputs/wemath2pro_mcts_label_analysis_v1/analysis_manifest.json`.

- Active phase completed (2026-08-16): label-only MCTS geometry and exact
  duplicated-BCE oracle analysis authorized by `plans/mcts_bce_analysis.md`.
- Decision: **Outcome C + Outcome E**. Raw MCTS and max-50 labels retain high
  route diversity, but the exact weighted per-sample BCE oracle has only 5.93%
  selected-valid Hit@1 (6,507/6,917 invalid hybrids). Separately, 95.83% of
  selected route occurrences are Pareto-dominated; diagnostic Pareto filtering
  raises oracle Hit@1 to 73.41% and lowers mean ON 17.21 -> 9.78.
- Outcome A rejected: raw mean pairwise Hamming is 13.36/28 and radius-4 mean
  effective mode count is 75.14. Outcome B rejected: selected Hamming 13.44 and
  entropy 0.5986 preserve raw geometry. Outcome D is not primary because the
  ideal per-input BCE target is already poor.
- Evidence: `reports/binary_mcts_label_geometry_and_bce_oracle_report.md` and
  `outputs/binary_mcts_label_geometry_v1/analysis_manifest.json` (SHA-256
  `0fd58601b811d5bfdd4785dc5bc804e90c1e90463fe228554ebee1d02257b36c`).
- No training, Qwen inference, label regeneration, selector change, or new
  research method was executed. The phase is closed; any matched
  Pareto-efficient BCE versus complete-route-coherent objective study requires
  explicit approval.

- Active phase (completed 2026-08-15): full10 POLAR-style duplicated-BCE
  comparator authorized by `plans/full_train_polar_bce.md`. Both ten-epoch
  trainings, the joint external preflight, both 22,307-record evaluations, and
  the merged analysis completed with integrity PASS. Exact-set-NLL full10
  artifacts remain unchanged.
- Runtime gate PASS: `outputs/binary_polar/full10_bce/preflight_v1.json`
  (Slurm `101019`) validated physical batch 128 on longest cached images,
  finite BCE/gradients for both modalities, zero frozen-encoder gradients,
  exact repeated logits, and matched shared initialization.
- Completed jobs: Question-only pipeline Slurm `101023` on node02 and
  Image+Question pipeline Slurm `101022` on node07. Best-Hit@1 selection is
  epoch 2 for both modalities. Final report:
  `reports/binary_polar_full10_bce_external_eval.md`.
- The completed BCE report now includes, for every benchmark and suite,
  FULL-normalized router accuracy ratio, fixes, regressions, mean visual-ON
  layers, distinct predicted-mask count, and ON-layer reduction.

- Active phase: full10 best-checkpoint external evaluation is complete with
  integrity PASS; no new predictor or evaluation action is authorized.
- Active source plan: `plans/dynamic_mllm_label_regeneration_plan.md`.
- Plan SHA-256:
  `634f2736d287c647cda7b21755b2ace753db29316ecc9c51523218b498380918`.
- Current step: stop after reporting. Both predictors completed exactly 22,307
  records. Image+Question selected ALL-ON universally; Question-only selected
  non-ALL-ON only 44 times, with no prediction, score, or correctness change.
- External baseline amendment: the first 192 live ALL-ON rows per job exposed
  7 ChartQA disagreements with the historical bundle cache (including one
  correctness change), despite the 9-row preflight passing. The cache is now
  audit-only. Every scientific comparison uses current live ALL-ON: reuse the
  predicted execution when its mask is ALL-ON; otherwise execute a paired live
  ALL-ON baseline. The 192 partial rows were atomically canonicalized with
  original backups and checksummed repair ledgers preserved.
- External jobs `100788`, `100787`, and aggregation job `100790` completed.
  Exact merged-result and analysis checksums pass. Final report:
  `reports/binary_polar_full10_external_eval.md`.
- Frozen execution contract SHA-256:
  `64f525f5d0a4333e1aeae27f41b9055c8da19a9a0fc566ab3c7db270ea37fc7d`.
- Frozen artifacts: `outputs/label_regeneration/v1/`; checksums pass for the
  contract, 8,000-record source manifest, and 15-record smoke manifest.
- Data scope: 8,000 historical-balanced records—4,000 GQA, 2,000 TextVQA,
  2,000 ChartQA; DocVQA excluded.
- Predictor split decision: exact image-group-disjoint 7,000 train / 1,000
  validation, seed 20260809, constructed without current route outcomes. The
  expanded 22,307-record active evaluation replaces the need for an internal
  test: ChartQA/TextVQA core VQA 7,500, MMStar/MMMU 5,807, and POPE 9,000,
  reported as three separate suites after checkpoint freezing. DocVQA is
  excluded by explicit user direction.
- Route semantics: unrestricted complete 28-bit layer-wise visual ON/OFF masks;
  POLAR segments are a derived predictor representation only.
- Label authority: fresh greedy output and benchmark score under the new frozen
  executor. Historical all-ON buckets and old cached masks are metadata or
  proposal-only; old valid/invalid labels cannot be copied.
- Processor contract: native Qwen defaults with no project-specific
  `max_image_tokens` override.
- Pre-extraction gate: exactly 15 smoke records (five per dataset); require
  15/15 binary ALL-ON/native token parity plus exact repeated mixed-route
  tokens/scores. Passed on Slurm job `99740`; report:
  `outputs/label_regeneration/v1/smoke_report_v1.json`.
- Search budget: 200 simulations for current all-ON-correct; 400 default and at
  most 600 adaptively for current all-ON-wrong.
- Raw-cache rule: retain every evaluated positive and negative mask and MCTS
  metadata. Target about 20 diverse positives when found; derive at most 50
  diverse positives for later training, matching POLAR's cap. Both objectives
  use the identical deterministic subset; the raw cache remains untruncated.
- Predictor status: audited and ready for the bounded P10 smoke only. P9 and
  all P10 static/real-encoder readiness checks pass. No optimizer step or real
  predictor training was run. Full mode is programmatically blocked by the
  readiness gate until a post-smoke decision explicitly opens it.
- Preserved evidence: BP-1 showed the old cache is not portable ground truth;
  two cached-positive fixtures became invalid under the repaired target
  executor. The direct binary head and exact valid-set NLL remain prospective
  downstream comparisons, not part of label generation.
- P0 provenance: the project root is not a Git checkout, so deterministic
  hashes of the active MCTS/executor/evaluator/runner source files are frozen
  in the contract.
- Initial P3 allocation: Slurm job `99741` on node07, 4 A6000 GPUs, 32 CPUs,
  240 GB RAM, and four workers. It was cleanly stopped at 2,291 completed
  records for the user-approved scale-up; no zero-byte, temporary, or error
  artifacts remained.
- Completed P3 runtime: Slurm job `99758` on node02, 8 NVIDIA RTX A6000 GPUs,
  64 CPUs, 400 GB RAM, and eight workers. Cross-shard-count resume discovery
  validates and skips completed records from the four-worker layout. All eight
  ranks have produced new contract-bound records with zero errors and no
  duplicates.
- Resume amendment: `outputs/label_regeneration/v1/p3_resume_amendment_v1.json`
  records the allocation change and the runner-only resume-index hash change;
  all other 13 frozen source hashes and the scientific contract are unchanged.
- P4 result: PASS on 4,000 GQA, 2,000 TextVQA, and 2,000 ChartQA records;
  WeMath2.0-Pro was explicitly excluded. Exactly 8,000 terminal records passed
  source binding, frozen contract, 28-bit route, anchor, score/validity,
  search-budget, and trace-linkage checks. Zero missing, unexpected, duplicate,
  invalid, error, temporary, or zero-byte records. Evidence:
  `outputs/label_regeneration/v1/post_generation/cache_audit_v1.json`.
- P5 result: PASS on all 8,000 checksum-reverified records. Current ALL-ON is
  correct for 4,045 and wrong for 3,955; 2,872/3,955 current-wrong records
  (72.62%) have a correcting evaluated route. Valid-route coverage is
  6,917/8,000 for at least one route and 4,877/8,000 for at least 20. Evidence:
  `outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json`
  and `reports/label_regeneration_p5_summary.md`.
- P6 result: PASS on all 528,047 valid masks from 6,917 positive samples and
  36,163,535 exact unordered within-sample route pairs. Sample-balanced means
  are 13.20 ON/OFF transitions and 13.36/28 pairwise Hamming distance; only
  1.02% of valid masks have at most three transitions. Evidence:
  `outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json`
  and `reports/label_regeneration_p6_route_diversity.md`.
- Initial P7 design audit: zero overlap with the original 5,807-record external
  bundle by UID, sample ID, benchmark, exact image SHA-256, normalized
  question/instruction, normalized prompt, and exact image-question pair.
  Evidence:
  `outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json`,
  `outputs/label_regeneration/v1/post_generation/predictor_split_design_audit_v1.json`,
  and `reports/binary_router_p7_split_and_external_eval_audit.md`.
- Expanded evaluation audit: the updated bundle passes full image verification.
  The active selection contains 7,500 ChartQA/TextVQA core-VQA, 5,807 prior
  multiple-choice, and 9,000 POPE records. ChartQA test, TextVQA validation,
  MMStar, MMMU, and MMMU-Pro have zero exact MCTS image overlap. DocVQA is
  excluded. POPE has one shared image repeated in 18 records; report full 9,000
  plus a pre-specified 8,982-record image-disjoint sensitivity. Evidence:
  `outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json`
  and `reports/binary_router_expanded_eval_suite_audit.md`.
- P7 split result: PASS. All 8,000 source records are frozen into exactly 7,000
  train and 1,000 validation records with zero cross-split image groups. The
  historical validation strata are exact: GQA 250 correct/250 wrong, TextVQA
  125/125, and ChartQA 125/125. Current outcomes were joined only after
  assignment and were not selection inputs. Manifest SHA-256:
  `4d12bf427f08b0cc55d21c82bf7eaac7d19d283dc514ffd4f59894d6faf1bd1a`.
  Evidence:
  `outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json`
  and `reports/label_regeneration_p7_predictor_split.md`.
- P8 result: PASS. The unchanged raw cache yielded 8,000 single-best and
  valid-set sample rows, 237,802 selected valid routes from 528,047 raw valid
  routes, 2,642,998 positive/negative ranking rows, and 237,802 exactly
  reconstructing POLAR-segment rows. There are 6,917 positive and 1,083
  zero-positive samples; 3,616 samples required the deterministic max-50 cap.
  Both predictor losses consume the identical selected route set with equal
  within-sample weights. Generation and independent streaming audits pass:
  `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json`
  and `derived_supervision_verification_v1.json`.
- Evaluation adapter: reuse bundle manifest/input/scorer/reporting contracts,
  but replace SW31, forced K=8, and admission with each trained question-only
  predictor's static 28-bit mask and verified executor. Do not aggregate core
  VQA, multiple choice, and POPE into one overall accuracy.
- P9 result: PASS. The final audit binds all 8,000 raw records through the P4
  checksum index and freezes 50 primary/code/provenance files, the final report,
  and a 53-entry checksum ledger. Independent checksum verification passed
  53/53. Evidence: `reports/label_generation_report.md` and
  `outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json`.
- Next bounded action: with explicit approval, run only the matched P10 smoke
  for duplicated BCE versus exact set-NLL. Do not begin full training or
  external evaluation automatically.
- P10 readiness audit: PASS. The active config is
  `configs/binary_polar_loss_comparison_v2.yaml`; the smoke freezes 300 positive
  train and 150 positive validation records (balanced 100/50 per dataset), two
  epochs, and 18 actual-execution records per objective. The real Qwen3 BF16
  encoder preflight produced finite losses and finite gradients for both
  objectives from the same initialization, with zero encoder gradients and
  zero optimizer steps. Evidence:
  `outputs/binary_polar/preflight/p10_readiness_gate_v1.json` and
  `reports/binary_polar_p10_readiness_final.md`.
- Updated at: 2026-08-12
- Concurrent action: We-Math2.0-Pro all-sample binary-route extraction is
  approved under `plans/dynamic_mllm_wemath2pro_label_extraction_plan.md`.
  The benchmark adapter and focused tests pass; the 4,552-record manifest is
  frozen before a five-record smoke. On smoke pass, launch node06 with 8 GPUs,
  96 CPUs, and 240 GB RAM without cancelling job `99758`.
- Combined scheduler ceiling for the two explicitly concurrent jobs: 16 GPUs.
- We-Math validity amendment: all 4,552 records remain in the frozen inventory;
  the exact eight records with an empty question and/or answer are marked
  technical-invalid, leaving 4,544 records in the MCTS manifest. No We-Math
  GPU outcome was inspected while applying the rule. The original node02 job
  `99758` remains unchanged.
- We-Math frozen contract SHA-256:
  `96b2c632ebc6e020c607b3d9a0eddd2a29f7aff1912f5219327ae96a507c3a50`.
- We-Math smoke status: passed 5/5 ALL-ON/native generated-token parity and all
  repeated mixed-mask token/score checks. Evidence:
  `outputs/label_regeneration/wemath2pro_v1/smoke_report_v1.json`.
- Superseded We-Math sweep: Slurm job `99850` initially ran on node06 with
  eight workers and produced 1,156 retained complete records before the
  scoring-stall repair below.
- We-Math scoring-stall repair (2026-08-12): old job `99850` was cancelled only
  after a checksum-bound audit preserved 1,156 complete records. MathRuler is
  now bounded at five seconds; timeout is explicitly recorded and scored
  conservatively incorrect. Amended contract SHA-256:
  `fc4a1df38925d20816770b861989b87d119bcdbf13b3bdff26a89b7abc90d485`.
  Active resumable job `100398` uses the six currently available node06 GPUs,
  72 CPUs, and 180 GB RAM. All six ranks completed a new record with zero
  errors; the two previously stalled samples each completed 600/600 simulations
  with zero scorer-timeout flags. The old stall's exact cause therefore remains
  unknown, while the bounded scorer prevents one important recurrence class.
- We-Math cap amendment (2026-08-13): the user removed the 600-simulation
  extension after only 25/528 completed extensions found a correction after
  simulation 400. The new hard cap is 400. Only terminal 200/400 predecessor
  records may be reused; all 600-simulation records must be rerun. The planned
  replacement was audited into
  `outputs/label_regeneration/wemath2pro_cap400_v2/`: 640 terminal records were
  retained (207 at 200 simulations and 433 at 400), 529 post-cap records were
  excluded, and 3,904 remain. Slurm job `100407` is now running seven workers
  on node06 with 7 GPUs, 84 CPUs, and 197 GiB. The active cap-400 contract is
  `80c7ea4ca2ca9df091696290dc644a4092508337f89cf85ecc5b849a0f4092c7`.
  The 197G request is the largest whole-GiB allocation that fits beside the
  existing 48,000 MiB job on node06; it does not change model/search semantics.
  Live startup validation passed on the first three new terminal records: all
  completed 400/400 under the active contract, with no extension, errors,
  temporary files, or overlap with retained UIDs.
- Portable MCTS handoff: `handoff/binary_visual_mcts_reproduction_v1/`
  contains the verified executor/search/evaluator sources, portable dataset
  manifest and contract tooling, Slurm template, detailed runbook, six passing
  CPU-only tests, and a complete SHA-256 inventory. Packaging did not alter
  either active extraction job or its frozen contract.
- Updated at: 2026-08-11
- Predictor loss-comparison implementation (2026-08-12): implemented matched
  duplicated-route BCE and exact valid-set NLL paths without training. Stable
  complete-mask `logsigmoid`/`logsumexp`, variable-K padding, equal route
  weights, duplicate rejection, 50-route-cap enforcement, split-group leakage
  checks, matched unique-input batching, route microbatching, and predictor
  initialization hashing are in place.
- Objective sanity: pass. Single-route and padding errors are `0.0`; exact
  set-NLL reduced the contradictory-set loss `2.7723 -> 0.7028` and selected a
  coherent valid mode; duplicated BCE converged to bit marginals. Frozen
  encoder gradients are zero and all 21 predictor parameter tensors have
  finite gradients. Evidence:
  `outputs/binary_polar/preflight/loss_comparison_sanity_v1.json`.
- Training status: not started. The regenerated-label P9 cache/split/derived-
  view gates now pass. The next action remains a separately approved bounded
  matched smoke; do not launch full training.
- Active phase memory:
  `workspace/phase_memory/phase_14_binary_polar_loss_comparison.md`.
- Updated at: 2026-08-12
- P10 matched smoke (2026-08-13): completed two epochs on frozen 300-train and
  150-validation positives per objective, followed by 18 actual executions
  each. Exact set-NLL improved validation Hit@1 `0.1333 -> 0.5733` and
  execution accuracy `0.2222 -> 0.5000` versus duplicated BCE. However, every
  BCE execution decoded ALL-OFF and every exact-set execution decoded ALL-ON;
  exact set-NLL matched FULL accuracy with zero compute reduction. A constant
  policy audit reproduces the route metrics, so the smoke advantage is
  predominantly a constant ALL-ON prior rather than demonstrated conditional
  routing. Full training remains unexecuted pending explicit user direction.
  Evidence: `reports/binary_polar_p10_smoke_results.md`.
- P11 result (2026-08-13): complete, Outcome C. Validation label geometry has
  58.12% ALL-ON coverage and 57.89% ALL-ON-plus-cheaper-valid prevalence.
  Weighted exact set-NLL improves aligned versus shuffled set-NLL (`14.8699`
  versus `15.5089`) but decodes ALL-ON on 147/150 route-validation records and
  57/60 bounded executions. Execution accuracy is 50%, identical to FULL;
  W→C=0, C→W=0, and the three non-FULL masks remain wrong. Full training was
  not launched. Evidence: `reports/binary_polar_p11_results.md`.
- P12 result (2026-08-13): complete, Outcome B. All 237,802 selected route
  occurrences pass exact canonical maximal-run round trip, but label geometry
  is weakly segment-compressible (mean/median 14.11/14 segments). The selected
  structured checkpoint is 150/150 ALL-ON on route validation and 60/60
  ALL-ON in actual execution. Aligned structured set-NLL remains better than
  shuffled (`17.4918` versus `18.1939`), but decoded metrics are identical.
  Actual accuracy is 50%, W→C=0, C→W=0, and compute reduction is zero. Full
  structured-head training was not launched. Evidence:
  `reports/binary_polar_p12_results.md`.
- Active phase memory:
  `workspace/phase_memory/phase_17_p13_multimodal_input_isolation.md`.
- P13 result (2026-08-13): complete, Outcome B. The experiment reused the
  direct P11 head, exact valid-set NLL, P11 weighting, 300/150 identities,
  two-epoch budget, and seed, changing only visible predictor input among
  Question, Image, and Image+Question. The cache contains 502 frozen native
  projected-visual feature records (500 image groups), with no answer/outcome
  leakage and all preflight/checksum gates passing. Selected Image+Question
  aligned set-NLL is `14.4944`, versus Question-only `14.8699`; shuffling its
  image worsens NLL to `14.8748`. However, Image+Question decodes ALL-ON on
  150/150, exactly matching the constant baseline's 57.33% Hit@1, 3.693
  nearest-valid Hamming, and 28 mean ON layers. The prospective execution gate
  failed, so no P13 Qwen route execution or full training ran. Evidence:
  `reports/binary_polar_p13_results.md` and
  `outputs/binary_polar/p13/analysis_manifest_v1.json`.
- Full10 result (2026-08-13): complete. Both runs used 6,043 positive train
  and 874 positive validation inputs, exact set-NLL, max-50 route sets, batch
  128, AdamW `5e-4`, cosine schedule, and 10 epochs. All 20 checkpoints and
  hashes are preserved. Best-Hit@1 is epoch 2 for Question-only and epoch 4
  for Image+Question; both equal constant ALL-ON at `58.12%`. Epoch-10
  diversity rises to 122/64 unique masks but Hit@1 falls to
  `55.03%`/`55.84%`. Actual frozen-60 best-checkpoint executions are both
  `50%`, W→C=0, C→W=0. Question-only epoch 10 has two uncached ChartQA
  corrections but is not the selected checkpoint. Do not admit the direct
  predictor to external evaluation. Evidence:
  `reports/binary_polar_full10_polar_matched_results.md` and
  `outputs/binary_polar/full10/`.
- Active phase memory:
  `workspace/phase_memory/phase_19_full10_external_evaluation.md`.
- Updated at: 2026-08-13
