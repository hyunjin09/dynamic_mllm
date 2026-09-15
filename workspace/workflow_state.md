# Workflow State

## Stopped Phase 89: 3B greedy corpus audit and current-server replay

**STOPPED by user on2026-09-15 11:41:56 KST.** Replay2994 and dependent
report2957 cancelled; saved outputs preserved. No automatic restart authorized.
V2 is incomplete. The execution details below are historical.
Transfer entry: `handoff/phase89_server_transfer/README.md`; exact durable
frontier and record verification: `progress_snapshot.json` in that directory.

User authorized V2; latest steering requests3 processes/GPU, brief monitoring
and a measured ETA, then user notification when jobs finish.
Output: `analysis/3b_greedy_route_corpus_replay_audit_v2/`.
Verified corpus:10,000 samples,9,273 image groups,3,907,717 unique routes.
Initial gate2939 failed31/32; diagnostic2941 supported a native FULL SDPA mask
repair. Repaired gate2942 PASSED32/32 and all72 recorded failing-anchor layer
boundaries agree numerically. Canonical package and old attempts are preserved.
Dense2954 and CPU certification2955 COMPLETED all10,000 samples. Current
dense5702C/4298W;220 labels changed(125C→W/95W→C). Routed2994 RUNNING on8GPUs
with24workers since2026-09-15 11:23:14 KST;2957 reporting now depends on2994.
Original2956 was deliberately stopped with720,540 records/1,834 complete
samples preserved. Original inference contracts unchanged; scheduling overlay
`concurrency_v1/contract.json` binds new24worker execution. Gate32/32 and all
9,323 pilot routes across24 saved samples match exactly. Pilot34.5routes/s;
initial expensive DocVQA production17.5routes/s. Eight new production samples
independently verify,8,414new routes saved with no failures at11:36:42 KST.
Preliminary remaining replay ETA25–55hours. Brief monitoring complete;2994
continues, and the user will notify us when finished. No geometry.
Memory: `workspace/phase_memory/phase_89_3b_greedy_route_replay_audit.md`.
Phase88 source completion remains unverified; Phase85 remains stopped.

## Active Phase 88: Benchmark-calibrated fixed READ/WRITE schedules

User authorized `plans/benchmark_calibrated_fixed_read_write_schedule_plan.md` on2026-09-14. Stage1 is disabled; no learning. Calibration-only independent bit selection, whole-group held-out evaluation, global and20random controls are required. Metadata split and reviewed contract frozen. Nine-case Dense parity smoke and all20,471 regenerated Dense rows passed; CAL-only action smoke and the complete READ sweep passed. Both calibration sweeps passed and all schedules are frozen. M1/M2/M3 and global are complete on all19,452TEST UIDs; four-GPU random controls are active (15,703/19,452 complete at 2026-09-14T19:42:52+09:00). Cross-server snapshot/runbook: `handoff/phase88_server_transfer/README.md`. Final interpretation/reporting remains pending. A missing-q edge case was repaired without excluding any UID. Current memory: `workspace/phase_memory/phase_88_benchmark_fixed_rw_schedule.md`. No top2 or old-search restart.


## Completed Phase 87: WRITE harm structure and learnability

Completed `plans/write_harm_structure_learnability_plan.md` on 2026-09-14 KST: 15,185 dense states / 1,413 UIDs, 360 local fits, 57,538 propagation records and 1,620 propagation fits. W-LOCAL-WEAK → qualified W-PROP-C / WRITE-S4. Full local F_ALL rho .04752 / AUROC .48188; exact 6,044-state H8-common delta rises from rho .04119 to .09732 (gain .05613 [.02670,.08523]), AUROC .53872, with no useful high-precision subset. Dense-W agrees. All reports and 12 figures complete; final integrity record: `analysis/write_harm_structure_learnability/final_verification.json`. No further experiment is authorized. One unexecuted recommendation: close tested local/H≤8 routing and reframe unresolved nonlocal intervention structure; no intrinsic-nonlocality or saturation claim. Source/LODO/routed/external work was gated off. Phase85 remains stopped. Memory: `workspace/phase_memory/phase_87_write_harm_structure_learnability.md`; entry report: `analysis/write_harm_structure_learnability/summaries/final_write_characterization.md`.

- Completed/stopped Phase 86 (2026-09-13): full 15,185-state / 1,413-UID frozen Stage1 READ branch-critic diagnostic, including all layer27 states. BC-C qualified: ON/OFF AUROC 0.6728/0.5949, balanced flip preference 47.77%, discordant AUROC 0.4684. Offline policy 30 W→C / 6 C→W, Net+24 [12,36]. Fixed-target cross-scores exclude attributing the branch AUROC gap to score collapse alone. Fresh smoke, full parity, four focused tests, independent result reconstruction, 850 artifact hashes and 1,413 cache hashes pass; manifest `9c033a70...427941c`. Phase85 remains stopped with evidence preserved. Exactly one unexecuted recommendation: paired branch-risk calibration/ranking. Memory: `workspace/phase_memory/phase_86_read_branch_critic.md`; report: `analysis/read_counterfactual_stage1_branch_critic/summaries/branch_critic_summary.md`.

- PRELIMINARY Phase 85 interim (2026-09-13 21:06 KST): exact common 612 completed W UIDs; 695 W pending and excluded. Completion composition is biased. Beam8 ANY/SELECTED@128 = 50.49%/39.71%, gaining 9.80/8.50 pp from 64; no overall saturation. Independent review supports finishing the original population and frozen B256 audit; B512 remains conditional. At that interim timestamp jobs were unchanged; they were subsequently stopped by the user. Report: `analysis/read_only_bounded_planning/interim/PRELIMINARY_20260913_210600/PRELIMINARY_report.md`.

- Historical Phase 85 launch (stopped 2026-09-13 by new user plan): user authorized `plans/read_only_bounded_planning_search_plan.md`, including all four occupied GPUs. Exact population is1307W/106C; cached single-READ-off rescue is223W; exact Hamming2 projection81058 new routes passes250000 cap. Independent review repairs cache binding, adaptive denominator, and enumeration reporting. Execution contract `67d2c6a1...fb0861`, supplemental analysis contract `8d143e37...5e9e39`;42 distinct focused tests pass. All40 smoke UIDs passed; full base128 and gated Hamming2 launched on GPUs0/1/2/3 but were subsequently stopped. No automatic continuation is active. State: `workspace/phase_memory/phase_85_read_only_bounded_planning.md`; evidence: `analysis/read_only_bounded_planning/`.

- Completed/stopped Phase 84 (2026-09-12): froze contract
  `3a0d2251...affebb6a`, passed fresh-cache repeat, swapped-order H8, exact
  Phase-82 H1, canonical ON, action-trace, global-census, and cache-readback
  gates, then extracted 47,133 paired horizon states / 94,266 branches from all
  15,185 dense states. All 2,115 fixed five-fold/three-seed fits completed.
  On the 6,916-state H8-common population, pooled DELTA Spearman is
  `0.0756/0.0688/0.0646/0.1097` and harmful AUROC is
  `0.5316/0.5287/0.5314/0.5483` for H1/H2/H4/H8. The H8-vs-H1 gains are only
  `+0.0340` Spearman and `+0.0167` AUROC, with image-group bootstrap lower
  bounds `-0.0009/-0.0020`; precision@10% is 0.5462 versus prevalence 0.4926.
  Token H8 Spearman is 0.0854, pooled metrics are non-monotone, and no horizon
  passes the prospective materiality/high-precision gates, yielding
  **H-READ-D**. All 51 declared repository files, 12 external caches, and
  2,115 checkpoint records verify under manifest `1f9005bf...35961`. No routed
  confirmation, WRITE study, router, search, deployment, or external
  evaluation ran. Evidence:
  `analysis/read_harm_short_horizon_propagation/`; phase memory:
  `workspace/phase_memory/phase_84_read_short_horizon_propagation.md`.

- Completed/stopped Phase 83 (2026-09-11): froze contract
  `eaaef860...e4d6`, passed exact state/post-state/branch/feature-repeat and
  full-query causal-SDPA validation, then extracted complete F1-F7 READ
  features for 15,185 dense states over 1,413 UIDs and 35,565 routed states on
  four GPUs. READ signs are 7,285 harmful and 7,900 beneficial. The fixed
  structure gates classify harm as **R-STRUCT-B** (mostly isolated); only
  1/34 matched feature effects excludes zero and the matched flip probe AUROC
  is 0.4186, giving **R-MECH-B**. The dense-selected F_ALL/MLP reaches OOF
  Spearman 0.0697 and harmful AUROC 0.5338, with weak source/LODO transfer and
  no material external-transfer gate, giving **R-LEARN-C**. Routed OOF
  Spearman 0.1623 is selection-qualified secondary evidence and does not
  override the dense-primary result. All 75 artifact hashes verify under
  manifest `35524bf3...05ab`. The sole unexecuted recommendation is a
  short-horizon READ effect-propagation/planning audit; no WRITE study,
  deployment router, external counterfactual, or follow-up ran. Evidence:
  `analysis/read_harm_structure_learnability/`; phase memory:
  `workspace/phase_memory/phase_83_read_harm_structure_learnability.md`.

- Completed/stopped Phase 82 (2026-09-10): froze one-step counterfactual-effect
  contract `a70e921a...2705`, passed stratified dense/routed deterministic and
  exact FULL-to-canonical post-state parity, then extracted all 15,185 dense
  states / 45,555 branches and 35,565 routed states / 106,695 branches. All
  570 primary OOF, 60 routed-secondary OOF, and 60 Dense-to-routed transfer
  tasks completed under the inherited five-fold image-group-disjoint registry.
  Dense OOF READ MLP Spearman is `0.0626/0.0533/0.0552/0.0523/0.0773/0.0568`
  for PRE/FULL/OFF/PAIR/DELTA/PAIR+DELTA; token is `0.0694`. WRITE is
  `0.0354/0.0364/0.0399/0.0421/0.0380/0.0399`; token is `0.0383`. Reported
  bootstrap gains include zero, harmful-flip ranking is below chance, and the
  modest routed token OOF (`0.1290/0.0726`) does not rescue dense-primary or
  Dense-to-routed identifiability. READ, WRITE, and joint decisions are
  **Case D**. All 96 declared repository artifact hashes and 52 focused tests
  verify; manifest `3a0eb918...143e`. The one unexecuted recommendation is a
  two-layer / short-horizon counterfactual-identifiability audit. No deployment,
  external evaluation, router retraining, MCTS, or follow-up ran. Evidence:
  `analysis/dense_failure_stage2/counterfactual_effect_identifiability/`;
  phase memory: `workspace/phase_memory/phase_82_counterfactual_effect_identifiability.md`.

- Completed/stopped Phase 81 (2026-09-09): froze Step-D contract
  `efa342f0...e571cf`, refit all 15 Stage-1/Stage-2 M0-X/M1/M3 models on the
  full internal corpora, and passed a bound seven-task/three-trigger-family
  smoke. Four direct GPUs recomputed all 19,960 external Dense rows and
  558,880 layer states with exact Phase-69 token/score/correctness/image and
  robust-trigger parity. The strict P90 domain is exactly 901 UIDs, 8,442
  post-trigger states, and zero POPE triggers. All prediction hashes were
  frozen before labels; all 33,768 four-action branches then passed live-state,
  FULL token/score/correctness/q, utility-algebra, and global-census checks.
  External Stage-1 M3 AUROC is `0.4998/0.6794/0.5399/0.5582` for
  ChartQA/TextVQA/MMMU-Pro/POPE (macro `0.5693`, pooled `0.6980`), yielding
  category **D1-C**. Stage-2 READ rho is `0.0848/0.1095/0.0246` and WRITE rho
  `0.0335/0.0158/0.0204` on the three triggered families, yielding **D2-A**.
  The result does not support robust external failure prediction or
  current-state local treatment prediction. All 72 artifact hashes and 51
  focused/inherited tests verify. No deployment, redesign, or next experiment
  ran. Evidence: `analysis/predictability_generalization/stepD_external_transfer/`;
  phase memory: `workspace/phase_memory/phase_81_predictability_stepD_external_transfer.md`.

- Completed/stopped Phase 80 (2026-09-09): froze label-blind Qwen3 question
  encoder contract `d574a8af...30a9` and full Step-C contract
  `fafd6442...07ecd`, then completed all 300/300 unchanged M0/M1/M3 fits on
  four direct GPUs across 22 semantic-cluster, source, LODO, and pairwise
  holdouts. Stage-1 M3 Q1/Q5 AUROC is `0.7828/0.8012`; the +0.0183 delta has
  95% image-group-bootstrap CI `[-0.0050, 0.0423]`. Concatenated K=100
  cluster-OOD AUROC is `0.7725` versus ID `0.7869`, but pooled
  Historical→Canonical/Canonical→Historical AUROC collapses to
  `0.4342/0.5563`, and LODO is only `0.5508-0.6043`, with severe threshold
  calibration drift for held-out ChartQA/GQA. Stage-2 cluster-OOD READ/WRITE
  Spearman is `0.0453/0.0164` and remains near chance under source/LODO.
  Therefore the plan-defined result is **S1-C source-specific signal / S2-A
  weak everywhere**, not semantic-neighbor dependence. Six sparse Stage-2
  within-dataset source cells were prospectively unsupported; four OOF kNN
  states were explicitly non-estimable at exact `k=5`. All 5,000-draw primary
  intervals and 5,000/5,000 Q1/Q5 contrast draws completed. Final artifact
  manifest `ca340f26...da875` verifies. `READY_FOR_STEP_D = true` is procedural
  only; no Step D, external evaluation, or redesign ran. Evidence:
  `analysis/predictability_generalization/stepC_generalization/`; phase memory:
  `workspace/phase_memory/phase_80_predictability_stepC_generalization.md`.

- Completed/stopped Phase 79 (2026-09-09): under frozen parent contract
  `792cc760...21bba4`, all four direct GPUs completed 600/600 five-fold OOF
  training tasks and 140/140 state-regime transfers using the exact Phase-78
  measurements. Stage-1 evaluated 10,399 UIDs/9,982 image groups/291,172
  states with zero group leakage: nuisance/linear/MLP/current-head AUROC is
  `0.6814/0.7665/0.7874/0.7869`, and current-head AUROC peaks at layer 20
  (`0.8257`). Primary dense Stage-2 READ/WRITE joint-router Spearman is only
  `0.0416/0.0347`, harmful AUROC `0.5193/0.5115`, with top-10% harmful
  precision near natural prevalence and no z_R/z_W specialization. Routed and
  bidirectional state-regime transfer results are likewise weak and secondary.
  All eight 5,000-draw image-group bootstrap intervals are valid, all final
  artifact hashes verify, and the evidence category is Case D: Stage-1 strong,
  Stage-2 weak. Two aggregation-only defects were repaired under chained child
  contract `fb211075...5d3d1` without changing the frozen parent code, fits,
  predictions, labels, folds, or metrics. `READY_FOR_STEP_C = true` is only a
  procedural status; Step C was not executed. Evidence:
  `analysis/predictability_generalization/stepB_id_learnability/`; phase memory:
  `workspace/phase_memory/phase_79_predictability_stepB_id_learnability.md`.

- Completed/stopped Phase 78 (2026-09-08): froze contract
  `6103b9b9...4613d` and constructed the full predictability Step-A
  measurement corpus without training. Stage 1 contains all 10,399 internal
  UIDs across 9,982 image groups and 291,172 `(sample, layer)` states; 12 live
  dense replays exactly match tokens, LMMS correctness, image hashes, and all
  three 28-layer BF16 feature blocks. Strict P90 triggers 1,413 UIDs. The
  primary census measured all 15,185 dense-origin states / 60,740 four-action
  branches; the secondary census measured 35,565 unique routed states /
  142,260 branches after deduplicating 69,178 route occurrences. Controlled
  utility has broad positive and negative support; primary local rescue and
  regression state counts are 1,048 and 83. Action semantics, FULL parity,
  repeatability, state/anchor parity, completeness, and utility algebra all
  pass. All 41 compact artifact hashes verify and all 1,413 external dense
  state files are present. `READY_FOR_STEP_B = true`, but no predictability
  claim is made and no Step-B probe/router training started. Evidence:
  `analysis/predictability_generalization/stepA_measurement/`; phase memory:
  `workspace/phase_memory/phase_78_predictability_stepA_measurement.md`.

- Completed/stopped Phase 77 (2026-09-07): under contract
  `3771fd97...a69be0f`, used the exact Phase-76 full-refit checkpoint on all
  569 training-side UIDs and the frozen internal-dev checkpoint on the original
  115 image-group-disjoint dev UIDs. All 35,565 unique routed states and 69,178
  route occurrences over 4,948 replay-valid programs were audited. Seen
  FULL/non-FULL/first-nonFULL top-1 recall is 99.47%/1.83%/2.54%; the selected
  highest-responsibility routes reach only 11.02% first-nonFULL recall. Training
  Dense-W R0/R1/R2 final-correct counts are `42/43/264` of 463: exact release
  on the expert corrective state does not help, while forcing the first
  corrective action yields a +47.73-point jump. Dense-C preservation is
  106/106. Held-out first-nonFULL recall is 1.79% and W success is 8/93, so
  generalization is weak but cannot be primary because seen fit is already
  weak. The fixed rule classifies the earliest bottleneck as
  `objective_action_learning`; on-policy relabeling is not justified as the
  first response. All seen/held-out UIDs, R0/R1/R2 rows, release-state/image
  hashes, required files, and artifact hashes verify; 21 focused/inherited
  tests pass. The one unexecuted recommendation is a bounded refit retaining
  the trajectory marginal plus a fixed-weight first-nonFULL auxiliary CE term.
  No training, search, relabeling, Stage-1 change, or external evaluation ran.
  Evidence: `analysis/dense_failure_stage2/teacher_forced_free_run_audit/`;
  phase memory:
  `workspace/phase_memory/phase_77_teacher_forced_free_run_audit.md`.

- Completed/stopped Phase 76 (2026-09-07): under contract
  `00639e7e...5f6a`, exact-replayed and cached all 4,948 retained successful
  programs over 569 UIDs as 35,565 unique closed-loop prefix states (69,178
  route occurrences), with zero quarantines and exact numerical/gradient
  parity for the per-UID trajectory-set marginal loss. The image-group-
  disjoint internal split selected epoch 1/20 at dev loss 0.425405, followed
  by a one-epoch full 569-UID refit. All 19,960 established ChartQA/TextVQA/
  MMMU-Pro/POPE rows completed with exact Dense/Stage-1 baseline and feedback-
  trace parity. Closed-loop W-to-C/C-to-W/net is `0/8/-8`, accuracy 0.780160,
  versus Dense 0.780561, Open-loop Program `3/8/-5`, and Sequential-A
  `3/19/-16`. Only 39/496 triggered Dense-W samples receive non-FULL and none
  rescue; median best-route geometric action probability is 0.7269. The fixed
  diagnostic rule therefore points next to on-policy state-distribution shift,
  but no follow-on is authorized or run. All 58 declared artifacts, 569 state
  files, and 19,960 evaluation rows verify. Evidence:
  `analysis/dense_failure_stage2/closed_loop_trajectory_set/`; phase memory:
  `workspace/phase_memory/phase_76_closed_loop_trajectory_set.md`.

- Completed/stopped Phase 75 (2026-09-06): froze contract
  `e274d24c...06c7fe7`, reproduced every one of 901 Phase-74 triggered top-1
  paths exactly (3 W-to-C, 8 C-to-W, net -5; zero beam-score error), then used
  four direct GPUs to execute all 6,944 unique frozen beam programs. W-to-C
  availability rises `3/10/24/33` at ranks `1/2/4/8`, giving 30 ranking
  failures, but 463/493 top-1 W failures (93.91%) remain generation failures.
  TextVQA contains 19 beam rescues versus zero top-1; MMMU-Pro contains 12
  versus two. All eight C-to-W regressions have a correct lower-ranked
  candidate; all-FULL is present in six of those beams and absent from two.
  Mean W beam support is 7.80 unique programs, so high all-FULL top-1 usage is
  conservatism rather than complete beam collapse. All 32 artifact hashes and
  65 focused/inherited tests pass. The one unexecuted recommendation is a
  minimal trigger-state representation-enrichment experiment; no retraining,
  reranking, search, or follow-on experiment ran. Evidence:
  `analysis/dense_failure_stage2/program_beam_oracle_audit/`; phase memory:
  `workspace/phase_memory/phase_75_program_beam_oracle_audit.md`.

- Completed/stopped Phase 74 (2026-09-06): built and exact-replayed 4,948
  eligible complete P90 suffix programs over 569 UIDs/560 image groups under
  contract `0c1696ed...597a49`, with all-UID cached/live trigger-state parity.
  A small autoregressive program decoder selected epoch 2 on an image-group-
  disjoint internal development split (beam-8 exact match 24/115, 20.87%) and
  was reinitialized and refit on the full corpus for exactly two epochs. The
  full paired four-family evaluation completed all 19,960 reference rows with
  exact Phase-69 Dense/Stage-1 parity. Program W-to-C/C-to-W/net is `3/8/-5`
  versus Sequential-A `3/19/-16`: it prevents 11 prior regressions but adds no
  rescues and remains 5 correct answers below Dense (accuracy 0.780311 versus
  0.780561). This is outcome B, not a deployment winner. All 65 declared
  artifact hashes and 60 focused/inherited tests pass. The one unexecuted
  recommendation is to retain program supervision as a conservative
  formulation while treating corrective-treatment transfer as unresolved; no
  follow-on experiment ran. Evidence:
  `analysis/dense_failure_stage2/polar_suffix_program/`; phase memory:
  `workspace/phase_memory/phase_74_polar_suffix_program.md`.

- Completed/stopped Phase 73 (2026-09-06): audited exactly 1,200 frozen
  Phase-72 states (500 old KEEP, 500 old INTERVENE, 200 old MIXED) across 422
  UIDs/415 image groups under contract `3c371c2f...08cc`. Four direct GPUs
  searched all 3,194 unobserved first-action branches with direct suffix,
  exhaustive one-later intervention, then MCTS@200. All states completed with
  zero quarantines; all 1,200 existing successes and 2,504 discoveries replayed
  correct with exact token parity. Old KEEP invalidation is 496/500 (99.2%,
  UID-bootstrap 95% CI [98.24%,99.81%]); old INTERVENE invalidation is 271/500
  (54.2%, [49.30%,58.89%]). Final labels are 4 AUDITED_KEEP, 229
  AUDITED_INTERVENE, and 967 AUDITED_MIXED; mean action-set cardinality rises
  1.338 to 3.425. MCTS discovery saturates under the frozen rule (1.80% after
  iteration 150). The planned audited five-fold binary probe is not estimable:
  only four clean KEEP UID/groups remain, so one test fold necessarily lacks a
  class. No fold/metric was changed post hoc. This directly supports severe
  label/target incompleteness but leaves representation separability unresolved.
  The one unexecuted recommendation is a prospectively specified set-valued
  Stage-2 target; no retraining or follow-on experiment ran. Evidence:
  `analysis/dense_failure_stage2/treatment_label_completeness/`; phase memory:
  `workspace/phase_memory/phase_73_treatment_label_completeness.md`.

- Completed/stopped Phase 72 (2026-09-05): reconstructed the frozen Phase-68
  exact-state supervision and extracted Experiment-A z_R/z_W/logits for all
  21,071 unique exact prefix-states (34,253 route occurrences, 569 UIDs) under
  contract `082c9f45...e44edd`, with exact router-logit parity and zero
  missing/duplicate states. Clean fitting uses 18,438 KEEP_REQUIRED and 1,657
  INTERVENE_REQUIRED states; 976 MIXED states remain descriptive only. Five
  UID/image-group-disjoint folds give AUROC/AUPRC: margin 0.4632/0.1050,
  four logits 0.4640/0.0939, READ 0.5370/0.0931, WRITE 0.5219/0.1088, and
  READ+WRITE 0.5620/0.1086. The prespecified optional RW MLP does not improve
  this (0.5583/0.1128). RW precision at 5/10/20% coverage is only
  0.123/0.135/0.123, with 0.0012 recall at 90% precision. Nuisance-only AUROC
  is 0.8150; matched RW remains weak at 0.5763 AUROC. Decision Case D applies:
  do not add another head; the one unexecuted recommendation is to diagnose
  representation/training-state diversity under separate authorization. All
  89 artifact hashes, 21,071 feature-index rows, and 23 focused/inherited tests
  pass. No deployment head, router retraining, search, Stage-1 change, or
  external evaluation ran. Evidence:
  `analysis/dense_failure_stage2/treatment_selectivity_separability/`; phase
  memory:
  `workspace/phase_memory/phase_72_stage2_treatment_selectivity_separability.md`.

- Completed/stopped Phase 71 (2026-09-05): calibrated one frozen global
  Stage-2 best-non-FULL-minus-FULL margin for Robust ALL-source Stage-1 P90
  plus Phase-66 Experiment A. Contract `87debbca...900b2` collected 3,907
  final-epoch routed-state margins over 1,048 training draws and froze 11
  candidates before development outcomes. All 1,210 required sequential
  rollouts completed over the 121 triggered Historical-800 UIDs, with exact
  δ=0 Phase-66 parity. δ=0 remains best at W-to-C/C-to-W/net `4/1/+3`;
  positive q10/q25/q40 points are `3/1/+2`, `2/1/+1`, and `2/0/+2`; q70+
  has zero answer-level changes, while q90+ has no non-FULL actions. All five image-group-disjoint cross-fit training
  folds select δ=0, pooled held-out `4/1/+3`. Rescue maximum-margin median
  0.6747 is not above regression 0.7210. Therefore a positive global margin is
  unsupported and the conditional external rerun was correctly not launched.
  Twenty-eight focused/inherited tests and all artifact hashes pass. The one
  unexecuted recommendation is to revisit the Stage-2 representation or
  training signal under separate authorization. Evidence:
  `analysis/dense_failure_stage2/abstention_margin/`; phase memory:
  `workspace/phase_memory/phase_71_stage2_abstention_margin_calibration.md`.

- Completed/stopped Phase 70 (2026-09-04): exhaustively decomposed the frozen
  Phase-69 full-benchmark run using all 19,960 stored paired traces and no new
  GPU inference. Stage 1 admits 496/4,380 Dense-W and 405/15,580 Dense-C;
  Stage 2 uses non-FULL on 119/496 triggered W and 92/405 triggered C. These
  activation rates are similar (0.2399 versus 0.2272), but outcomes are not:
  only 3/119 treated W are rescued while 19/92 treated C regress. The fixed
  descriptive classification is preservation-limited for ChartQA/TextVQA,
  treatment-quality-limited for MMMU-Pro (0 rescues from 82 treated W), and
  inactive for POPE (maximum Stage-1 score 0.828423, still 0.077710 below P90).
  TextVQA contributes 12/19 regressions and -10/-16 net. All 22 changed rows,
  full score/action traces, required tables, six figures, and 28 artifact
  hashes verify; 12 focused/inherited tests pass. The one unexecuted
  recommendation is a development-only preservation-calibrated Stage-2
  abstention margin. Evidence:
  `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/`; phase
  memory:
  `workspace/phase_memory/phase_70_full_benchmark_exhaustive_regression_rescue_audit.md`.

- Completed/stopped Phase 69 (2026-09-04): ran the first prospectively frozen
  full paired external evaluation of Robust ALL-source Stage-1 at strict P90
  followed by Phase-66 Stage-2 Experiment A. Contract
  `63379eef...27e83` uses the exact `shared_prefix_eval_20260812` manifests,
  prompt builder, native dense generation, and LMMS-compatible scorers for all
  19,960 unique ChartQA/TextVQA/MMMU-Pro/POPE UIDs. Dense versus routed results
  are W-to-C/C-to-W/net `1/3/-2` ChartQA, `2/12/-10` TextVQA, `0/4/-4`
  MMMU-Pro, and `0/0/0` POPE; pooled `3/19/-16`, delta accuracy -0.000802 with
  paired-bootstrap 95% CI [-0.001253,-0.000351]. Stage 1 triggers 901/19,960
  (4.51%); only 211 triggered samples use any non-FULL action, and POPE has no
  triggers. Decision D applies: this frozen candidate is regression-dominated
  at full scale. The one unexecuted recommendation is conservative Stage-2
  action-selection calibration focused on preservation. All 36 final artifact
  hashes and 19,960 UID/contract checks pass; no retraining, threshold change,
  new search, or follow-on evaluation ran. Evidence:
  `analysis/dense_failure_stage2/full_benchmark_eval/`; phase memory:
  `workspace/phase_memory/phase_69_full_benchmark_end_to_end_evaluation.md`.

- Evaluation-scope amendment (2026-09-04): future external evaluation defaults
  to four benchmark families: ChartQA; TextVQA; MMMU-Pro Standard and Vision;
  and POPE adversarial, popular, and random. This supersedes the earlier
  three-family ChartQA/MMMU-Pro/POPE restriction. DocVQA, MMStar, and base MMMU
  remain excluded unless explicitly added. The amendment is scope memory, not
  authorization to launch evaluation, and TextVQA evaluation assets must be
  independently verified before use.

- Completed/stopped Phase 68 (2026-09-04): replaced only Phase-66 B's one-hot
  CE with exact-prefix observed-valid-set loss under contract
  `c3e6adfd...2bd8421`; model, router, seed, optimizer, 3,144-update schedule,
  sampler, union routes, Stage-1 thresholds, Historical-800 validation,
  executor, and LMMS evaluator remained fixed. The exact-state audit found
  21,071 states over 34,253 occurrences, with valid-set sizes 1/2/3/4 =
  19,772/785/377/137. The deliberate smoke passed all prospective gates.
  Experiment C raises MCTS nominal/observed-valid non-FULL recall from
  8.45%/10.36% to 16.16%/21.26%; first/later observed-valid recall rises from
  13.52%/9.00% to 25.93%/19.25%. It does not repair single-source negative
  transfer: nominal recall remains 1.36% versus A's 10.78%. Historical-800
  free rollouts yield P98/P95/P90 net corrections -1/-1/0; P90 has one rescue
  and one regression. Thus Decision Case B applies: accepting-set ambiguity is
  a causal oracle-learning bottleneck, but not the sole deployment bottleneck.
  The 86-file artifact audit and 26 focused/inherited tests pass. No test set,
  new search, Stage-1/threshold/architecture change, or external evaluation
  ran. The one unexecuted recommendation is a small partial-prefix on-policy
  collection. Evidence:
  `analysis/dense_failure_stage2/observed_valid_set_loss/`; phase memory:
  `workspace/phase_memory/phase_68_stage2_observed_valid_set_loss.md`.

- Completed/stopped Phase 67 (2026-09-04): diagnosed the Phase-66 MCTS failure
  under authoritative contract `ffc2b02b...23bbcf` without retraining, new
  search, threshold changes, or held-out test access. Four exact-replay workers
  covered 2,519 routes and 34,253 oracle states per checkpoint; the 725 P90
  MCTS routes also produced 3,863 controlled-prefix rollouts and 6,653 drift
  rows with every oracle correctness control passing. B recognizes only 8.45%
  of MCTS non-FULL actions on exact oracle states (first/later 9.93%/7.82%) and
  first disagrees at the first intervention on 88.83% of routes. Adding MCTS
  reduces single corrective recall from A's 10.78% to B's 1.36%; the UID-level
  delta is -21.74 points, 95% CI [-25.56,-18.01]. B's multi-valid-state nominal
  error is 30.96% versus 5.65% for single-valid states, and accepting any exact-
  prefix observed-successful action recovers 21.63% of multi-valid rows. C1-C3
  forcing raises correctness mainly by executing forced corrections; after
  release B reproduces only 8.35%/6.75%/8.33% of remaining oracle corrective
  actions. Thus action learning, negative transfer, and label ambiguity are
  supported; exposure drift is secondary and no later-only representation
  deficit is isolated. The 50-file artifact audit passes. The one unexecuted
  recommendation is an unchanged-router observed-valid-set loss experiment.
  Evidence: `analysis/dense_failure_stage2/mcts_failure_diagnosis/`; phase
  memory: `workspace/phase_memory/phase_67_stage2_mcts_failure_diagnosis.md`.

- Completed/stopped Phase 66 (2026-09-04): froze shared Stage-2 contract
  `b17a81d...40f8`, deduplicating 106 preservation bases, 1,688 successful
  single routes over 270 W bases, and 725 MCTS routes over 216 W bases. Both
  matched shared-router arms passed native-dense implementation smoke and
  four-action overfit gates, then completed 3,144 fixed updates and P98/P95/P90
  free rollouts over all 800 leakage-free Historical validation UIDs. Single-
  only A produced W-to-C/C-to-W/net `0/0/0`, `0/0/0`, and `4/1/+3`; its P90
  gain was +0.00375 accuracy. Adding MCTS in B produced zero W-to-C and zero
  C-to-W at every threshold, while final non-FULL recall fell from 0.1920 to
  0.1687. Thus MCTS label abundance did not improve learned rollout behavior.
  The final planned B condition selects P98 only by conservative tie-break
  among three unchanged-accuracy points and is Decision C (under-generalized),
  not a positive deployment selection. Canonical OOF rows overlap Stage-2
  supervision, so their requested validation cells are explicitly unavailable;
  conclusions are Historical-800-only. The 145-file artifact audit passes. No
  held-out test, Stage-1 change, new threshold, search, or external evaluation
  ran. Evidence: `analysis/dense_failure_stage2/shared_union_training/`; phase
  memory: `workspace/phase_memory/phase_66_stage2_shared_union_training.md`.

- Completed/stopped Phase 65 (2026-09-04): froze contract
  `767284f1...57ab4`, passed a four-GPU 12-UID smoke, and completed all 1,413
  robust-gate work UIDs (1,104 new-search UIDs plus existing-route recapture and
  106 triggered-C preservation UIDs) with zero failures or duplicates. Shared
  search resolved 333/1,906 previously missing threshold pairs: 55 by exhaustive
  single intervention and 278 additional by MCTS@200. Final known-corrective
  coverage is P98 108/344 (0.3140), P95 234/727 (0.3219), and P90 463/1,307
  (0.3542); 236/493/844 remain unresolved at budget. The run retained 2,519
  exact-replay-valid routes and 34,253 routed state rows in 569 verified shards.
  Search used 34,023 single terminal routes, 1,799 MCTS roots, 327,832 MCTS
  iterations, and 15.40 sample GPU-hours; sharing saved 18,777 single terminal
  evaluations and 52 MCTS roots. Correctability remains much lower on GQA and
  late L19-L27 triggers. The 611-file artifact audit passes. No Stage-2 model
  was trained and no threshold, Stage-1, test, or external evaluation changed.
  Evidence: `analysis/dense_failure_stage2/robust_gate_corrective_search/`;
  phase memory:
  `workspace/phase_memory/phase_65_robust_missing_corrective_search.md`.

- Completed/stopped Phase 64 (2026-09-03): kept the Phase-63 ALL-source head
  frozen and audited treatment-dependent P98/P95/P90 operating points under
  contract `ab254e98...546674`. The held-out frontier gives worst-source C
  preservation 0.9808/0.9501/0.9003 and pooled W recall
  0.1054/0.2179/0.3572, with five-fold threshold ranges
  0.00361/0.01000/0.01465. Exact five-head-mean scoring reproduced 12,000
  prior held-out trajectory checks with zero error and mapped all 6,399
  Historical plus 4,000 Canonical train rows. P98/P95/P90 trigger
  344/727/1,307 W and 7/30/106 C; exact replay preserves 67/146/259 W with at
  least one reusable route. Every one of 2,618 route×operating-point replays
  remained LMMS-correct with exact stored-token parity. The residual workloads
  are 277/581/1,048 W, or 1,104 unique W across all points. Thirty-one artifact
  hashes and 54 focused/inherited tests pass. No corrective search, Stage-2
  training, Stage-1 retuning, or external evaluation ran. Evidence:
  `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/`;
  phase memory:
  `workspace/phase_memory/phase_64_robust_operating_points_and_compatibility.md`.

- Completed/stopped Phase 63 (2026-09-03): calibrated one strict any-layer
  threshold for the frozen Phase-62 ALL-source Shared Random-4 system using
  Historical validation plus 4,000 Canonical OOF trajectories, while keeping
  the 800-record Historical test outside selection. Under contract
  `30288d8a...262a38`, the prospective 98%-worst-source rule selects and freezes
  `tau=0.9711347410314399`. Historical/Canonical C preservation is
  0.9875/0.9808; W recall is 0.0650/0.1240 (pooled 0.1054), with pooled trigger
  precision 0.6734 and median first trigger L23. The untouched Historical test
  gives 0.9975 C preservation, 0.0650 W recall, and 0.9630 precision. Five
  cross-fit thresholds span 0.96859-0.97220 (IQR 0.00254), minimum held-out
  Canonical-fold C preservation is 0.9728, and no adequately supported
  dataset/source cell falls below 90% preservation. Canonical ChartQA preserves
  0.9842 C with 0.1034 W recall; Canonical TextVQA detects 0/19 W (Wilson upper
  0.1682), so its benefit remains unresolved. Decision A freezes gate hash
  `d3b019b3...cae4f`; 33 artifact hashes and 24 focused/inherited tests pass.
  No trigger map, Stage-2 work, search, retraining, or external evaluation ran.
  Evidence:
  `analysis/dense_failure_stage1/all_source_threshold_calibration/`; phase
  memory:
  `workspace/phase_memory/phase_63_stage1_all_source_threshold_calibration.md`.

- Completed/stopped Phase 62 (2026-09-03): trained the unchanged Shared
  Random-4 head with exact 25/25/25/25 Historical-C/Historical-W/Canonical-C/
  Canonical-W epoch quotas and the frozen old normalization under contract
  `08dbf146...90a4fa`. Five canonical OOF folds plus a five-model Historical
  probability ensemble yield AUROC 0.7689 Canonical and 0.8394 Historical;
  average/worst source AUROC is 0.8042/0.7689, improving worst-source behavior
  over both specialists (0.4056 historical-only, 0.5511 canonical-only).
  Historical GQA/ChartQA/TextVQA is 0.7370/0.9487/0.9488; Canonical is
  0.7538/0.5826/0.6111, with only 19 Canonical TextVQA-W. Full target-blind
  LODO Historical/Canonical AUROC is 0.7016/0.5691 ChartQA,
  0.6243/0.6259 TextVQA, and 0.6034/0.5931 GQA. Thus the frozen rule selects
  Decision B: source-robust for the current mixture but benchmark-OOD limited.
  L26 is post-hoc diagnostically consistent across all six LODO cells (minimum
  0.6270) but was not selected. All 4,000 Canonical OOF and 1,600 Historical
  ensemble rows are complete; 85 artifact hashes, eight checkpoint provenance
  records, exact sampler quotas, target exclusion, and 22 tests pass. No
  threshold, trigger map, Stage-2 change, search, or external evaluation ran.
  Evidence: `analysis/dense_failure_stage1/all_source_robustness/`; phase
  memory: `workspace/phase_memory/phase_62_stage1_all_source_robustness.md`.

- Completed/stopped Phase 61 (2026-09-03): refit the exact historical Shared
  Random-4 Stage-1 head on frozen canonical current-runtime labels while
  retaining the exact old global normalization, under contract
  `c1a0420...0c826`. Five image-group-disjoint 800-record folds produced one
  OOF 28-layer trajectory for every 4,000 canonical UIDs. Max-score OOF AUROC
  recovered from 0.4056 for the frozen old head to 0.8178 (paired gain +0.4123,
  95% CI [+0.3862,+0.4376]); GQA/ChartQA/TextVQA are
  0.7214/0.7021/0.4640. Thus ChartQA inversion disappears and the same
  architecture retains canonical signal on GQA/ChartQA, while TextVQA remains
  unresolved with only 19 W. Canonical Dense-C mean/p95 max risk fell from
  0.8056/1.0000 to 0.4677/0.9186. A full-canonical fit scored only 0.5511 on
  frozen historical val+test versus 0.8885 for the old head, confirming that
  neither source-specific boundary is robust across both regimes. The bounded
  old-recipe reproduction error is 6.22e-08; 23 artifact hashes and 13 tests
  pass. Decision A is retained with a dataset caveat: recommend a separately
  authorized source-balanced old+canonical experiment, not Arm B or Stage-2
  regeneration. No threshold, trigger map, Stage-2 artifact, search, or routing
  run changed. Evidence:
  `analysis/dense_failure_stage1/canonical_refit_diagnostic/`; phase memory:
  `workspace/phase_memory/phase_61_stage1_canonical_refit_diagnostic.md`.

- Completed/stopped Phase 60 (2026-09-03): reconstructed the historical Stage-1
  population and audited the frozen Shared Random-4 head using existing
  artifacts only under protocol `411a7530...99dd`. The old 8K population was
  explicitly selected as previous-Qwen Dense-C/Dense-W quotas (GQA 2K/2K,
  ChartQA and TextVQA 1K/1K); the executable 7,999 and its 6,399/800/800
  Phase-48 splits therefore share one selection regime despite zero UID and
  image-SHA-group overlap. Within current Dense-C, L21 old-vs-canonical source
  AUROC is 0.586 GQA, 0.938 ChartQA, and 0.937 TextVQA. Canonical-minus-old-C
  frozen max-score shifts are +0.089/+0.689/+0.712. Metadata-only old-test
  correctness AUROC is 0.673/0.788/0.630 and canonical ChartQA/overall invert
  to 0.381/0.272. Frozen max-score old-test AUROC falls from 0.886 overall to
  0.840 under exact visual-token matching and 0.820 under fixed coarsened
  token/aspect/question matching, so measured nuisances explain part but not
  most of the historical signal. New correct ChartQA/TextVQA states project
  toward the historical wrong geometry; normalized source shift is
  concentrated (conditioned median/max RMS 0.063/1.323), not a uniform proof
  that old normalization is the primary cause. All 30 artifacts and 34 focused
  tests pass. No inference, retraining, threshold change, Stage-2 work, or
  corrective search ran. Evidence:
  `analysis/dense_failure_stage1/historical_population_shortcut_audit/`; phase
  memory: `workspace/phase_memory/phase_60_stage1_historical_population_shortcut_audit.md`.

- Completed/stopped Phase 59 (2026-09-01): froze an outcome-blind,
  metadata-stratified canonical-source pool of exactly 4,000 new train
  identities (2,000 GQA / 1,000 ChartQA / 1,000 TextVQA), with unique image
  SHA groups and zero overlap with all 8,000 legacy candidates, under search
  contract `d85b5e9b...c4d94`. Four shared direct GPUs completed 4,000/4,000
  current-runtime dense rows with zero skips (3,129 C / 871 W). The unchanged
  Shared Random-4 gate produced 257 triggered W and 1,691 triggered C; exact
  search yielded 75 SINGLE_FIXABLE, 33 additional MCTS_ONLY_FIXABLE, and 149
  UNRESOLVED. All 2,707 retained routes replay exactly; all 1,799 new state
  shards and all 1,837 accepted artifact hashes validate. Expanded A/B/C now
  contain 1,730/8,578/508 routes over 1,730/773/242 bases; 44 new bounded-
  fixable GQA bases meet the prospective material-increase criterion. New
  bounded fixability among triggered W is 0.4202 versus 0.4822 old, within the
  frozen ±0.10 tolerance, but the gate shows major canonical-source shift:
  `P(trigger|C)=0.5404` versus 0.0122 old and `P(trigger|W)=0.2951` versus
  0.5878 old. Thus this is canonical-source scale-up, not a pure sample-count
  replication. The focused/regression suite passes 39/39. No Stage-2 model
  was trained and no validation/test search, test evaluation, Stage-1 retune,
  or external evaluation ran. Evidence:
  `analysis/dense_failure_stage2/data_scale_search/`; phase memory:
  `workspace/phase_memory/phase_59_stage2_data_scale_search.md`.

- Completed/stopped Phase 58 (2026-09-01): implemented the revised 698-first
  shared Stage-2 router with exact routed text/visual-token replay and froze
  contract `8f090f7d...bdb14`, a 3,144-update final-checkpoint-only schedule, and
  one validation opening. The 1,792-token peak-memory smoke passed exact native
  dense token parity with 16,778 MiB process-reserved memory and 13,849 MiB
  shared-GPU headroom. The 48-W/24-C overfit gate passed (0.7292 non-FULL
  recall; all three non-FULL actions learned). Full Corpus A+B training was
  numerically clean after moving only the small router to FP32, but the final
  train diagnostic remained near-FULL (0.0315 non-FULL recall). Validation
  completed all 800 UIDs/235 triggered UIDs exactly once: W→C/W→W/C→C/C→W
  were `5/395/400/0`, raising accuracy `0.5000→0.50625` with perfect C
  preservation. Behavior is narrow: 98.397% post-trigger FULL, 96.30% of
  intervened samples act immediately at trigger, no WRITE_ONLY rollout action,
  and no GQA intervention; the five rescues are 2 ChartQA + 3 TextVQA. An
  earlier BF16-router attempt was stopped/quarantined at non-finite epoch-5
  loss and produced no checkpoint. All 53 accepted artifact hashes and 800-UID
  coverage checks pass; 48 focused/regression tests pass. Repository-wide
  collection remains unavailable because transferred historical modules and
  reference-package paths are absent. No V1.5, Corpus C, test evaluation,
  Stage-1 change, or external evaluation ran. Evidence:
  `analysis/dense_failure_stage2/v1_training_revised/`; phase memory:
  `workspace/phase_memory/phase_58_stage2_v1_training_revised.md`.

- Completed/stopped Phase 57 (2026-09-01): audited the authoritative Phase-56
  Corpus A+B single-label supervision under analysis contract
  `32261c07...f7bc5` after reverifying all 977 Phase-56 source hashes. The
  frozen inputs are exactly 39 preservation samples/routes with 400 states and
  698 SINGLE_FIXABLE samples with 7,628 successful single routes and 199,193
  states. Routes/sample have mean 10.93, median 7, IQR 2-15, and range 1-60;
  sample-weighted RO/WO/IGNORE fractions are 0.2666/0.2649/0.4685. Correction
  is mainly delayed: 491/698 = 0.703 are delayed-only and median
  trigger-to-intervention delay is 13 layers. Naive Corpus B is 0.9617 FULL
  (25.11:1), with 90,609 duplicated semantic rows and 4,151 exact entering
  states carrying multiple observed route labels. The simplest recommended V1
  loader is one uniformly sampled route per W sample, one corrective plus up
  to two pre- and two post-FULL states (expected FULL 0.7847), and C:W = 1:2
  sample mixing; alternatives remain metadata and multi-label loss is deferred.
  All 28 output hashes and 18 focused/inherited tests pass. No training,
  inference, search, Corpus C use, validation/test labeling, rollout, or
  external evaluation ran. Evidence:
  `analysis/dense_failure_stage2/single_label_audit/`; phase memory:
  `workspace/phase_memory/phase_57_stage2_single_label_distribution_audit.md`.

- Completed/stopped Phase 56 (2026-09-01): froze full corrective-label
  generation under contract `6489a0b...9905b`, imported the exact cap-200
  interpretation of the 120-row Phase-55 pilot, and executed the remaining
  1,761 triggered Dense-W searches plus all 39 triggered Dense-C FULL-suffix
  preservation rows on four direct GPUs. All 1,800 fresh UIDs completed once
  with zero failures. The final 1,881-W classes are 698 SINGLE_FIXABLE
  (0.3711), 209 MCTS_ONLY_FIXABLE (0.1111), and 974 UNRESOLVED (0.5178), for
  bounded correctability 907/1,881 = 0.4822, only +0.0155 above the cap-200
  pilot projection. Support is highly nonuniform: GQA/ChartQA/TextVQA total
  rates are 0.1986/0.4854/0.6448, and L0/L1-8/L9-18/L19-27 rates are
  0.5923/0.4560/0.3975/0.2507. All 7,628 single routes, 442 MCTS routes, and
  39 preservation routes replayed exactly into 946 provenance-bound shards
  with 208,280 routed state rows. Corpus A/B/C contain 400/199,193/8,687
  state rows and remain separate. The 977-file artifact audit passes under
  manifest SHA-256 `a26d31e1...0ff4d4`; 57 scoped tests pass. One
  implementation-only trigger-field alias smoke failed closed, was
  quarantined, regression-tested, and produced no accepted scientific row.
  No Stage-2 model was trained, Stage 1 was unchanged, validation/test were
  not searched, and no external/downstream evaluation ran. Evidence:
  `analysis/dense_failure_stage2/full_corrective_labels/`; phase memory:
  `workspace/phase_memory/phase_56_full_corrective_label_generation.md`.

- Completed/stopped Phase 55 (2026-08-31): froze and executed the 120-row
  trigger-conditioned corrective-search pilot under contract
  `6b4eb6de...afa2c`. The prospective image-group-unique diagnostic allocation
  uses 40 train triggered Dense-W samples per dataset and reports both balanced
  and Phase-54 dataset×depth-cell-weighted estimates. Four direct GPUs passed a
  12-row parity/provenance smoke and completed 120/120 UIDs with zero failures.
  Exhaustive singles found 35/120 fixable; ordered trajectory-conditioned MCTS
  added 22/120, for balanced total 57/120 (0.4750) and cell-weighted total
  0.5309. Fixable@100/200/300 was 52/56/57; the frozen 0.01 absolute-gain rule
  selects 200 iterations. Preferred correct routes use median one non-FULL
  action (IQR 1–3), and 14/57 fixable samples exposed multiple successful
  trigger actions. GQA/ChartQA/TextVQA support is 0.400/0.475/0.550; depth-bin
  support is 0.652/0.438/0.562/0.303. All 360 retained correct routes replayed
  exactly into 57 provenance-bound shards with 7,776 routed state rows; 106
  artifact hashes and 46 focused/regression tests pass. Projected scale-up is
  about 362,007 terminal routes, 31.15 GPU-hours, or 7.79 four-GPU wall-hours.
  No remaining 1,761-row search, triggered-C search, Stage-2 training, Stage-1
  change, validation/test search, W-to-C repair, or external evaluation ran.
  Evidence: `analysis/dense_failure_stage2/corrective_search_pilot/`; phase
  memory: `workspace/phase_memory/phase_55_trigger_conditioned_corrective_search_pilot.md`.

- Completed/stopped Phase 54 (2026-08-31): froze and audited the existing
  Shared Random-4 dynamic Stage-1 trigger map under contract
  `d9dd591a...5653a7`, without Qwen inference or new four-action search. Four
  direct GPU workers reconstructed only the missing 6,399 train score
  trajectories from hash-verified stored dense features; saved Phase-51
  validation/test trajectories supplied the other 1,600 rows. The complete
  7,999-row partitions are train C/no-trigger, C/trigger, W/no-trigger,
  W/trigger `3160/39/1319/1881`; validation `377/23/188/212`; and test
  `376/24/196/204`. Preservation/recall/precision are
  `0.9878/0.5878/0.9797` train, `0.9425/0.5300/0.9021` validation, and
  `0.9400/0.5100/0.8947` test. GQA wrong recall is only `0.180/0.185` on
  validation/test versus `0.880/0.800` ChartQA and `0.880/0.870` TextVQA.
  The deterministic 24-record score check passed within `2.81e-7`, all trigger
  decisions matched, all 1,600 Phase-53 trigger rows and 18 Phase-52 aggregate
  fields matched exactly, all 41 artifact hashes match, and 36 focused tests
  pass. The authoritative future train workloads are 1,881 triggered-W search
  candidates and 39 triggered-C FULL-suffix preservation candidates; val/test
  remain non-training cohorts. Two implementation-only attempts stopped
  fail-closed and were quarantined before accepted outputs. All GPUs are idle.
  No Stage-2 labels/training, threshold change, persistence/EMA, W-to-C repair,
  or external evaluation ran. Evidence:
  `analysis/dense_failure_stage1/trigger_map/`; phase memory:
  `workspace/phase_memory/phase_54_stage1_trigger_map_audit.md`.

- Completed/stopped Phase 53 (2026-08-31): executed current-runtime
  treatment-correctability for the Phase-52 shared Random-4, independent
  sequential, and fixed-L27 gates under frozen contract
  `2f935dd0...c956f1e`. A 12/12 smoke passed native-dense token parity and
  31/31 cached-suffix/complete-route parity. Four direct GPUs completed all
  443 validation and 433 test execution samples, 2,196 regimes, and 59,239
  unique bounded-search routes with zero final failures. Validation froze
  shared Random-4 before test: validation population rescue/conditional
  correctability were `0.2025/0.3821`, versus `0.1825/0.3544` independent and
  `0.1975/0.3657` fixed L27. Held-out values were `0.2100/0.4118`,
  `0.1975/0.3709`, and `0.2200/0.4171`, respectively. Shared-gate full-replay
  enrichment was 1.1211 validation and 1.1132 test; triggered-correct
  preservability was 1.0 for every gate. Early shared triggers were more
  conditionally correctable, but fixed-L27 replay slightly exceeded dynamic
  population rescue on test and remains a serious fallback. The 463-record
  selected-gate Stage-2 handoff is frozen; proceeding to an action head is
  supported only provisionally and requires separate authorization. A
  regime-status schema defect and native/materialized BF16 SDPA parity defect
  were caught fail-closed, fixed, regression-tested, and their two invalid
  contracts quarantined. All 22 required artifact hashes and 35 focused tests
  pass; all GPUs are idle. No Stage-2 training, broad W2C repair/MCTS, routing
  training, or external evaluation ran. Evidence:
  `analysis/dense_failure_stage1/treatment_correctability/`; phase memory:
  `workspace/phase_memory/phase_53_stage1_treatment_correctability.md`.

- Completed/stopped Phase 52 (2026-08-31): reused the frozen Phase-50/51
  validation/test score trajectories and selected one Stage-1 admission gate
  under contract `f065d372...f2288135`, without retraining or model inference.
  Full validation sweeps maximized `(wrong detected - correct false triggers) /
  800` subject to aggregate failure precision at least 0.90. Shared fixed L27
  won at raw threshold `0.8497647428417646`, with validation
  preservation/recall/precision/utility `0.9400/0.5400/0.9000/0.2400`; shared
  Random-4 was three net utility samples behind. The frozen winner transferred
  to `0.9500/0.5275/0.9134/0.23875` on the selection-held-out test. Independent
  sequential tied test utility at `0.23875`; shared Random-4 and All-28 reached
  `0.2250` and `0.2175`, so sequential gating has no demonstrated aggregate
  utility advantage. Fixed-L27 minus the validation runner-up has test 95%
  bootstrap intervals spanning zero for utility `[-0.00375, 0.03125]` and
  recall `[-0.0125, 0.0475]`. Performance remains task-dependent: winner test
  utility is `0.0775/0.3850/0.4150` on GQA/ChartQA/TextVQA and GQA precision is
  only `0.7460`. Carry fixed L27 and this threshold only into a separately
  authorized treatment-feasibility phase; it is not a final deployment
  threshold or a universal fixed-layer claim. All 20 artifact hashes and 21
  focused/regression tests pass; no treatment, W-to-C/four-action work, MCTS,
  OOD, external evaluation, or GPU job ran. Evidence:
  `analysis/dense_failure_stage1/gate_winner_selection/`; phase memory:
  `workspace/phase_memory/phase_52_stage1_gate_winner_selection.md`.

- Completed/stopped Phase 51 (2026-08-31): executed the authorized shared
  Stage-1 predictor/global-risk gate under frozen contract
  `264a9407...65e095b`, reusing the exact Phase-48 6,399/800/800 split and
  compact current-dense features. Four direct GPUs trained the four
  prospectively fixed models: state-only All-28, layer-only All-28,
  state+layer All-28, and state+layer Random-4. Validation designated
  state+layer Random-4 with the full 0-27 gate window before one aggregate test
  pass; state/layer-only controls never received test scores. The shared
  predictor matches independent-probe ranking (mean test layer AUROC
  0.8769 versus 0.8737), while layer-only is exactly chance and state+layer
  does not materially exceed state-only. At the 99% validation target,
  Random-4 transfers from 0.9900/0.3950 validation preservation/recall to
  0.9875/0.4050 test, reducing absolute preservation drift from Phase-50's
  0.0425 to 0.0025. However, test preservation/wrong-recall spread across
  GQA/ChartQA/TextVQA is 0.0300/0.7600, not better than Phase 50's
  0.0250/0.7250. A validation-selected shared fixed L27 also gives
  0.9825/0.4200 versus sequential 0.9875/0.4050, so sequential deployment has
  no clear advantage. The frozen readiness decision is **NO**: do not connect
  this gate to treatment. All 33 required hashes, 4x800 validation and 2x800
  test trajectories with 28 scores, five figures, and 26 focused tests pass;
  all GPUs are idle. One pre-test implementation error was quarantined under
  `shared_global_gate.invalid_6dd970d9` and no checkpoint crossed contract
  hashes. No OOD, treatment, W-to-C/four-action work, MCTS, or external
  evaluation ran. Evidence: `analysis/dense_failure_stage1/shared_global_gate/`;
  phase memory:
  `workspace/phase_memory/phase_51_shared_stage1_global_risk_gate.md`.

- Completed/stopped Phase 50 (2026-08-31): reused the exact 28 frozen Phase-48
  linear probes and its 800/800 validation/test cohorts to evaluate the
  authorized independent layer-wise first-trigger gate under frozen contract
  `6b1e4812...653831d`. Validation-only shared-alpha calibration selected
  `alpha=0` for both 99% and 98% targets (1.000 correct preservation, 0.4100
  wrong recall) and `alpha=0.00250627` for 95% (0.9675 preservation, 0.4750
  recall), after which test was opened once. On test, the 99%/98% gate retained
  only 0.9575 of correct samples while detecting 0.4175 of wrong samples; the
  95% gate achieved 0.9550/0.4775. Median first triggers were layers 3 and 2,
  respectively. The sequential gate improved over the best fixed L14/L21/L27
  baseline by only +0.0200 recall at the 99% target and was worse by 0.0225 and
  0.0500 at 98% and 95%. Test wrong-recall spread across datasets was
  0.7250/0.7250/0.6650 (GQA much weaker than ChartQA/TextVQA), so the common
  calibration is not a robust final gate under the prospective criteria. All
  1,600 score rows contain 28 scores, 17 required artifact hashes match, 17
  focused tests pass, and all four GPUs are idle. No probe was retrained and no
  treatment, shared predictor, global risk-budget model, routing, or external
  evaluation ran. Evidence:
  `analysis/dense_failure_stage1/independent_sequential_gate/`; phase memory:
  `workspace/phase_memory/phase_50_independent_sequential_gate.md`.

- Completed/stopped Phase 49 (2026-08-31): executed the authorized OOD-first
  dense-failure diagnostic under frozen contract `dfef70e6...c64fff`. The
  native pre-language-decoder control was validated on a 12/12 exact-repeat
  four-GPU smoke with one pre-hook capture and zero decoder forward firings,
  then extracted for all 7,999 current-label rows into 128 audited shards.
  Cross-dataset image-group overlap is zero. Eighty-seven source-only linear
  probes were fitted for the three leave-one-dataset-out runs; source
  validation froze representative layers 22 (TextVQA target), 26 (ChartQA),
  and 20 (GQA) before one-pass target scoring. Representative OOD AUROC is
  0.7236 on TextVQA, 0.4502 on ChartQA, and 0.6160 on GQA, versus pre-decoder
  0.5219/0.5472/0.5432. Target-descriptive layer-21 AUROC is
  0.8046/0.8018/0.5975, showing real but nonuniform and layer-unstable transfer.
  Source 99%-preservation thresholds transfer conservatively only to TextVQA
  (actual preservation 0.995, wrong recall 0.092); preservation collapses on
  ChartQA (0.305) and GQA (0.138), making their high recalls nonconservative.
  The fixed prospective interpretation gate therefore rejects a strong
  benchmark-general computation-dependent claim and does not authorize the
  shared Stage-1 predictor on that rationale. All 87 tasks, required artifacts,
  and SHA-256 checks pass; 28 focused tests pass. No shared predictor, layer
  embedding, W-to-C work, routing, Stage 2, or external evaluation ran.
  Evidence: `analysis/dense_failure_stage1/ood_signal_diagnostic/`; phase
  memory: `workspace/phase_memory/phase_49_ood_failure_signal_diagnostic.md`.

- Completed/stopped Phase 48 (2026-08-30): executed the authorized 28-layer
  current-dense failure-predictability diagnostic under frozen contract
  `3cf49a46...0234cd`. The exact image-group-disjoint split is 6,399 train / 800
  validation / 800 test with zero UID or group overlap. Four direct GPUs fitted
  one identical regularized linear probe per layer over the concatenated
  `text_final`/`text_mean`/`visual_mean` summaries; validation froze the full
  layers-0-27 informative region before a single test evaluation. Test
  AUROC/AUPRC are already 0.8421/0.8519 at layer 0 and best AUROC is 0.8992 at
  layer 21, before Phase-47 answer commitment at layers 25-27. Every dataset is
  informative from layer 0, but GQA is weaker (0.6981, peak 0.7713) than
  ChartQA/TextVQA (about 0.94 at layer 0, peaks above 0.97). Among transferred
  validation thresholds that still meet the requested preservation on test,
  best wrong recall is 0.3350 at 99%, 0.4275 at 98%, and 0.5225 at 95%.
  All required artifacts and 28 checkpoints pass SHA-256 audit; 21 focused
  tests pass. Because informative-only 0-27 equals all-layer, a future
  separately authorized comparison should prioritize all-layer versus random-k
  over 0-27; layers 16-27 may be a predeclared stronger-plateau sensitivity.
  No shared predictor, W→C work, routing, MCTS, or external evaluation ran.
  Evidence: `analysis/dense_failure_stage1/layerwise_failure_probe/`; phase
  memory:
  `workspace/phase_memory/phase_48_layerwise_dense_failure_predictability.md`.

- Completed/stopped Phase 47 (2026-08-30): corrected dense answer-logit
  emergence at the actual assistant answer-start position over all 7,999
  current LMMS-labeled native-dense samples. Frozen contract
  `c9a6d630...cbdcce` passed a 72/72 stratified first-token gate plus cached
  shared-prefix checks, and four direct GPUs completed 7,999/7,999 records with
  zero execution failures. Layer-27 raw top-1 reproduced 7,926/7,999 stored
  first tokens (99.09%); residual differences reflect frozen generation
  processors rather than the old position error. Of 4,000 wrong samples, 3,993
  first-divergence comparisons are usable and 7 unreplayable collisions are
  explicitly excluded. Correct GT first becomes raw top-1 at median layer 26
  (IQR 26–27), with 94.95% top-1 at layer 27. The fixed three-layer persistent
  rule is right-censored and is defined for only 851/3,999 correct samples
  (median 24 among defined). The literal wrong zero-crossing median is layer 2,
  but early targets have extremely low ranks and near-zero, balanced margins;
  material wrong-answer separation occurs around layers 23–27 and is strongest
  at 25–27. Fixed wrong taxonomy: 1,977 early, 33 progressive, 1,711 answer
  erosion, and 272 ambiguous over 3,993 usable samples; early/erosion counts are
  sign patterns, not proof of semantic early answers. The evidence supports
  layers 25–27 (especially 26–27) as a candidate late supervision region but
  does not select a Stage-1 strategy. No predictor training, W→C work, routing,
  MCTS, or external evaluation ran. Evidence:
  `analysis/dense_failure_stage1/answer_logit_emergence_v2/`; phase memory:
  `workspace/phase_memory/phase_47_answer_position_logit_emergence_v2.md`.

- Completed/stopped Phase 46 (2026-08-30): executed the user-authorized first-answer-token
  logit-lens analysis in `plans/dense_answer_logit_emergence_analysis_plan.md`
  over all 7,999 completed current-dense samples. The prospective analysis uses
  the frozen Qwen2.5-VL final norm/head on the saved final literal user-query
  token state, raw-logit delta 1.0, three-layer persistence, early cutoff layer
  4, and companion persistent zero crossings. It uses canonical GT first tokens
  and exact generated first tokens and fit no learned probe. Four direct GPUs
  scored 128/128 shards and 7,999/7,999 unique UIDs with finite logits under
  frozen contract `73daef6c...`. Correct GT-vs-strongest-token delta and zero
  emergence were both 0/3,999; wrong delta emergence was 2,215/4,000, reaching
  50% coverage at layer 21 but never 75%. Wrong taxonomy was 1,203 early, 425
  progressive, 587 answer erosion, and 1,785 ambiguous, including 594 exact
  GT/predicted first-token collisions. The one allowed validity diagnostic found
  layer-27 `<|im_end|>` top-1 for 54/54 checked correct records, confirming that
  the saved final literal user-token position predicts chat structure rather
  than the first assistant answer. Therefore no supervision start range or
  training strategy is selected; optional sequence analysis and all training
  were stopped. Evidence: `analysis/dense_failure_stage1/logit_emergence/`;
  phase memory:
  `workspace/phase_memory/phase_46_dense_answer_logit_emergence.md`.

- Completed Phase 45 (2026-08-30): the user stopped the Phase-44 fail-closed
  repair/recheck workflow and authorized one direct current-dense Stage-1 data
  action. The complete recovered population is 8,000 GQA/ChartQA/TextVQA
  candidates. Native Qwen2.5-VL dense all-on inference will be scored by the
  official `lmms-eval==0.7.3` task implementations, retaining raw fractional
  TextVQA consensus and the repository's existing 0.5 binary threshold. An
  18/18 functional smoke passed, followed by a four-GPU full run. All 8,000
  candidates were attempted; 7,999 completed and one ChartQA sample was
  explicitly skipped for a missing image. Current LMMS labels are 3,999 correct
  and 4,000 wrong (GQA 2,000/2,000, ChartQA 999/1,000, TextVQA 1,000/1,000).
  All completed samples have finite BF16 `[28,3584]` text-final, text-mean, and
  visual-mean features across 128 shards. The completed population has 7,476
  image groups. No split or predictor training ran. Evidence root:
  `analysis/dense_failure_stage1/current_dense_8k/`; phase memory:
  `workspace/phase_memory/phase_45_current_dense_8k_lmms.md`.

- Stopped Phase 44 (2026-08-30): the user replaced historical-label recovery
  with authoritative current-runtime native-dense regeneration. The tracked
  portable reproduction manifest restores all 8,000 requested GQA/ChartQA/
  TextVQA identities and annotations; all required semantic fields match the
  independent 6,917-row transferred overlap. The implementation loads native
  Qwen2.5-VL directly with no routed wrapper, freezes physical content-hash
  image groups, and gates optional 28-layer passive pooled-state extraction on
  exact 24-record token parity. The next boundary is contract freeze followed
  by repeatability/hook smoke; four-GPU full execution is authorized only after
  those gates pass. Stop before predictor training. Phase memory:
  `workspace/phase_memory/phase_44_current_dense_stage1_regeneration.md`.

- Stopped Phase 43 audit (2026-08-30):
  `plans/stage1_dense_failure_8k_label_audit_plan.md` (SHA-256
  `1e645d2f59df580606be067812d78d4e2c3e19e8973ce78387e9cc727fcdcc6d`).
  The physical active pool is complete at 8,000 images with historical 4K/4K
  filename buckets, but the six source JSONLs and canonical 8K regenerated
  label/contract bundle were not transferred. The available source derivative
  contains only 6,917 positive-route VQA rows and omits all 1,083 zero-positive
  rows plus the original dense predictions. Current dense counts reconstruct
  to 4,045 correct / 3,955 wrong. All image bytes form 7,477 SHA-256 groups,
  including 513 repeated-content groups, so future splitting must be image
  group/content disjoint. The authority decision is
  `EXECUTION_CONTRACT_UNRESOLVED`; no replay subset, GPU inference, split,
  hidden-state extraction, or training was started. Evidence:
  `analysis/dense_failure_stage1/8k_label_audit/`; phase memory:
  `workspace/phase_memory/phase_43_stage1_dense_failure_label_audit.md`.

- Completed/stopped audit (2026-08-30): the exact dirty-worktree source that
  generated `mcts_labels_4action/sequential_branching_v1` is reconstructed and
  matches all 16/16 contract-bound SHA-256 values plus the frozen YAML hash.
  Recorded `HEAD` was `a3c6a411...`, but two files were modified and six were
  untracked relative to that commit. The historical executor implements the
  intended READ/WRITE truth table; same-layer text READ uses pre-layer visual
  K/V. The fixed complete-route path has no semantic source difference from
  current code capable of explaining the Phase-42 replay mismatch. H100 versus
  RTX BF16 numerical execution is the leading inference but remains unverified:
  audit-only H100 job 1763 was canceled before allocation at explicit user
  instruction and consumed zero GPU time. End-to-end decision is fail-closed C,
  while source recovery and scientific semantics are exact/valid. No executor,
  label, regeneration, or W2C repair change was made. Evidence:
  `analysis/executor_provenance_audit/`; phase memory:
  `workspace/phase_memory/phase_42_w2c_when_label_repair.md`.

- Stopped phase (2026-08-30):
  `plans/w2c_when_label_repair_plan.md` (SHA-256
  `19d750c7acca5caaf37a85438f432e566dd980cbc29ddb1e6cf7d3c8e0c23e88`).
  Phase 42 verified all 640 source records and froze one iterative known-suffix
  plus capped one-edit repair contract, but its 12-sample four-GPU smoke failed
  exact old-route replay. 37/312 cached-correct routes now execute incorrectly,
  affecting 10/12 samples across all datasets. A bounded diagnostic repeated
  one failure per GPU twice: 4/4 current pairs were exact and 0/4 matched the
  original cached generated tokens, supporting reproducible runtime/cache
  drift. Original label-record executor hashes differ from current code. The
  complete repair, post-repair audit, router/gate training, Stage 2, and
  external evaluation did not start. Evidence:
  `analysis/w2c_when_repair/decision_summary.md`; phase memory:
  `workspace/phase_memory/phase_42_w2c_when_label_repair.md`.

- Completed/stopped phase (2026-08-30):
  `plans/selective_continue_deviate_expanded_plan.md` (SHA-256
  `20a7517dc61197c8d3914cf8cf45183af7438e514ccdb4cba1583f9b25da34e9`).
  Phase 1 audited the complete 128-state held-out W2C census by inserting
  `FULL` at each mandatory boundary and executing all 252 deduplicated routes
  induced by every compatible frozen suffix on four direct GPUs. It found
  39/128 bounded rescues (30.47%, 10,000-draw 95% UID-bootstrap CI
  [22.66%, 38.28%]), 89 bounded invalid states, and zero unresolved states,
  across all three datasets. The prospective label-trust gate therefore fails:
  only 89 trusted validation DEVIATE positives remain versus the required 128.
  No gate dataset/training, threshold sweep, oracle-WHAT execution, Stage 2, or
  external evaluation ran. Evidence:
  `analysis/selective_continue_deviate/stage1_decision_summary.md`; phase memory:
  `workspace/phase_memory/phase_41_selective_continue_deviate.md`.

- Completed phase (2026-08-30):
  `plans/four_action_generalization_diagnostic_plan.md` (SHA-256
  `cdc39940a1e19c22f17771bc535d22be792b97cd7f45d7c6128ce816e933e446`).
  All 640 frozen W2C mandatory boundaries were exactly matched by split,
  dataset, and layer to 640 different-UID `FULL`-unique W2C trajectory states.
  Four direct RTX 6000 Ada ranks extracted finite selected-checkpoint outputs
  for all 1,280 states. POLAR/online validation KEEP-vs-DEVIATE AUROC is
  0.542877/0.507751 and argmax deviation recall is 0.054688/0.148438, versus
  train AUROC 0.915585/0.994514. Both READ_OFF and WRITE_OFF generalization is
  weak and IGNORE-only recall is zero. The online frozen-state linear WHEN
  probe reaches 0.737976 AUROC, but within-cell joint state shuffle leaves
  83.6% of predictions unchanged and does not reduce validation bit AUROCs.
  k=10 exact mechanism purity is only 0.371--0.431. The final bounded four-GPU
  audit executed all 19 frozen known-suffix routes for 14 selected
  cached-invalid states; 6/14 have an execution-correct supposedly invalid
  action, proving conditional WHAT-label incompleteness. Dominant deployed
  failure: WHEN, with online trained-signal-use failure, broad WHAT collapse,
  weak exact-label smoothness, and incomplete cached actions as coexisting
  supported findings. No new training or external evaluation ran. A future
  bounded audit inserting `FULL` at mandatory boundaries is recommended only
  with explicit approval. Evidence:
  `analysis/4action_generalization_diagnostics/decision_summary.md`; phase
  memory:
  `workspace/phase_memory/phase_40_four_action_generalization_diagnostics.md`.

- Completed phase (2026-08-30): `plans/four_action_generalization.md`
  (SHA-256
  `79c159af4aa451cdbb153e95b7145566f77835770c1408765f1fafe1d35837b5`).
  A deterministic matched subset froze 512 W2C + 512 C2C training records and
  128 W2C + 128 C2C validation records across GQA/ChartQA/TextVQA. Both the
  unchanged POLAR exact-set-NLL and online set-valued routers trained for 20
  epochs with one mandatory-boundary term for every W2C every epoch, using all
  four local RTX 6000 Ada GPUs through direct execution. All 20 checkpoints
  per substrate were internally executed on the same 256 held-out records.
  POLAR selected epoch 15 with 7/128 W2C rescues and 124/128 C2C preservation;
  online selected epoch 14 with 6/128 and 122/128. The paired online-minus-
  POLAR W2C difference was -0.0078125 (10,000-draw 95% bootstrap interval
  [-0.0625, 0.0390625]), so the prospective decision is **no supported
  architecture advantage; operationally prefer POLAR**. A direct all-FULL
  audit found one current-runtime mismatch in the frozen C2C cohort; excluding
  it gives POLAR 124/127 and online 122/127 C2C preservation and leaves both
  selections and the decision unchanged. No external evaluation or follow-up
  action ran. Evidence: `analysis/persistent_corrective_supervision/decision_summary.md`
  and `analysis/persistent_corrective_supervision/runtime_cohort_sensitivity.md`;
  phase memory:
  `workspace/phase_memory/phase_39_persistent_corrective_supervision.md`.

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
