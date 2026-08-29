# Online Four-Action Router Experiment Log

- 2026-08-28: froze the separate V1 online-router scope from
  `plans/four_action_train.md`; existing pending POLAR job 1662 is unchanged.
- Verified 6,811 GQA/ChartQA/TextVQA samples, 248,804 routes, one executor
  contract, source evaluator metadata, Qwen revision, evaluation population,
  eight-H100 topology, project `.venv`, and live Slurm state.
- Implemented current-state READ/WRITE branches, structured four-action logits,
  exact prefix tries, stable set-valued loss, deterministic balanced route
  replay, eight-way DDP training, every-epoch routed validation, checkpoint
  selection, resumable external evaluation, and result merging.
- Label/trie audit: PASS; 7,621,638 trainable router parameters and 225,533
  multi-valid trie nodes.
- GPU execution is fail-closed as smoke -> training -> external evaluation.
- Final repository verification: 476 tests passed in 33.48 seconds.
- Submitted smoke job 1663 (8 H100, 64 CPU, 512G, four hours). Fresh live state:
  PENDING/`AssocGrpGRES`; scheduler estimate 2026-08-28 17:29:56 KST is not a
  guarantee.
- Submitted training job 1664 (8 H100, 64 CPU, 512G, seven days) with exact
  `afterok:1663` dependency.
- Submitted external-evaluation job 1665 (8 H100, 64 CPU, 512G, seven days)
  with exact `afterok:1664` dependency. Its preflight precedes eight concurrent
  shards and merge/bootstrap reporting.
- Existing upfront POLAR job 1662 remains live-PENDING/`AssocGrpGRES` and was
  not changed, canceled, or made a dependency of the online-router chain.
- 2026-08-28 terminal update: smoke 1663 ran for 39 seconds and failed before
  semantic validation because the fail-closed guard found the existing output
  directory `outputs/four_action_online_router/smoke_v1`.
- At the user's explicit request, pending never-started training job 1664 and
  dependent evaluation job 1665 were canceled at 11:00:54 KST. A fresh
  `squeue -u $USER` was empty immediately afterward.
- Slurm accounting also established that separate POLAR job 1662 had already
  failed before this cancellation; it was not changed by the cancellation.
  No fix, output reuse/deletion, or relaunch is authorized.
- 2026-08-29 authorization: the user requested the online-router training and
  subsequent ChartQA/MMMU-Pro/POPE evaluation using all eight H100s.
- Root-cause repair: a red/green test reproduced the DDP directory race in
  which late ranks observed the directory just created by rank 0. Rank 0 alone
  now checks/creates the shared smoke root before the barrier. The focused test
  and all 480 project tests pass; fix commit `23ed41c` is pushed.
- Fresh output roots are `smoke_v2`, `training_v2`, `external_v2`, and
  `external_analysis_v2`. The empty historical `smoke_v1` directory was not
  deleted, renamed, or reused.
- Submitted all-eight-H100 job chain: smoke 1684; training 1685 with
  `afterok:1684`; evaluation 1686 with `afterok:1685`. Direct `scontrol`
  verification confirmed both dependency fields. Smoke is pending for
  `AssocGrpGRES`; downstream jobs are dependency-blocked.
- Exact launch hashes, resources, paths, and continuation commands are saved in
  `reports/four_action_online_router_h100_relaunch_20260829.md`.
- 2026-08-29 v2 runtime update: smoke 1684 started on all eight H100s and
  preserved a complete report/checkpoint, but failed the unchanged loss-
  decrease gate after mean loss rose `1.41447294 -> 1.45719039`. All semantic,
  gradient, frozen-backbone, deterministic-route, and checkpoint-roundtrip
  evidence remained healthy. Training 1685 became dependency-never-satisfied;
  evaluation 1686 remained blocked.
- Supported diagnosis: smoke constructed AdamW directly at full LR `5e-4`,
  while main training's cosine-with-warmup scheduler initializes LR at zero and
  uses `0`, `5e-5`, `1e-4`, and `1.5e-4` over the first four steps. Thus the
  smoke did not test the frozen training optimization contract.
- Repair: shared optimizer/scheduler construction is now used by smoke and
  training. A regression failed before the helper existed and passes after the
  repair; the online-router file passes 16/16 and the full project suite passes
  481/481. Portable commit `f6a0c42` is pushed.
- Dead jobs 1685/1686 were canceled. Fresh v3 roots were verified absent and
  machine-local wrappers pass `bash -n`.
- Submitted fresh all-eight-H100 chain at 2026-08-29 13:07:17 KST: smoke 1690,
  training 1691 with `afterok:1690`, and evaluation 1692 with `afterok:1691`.
- Smoke 1690 completed `0:0` in 42 seconds and passed all gates. Mean loss fell
  `1.41447294 -> 0.98042297`; exact checkpoint roundtrip, multi-valid
  supervision, routed-state conditioning, nonzero READ/WRITE query gradients,
  and zero backbone gradients all pass. Smoke report SHA-256:
  `b0a08073dbac9b4e2d8a17c165bcbdad7275cb5ff897b81037e4ec0b60ce6a61`.
- Training 1691 started automatically from exact Git commit `f6a0c42`. All
  eight ranks emitted finite teacher-forced losses through global step 3;
  evaluation 1692 remains dependency-blocked. Partial sample losses are runtime
  health evidence only, not a scientific result.
- Smoke-calibrated estimate is 1.59 wall-hours for teacher replay, 0.45 hours
  for ten epoch-validations, and 0.77 hours for external evaluation on eight
  GPUs (2.81 wall-hours / 22.47 allocated GPU-hours combined), with documented
  generation/backward-overhead caveats in
  `analysis/4action_router/calibrated_compute_estimate_v3.json`.
