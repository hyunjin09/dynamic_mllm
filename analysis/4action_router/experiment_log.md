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
