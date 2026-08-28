# Four-Action POLAR Node07 Launch Handoff

## Status

- Date: 2026-08-28
- Current Git commit before local implementation changes: `4818b98`
- Slurm job: `105063`
- Job name: `fa4polar_n07`
- Requested placement: node07, A6000 partition, four GPUs, 48 CPUs, 240 GB
- Current state at handoff: `PENDING`
- Pending reason: `ReqNodeNotAvail, UnavailableNodes:node07`
- Training started: no
- Evaluation started: no

The four node07 GPUs reported free at the start of the action were occupied by
four other one-GPU jobs before submission. No other A6000 node had four free
GPUs at the final capacity check. The job remains safely queued for node07.

## Frozen training comparison

- Architecture: Image+Question four-action POLAR, categorical `[B,28,4]`.
- Action order: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- Objective A: duplicated one-hot action BCE.
- Objective B: exact complete-valid-set NLL.
- Training: 10 epochs, physical/effective batch 128, AdamW, learning rate
  `5e-4`, cosine schedule, warmup 10, seed `20260809`.
- Placement after cache extraction: BCE GPUs 0--1; NLL GPUs 2--3.
- Checkpointing: every epoch; selection on the frozen internal validation
  order before external evaluation.

## Data and executor contract

- Source VQA records: 6,917.
- Eligible records: 6,811.
- Train/validation: 5,945 / 866.
- Valid routes: 248,804.
- Unique image groups: 6,490.
- Explicit zero-valid exclusions: 106.
- Dataset counts: GQA 3,333; ChartQA 1,756; TextVQA 1,722.
- Executor contract:
  `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
- Local manifest SHA-256:
  `73919effd8b412ed264491c98ab136905e4057acb483459b77b5b63d50b2c7d3`.
- Local manifest-audit SHA-256:
  `5d95dd140b66510e0efed5efb7edda98e9712e4d3877e4247a85ba0de5b1a189`.

The only manifest difference from the other-server freeze is the explicit
machine-local absolute-path rebase from `/data/research/datasets/dynamic_mllm`
to `/data/dataset/dynamic_mllm`. All scientific counts, routes, splits, image
groups, exclusions, and executor contracts match.

## Runtime configs and implementation

- BCE runtime config:
  `runs/four_action_polar_node07_20260828/configs/bce.yaml`
  (`8f65261ff314a2407e6c38ac714a0269651cbbff61064f78a99b8f97ff8245c9`).
- NLL runtime config:
  `runs/four_action_polar_node07_20260828/configs/nll.yaml`
  (`20a68ee78a1920a91123f78a002d535483b6071150075a3005f3d6d653d4dc41`).
- Machine-local pipeline:
  `infra/run_four_action_polar_pipeline.sh`
  (`0dbc061e5ccf7269550a1ca27b3560a0acb774deb14044fc2bbc159b7047a2b9`
  at initial submission; re-check after any repair).
- Portable path-remap implementation:
  `experiments/build_four_action_polar_manifest.py`.
- Focused verification: 31 tests passed.

## Pipeline order

1. Extract a fresh projected-visual-row cache using four deterministic shards.
2. Finalize and checksum-audit the cache over every image group.
3. Run cache-bound BCE and NLL preflights.
4. Train BCE and NLL concurrently with two GPUs each.
5. Require three finite/progressing batch events from each objective.
6. Select each checkpoint using internal validation only.
7. Run concurrent external preflights.
8. Evaluate ChartQA, MMMU-Pro Standard/Vision, and POPE adversarial/popular/
   random with two shards per objective.
9. Merge each objective and produce the BCE-versus-NLL report.

## Evaluation contract

- ChartQA: 2,500.
- MMMU-Pro Standard: 1,730.
- MMMU-Pro Vision: 1,730.
- POPE adversarial/popular/random: 3,000 each.
- Total: 14,960 records.
- TextVQA, DocVQA, MMStar, and base MMMU are excluded prospectively.

## Monitoring commands

```bash
squeue -j 105063 -o '%.18i %.12P %.28j %.2t %.10M %.4D %R'
tail -f runs/four_action_polar_node07_20260828/slurm.log
tail -f runs/four_action_polar_node07_20260828/cache_shard_0.log
tail -f runs/four_action_polar_node07_20260828/bce_train.log
tail -f runs/four_action_polar_node07_20260828/nll_train.log
```

Do not report the training as started until the cache-bound preflights pass and
both startup-monitor JSON files exist with `passed: true`.
