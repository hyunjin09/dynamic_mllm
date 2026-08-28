# Online Four-Action Router H100 Relaunch Handoff

## Scope and authorization

On 2026-08-29 the user explicitly authorized the online state-conditioned
four-action router from `plans/four_action_train.md`: run the real semantic
smoke, train for ten epochs using all eight H100s, and then evaluate the
validation-selected checkpoint on ChartQA, MMMU-Pro Standard/Vision, and POPE.
This is separate from the completed upfront POLAR BCE/NLL training.

## Portable source boundary

- Git branch: `main`.
- Portable repair commit: `23ed41c` (`fix: serialize online-router smoke setup`).
- Training source SHA-256:
  `5782b04dbe2f43ec2f3fb30629f41d47c1665512eefa1f005de7ba80575e432c`.
- Frozen config SHA-256:
  `37635f5acbd8842c387683e57de9e76d4a69dced1b23f57010f23e8ff579f21d`.
- Manifest SHA-256:
  `6a50ca1a2d5c512d7bd8cededfc4732c93258264ae471e1fd4653c6c16637c28`.
- Manifest-audit SHA-256:
  `8740c87b900d3413e5aebfec4a35cf0f6ddfcebf8ebdfde80e5d06e1191a8823`.
- Source-manifest SHA-256:
  `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7`.
- Verification: the focused regression failed before the fix, passed after it,
  the complete router test file passed 15/15, and the project suite passed
  480/480 in 32.54 seconds.

## Failure repaired

Historical smoke 1663 left an empty
`outputs/four_action_online_router/smoke_v1` directory. Every DDP rank checked
whether the directory existed, but rank 0 created it before some other ranks
performed that check. Those ranks then raised the overwrite guard. The repair
makes rank 0 alone check and create the shared smoke directory before all ranks
synchronize. The historical v1 directory remains untouched.

## Machine-local launch wrappers

These files are intentionally ignored by Git and must be recreated or verified
on another server:

| Stage | Wrapper | SHA-256 |
|---|---|---|
| Smoke | `infra/run_four_action_online_router_smoke_v2.sh` | `bdb1aac80dc9895238247f18f3df050e760b2bce9c7b8385ce8fb7df2e0b9125` |
| Train | `infra/run_four_action_online_router_train_v2.sh` | `4d6677975c73325b113d4b0787407e4d323240fccaf06305e02d4e1ce51579ae` |
| Evaluate | `infra/run_four_action_online_router_eval_v2.sh` | `72074d52fe8f5925cb5d64197bfe2c2819f29027c220af51a13a60281aa19d16` |

All three pass `bash -n`. They invoke the project-local `.venv`, export
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and use fresh v2 output paths.

## Environment and assets verified

- Project-local `.venv`: Python 3.12.7, PyTorch 2.6.0+cu124, CUDA userspace
  12.4, Transformers 5.3.0, and `uv pip check` PASS across 84 packages.
- Model link resolves under the allowed `/data/research/models` root to the
  pinned Qwen2.5-VL revision `cc594898...`.
- `datasets` resolves under `/data/research/datasets/dynamic_mllm`.
- The 6,811-row training manifest, 866-row validation split, 248,804 routes,
  model snapshot, source inventory, and exact 14,960-row external data root are
  present with the frozen hashes above.

## Slurm chain

Submitted at 2026-08-29 01:12:56 KST:

| Job | Stage | Resources | Dependency | Initial state |
|---:|---|---|---|---|
| 1684 | eight-sample semantic smoke | 8 H100, 64 CPU, 512G, 4h | none | PENDING / `AssocGrpGRES` |
| 1685 | ten-epoch DDP training | 8 H100, 64 CPU, 512G, 7d | `afterok:1684` | PENDING / dependency |
| 1686 | restricted external evaluation | 8 H100, 64 CPU, 512G, 7d | `afterok:1685` | PENDING / dependency |

`scontrol show job` directly verified both dependency fields. At submission,
all eight physical H100s were at 98--99% utilization under other work. The
replacement jobs do not interfere and wait under normal Slurm policy.

## Fresh output and log boundaries

- Smoke: `outputs/four_action_online_router/smoke_v2/`.
- Training/checkpoints: `outputs/four_action_online_router/training_v2/`.
- External rows: `outputs/four_action_online_router/external_v2/`.
- Merged external analysis:
  `outputs/four_action_online_router/external_analysis_v2/`.
- Slurm logs:
  `logs/slurm/four-action-online-router-{smoke,train,eval}-v2-<job>.log`.
- Per-shard evaluation logs: `logs/four_action_online_router/v2/`.

All four v2 output roots were absent immediately before submission. No ignored
checkpoint or result is implied to exist on another server merely because this
report records the launch.

## Continuation and interpretation boundary

Monitor smoke 1684 first. Require `smoke_v2/smoke_report.json` to pass every
semantic, gradient, frozen-backbone, determinism, loss-decrease, routed-state,
and checkpoint-roundtrip gate before relying on training. Training 1685 and
evaluation 1686 start automatically only through their `afterok` dependencies.
Do not interpret partial validation or external rows. Final scientific evidence
requires ten atomically completed epochs, a pre-external selected checkpoint,
all eight exact external shards, and the checksum-bound merge/report.
