# Cross-Server Follow-Up and Online Router V3 Launch

## Git synchronization boundary

- The H100 server fast-forwarded `main` from `34e98b3` to other-server commit
  `293c425` (`docs: record four-action router collapse`).
- The online-router smoke repair was then committed and pushed as `f6a0c42`
  (`fix: align online-router smoke warmup`).
- No force update, local reset, or discarded worktree change was used.

## Imported Phase 36 result

The pulled A6000-server reports establish that four-action upfront POLAR BCE
and exact-set NLL both completed ten epochs and complete 14,960-record external
evaluation on ChartQA, MMMU-Pro Standard/Vision, and POPE. Both objectives
selected all-FULL at all 28 layers for every internal-validation and external
record:

| Objective | External records | FULL decisions | Non-FULL decisions | Unique masks |
|---|---:|---:|---:|---:|
| Duplicated BCE | 14,960 | 418,880 | 0 | 1 |
| Exact set NLL | 14,960 | 418,880 | 0 | 1 |

This is confirmed top-1 policy collapse with zero corrections or regressions.
Its cause remains unknown. The A6000 job IDs are historical provenance on this
server. The four referenced merged result/checkpoint payloads under
`outputs/four_action_polar/node07_20260828/` were explicitly checked and are
absent here; Git transferred only compact reports/checksums.

Evidence:

- `reports/four_action_polar_action_collapse_audit_20260829.md`
- `reports/four_action_polar_node07_bce_external.md`
- `reports/four_action_polar_node07_nll_external.md`
- `workspace/phase_memory/phase_36_four_action_polar_training.md`

## Phase 37 v2 smoke failure

The separate online state-conditioned router remained authorized. V2 smoke
job 1684 subsequently ran on all eight H100s and failed the unchanged repeated-
optimization loss gate:

```text
global initial mean loss: 1.4144729376
global final mean loss:   1.4571903944
```

The smoke preserved a complete report and checkpoint. Multi-valid
supervision, routed-state conditioning, deterministic route selection,
READ/WRITE query gradients, frozen-backbone checks, and checkpoint roundtrip
were otherwise healthy.

Direct source comparison plus a local scheduler probe supported the cause:
smoke constructed AdamW at full LR `5e-4` for every one of its four updates,
whereas the frozen main-training cosine/warmup contract starts those updates at
`0`, `5e-5`, `1e-4`, and `1.5e-4`. The smoke was therefore testing a different
optimization path.

## Repair and verification

Smoke and main training now use one optimizer/scheduler constructor. The loss-
decrease criterion was not weakened.

- Regression: failed before the shared constructor existed, passed afterward.
- Online-router tests: 16/16 passed.
- Full project tests: 481/481 passed.
- Portable repair commit: `f6a0c424010c57bf91bd257195fe9fd52379a6b6`.
- Training source SHA-256:
  `de4c9bd1d150df845d86f900840b60cfd9d15cd311f9eecb1219f87c4eed09f9`.
- Frozen config SHA-256 remains:
  `37635f5acbd8842c387683e57de9e76d4a69dced1b23f57010f23e8ff579f21d`.

Dead dependency jobs 1685/1686 were canceled. The failed v2 smoke artifacts
remain preserved and were not overwritten.

## V3 smoke result and live chain

Fresh eight-H100 smoke 1690 completed `0:0` in 42 seconds and passed every
gate:

```text
global initial mean loss: 1.4144729376
global final mean loss:   0.9804229736
checkpoint roundtrip:     exact
multi-valid supervision:  exercised
routed-state conditioning: exercised
READ/WRITE query gradients: nonzero on every rank
backbone gradients:       zero
```

- Smoke report SHA-256:
  `b0a08073dbac9b4e2d8a17c165bcbdad7275cb5ff897b81037e4ec0b60ce6a61`.
- Compute estimate SHA-256:
  `b6384413fec3b4ff311c77fc657e6687b9fec9295a2cc31168db5d4eff0f9167`.

The current fail-closed chain is:

| Job | Stage | Resources | Dependency | Handoff state |
|---:|---|---|---|---|
| 1690 | semantic smoke | 8 H100, 64 CPU, 512G | none | COMPLETED `0:0` |
| 1691 | ten-epoch DDP training | 8 H100, 64 CPU, 512G | `afterok:1690` | RUNNING |
| 1692 | restricted external evaluation | 8 H100, 64 CPU, 512G | `afterok:1691` | PENDING / dependency |

Training 1691 initialized from exact commit `f6a0c42`, the frozen 6,811-row
manifest (5,945 train / 866 validation), and eight-way DDP. At the handoff
boundary all eight ranks had emitted finite samples through global step 3.
There was no completed epoch yet, so no validation or checkpoint-quality
interpretation is available.

## Machine-local wrappers and output boundaries

The intentionally ignored H100 wrappers are:

| Stage | Wrapper | SHA-256 |
|---|---|---|
| Smoke | `infra/run_four_action_online_router_smoke_v3.sh` | `b9a0a12d39bd4dab57560aba28c7e45e30a3dae8535e88eb62f624bd5ccbd13f` |
| Train | `infra/run_four_action_online_router_train_v3.sh` | `8632a5f86dce992a20311ae19a4f674cfb58657999acad378ad9989aaa486e0b` |
| Evaluate | `infra/run_four_action_online_router_eval_v3.sh` | `b0ac470a8f0f73d88cd2f388667bcc48a1d7af179b26398973df7e367436c022` |

They all pass `bash -n`. Important local paths:

```text
outputs/four_action_online_router/smoke_v3/
outputs/four_action_online_router/training_v3/
outputs/four_action_online_router/external_v3/
outputs/four_action_online_router/external_analysis_v3/
logs/slurm/four-action-online-router-{smoke,train,eval}-v3-<job>.log
logs/four_action_online_router/v3/
```

## Compute estimate and continuation boundary

The passed smoke calibration estimates approximately 1.59 wall-hours for
teacher replay, 0.45 hours for ten routed epoch validations, and 0.77 hours for
external evaluation on eight GPUs: 2.81 wall-hours / 22.47 allocated GPU-hours
combined. This excludes model-load overhead and may understate backward or
long-generation cost.

Continue by monitoring atomic epoch artifacts from job 1691. Do not interpret
partial training-sample losses. Evaluation 1692 may open external outcomes only
after training completes and selects a checkpoint from internal routed
validation. Any job IDs in the pulled A6000 reports remain historical on this
H100 server.
