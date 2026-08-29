# Phase 37: Online Four-Action Router Memory

## Current Objective
Implement and queue the `plans/four_action_train.md` online state-conditioned
four-action router using the authoritative GQA, ChartQA, and TextVQA labels,
then select by internal routed execution and evaluate the best checkpoint on
ChartQA, MMMU-Pro Standard/Vision, and POPE.

## Active Constraints
- Treat the separately authorized POLAR train/eval job 1662 as historical; it
  had already failed before the online chain was canceled and must not be
  relaunched implicitly.
- Use only GQA, ChartQA, and TextVQA training labels; exclude both WeMath
  datasets from this run by explicit user instruction.
- Use the frozen Qwen2.5-VL-7B revision and exact unified four-action executor
  that generated the labels; Qwen remains frozen.
- Train one online router with 8-way DDP, one Qwen/router replica per H100.
- Supervise exact routed-prefix states with a prefix trie and set-valued next
  action loss; do not train from all-FULL states or collapse multi-route labels.
- Save every epoch and select by balanced W2C rescue/C2C preservation before
  external outcomes are read.

## Current State
- Done: architecture, checksum-frozen label/trie audit, online executor hook,
  router, exact supervision, DDP training/validation, checkpointing, external
  evaluation, reporting chain, smoke-scheduler repair, and all 481 CPU tests.
- Historical runtime: smoke job 1663 failed after 39 seconds because its
  fail-closed output guard found the existing directory
  `outputs/four_action_online_router/smoke_v1`.
- Canceled at the user's request on 2026-08-28: never-started training job 1664
  and never-started external-evaluation job 1665. The fresh user queue was
  empty immediately after cancellation.
- Historical v2 runtime: smoke 1684 ran for 42 seconds and failed only the
  repeated-tiny-batch loss-decrease gate (1.414473 -> 1.457190). It preserved
  a complete report and checkpoint. Dead dependents 1685/1686 were canceled.
- Supported diagnosis and repair: smoke had used full LR `5e-4` for all four
  updates while main training uses cosine warmup (LRs 0, `5e-5`, `1e-4`,
  `1.5e-4` for its first four steps). Shared optimizer/scheduler construction
  now enforces the frozen training contract in both modes. Regression and full
  suite pass; portable fix commit `f6a0c42` is pushed.
- Final v3 runtime: eight-H100 smoke 1690 completed `0:0` and passed every
  semantic/gradient/checkpoint gate. Training job 1691 then completed nine
  atomic epochs/validations before the user authorized early stopping at epoch
  10 step 478/480. Jobs 1691 and 1692 were canceled at 14:19:22 KST; external
  evaluation never started and `external_v3` remains absent.
- Preserved boundary: epochs 1--9 each contain a checksum-valid checkpoint and
  exactly 866 unique validation outputs; there are no temporary or zero-byte
  files. No official `best_checkpoint.json` or `training_summary.json` exists
  because the ten-epoch transaction intentionally did not complete.
- Final behavioral observation: every completed epoch has zero W2C rescues.
  Epochs 2--8 execute exactly all-FULL for all 24,248 validation layer
  decisions; epoch 9 has 24,247 FULL and one IGNORE. C2C preservation is 1.0
  from epoch 2 onward. The cause of policy collapse remains unknown.
- Current bottleneck: none. Phase 37 is stopped with a preserved negative
  internal-validation result; no external evaluation or replacement method is
  authorized.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 6,811 samples and 248,804 routes with no image-group leakage | `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json` | Supplies checksum-frozen VQA supervision without rebuilding labels | confirmed |
| Executor contract `d8f524...` for every included sample | same audit and source records | Binds replay states to current unified semantics | confirmed |
| Direct upfront predictors can fit likelihood without useful complete routes | `reports/binary_polar_full10_polar_matched_results.md` | Supports execution-based selection and the approved online architecture | confirmed |
| POLAR job 1662 failed before this cancellation | Slurm accounting and `logs/slurm/four-action-polar-train-eval-v1-1662.log` | It was terminal and was not one of the canceled queue entries | confirmed |
| Nine complete online-router validation histories | `outputs/four_action_online_router/training_v3/history.json` (SHA-256 `a1b9961e...`) | Establishes repeated zero-rescue/all-FULL behavior before early stop | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Direct upfront factorized binary prediction | Best internal checkpoint remained effectively ALL-ON with no frozen-60 rescue | supported architecture/objective limitation for that run | binary full10 report | Select the online router by actual routed execution, not node likelihood alone | Do not reuse upfront POLAR states as online training states |
| V2 eight-rank online smoke | Mean loss increased 1.414473 -> 1.457190 despite finite gradients and valid semantics | supported: smoke skipped the main warmup/cosine schedule and applied full LR `5e-4` four times | `smoke_v2/smoke_report.json`, source comparison, local LR probe | Share optimizer/scheduler construction; preserve the strict loss gate | Do not weaken the loss gate or repeat full-LR tiny-batch smoke |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Approved online READ/WRITE router | Uses route-conditioned hidden states and function-specific features | Tests the plan's primary hypothesis | high | stopped after nine zero-rescue validations |

## Next-Step Decision
- Deliberation mode: standard; the user proposed stopping the near-complete
  training and canceling external evaluation after repeated all-FULL behavior.
- Active objective and bottleneck: determine whether any remaining compute can
  change the primary internal routed-execution decision.
- Confirmed observation: nine atomic epochs pass runtime/integrity checks, but
  every epoch has zero W2C rescues. Epochs 2--8 execute exactly all-FULL over
  all 24,248 validation layer decisions; epoch 9 differs by one IGNORE. C2C
  preservation is 1.0 from epoch 2 onward. At the decision boundary epoch 10
  had reached step 478/480 but had no atomic checkpoint or validation result.
- Unverified interpretation: why the online router converged to FULL remains
  unknown. The repeated behavior is sufficient to reject this run's primary
  rescue objective without diagnosing the cause.
- Viable alternatives considered: finish epoch 10 then stop; cancel training
  and evaluation immediately; or complete external evaluation. Finishing
  external evaluation is not decision-relevant after nine zero-rescue internal
  validations. Completing epoch 10 would preserve the planned schedule but is
  unlikely to reverse nine epochs during the near-zero-LR tail.
- Chosen action and strongest objection: cancel jobs 1691 and 1692 now,
  preserve nine complete checkpoints/validations, and do not run external
  evaluation. The strongest objection is losing the final atomic epoch, but it
  does not justify continued eight-GPU use given the repeated result.
- How this differs from failed attempts: this stops on repeated valid execution
  evidence rather than bypassing a semantic gate or modifying the objective.
- Authorization and stop condition: explicitly authorized by the user's
  2026-08-29 stop request. Stop when both jobs are terminal and the nine-epoch
  artifact boundary is verified intact.

## Latest Research-Action Result
- Action taken: after nine repeated zero-rescue validations, compared immediate
  stop, finishing epoch 10 only, and full external evaluation. The user-
  authorized immediate stop was selected; jobs 1691 and 1692 were canceled.
- Result: job 1691 is terminal `CANCELLED` after 1:11:23 and job 1692 is
  terminal `CANCELLED` without starting. Nine checkpoints and 7,794 validation
  rows are intact and checksum/UID audits pass. Epoch 10 produced no partial
  checkpoint, and no external row was generated.
- Evidence saved: `outputs/four_action_online_router/training_v3/history.json`,
  nine epoch metadata/checkpoint/validation triplets, Slurm accounting/logs,
  `reports/four_action_online_router_early_stop_20260829.md`, and this memory.
- Failure or issue: the implementation and optimization ran correctly, but the
  deployed validation policy was essentially all-FULL and achieved zero W2C
  rescue at every completed epoch. Why it collapsed is not diagnosed.
- Next implication: preserve the negative result and stop. Any objective,
  architecture, weighting, or supervision change is a new research action and
  requires explicit approval.
