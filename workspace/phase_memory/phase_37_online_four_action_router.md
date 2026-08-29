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
- Current runtime: eight-H100 smoke 1690 completed `0:0` in 42 seconds and
  passed every semantic/gradient/checkpoint gate with mean loss
  1.414473 -> 0.980423. Ten-epoch job 1691 is running on all eight H100s from
  exact commit `f6a0c42`; all ranks emitted finite samples through global step
  3. Evaluation 1692 is pending on exact dependency `afterok:1691`.
- Current bottleneck: complete ten atomically saved epochs and routed
  validation before opening external outcomes. No completed epoch exists yet.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 6,811 samples and 248,804 routes with no image-group leakage | `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json` | Supplies checksum-frozen VQA supervision without rebuilding labels | confirmed |
| Executor contract `d8f524...` for every included sample | same audit and source records | Binds replay states to current unified semantics | confirmed |
| Direct upfront predictors can fit likelihood without useful complete routes | `reports/binary_polar_full10_polar_matched_results.md` | Supports execution-based selection and the approved online architecture | confirmed |
| POLAR job 1662 failed before this cancellation | Slurm accounting and `logs/slurm/four-action-polar-train-eval-v1-1662.log` | It was terminal and was not one of the canceled queue entries | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Direct upfront factorized binary prediction | Best internal checkpoint remained effectively ALL-ON with no frozen-60 rescue | supported architecture/objective limitation for that run | binary full10 report | Select the online router by actual routed execution, not node likelihood alone | Do not reuse upfront POLAR states as online training states |
| V2 eight-rank online smoke | Mean loss increased 1.414473 -> 1.457190 despite finite gradients and valid semantics | supported: smoke skipped the main warmup/cosine schedule and applied full LR `5e-4` four times | `smoke_v2/smoke_report.json`, source comparison, local LR probe | Share optimizer/scheduler construction; preserve the strict loss gate | Do not weaken the loss gate or repeat full-LR tiny-batch smoke |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Approved online READ/WRITE router | Uses route-conditioned hidden states and function-specific features | Tests the plan's primary hypothesis | high | selected |

## Next-Step Decision
- Deliberation mode: fast; the user explicitly authorized the already-approved
  online-router training and its three-family external evaluation on all eight
  H100s.
- Active objective and bottleneck: allow validated ten-epoch training 1691 to
  finish every epoch/checkpoint/validation transaction, then run the already-
  queued restricted external evaluation 1692.
- Relevant failure used: the v2 loss gate caught a smoke-only optimization
  mismatch; preserving the gate distinguished a scheduler defect from a
  semantic/executor failure.
- Confirmed repair: smoke and training share one optimizer/scheduler builder;
  the focused regression and all 481 tests pass. Real v3 smoke 1690 then
  passed every gate with a 30.7% mean-loss reduction.
- Chosen action: monitor job 1691 for finite routed samples and the first
  atomic epoch boundary. Evaluation 1692 starts only through `afterok:1691`.
- Automatic execution authorized: yes.
- Authorization basis: explicit user instruction on 2026-08-29 to perform the
  training and then evaluate ChartQA, MMMU-Pro Standard/Vision, and POPE using
  eight GPUs.
- Stop condition: frozen-contract mismatch, backbone gradients, invalid multi-
  route supervision, non-finite training, epoch transaction failure, or
  external preflight/merge failure. Resource pending alone is not a failure.

## Latest Research-Action Result
- Action taken: diagnosed the v2 smoke loss failure, added a red/green schedule-
  contract regression, shared optimizer/scheduler construction across smoke
  and training, passed 481 tests, pushed commit `f6a0c42`, canceled dead jobs
  1685/1686, and launched the fresh v3 chain 1690/1691/1692.
- Result: smoke 1690 passed in 42 seconds with exact checkpoint roundtrip,
  multi-valid supervision, routed-state conditioning, nonzero READ/WRITE query
  gradients, zero backbone gradients, and loss 1.414473 -> 0.980423. Training
  1691 is live-RUNNING; all eight ranks have emitted finite samples through
  global step 3. Evaluation 1692 is dependency-blocked as intended.
- Evidence saved: `outputs/four_action_online_router/smoke_v3/smoke_report.json`,
  `analysis/4action_router/calibrated_compute_estimate_v3.json`,
  `reports/four_action_online_router_h100_v3_launch_20260829.md`, Slurm logs,
  and this phase memory.
- Failure or issue: no current training failure. No completed epoch exists yet,
  so validation behavior and checkpoint quality remain unknown.
- Next implication: continue job 1691; inspect only atomic epoch artifacts and
  do not interpret partial training samples as scientific results.
