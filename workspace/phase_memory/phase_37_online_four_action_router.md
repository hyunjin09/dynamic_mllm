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
  evaluation, reporting chain, and all 476 CPU tests.
- Historical runtime: smoke job 1663 failed after 39 seconds because its
  fail-closed output guard found the existing directory
  `outputs/four_action_online_router/smoke_v1`.
- Canceled at the user's request on 2026-08-28: never-started training job 1664
  and never-started external-evaluation job 1665. The fresh user queue was
  empty immediately after cancellation.
- Authorized and queued on 2026-08-29: fresh semantic smoke 1684, ten-epoch
  training 1685 after successful smoke, and external evaluation 1686 after
  successful training. Every job requests all eight H100s.
- Current bottleneck: smoke 1684 is pending for `AssocGrpGRES`; training and
  evaluation are dependency-blocked. No output exists under the fresh v2
  roots yet.
- Most recent useful observation: the existing audited VQA manifest contains
  6,811 replay-valid samples (5,945 train / 866 validation), 248,804 complete
  routes, and one executor contract across GQA, ChartQA, and TextVQA.

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

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Approved online READ/WRITE router | Uses route-conditioned hidden states and function-specific features | Tests the plan's primary hypothesis | high | selected |

## Next-Step Decision
- Deliberation mode: fast; the user explicitly authorized the already-approved
  online-router training and its three-family external evaluation on all eight
  H100s.
- Active objective and bottleneck: pass the real eight-rank semantic smoke,
  then complete ten-epoch training and checksum-bound external evaluation.
- Relevant failure used: smoke 1663 left only an empty output directory. Code
  inspection and a red/green regression showed that every rank checked for the
  directory while rank 0 created it, so late ranks mistook the new directory
  for stale output.
- Confirmed repair: only rank 0 now checks/creates the shared smoke directory;
  all ranks synchronize afterward. The focused test failed before the repair,
  then passed, and the full project suite passes 480 tests. Portable fix commit:
  `23ed41c`.
- Chosen action: submit a fresh fail-closed eight-H100
  `smoke_v2 -> training_v2 -> external_v2` Slurm chain. Each downstream job
  uses an exact `afterok` dependency; the prior empty `smoke_v1` directory and
  historical jobs remain untouched.
- Automatic execution authorized: yes.
- Authorization basis: explicit user instruction on 2026-08-29 to perform the
  training and then evaluate ChartQA, MMMU-Pro Standard/Vision, and POPE using
  eight GPUs.
- Stop condition: semantic smoke failure, frozen-contract mismatch, backbone
  gradients, invalid multi-route supervision, training integrity failure, or
  external preflight/merge failure. Resource pending alone is not a failure.

## Latest Research-Action Result
- Action taken: reproduced the smoke-directory startup race with a failing
  test, repaired rank-zero-only directory creation, passed the focused test and
  all 480 project tests, pushed fix commit `23ed41c`, and submitted a fresh
  eight-H100 fail-closed v2 chain.
- Result: jobs 1684/1685/1686 are live-PENDING with exact dependencies
  `1685=afterok:1684` and `1686=afterok:1685`. No v2 scientific output exists
  yet; all eight H100s are currently allocated to other work.
- Evidence saved: `reports/four_action_online_router_h100_relaunch_20260829.md`,
  `analysis/4action_router/experiment_log.md`, the Slurm job records, and this
  phase memory.
- Failure or issue: none in the replacement run yet. Resource pending is
  expected and does not relax any smoke or semantic gate.
- Next implication: monitor smoke 1684 when allocated. Training and evaluation
  start automatically only after their exact predecessor succeeds.
