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
- Paused: no fix, replacement smoke, training, or evaluation launch is
  authorized by the cancellation request.
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
- Deliberation mode: none; this was an explicit scheduler cancellation.
- Active objective and bottleneck: paused after cancellation; the latest smoke
  did not reach semantic validation because an existing-output guard stopped it.
- Confirmed observation: jobs 1664 and 1665 were pending, never started, and
  are now `CANCELLED by 1003`; the fresh user queue is empty.
- Diagnosis: the direct smoke failure is confirmed; whether the existing smoke
  payload is complete and reusable is unverified and was not investigated.
- Chosen action: preserve the cancellation and terminal-job evidence only.
- Automatic execution authorized: no.
- Authorization basis: the user's explicit request to cancel the two queued
  jobs did not authorize a fix or replacement run.
- Stop condition: satisfied when both exact IDs were canceled and absence from
  the fresh queue was verified.

## Latest Research-Action Result
- Operational action taken: canceled pending jobs 1664 and 1665 on the user's
  explicit request and verified a fresh empty user queue.
- Result: neither job started; smoke 1663 and POLAR job 1662 were already
  terminal failures and were not canceled by this action.
- Evidence saved: Slurm accounting, the two Slurm logs, this phase memory,
  `workspace/workflow_state.md`, and
  `analysis/4action_router/experiment_log.md`.
- Failure or issue: smoke 1663 found an existing output directory before
  semantic validation. No diagnosis beyond that direct observation is claimed.
- Next implication: do not fix, delete/reuse the existing smoke output, or
  relaunch any part of the chain without a new explicit request.
