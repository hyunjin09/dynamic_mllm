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
  from epoch 2 onward. A subsequent deterministic label/sampler audit supports
  action/prefix imbalance and teacher-forcing coverage as contributors, while
  the exact C2C all-FULL route remains a plausible additional shortcut; no
  single sole cause is established.
- Current bottleneck: the unchanged sampler never trains the latest valid
  all-FULL-prefix deviation boundary for 1,045/2,397 W2C samples. A small
  isolated mandatory-boundary coverage/capacity pilot is the recommended next
  diagnostic, but no new training or replacement method is authorized.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 6,811 samples and 248,804 routes with no image-group leakage | `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json` | Supplies checksum-frozen VQA supervision without rebuilding labels | confirmed |
| Executor contract `d8f524...` for every included sample | same audit and source records | Binds replay states to current unified semantics | confirmed |
| Direct upfront predictors can fit likelihood without useful complete routes | `reports/binary_polar_full10_polar_matched_results.md` | Supports execution-based selection and the approved online architecture | confirmed |
| POLAR job 1662 failed before this cancellation | Slurm accounting and `logs/slurm/four-action-polar-train-eval-v1-1662.log` | It was terminal and was not one of the canceled queue entries | confirmed |
| Nine complete online-router validation histories | `outputs/four_action_online_router/training_v3/history.json` (SHA-256 `a1b9961e...`) | Establishes repeated zero-rescue/all-FULL behavior before early stop | confirmed |
| Deterministic collapse label/sampler audit | `reports/four_action_router_collapse_label_audit_20260829.md` | Distinguishes C2C route shortcut, action imbalance, and missing W2C boundary exposure | confirmed |

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
- Deliberation mode: deep. Both the upfront POLAR objectives and the online
  state-conditioned router converged to essentially all-FULL, so this is a
  repeated outcome across two architectures and the proposed remedies alter
  the training population or supervision.
- Active objective and bottleneck: distinguish the user's proposed C2C
  universal-FULL shortcut from broader label geometry, especially whether W2C
  prefixes and sampled teacher routes are themselves overwhelmingly FULL.
- Confirmed observation: the online run produced zero W2C rescues for nine
  completed epochs and was exactly all-FULL in epochs 2--8. Both completed
  upfront four-action POLAR variants also selected all-FULL everywhere.
- Unverified interpretations: C2C may dominate because all-FULL is a valid
  route for every C2C sample; alternatively, rare W2C deviation points,
  route/action imbalance, per-layer factorization, teacher-forcing exposure,
  or greedy 28-step decoding may independently make FULL the easiest policy.
- Chosen action and strongest objection: perform one bounded, CPU-only audit of
  the existing manifest and the exact epoch sampler. Quantify all-FULL route
  prevalence, non-FULL alternatives, complete-route action frequencies, and
  prefix-level valid-action masks separately for W2C and C2C, including
  counterfactual W2C-only and C2C-without-all-FULL label views. The strongest
  objection is that label statistics cannot prove an optimization mechanism,
  but they can decide whether removing C2C or only its trivial route is a
  coherent next experiment before spending GPUs.
- How this differs from failed attempts: this does not retrain a renamed
  variant; it tests a concrete supervision-level explanation using the frozen
  labels and exact sampler that produced the failed run.
- Review and final ranking: an independent read-only review revised the
  provisional bundled pilot. The clean next diagnostic is to retain C2C and
  the unchanged loss while guaranteeing one visit to every W2C latest
  all-FULL-prefix mandatory-deviation node. Removing only the C2C all-FULL
  route is the next separate ablation; dropping C2C is not recommended. Do not
  bundle boundary coverage, class weighting, and on-policy exposure in the
  first retry because the result would be causally ambiguous.
- Authorization and stop condition: authorized by the user's request to analyze
  the collapse and identify what to check. The label/sampler audit is complete.
  Do not launch the recommended pilot without explicit authorization.

## Latest Research-Action Result
- Action taken: deterministically audited the frozen train/validation manifest,
  the exact ten-epoch online sampler, prefix-trie valid-action masks, upfront
  objective geometry, and deployed versus teacher-forced action behavior. No
  inference, training, or label mutation was run.
- Result: 98.675% of C2C train samples contain all-FULL, but W2C sampled teacher
  actions are themselves 76.782% FULL. Under exact 50:50 sampling, FULL is
  valid at 72.880% and uniquely valid at 55.360% of nodes, versus singleton
  READ/WRITE rates of 3.107%/2.216%. The ten-epoch sampler never visits the
  latest all-FULL-prefix deviation boundary for 1,045/2,397 W2C samples. At
  those boundaries FULL is invalid, while READ/WRITE are valid for
  43.388%/52.733% of samples.
- Diagnosis: supported contributors are action/prefix imbalance, zero C2C
  READ/WRITE positives, teacher-forcing/free-rollout exposure, and upfront
  checkpoint/objective geometry. Whether the state features/head can fit the
  explicitly covered boundaries remains unknown.
- Evidence saved: `reports/four_action_router_collapse_label_audit_20260829.md`,
  the frozen manifest/configs, upfront histories, online history, and this
  memory.
- Next implication: do not remove C2C wholesale and do not assume deleting its
  all-FULL route is sufficient. Pending explicit authorization, first isolate
  W2C mandatory-boundary coverage with the unchanged router/loss/C2C set, then
  test C2C all-FULL removal separately if the capacity/coverage gate passes.
