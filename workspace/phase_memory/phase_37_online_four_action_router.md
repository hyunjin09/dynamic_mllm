# Phase 37: Online Four-Action Router Memory

## Current Objective
Implement and queue the `plans/four_action_train.md` online state-conditioned
four-action router using the authoritative GQA, ChartQA, and TextVQA labels,
then select by internal routed execution and evaluate the best checkpoint on
ChartQA, MMMU-Pro Standard/Vision, and POPE.

## Active Constraints
- Leave the separately authorized POLAR train/eval job 1662 pending unchanged.
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
- Queued: smoke job 1663; training job 1664 after successful smoke; external
  evaluation job 1665 after successful training.
- Blocked: no implementation blocker. Smoke 1663 is live-PENDING for
  `AssocGrpGRES` while the physical node is allocated to another user.
- Most recent useful observation: the existing audited VQA manifest contains
  6,811 replay-valid samples (5,945 train / 866 validation), 248,804 complete
  routes, and one executor contract across GQA, ChartQA, and TextVQA.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 6,811 samples and 248,804 routes with no image-group leakage | `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json` | Supplies checksum-frozen VQA supervision without rebuilding labels | confirmed |
| Executor contract `d8f524...` for every included sample | same audit and source records | Binds replay states to current unified semantics | confirmed |
| Direct upfront predictors can fit likelihood without useful complete routes | `reports/binary_polar_full10_polar_matched_results.md` | Supports execution-based selection and the approved online architecture | confirmed |
| Current POLAR job 1662 remains pending | live Slurm queue | The new run is separate and must not cancel or mutate it | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Direct upfront factorized binary prediction | Best internal checkpoint remained effectively ALL-ON with no frozen-60 rescue | supported architecture/objective limitation for that run | binary full10 report | Select the online router by actual routed execution, not node likelihood alone | Do not reuse upfront POLAR states as online training states |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Approved online READ/WRITE router | Uses route-conditioned hidden states and function-specific features | Tests the plan's primary hypothesis | high | selected |

## Next-Step Decision
- Deliberation mode: fast; the user explicitly approved the architecture and
  requested training plus external evaluation.
- Active objective and bottleneck: produce a semantically gated online router;
  the immediate bottleneck is implementation, then queued GPU availability.
- Relevant memory item used: route utility is context-dependent and direct
  upfront prediction previously failed execution selection.
- Confirmed observation: all required labels, model, evaluation data, and
  executor code are present; another user's exclusive job currently owns the
  server and job 1662 is already pending.
- Unverified interpretation: physical batch one per DDP rank with 16-step
  accumulation will fit every native visual-token geometry.
- Diagnosis: unknown until the eight-sample real-GPU smoke.
- Chosen action: implement the exact V1 architecture and queue a fail-closed
  smoke -> ten-epoch train -> best-checkpoint external-eval chain using all
  eight H100s at each GPU stage.
- Automatic execution authorized: yes.
- Authorization basis: the user's explicit request to follow
  `plans/four_action_train.md`, train on the three VQA label sets, preserve the
  current pending run, and queue this new train/eval.
- Stop condition: smoke semantic/gradient/determinism failure, executor-contract
  mismatch, non-reproducible sampling, backbone gradients, or invalid
  multi-route supervision.

## Latest Research-Action Result
- Action taken: implemented, CPU-validated, and queued the complete online
  smoke -> train -> external-evaluation action.
- Result: 476 tests pass; label/trie audit passes; jobs 1663/1664/1665 are live
  in Slurm with exact fail-closed dependencies. Existing job 1662 is unchanged.
- Evidence saved: `analysis/4action_router/implementation_audit.md`,
  `label_and_trie_audit.{json,md}`, `provisional_compute_estimate.md`, and
  `experiment_log.md`.
- Failure or issue: none.
- Next implication: wait for scheduler allocation; inspect the real-GPU smoke
  before interpreting or relying on the downstream training run.
