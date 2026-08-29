# Phase 36: Four-Action POLAR Training Memory

## Current Objective
Run two Image+Question four-action POLAR objectives—duplicated one-hot BCE and
exact valid-set NLL—sequentially with the same two GPUs, select checkpoints on
internal validation, and complete the prospectively restricted external
evaluation.

## Active Constraints
- Inputs are Image+Question only: frozen Qwen3 question-token features plus
  cached Qwen2.5-VL projected visual rows entering decoder layer 0.
- Actions are categorical in executor order `IGNORE`, `READ_ONLY`,
  `WRITE_ONLY`, `FULL`; routes have 28 actions.
- Match the binary full10 optimizer/schedule settings, save and validate every
  epoch, and evaluate only ChartQA, MMMU-Pro Standard/Vision, and POPE.
- GPU work must use Slurm. The current-server run uses one two-GPU allocation;
  BCE uses both GPUs first, then NLL uses the same two GPUs.

## Current-Server Resume (2026-08-28)

- Other-server job `1662` is historical here, not live. The latest remote
  evidence confirms that it completed its server-local visual cache and both
  training processes before failing in external-evaluation preflight. Those
  machine-local artifacts were not imported to this server.
- The complete 6,917-record VQA label cache is locally present. A deterministic
  path-rebased manifest reproduces the frozen 6,811 eligible records,
  5,945/866 split, 248,804 routes, 6,490 image groups, 106 exclusions, and
  executor contract exactly; only machine-local absolute paths and their file
  hashes differ.
- Static BCE and NLL preflights pass against the local Qwen2.5-VL/Qwen3
  snapshots and the frozen 14,960-row ChartQA/MMMU-Pro/POPE population.
- The authorized current action is a fail-closed two-GPU pipeline: two-shard
  fresh visual-cache extraction, cache-bound preflights, sequential two-GPU
  BCE/NLL training, checkpoint selection, then sequential two-shard evaluation
  per objective. Initial batch logs must be inspected before relying on the
  full run.

## Current State
- Done: action/model/loss/decoder/data contracts; checksum-audited training
  manifest; both ten-epoch BCE/NLL runs; checkpoint selection; both complete
  14,960-record external evaluations; checksum-bound merges; and deterministic
  action-collapse audit.
- Final result: BCE epoch 8 and NLL epoch 6 both select all-FULL on every one of
  866 internal-validation records and every one of 14,960 external records.
  Each external output contains 418,880/418,880 FULL layer decisions, one
  unique mask, zero non-FULL decisions, and zero corrections or regressions.
- Interpretation: top-1 policy collapse is confirmed. The cause remains
  `unknown`; this result does not prove that non-argmax logits or internal
  features are input-independent.
- Cross-server boundary: the merged per-record outputs and checkpoints remain
  machine-local to the A6000 server. They are not present on this H100 server
  merely because their reports and checksums are tracked in Git.
- Bottleneck: none for the authorized Phase 36 run. No new training,
  architecture change, or diagnostic experiment is selected.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 6,811 eligible records: 5,945 train, 866 validation; 248,804 routes | `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json` | Freezes the actual supervised population and split | confirmed |
| 106 zero-valid records have zero replay-valid routes | same audit, `excluded_zero_valid_records` | They cannot supervise BCE or NLL without fabricating labels | confirmed |
| Binary full10 hyperparameters and checkpoint cadence | `configs/binary_polar_full10_polar_bce_v1.yaml`, `reports/binary_polar_training_architecture_inference.md` | Defines the matched training comparison | confirmed |
| Imported external data and local Qwen3/Qwen2.5 snapshots are present | `workspace/dataset_inventory.md` and direct local load checks | External evaluation can be prepared without importing more assets | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Require every one of 6,917 source records in the training manifest | Builder stopped at the first record with zero valid routes | supported | manifest builder output and exclusion audit | Exclude all zero-valid records explicitly and preserve their identities/reasons | Do not fabricate routes or silently drop records |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Fresh projected-visual-row cache | Matches the proven binary Image+Question representation | Supplies predictor image input without outcome leakage | medium | completed on both servers; payloads remain machine-local |
| Duplicated one-hot action BCE | Direct four-class generalization of prior POLAR BCE | Tests matched POLAR-style supervision | high | training complete; external evaluation in progress here |
| Exact complete-valid-set NLL | Optimizes mass over all valid routes without route duplication | Tests categorical set supervision | high | training complete; external evaluation waits behind BCE here |

## Next-Step Decision
- Deliberation mode: none; the authorized action is complete.
- Confirmed observation: both objectives collapse to all-FULL on the complete
  internal-validation and external populations, with positive FULL-versus-
  runner-up margins at every external sample-layer decision.
- Evidence: `reports/four_action_polar_action_collapse_audit_20260829.md`,
  `reports/four_action_polar_node07_bce_external.md`, and
  `reports/four_action_polar_node07_nll_external.md`.
- Chosen action: stop Phase 36 after preserving the negative result. Do not
  infer a cause or launch another predictor change without a separate bounded
  decision and explicit authorization.

## Latest Research-Action Result
- Action taken: A6000 job `105451` completed both frozen external evaluations,
  exact shard merges, integrity checks, and deterministic action parsing.
- Result: both BCE and NLL are all-FULL for all 14,960 records. Predicted and
  unified-FULL accuracy are therefore identical mechanically for ChartQA,
  MMMU-Pro Standard/Vision, and all POPE splits.
- Evidence saved: `reports/four_action_polar_action_collapse_audit_20260829.md`
  plus the two objective-specific external reports and checksum sidecars.
- Failure or issue: none in final integrity. The scientific outcome is a
  negative routing result: no conditional non-FULL policy behavior appeared.
- Next implication: preserve the result and stop; the separate online-router
  Phase 37 remains the currently running architecture experiment.

## Current-Server Launch Result (2026-08-28)

- Action taken: reconstructed the frozen VQA manifest with an explicit
  `/data/research/datasets/dynamic_mllm` to `/data/dataset/dynamic_mllm` path
  rebase, validated local model/evaluation assets, restored the machine-local
  project scheduler, and submitted the four-GPU node07 pipeline as job
  `105063`.
- Result: both static preflights pass. Job `105063` is pending with reason
  `ReqNodeNotAvail, UnavailableNodes:node07`; four one-GPU jobs occupied the
  previously free half of node07 before this submission. No cache extraction
  or training batch has started.
- Frozen current-server placement: one four-GPU allocation; shared cache uses
  four shards; BCE uses visible GPUs 0--1; NLL uses 2--3; full evaluation uses
  two shards per objective over ChartQA, MMMU-Pro Standard/Vision, and all
  three POPE splits.
- Startup gate: each training process must emit three finite, progressively
  larger batch events with the correct objective within 30 minutes of training
  process launch. The pipeline stops both objectives if either startup monitor
  fails.
- Evidence: `reports/four_action_polar_node07_launch_20260828.md` and
  `runs/four_action_polar_node07_20260828/slurm.log`.
- Superseded: job `105063` was cancelled before execution after the user moved
  the run to a two-GPU tmux allocation. See the latest result and
  `reports/four_action_polar_tmux2_launch_20260828.md`.

## Preparation Result
- The action/model/loss/data/metric interfaces, resumable cache extractor,
  resumable ten-epoch trainer, per-epoch full validation/checkpoint selection,
  restricted external evaluator, shard merger, and objective comparison are
  implemented.
- Static BCE and NLL preflights pass and independently resolve the same cache
  contract SHA-256
  `12dfb9d50ee2e962ee318acf7bd1f08cb43b7e05d24a906d205fdd4b74233045`.
- Verification: 460 project tests pass. Whole-repository pytest discovery also
  enters the imported reference bundle and fails collection because that
  bundle expects its own Python path; the project-local `tests/` suite is
  clean.
- No GPU workload was launched. The next authorized runtime action is the
  eight-shard fresh visual-feature extraction inside one Slurm allocation,
  followed by CPU finalization and both full preflights.
- Historical note: the preceding bullets describe the preparation boundary;
  jobs `1662` and `105068` subsequently executed independent cache and training
  stages as recorded in `Latest Research-Action Result` above.

## Evaluation-Only Resume (2026-08-28)

- Authorized action: repair the external-evaluation runtime contract and run
  both frozen checkpoint evaluations sequentially on one node06 GPU. Training
  is not rerun.
- Confirmed root cause: Transformers 5.3 3-D position-ID construction received
  GPU prompt/attention tensors but CPU `mm_token_type_ids` from the shared-
  prefix input builder.
- Repair: move `mm_token_type_ids` to the prompt-embedding device inside
  `_position_ids`; a focused red/green regression and all nine focused tests
  pass.
- Validity result: BCE preflight passed 6/6 exact generated-token parity,
  prediction/evaluator parity, and repeated-execution determinism fixtures.
- Monitoring repair: job `105448` stopped after its first atomic 32-row chunk
  because the monitor expected the wrong result-field nesting. The evaluator
  itself completed 43 healthy rows; only the committed 32 were retained. The
  monitor now validates `predicted.score`, and replacement job `105451` resumed
  without duplicate UIDs or incomplete rows.
- Live boundary: job `105451`, node06, one A6000, BCE then NLL. At the
  2026-08-29 Git handoff boundary, 6,240/14,960 BCE rows were present exactly
  once. NLL had not started.
- Evidence: `reports/four_action_polar_tmux2_launch_20260828.md`,
  `runs/four_action_polar_eval_node06_20260828/`, and
  `outputs/four_action_polar/node07_20260828/eval_bce/`.
- Next implication: allow job `105451` to finish BCE and NLL, then inspect only
  checksum-bound merged reports. Do not interpret the partial rows.

## Completed External Evaluation and Collapse Audit (2026-08-29)

- Completion: BCE and NLL each completed all 14,960 prospectively selected
  records exactly once; both merged integrity manifests report `PASS`.
- Confirmed observation: both objectives selected the all-FULL 28-layer mask
  for every external record. Each has one unique top-1 mask, 418,880 FULL
  layer decisions, zero non-FULL decisions, and zero samples with any non-FULL
  action.
- Confidence: high. This is direct parsing of both merged per-record outputs,
  and every FULL-versus-runner-up logit margin is positive.
- Margin evidence: duplicated BCE minimum/median/mean FULL margin is
  0.12549/0.59766/1.09239; exact set NLL is
  0.36719/4.52734/4.43887.
- Behavioral implication: predicted and baseline accuracy are identical by
  construction because every predicted route is the baseline FULL route;
  W-to-C and C-to-W counts must both be zero.
- Interpretation: supported top-1 policy collapse. It is not yet evidence that
  every internal feature or non-argmax logit is input-independent. The cause of
  FULL dominance remains `unknown`.
- Evidence: `reports/four_action_polar_action_collapse_audit_20260829.md` and
  both merged `external_results_v1.jsonl` files.
- Next-step decision: none selected in this result-interpretation action. Do
  not start another training or architecture change without a separate bounded
  decision.
