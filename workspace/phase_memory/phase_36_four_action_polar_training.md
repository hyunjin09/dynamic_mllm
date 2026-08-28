# Phase 36: Four-Action POLAR Training Memory

## Current Objective
Run two Image+Question four-action POLAR objectives—duplicated one-hot BCE and
exact valid-set NLL—with four GPUs each, select checkpoints on internal
validation, and complete the prospectively restricted external evaluation.

## Active Constraints
- Inputs are Image+Question only: frozen Qwen3 question-token features plus
  cached Qwen2.5-VL projected visual rows entering decoder layer 0.
- Actions are categorical in executor order `IGNORE`, `READ_ONLY`,
  `WRITE_ONLY`, `FULL`; routes have 28 actions.
- Match the binary full10 optimizer/schedule settings, save and validate every
  epoch, and evaluate only ChartQA, MMMU-Pro Standard/Vision, and POPE.
- GPU work must use Slurm. Training and full external evaluation both use one
  eight-GPU allocation split 4/4 between the two objectives.

## Current State
- Done: action/model/loss/decoder/data contracts; checksum-audited training
  manifest; Qwen3/Qwen2.5 model and external-evaluation asset inventory.
- Historical runtime: Slurm job 1662 completed visual-cache extraction and both
  BCE/NLL training processes, then failed during external-evaluation preflight.
- Confirmed failure: Qwen position-ID construction indexed a CPU tensor with an
  attention mask on another device. The external evaluation did not begin.
- Paused: no fix or evaluation relaunch is currently authorized.
- Most recent useful observation: 106 of 6,917 source records have no
  replay-valid four-action route; all are W2C records with replay failures.
- Latest infrastructure observation: the host scheduler reports one idle node,
  eight idle H100 80GB GPUs, and no user jobs, while this restricted process
  has no GPU devices because it is outside a Slurm allocation.

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
| Fresh projected-visual-row cache | Matches the proven binary Image+Question representation | Supplies predictor image input without outcome leakage | medium | pending authorized GPU extraction |
| Duplicated one-hot action BCE | Direct four-class generalization of prior POLAR BCE | Tests matched POLAR-style supervision | high | implementation in progress |
| Exact complete-valid-set NLL | Optimizes mass over all valid routes without route duplication | Tests categorical set supervision | high | implementation in progress |

## Next-Step Decision
- Deliberation mode: none after the terminal runtime observation.
- Active objective and bottleneck: training completed, but external evaluation
  is blocked by a device-placement runtime failure in its preflight.
- Confirmed observation: job 1662 is `FAILED`/`1:0`; no external-evaluation
  sample completed in the failing preflight shown by the Slurm log.
- Diagnosis: the direct device mismatch is supported by the traceback; its
  underlying code-level cause has not been diagnosed.
- Evidence: Slurm accounting and
  `logs/slurm/four-action-polar-train-eval-v1-1662.log`.
- Chosen action: preserve the terminal state without a fix or relaunch.
- Automatic execution authorized: no.
- Authorization basis: the latest user request authorized cancellation of the
  separate queued online jobs only.
- Stop condition: wait for an explicit request to diagnose, fix, or resume the
  POLAR evaluation.

## Latest Research-Action Result
- Action taken: historical job 1662 ran the resumable cache/train/evaluate
  pipeline.
- Result: cache extraction and both ten-epoch training processes reached the
  pipeline's `training_complete` stage; the job then failed at the first
  external-evaluation preflight and exited `1:0` at 05:02:27 KST.
- Evidence saved: `logs/slurm/four-action-polar-train-eval-v1-1662.log` and
  per-task logs under `logs/four_action_polar/pipeline_v1/`; these are
  machine-local and their presence on another server must be verified.
- Failure or issue: `RuntimeError: indices should be either on cpu or on the
  same device as the indexed tensor (cpu)` from `build_binary_inputs` during
  native/unified FULL parity setup.
- Next implication: do not infer external results exist and do not relaunch or
  fix this failed evaluation without explicit authorization.

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
  job 1662 subsequently executed the cache and training stages as recorded in
  `Latest Research-Action Result` above.
