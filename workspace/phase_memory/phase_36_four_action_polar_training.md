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
- In progress: Slurm job 1662 contains the authorized end-to-end runtime
  execution; it is pending until the current exclusive eight-GPU job releases
  the server. All code, configs, and CPU/static preflights are complete.
- Bottleneck: training readiness first requires a new GPU-extracted visual
  cache bound to the 6,811 eligible records.
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
- Deliberation mode: fast (the user specified the complete runtime action).
- Active objective and bottleneck: execute both frozen training objectives and
  their external evaluation; the fresh visual cache is the first dependency.
- Relevant memory item used: training must remain fail-closed until the cache
  finalizes and both config-bound full preflights pass.
- Confirmed observation: all eight H100s are live and idle, the user queue is
  empty, and `cache_audit_v1.json` is absent.
- Diagnosis: supported.
- Evidence: live `sinfo`/`squeue` on 2026-08-28 and
  `outputs/four_action_polar/visual_features_v1/`.
- Chosen action: submit one resumable eight-GPU pipeline that extracts/finalizes
  the cache, gates both configs, trains BCE on GPUs 0–3 and NLL on GPUs 4–7,
  then evaluates BCE on GPUs 0–3 and NLL on GPUs 4–7 concurrently and merges
  the objective reports.
- Automatic execution authorized: yes.
- Authorization basis: the user explicitly said to start training, use four
  GPUs per objective, run evaluation afterward, and monitor early training.
- Stop condition: the pipeline and evaluation are complete, or an unresolved
  semantic/runtime correctness failure makes continued execution invalid.

## Latest Research-Action Result
- Action taken: submitted the resumable cache/train/evaluate pipeline as Slurm
  job 1662 after 461 project tests and shell validation passed.
- Result: job 1662 is pending with 8 H100s, 64 CPUs, and 512 GB requested; BCE
  and NLL use GPUs 0–3 and 4–7 respectively for both training and evaluation.
- Evidence saved: machine-local scheduler queue, Slurm job 1662, and
  `infra/run_four_action_polar_pipeline.sh`.
- Failure or issue: none in this pipeline; another user's exclusive job 1654
  currently owns the server resources.
- Next implication: job 1662 starts automatically when the allocation becomes
  available and performs an early three-batch liveness/loss audit for both
  objectives before continuing through all epochs and external evaluation.

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
