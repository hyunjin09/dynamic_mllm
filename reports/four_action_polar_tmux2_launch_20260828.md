# Four-Action POLAR Two-GPU Tmux Launch Handoff

## Live status

- Date: 2026-08-28
- Tmux session: `1`
- Live Slurm job: `105068`
- Job name: `fa4polar_t2`
- Placement: node06, A6000 partition, two GPUs, 16 CPUs, 100 GB
- Launch mode: foreground `srun` retained inside tmux session `1`
- Pipeline log: `runs/four_action_polar_tmux2_20260828/pipeline.log`
- Runtime wrapper: `runs/four_action_polar_tmux2_20260828/run_pipeline.sh`
- Runtime-wrapper SHA-256:
  `c3accbef992c6de753524c898fd3b69ebb54a3a7930408d2056f136c6a0f7f68`
- Current stage after startup monitoring: BCE epoch 1 training.

The former four-GPU node07 request `105063` was cancelled before it started.
It produced no cache, checkpoint, or evaluation output.

## Placement repair

The original tmux allocation `105067` exposed two logical GPUs on node03, but
one device returned `Unable to determine the device handle`, and PyTorch
reported zero CUDA devices because CUDA initialization failed. No project
workload was launched in that allocation. It was released, and the replacement
request excluded node03. Job `105068` then started on node06 and passed:

- exactly two visible CUDA devices;
- allocation on both devices with PyTorch;
- project-local Python at `.venv/bin/python`;
- node06 NCCL/P2P safeguards from `infra/gpu_policy.md`.

The local environment has no `.venv/bin/activate`; the wrapper deliberately
invokes `.venv/bin/python` directly. This preserves the same project-local
environment and avoids the missing activation-script failure.

## Frozen scientific comparison

No scientific or training configuration changed. The two runs remain:

1. duplicated one-hot action BCE;
2. exact complete-valid-set NLL.

Both use the same Image+Question four-action POLAR architecture, manifest,
route weighting, ten epochs, seed, optimizer/schedule, and validation/checkpoint
selection. The output is categorical `[batch, 28, 4]` in executor order
`IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.

## Sequential two-GPU execution

1. Verify all 6,490 image groups and extract the fresh visual cache in two
   deterministic shards, one process per GPU.
2. Finalize and checksum-audit the cache.
3. Run cache-bound BCE and NLL preflights.
4. Train BCE on both GPUs with `DataParallel`.
5. Require the first three BCE batch events to have finite nonnegative loss
   and strictly increasing sample counts.
6. Train NLL on both GPUs with the same check.
7. Evaluate BCE in two parallel shards.
8. Evaluate NLL in two parallel shards.
9. Merge ChartQA, MMMU-Pro Standard/Vision, and POPE results and compare the
   objectives.

The change from the cancelled four-GPU job is placement only: BCE and NLL run
sequentially instead of concurrently. Each objective still uses two GPUs.

## Frozen data and evaluation

- Training population: 6,811 records, 5,945 train / 866 validation.
- Valid routes: 248,804.
- Image groups: 6,490.
- Explicit zero-valid exclusions: 106.
- Manifest SHA-256:
  `73919effd8b412ed264491c98ab136905e4057acb483459b77b5b63d50b2c7d3`.
- External population: 14,960 records.
- Benchmarks: ChartQA; MMMU-Pro Standard; MMMU-Pro Vision; POPE adversarial,
  popular, and random.

## Monitoring evidence

The wrapper fails closed if either training process exits, stalls for 30
minutes, emits a nonfinite/negative loss, reports the wrong objective, or does
not advance through three distinct batch events. Passed evidence is written to:

- `outputs/four_action_polar/node07_20260828/preflight/bce_startup_monitor.json`
- `outputs/four_action_polar/node07_20260828/preflight/nll_startup_monitor.json`

Training must not be described as started until the shared cache finalizes and
the corresponding real batch events appear.

### BCE startup result

The fresh cache finalized over all 6,490 image groups, both cache-bound
preflights passed, and BCE training began at `2026-08-28T12:05:27+09:00`.
The frozen startup monitor passed after three real optimizer steps:

| Batch | Samples seen | Mean loss so far | Learning rate |
|---:|---:|---:|---:|
| 1 | 128 | 0.7178628163 | 0.00005 |
| 2 | 256 | 0.7180740249 | 0.00010 |
| 3 | 384 | 0.6954456768 | 0.00015 |

All losses were finite and nonnegative, sample counts strictly increased, and
all events reported objective `duplicated_action_bce`. Evidence:
`outputs/four_action_polar/node07_20260828/preflight/bce_startup_monitor.json`.
NLL remains sequentially downstream and will receive the identical monitor
when BCE finishes.

Useful live checks:

```bash
squeue -j 105068
tmux attach -t 1
tail -f runs/four_action_polar_tmux2_20260828/pipeline.log
tail -f runs/four_action_polar_tmux2_20260828/cache_shard_0.log
tail -f runs/four_action_polar_tmux2_20260828/bce_train.log
tail -f runs/four_action_polar_tmux2_20260828/nll_train.log
```

## Completion status (2026-08-28 23:03 KST)

The Slurm pipeline exited with code `1` after both training runs completed and
before external evaluation processed a sample.

### Completed training

| Objective | Epochs | Steps | Selected epoch | Validation Hit@1 | Top-1 all-FULL | Unique top-1 masks |
|---|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | 10 | 470 | 8 | 0.585450 | 1.000000 | 1 |
| Exact set NLL | 10 | 470 | 6 | 0.585450 | 1.000000 | 1 |

Both training summaries report `passed: true`; both startup monitors passed;
all ten epoch checkpoints and checksum-bound best-checkpoint selections exist.
The selected validation predictions are descriptively all-FULL for both
objectives. This does not provide an external behavioral evaluation.

### External-evaluation blocker

BCE external preflight failed on its first of six technical samples before any
external result was saved. The direct observation is:

```text
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

The failure occurs in Qwen2.5-VL `get_rope_index` when a GPU attention mask
indexes a CPU `input_token_type`, reached through
`binary_policy.executor.inputs._position_ids`. NLL preflight and both full
evaluations did not start. Therefore these remain missing:

- both merged external analyses;
- both external reports;
- the BCE-versus-NLL comparison report.

Evidence: `runs/four_action_polar_tmux2_20260828/bce_eval_preflight.log`.
Diagnosis status: supported for the device mismatch and affected code path;
the minimal repair has not yet been implemented or rerun.

## External-evaluation repair and resume (2026-08-28 23:53 KST)

The device mismatch was repaired surgically in
`binary_policy/executor/inputs.py`: `mm_token_type_ids` is now moved to the
same device as prompt embeddings before Transformers 5.3 computes 3-D
position IDs. A focused regression reproduced the prior CPU/device mismatch
before the change and passes after it; the complete focused executor and
external-evaluation tests pass 9/9.

The frozen BCE and NLL checkpoints, configs, data population, scoring, and
decoding rules are unchanged. Evaluation-only job `105451` is running on one
node06 A6000 with a three-day limit and node06 safeguards. The BCE technical
preflight passed all six fixtures:

- 6/6 exact native-versus-unified generated-token parity;
- 6/6 prediction and evaluator-correctness parity;
- 6/6 exact repeated predicted and baseline executions.

An initial evaluation job `105448` executed 43 healthy BCE rows but its wrapper
stopped because the newly added startup monitor looked for a top-level
`predicted_score` instead of the evaluator's nested `predicted.score`. This
was an observability-script failure, not a model or evaluation failure. The
first 32 rows had already been atomically serialized; 11 buffered rows were
not committed and are deterministically rerun. The monitor schema was repaired
and validated against the saved chunk without changing evaluation logic.

Replacement job `105451` resumed from the 32 committed rows. Its first-chunk
gate passed unique-UID, required-field, and finite-score checks. At the latest
initial inspection, 256 unique BCE rows were atomically present and the live
evaluator had processed at about 1.4--2.5 samples/s. At the 2026-08-29 Git
handoff boundary, 6,240/14,960 BCE rows were atomically committed and NLL had
not started. NLL runs sequentially after the
complete BCE evaluation. No external aggregate or objective comparison is yet
available and no external result should be interpreted until both 14,960-row
passes and merges complete.

Live evidence:

```text
runs/four_action_polar_eval_node06_20260828/slurm.log
runs/four_action_polar_eval_node06_20260828/bce_eval_preflight.log
runs/four_action_polar_eval_node06_20260828/bce_eval_full.log
runs/four_action_polar_eval_node06_20260828/bce_startup_monitor.json
outputs/four_action_polar/node07_20260828/eval_bce/preflight_v1.json
outputs/four_action_polar/node07_20260828/eval_bce/shard_000_of_001/
```
