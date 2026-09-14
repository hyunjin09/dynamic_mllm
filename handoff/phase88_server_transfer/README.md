# Phase88 server transfer snapshot

This is a Git handoff taken while the source server is still running random
controls. It is not a completed-experiment release. Read `research_handoff.md`
first. `progress_snapshot.json` is timestamped historical status, not a live
queue on the destination. PIDs and the four-GPU layout belong to the source.

## Included in Git

- Research source, tests, plans, project skill, phase memories and decision log.
- Available regular Markdown analysis reports, including earlier negative
  results and the explicitly PRELIMINARY Phase85 interim report.
- All 59 regular Phase88 metadata files present at packaging: exact frozen
  split, protocol, execution config/contract, CAL tables, frozen schedules,
  parity evidence and completed-stage checksum manifests. Files over 1 MB
  are gzip-compressed without changing their uncompressed bytes.
- Exact installed package versions in this directory's `requirements-lock.txt`.
  The root lock predates several research dependencies; use this snapshot for
  Phase88. Source runtime is Python 3.12.7, Torch 2.6.0 (CUDA 12.4 runtime),
  Transformers 5.3.0, NumPy 2.5.1, SciPy 1.18.1, LMMS-Eval 0.7.3.

`metadata_manifest.json` hashes every original Phase88 metadata file.
`included_analysis_paths.txt` lists the explicitly included analysis artifacts
and compressed payloads. This is a partial evidence bundle: links from historical
reports to raw outcomes, checkpoints and plots generally require external data.
Old Dense repair contracts/manifests are retained only as superseded provenance;
the sole active contract is `b995483d75b300ef0170db9f06eff2a68801ba4f78fcb922e17951e96ee7492e`.

## Bootstrap on the destination

1. Have the server owner establish the Git-ignored `ACCESS_POLICY.md` before
   the agent inspects other files. Define the actual allowed checkout/data roots.
   Create `infra/gpu_policy.md` from `infra/gpu_policy.example.md`, and record
   actual local runtime in `workspace/env_state.md`. Do not copy the old queue.
2. Read the handoff, active Phase88 memory, key summaries and active plan in
   the handoff's order. Do not start a supervisor: the source run remains active.
3. Check for `uv`. Create a project-local environment and install the snapshot:

   ```bash
   uv venv .venv --python 3.12.7
   uv pip install --python .venv/bin/python -r handoff/phase88_server_transfer/requirements-lock.txt
   .venv/bin/python handoff/phase88_server_transfer/restore_metadata.py
   ```

   Follow the destination's access/cache policy for uv storage. If uv or the
   required wheels are unavailable, report the concrete dependency blocker.
   Never modify system Python. Package versions alone do not establish source
   parity: the frozen contract also hashes two installed Transformers files.
4. Verify the restored split and metadata with `restore_metadata.py --check-only`.
   The helper refuses to overwrite different existing artifacts. After a later
   final-results sync, use that later snapshot's verification instead of trying
   to overwrite it with this in-progress snapshot.
5. Obtain the external payloads below when raw verification/reporting is needed.
   Reading the handoff and summaries needs no GPU or external payload.

## External payloads: transfer separately, not through Git

| Payload on source server | Destination role |
|---|---|
| `/mnt/hyemin/qwen_train_eval/outputs/benchmark_calibrated_fixed_rw_schedule_v1/` | Required Phase88 raw records for dense/R/W/methods/global/random, worker completion/timing files, launch evidence, logs, calibration images and preserved failed attempts |
| `analysis/benchmark_calibrated_fixed_rw_schedule/` in the source checkout, after its pipeline finishes | Final metadata, random completion manifest and aggregate results newer than this Git snapshot; excludes/recreates the `work` symlink |
| `/mnt/hyemin/qwen_train_eval/datasets/` | Dataset root; for Phase88 inference copy the exact images referenced by the frozen split, not every historical label/feature cache |
| `/mnt/hyemin/qwen_train_eval/eval/reference/shared_prefix_eval_20260812/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5/` | Exact model/processor snapshot for inference and the final model hash audit |
| Other `analysis/`, `outputs/`, and external work roots referenced by historical phase memories | Only if revisiting a prior phase's raw evidence; not required for Phase88 inference |

Do not dereference the entire analysis tree: some links lead to very large
feature and checkpoint stores. Do not migrate `.venv`, caches, credentials,
machine policy, GPU queues or process IDs as runnable state.

The source is still writing. A copy made now is only a partial snapshot; do a
final incremental sync after `analysis_ready_for_interpretation`, or explicitly
coordinate a stopped source before any resume elsewhere. Do not use `--delete`
or overwrite a destination run. Example transfer shapes, with host and local
directories supplied by the server owner:

```bash
rsync -a --info=progress2 SOURCE_HOST:/mnt/hyemin/qwen_train_eval/outputs/benchmark_calibrated_fixed_rw_schedule_v1/ LOCAL_DATA_ROOT/outputs/benchmark_calibrated_fixed_rw_schedule_v1/
rsync -a --exclude=work --info=progress2 SOURCE_HOST:/home/aix7101/hyemin/0830/dynamic_mllm/analysis/benchmark_calibrated_fixed_rw_schedule/ analysis/benchmark_calibrated_fixed_rw_schedule/
```

No remote data transfer has been performed by this Git push.

## Path and contract constraints

The old implementation is not location-independent. The split/config contain
absolute image/model paths and execution manifests contain absolute record
paths. `EXT` is derived from the resolved `datasets` target's parent plus
`outputs/benchmark_calibrated_fixed_rw_schedule_v1`. Simply creating a `work`
link to an arbitrary directory is insufficient.

Prefer a destination layout that exposes the same absolute asset paths (only
if allowed by destination policy) and links `datasets` to the matching data
root. Recreate `analysis/benchmark_calibrated_fixed_rw_schedule/work` to the
matching output root. Otherwise, document and implement a relocation mapping
with preserved original hashes before attempting resume or final verification.
Do not edit the frozen split/config, refreeze the contract, regenerate splits,
or turn off checks to make paths fit. Relocation is a provenance task, not a
new experiment, and has not been implemented in this snapshot.

## What remains after computation

The supervisor runs random finalization and aggregation, then stops at
`analysis_ready_for_interpretation`. It does not generate final reports.
After the user resumes this authorized work, verify all complete cohorts,
interpret the frozen full results, then run:

```bash
.venv/bin/python -m experiments.summarize_benchmark_fixed_schedule report
.venv/bin/python -m experiments.summarize_benchmark_fixed_schedule verify
```

Review the report before final verification: its CAL-D recommendation text is
provisional. The CAL stability predicates already preclude A/B/C, so residual
qualified D does not by itself mean no held-out accuracy benefit. Report actual
efficacy and controls separately. Inspect figures, reconcile interpretation,
and update phase/global state and handoff. Recommend exactly one unexecuted
next step. No Phase85 restart, new router, top2 schedule or follow-on experiment.
