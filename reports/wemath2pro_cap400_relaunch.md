# We-Math2.0-Pro cap-400 relaunch

Date: 2026-08-13

## Outcome

The We-Math2.0-Pro label extraction was migrated from the prior 600-extension
protocol to a hard maximum of 400 MCTS simulations per sample. The replacement
run is active as Slurm job `100407` on node06 with seven one-process-per-GPU
workers.

## Preservation and coverage

- Frozen valid manifest: 4,544 records; the previously documented eight
  technical-invalid source records remain excluded.
- Retained cache: 640 checksum-valid terminal records, comprising 207 records
  with 200 simulations and 433 records with 400 simulations.
- Excluded cache: all 529 completed records that requested 600 simulations.
  They are preserved in the predecessor output tree but are not visible to the
  new scientific run.
- Remaining work at staging time: 3,904 records.
- Coverage invariant: retained UIDs plus remaining UIDs equal the complete
  4,544-record manifest, with unique source UIDs and no temporary records.
- Publication invariant: a record is reusable only after an atomic JSON rename
  and only when requested and completed simulation counts agree. An interrupted
  in-memory sample is recomputed on resume.

The detailed evidence is
`outputs/label_regeneration/wemath2pro_cap400_v2/cap400_resume_audit_v1.json`.

## Frozen execution

- Active contract SHA-256:
  `80c7ea4ca2ca9df091696290dc644a4092508337f89cf85ecc5b849a0f4092c7`.
- Contract artifact:
  `outputs/label_regeneration/wemath2pro_cap400_v2/frozen_execution_contract_cap400_v5.json`.
- Correct current ALL-ON samples receive 200 simulations.
- Incorrect current ALL-ON samples receive 400 simulations.
- `--max-simulations-per-sample 400` prevents any extension.
- Route masks remain unrestricted 28-bit layer-wise visual ON/OFF masks.
- Model, native image processing, deterministic generation, MathRuler scoring,
  scoring timeout, route semantics, sample-derived seeds, and cache schema are
  unchanged.

## Launch validation

- Job: `100407` (`wemath20pro_cap400_r7`).
- Allocation: node06, 7 RTX A6000 GPUs, 84 CPUs, 197 GiB Slurm memory.
- Seven `torchrun` children are active, one per GPU; each GPU loaded the pinned
  model and showed active compute.
- The first three new terminal records (`wemath2pro:2`, `wemath2pro:3`, and
  `wemath2pro:4`) passed live validation: each stores the active contract and
  runtime cap, requested and completed exactly 400 simulations, records no
  extension, and has no UID overlap with the retained cache. At this check
  there were zero error records and zero temporary files.
- The original 200G request was cancelled while pending. Slurm expands 200G to
  204,800 MiB, which cannot fit beside node06's existing 48,000 MiB allocation
  on a 250,000 MiB node. The 197G allocation fits without changing the run.
- One corrected relaunch (`100406`) failed before model loading because the
  command used `--model-revision` instead of the runner's `--revision`. No
  sample or temporary file was written. Job `100407` uses the correct flag.

No predictor training or post-label scientific interpretation was started.
