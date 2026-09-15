# Phase89 transfer: stopped3B corpus audit

This snapshot preserves a **user-stopped, incomplete** experiment. Jobs2994
(replay) and2957(report) were cancelled on2026-09-15 11:41:56 KST. The source
queue is empty. No restart, geometry analysis, training or new experiment is
authorized by this handoff. Job IDs, PIDs and old ETA estimates are historical.

Read `research_handoff.md`, the current Phase89 memory, this runbook,
`research_summary.md`, and the V2 plan. The original Phase88 transfer remains
available, but its source-server completion has not been rechecked. Phase85
remains stopped. Do not read every historical plan or cache by default.

## Exact stopped frontier

`progress_snapshot.json` was constructed from the stopped files, validating
every durable route's identity, source binding, record checksum and Dense reuse.
It takes precedence over the last live monitoring counters.

| Item | State |
|---|---:|
| Canonical samples / image groups |10,000 /9,273|
| Canonical unique routes |3,907,717|
| Dense/FULL completed |10,000 /10,000|
| Records persisted in routed sample streams |734,481|
| Complete / partial routed samples |1,858 /31|
| Samples without a routed checkpoint |8,111|
| Saved error records |0|
| Unique available results, including separate Dense records |742,592|
| Fresh routed inferences still needed |3,165,125|

The8,111 samples without a routed checkpoint already have separate Dense
records. Do not double-count these or regenerate Dense merely to resume.
Routed coverage is18.8%; it is dominated by ChartQA and a small expensive-first
DocVQA block. It cannot support final whole-corpus drift or cohort conclusions.
Final reporting never ran. The source code's final reporter would still require
qualitative RS-A/B/C and READY/NOT READY review after all routes complete.

## What Git includes

- Both user plans, exact V1/V2 diagnostic/replay/reporting code and tests, and
  the separate24-worker scheduling implementation. The61 direct analysis
  source/Markdown paths are listed in `included_git_paths.txt`.
- Original diagnostic, repair and replay contracts; the concurrency contract;
  input/source indices; corpus census; exact10,000 Dense records and certified
  Dense manifest; transition summaries; parity traces;24 complete pilot
  references; small logs/status/submission/cancellation evidence.
- `metadata.tar.gz` preserves the uncompressed bytes of the larger metadata
  and multi-file Dense evidence. `metadata_manifest.json` covers10,467 original
  analysis files, either stored directly in Git or inside that archive.
- `runtime_snapshot/` contains the exact51 hash-bound packaged executor files
  for review. Its manifest preserves their source paths and hashes. This is a
  reference copy, not an automatic replacement of the canonical package.
- An84-package environment lock, environment snapshot, full external-file
  checksums and rsync file lists. No credentials or live machine policies.

`analysis/.../artifact_manifest.json` is a historical preparation artifact;
it is **not** a final result certificate. The authoritative transfer manifest
is this directory's `metadata_manifest.json`, with `transfer_verification.json`
recording the packaging checks. Older summaries saying a gate was queued or
that jobs are running are preserved historical evidence, superseded here.

## Bootstrap and metadata verification on the destination

1. Have the server owner establish local `ACCESS_POLICY.md` first. If missing,
   stop and request allowed roots before inspecting other files. Read local
   `AGENTS.md`, `infra/gpu_policy.md`, then README. Retain this server's policies;
   never overwrite them with another server's ignored policy/environment files.
2. Inspect Git status/branch/remotes and local commits. Preserve destination
   work; fetch origin and fast-forward only when safe. Read the current handoff.
3. If an environment is needed, use project-local uv only. The source had
   Python3.12.7, Torch2.6.0+cu124, Transformers5.3.0, NumPy2.5.1 and H10080GB.
   Follow destination cache/compute policy; do not modify system Python.

   ```bash
   uv venv .venv --python 3.12.7
   uv pip install --python .venv/bin/python -r handoff/phase89_server_transfer/requirements-lock.txt
   uv pip check --python .venv/bin/python
   ```

   Do not rebuild an existing working environment blindly. Stop on missing uv
   or unavailable required wheels. The source lock was validated for Phase89;
   it lacks some packages in the older Phase88 lock. Installing version numbers
   alone does not verify CUDA build, installed implementation hashes or parity.
4. Verify the packaged metadata, restore missing files, and check again:

   ```bash
   .venv/bin/python -B handoff/phase89_server_transfer/restore_metadata.py --check-only
   .venv/bin/python -B handoff/phase89_server_transfer/restore_metadata.py
   .venv/bin/python -B handoff/phase89_server_transfer/restore_metadata.py --check-only
   ```

   The first command may report missing destinations in a fresh clone; after
   restoration it must report zero missing. All archive hashes and existing
   destinations are checked before any writes. Different existing files,
   symlinks and unsafe archive entries are rejected. Do not replace newer local
   work with this snapshot. The helper uses only the standard library and
   never submits a job or edits absolute paths inside the preserved files.

## External assets:24.8GB, transfer separately

No large payload was uploaded by the Git push. Exact per-file SHA256 and sizes
are in `external_files.jsonl.gz`; all22,035 files were hashed on the stopped
source. `external_assets.json` describes the source roots and role totals.

| Required payload | Approximate decimal size |
|---|---:|
| V1 `source_inventory/by_sample/`:10,000 prepared source shards |4.55GB|
| V2 `replay/by_sample/`:1,889 stopped checkpoint files |1.87GB|
| Canonical package: indexed metadata/routes/V3.1 pairs, selected lineage and executor |8.75GB|
| Exact10,000 input-image paths |2.09GB|
| Exact3B model snapshot, processor/tokenizer and2 weight shards |7.52GB|

The canonical3B package root on this source is:
`/data/research/datasets/Sparse_Visual_Contextualization/Qwen2.5-VL-3B-Instruct`.
The model snapshot is:
`/data/research/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`.
The source checkout is `/home/hyunjin/projects/dynamic_mllm`.

Inventory destination assets under its allowed roots first. Transfer only
missing needed files, preserving conflicting destination work. The following
are example shapes; supply the source host and actual allowed destination roots:

```bash
rsync -a --ignore-existing --info=progress2 --files-from=handoff/phase89_server_transfer/transfer_lists/project.txt SOURCE_HOST:/home/hyunjin/projects/dynamic_mllm/ ./
rsync -a --ignore-existing --info=progress2 --files-from=handoff/phase89_server_transfer/transfer_lists/package.txt SOURCE_HOST:/data/research/datasets/Sparse_Visual_Contextualization/Qwen2.5-VL-3B-Instruct/ DESTINATION_PACKAGE_ROOT/
rsync -aL --ignore-existing --info=progress2 --files-from=handoff/phase89_server_transfer/transfer_lists/model.txt SOURCE_HOST:/data/research/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/ DESTINATION_MODEL_ROOT/
```

The model command dereferences the source snapshot's weight links so that blob
targets are actually copied. Do not use `--delete`. `--ignore-existing` preserves
different existing files; the verifier reports those conflicts rather than
treating them as successfully restored. Do not copy the entire `.venv`, caches,
GPU queues, policy files or unrelated historical datasets.

After copying, verify using explicitly policy-approved roots:

```bash
.venv/bin/python -B handoff/phase89_server_transfer/verify_external.py \
  --package-root DESTINATION_PACKAGE_ROOT \
  --model-root DESTINATION_MODEL_ROOT \
  --allowed-root DESTINATION_ALLOWED_DATASET_ROOT \
  --allowed-root DESTINATION_ALLOWED_MODEL_ROOT
```

The verifier maps inventory paths without rewriting original contracts or data.
It rejects paths/symlink targets outside its explicit allowed roots. All22,035
entries must verify for the complete listed asset set; missing files are a
transfer blocker, not grounds to regenerate a population or bypass verification.

## Path, label and resume constraints

The archived execution is **not location-independent**. `common.py` names this
server's package/model roots. Frozen contracts hash files at absolute source
paths, including installed Transformers implementation files. Prepared-index
entries point to V1's source shards; input manifests contain absolute images.
The24-worker coordinator also consults historical Slurm job2956 and forbids
reuse of an existing job's gate directory. Never submit these archived Slurm
scripts unchanged on another server, where job numbers may identify other work.

Metadata restoration and the external verifier are safe relocation tools for
evidence only. They do not adapt the runtime. A later authorized continuation
must explicitly resolve the destination layout through allowed compatible
paths or a versioned relocation/execution adapter preserving original hashes.
Keep old contracts immutable; never edit frozen files, reset hashes, regenerate
splits, or rewrite saved records simply to make verification pass.

“Current” labels in this snapshot mean **the stopped H100 source runtime**.
Do not silently merge them into destination-server outcomes. After authorization,
establish compatible model/environment/implementation hashes, full32-anchor
HF/custom parity, sparse cross-worker repeatability and saved-route comparison
under the destination configuration before deciding what can be reused. Any
observed drift must remain explicit and may require a separate versioned
destination replay. Matching a small anchor set is not proof of whole-corpus
cross-server identity.

The archived scheduler uses24 disjoint cost-balanced assignments and records the
original source rank separately from worker/GPU IDs. Completed records are
reused unchanged; every new record carries the scheduling-overlay hash. Its
eight aggregate rank markers are created only after all24workers complete and
exact sample/route counts match. Preserve its partial progress and source
shards when designing a destination continuation. Never fabricate completion
markers or rerun `submit_pipeline.py` against existing submission files.

## Work remaining, only after a new user instruction

Resolve destination compatibility and execution ownership; then finish the
authorized V2 scope: routed replay, complete old/current transitions, current
Dense/route category reconstruction, geometry-eligibility and action-distance
census, storage estimates and V3.1 pair validity audit. Review qualitative
RS-A/B/C and READY/NOT READY using actual cohort impact without new numerical
cutoffs. The existing final reporter leaves nonzero drift pending this review.
Final summaries and artifact verification must follow the reviewed results.
Do not start geometry, hidden-state corpus capture, finetuning or pair rebuilding.

For a new agent, use `next_server_prompt.md`; it ends at handoff verification
and explicitly preserves the user's stop instruction.
