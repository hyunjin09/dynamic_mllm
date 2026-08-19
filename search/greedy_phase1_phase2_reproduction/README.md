# Greedy Phase 1+2 Reproduction Package

This directory reproduces the Qwen2.5-VL-7B visual-on/off candidate
collection used in `10k_dataset_mask/final_phase1_phase2` on another server.
It contains the frozen manifest, exact search configuration, original core
collectors, a path relocation utility, fail-fast validity checks, and a
resumable multi-GPU launcher.

## What Is Reproduced

Phase 1 starts from the all-visual-on 28-layer route and evaluates:

- all-on and all-off anchors;
- `early_to_late`;
- `late_to_early`;
- `center_out`;
- `outside_in`;
- six UID-conditioned random permutations with seeds 20260714--20260719;
- every accepted and rejected greedy removal in all ten traces.

A removal is accepted when its task score is no lower than the current target
within `1e-9`. Phase 2 is not another greedy pass. It evaluates
budget-stratified random masks, same-budget swaps, add/remove-one neighbors,
and unions/intersections of successful Phase-1 masks.

The run is valid only after the current HF all-on route and binary DVR all-on
route produce identical token IDs, predictions, and task scores on the gate.

## Frozen Inputs

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Snapshot: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Samples: 10,000, with 5,000 `complete_correct` and 5,000
  `complete_wrong` provenance rows
- Benchmarks: GQA 4,000; ChartQA, TextVQA, and DocVQA 2,000 each
- Language layers: 28
- Attention: SDPA
- Dtype: bfloat16
- Processor: `use_fast=false`
- Generation: greedy, with per-row `max_new_tokens` and image cap
- Manifest semantic SHA-256:
  `2e03a1705c455ff56e23206c2ff3e9a6ef1653354387c523ca54ba0390d1571e`

The semantic hash excludes only `local_image_path`, so relocating images is
allowed while changing IDs, prompts, answers, scores, image hashes, split,
DocVQA pixel caps, or generation settings is rejected.

## Directory Layout

```text
greedy_phase1_phase2_reproduction/
├── README.md
├── config/
│   ├── collection_config.json
│   └── paths.env.example
├── manifests/
│   └── all_samples.jsonl
├── reference/
│   ├── CHECKSUMS.sha256
│   └── runtime_versions.txt
└── scripts/
    ├── core/
    │   ├── collect_phase1_candidates.py
    │   ├── collect_phase2_candidates.py
    │   ├── aggregate_phase1.py
    │   ├── aggregate_phase1_phase2.py
    │   └── audit_final_phase1_phase2.py
    ├── preflight.py
    ├── relocate_manifest.py
    ├── run_pipeline.sh
    └── status.sh
```

The core files are frozen copies of the scripts that built the canonical
dataset. Do not edit them for a reproduction run.

## Files That Must Exist on the New Server

1. This complete reproduction directory.
2. The complete `0618_visual_on` project, including:
   - `analysis_outputs/harmful_validation_common.py`
   - `analysis_outputs/run_harmful_interventions.py`
   - `dvr_qwen/generate.py`
   - `dvr_qwen/binary_generate.py`
3. The exact Qwen snapshot listed above.
4. The four-benchmark image pool corresponding to the frozen manifest.
5. A Python environment matching `reference/runtime_versions.txt`.

The canonical raw Phase-1 payload records PyTorch `2.9.1+cu128`. The validated
source environment used Transformers 4.57.1, Accelerate 1.11.0,
qwen-vl-utils 0.0.14, and Pillow 12.0.0. Install the CUDA-compatible PyTorch
2.9.1 wheel first, then install `requirements.txt`. `STRICT_RUNTIME=1` is the
default and rejects a different package environment before GPU search begins.

The current server's newer environment was used only to test this portable
wrapper and passed its structural preflight. It is not mislabeled as the
canonical collection runtime. Set `STRICT_RUNTIME=0` only for an explicitly
reported compatibility replication; that run must still pass the token-level
generation anchor gate and must not be called an exact runtime reproduction.

## 1. Transfer

Transfer the project and dataset separately. For example:

```bash
rsync -a /source/0618_visual_on/ user@new-server:/new/root/0618_visual_on/
rsync -a /source/complete_correct_wrong_pools_20260713/ \
  user@new-server:/new/data/complete_correct_wrong_pools_20260713/
```

The package is already inside `0618_visual_on`, so the first command transfers
the scripts and frozen manifest.

## 2. Relocate Image Paths

The frozen manifest uses this original prefix:

```text
/data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713
```

Create a relocated manifest rather than editing the frozen file:

```bash
cd /new/root/0618_visual_on/greedy_phase1_phase2_reproduction

/new/root/0618_visual_on/dvr_qwen/.venv/bin/python \
  scripts/relocate_manifest.py \
  --input manifests/all_samples.jsonl \
  --output manifests/all_samples.relocated.jsonl \
  --old-prefix /data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713 \
  --new-prefix /new/data/complete_correct_wrong_pools_20260713 \
  --verify-images
```

`--verify-images` hashes all 10,000 images and is recommended once after
transfer. The command writes a relocation audit next to the output manifest.

## 3. Configure Paths

```bash
cp config/paths.env.example config/paths.env
```

Edit `config/paths.env`. In particular, set:

- `PROJECT_ROOT` to the relocated `0618_visual_on` directory;
- `PYTHON_BIN` to its environment Python;
- `MODEL_SOURCE` to the exact snapshot directory;
- `MANIFEST` to `manifests/all_samples.relocated.jsonl`;
- `OUTPUT_ROOT` to a new, empty output directory;
- `GPU_IDS` to the physical GPUs to use;
- Hugging Face cache paths.

For 48 GB GPUs, the recorded launcher value is
`FIRST_GPU_MAX_MEMORY_GB=42`. For a 32 GB GPU, lower it to approximately 28--30.
Each worker sees one physical GPU, so the model's local device is always CUDA
device 0 inside that worker.

Do not change `GPU_IDS` count after Phase 1 or Phase 2 has started. The output
shard topology is locked under `OUTPUT_ROOT/state` to prevent duplicate sample
files during resume.

## 4. Preflight

```bash
./scripts/run_pipeline.sh preflight
```

To hash all images again during preflight:

```bash
VERIFY_IMAGE_HASHES=1 ./scripts/run_pipeline.sh preflight
```

Preflight checks:

- exact model snapshot directory name;
- 28 language layers;
- required DVR/scoring modules;
- canonical ten permutation orders;
- exact frozen config and core-script checksums;
- 10,000 unique UIDs and exact benchmark/split/source-bucket counts;
- path-invariant manifest SHA-256;
- existence of all images;
- canonical runtime package versions when `STRICT_RUNTIME=1`;
- collector self-tests.

Do not proceed if preflight fails.

## 5. Run

The complete foreground pipeline is:

```bash
./scripts/run_pipeline.sh all
```

For cluster use, run it inside `tmux`, `screen`, or the site's scheduler. The
script itself does not assume Slurm and does not detach silently.

The stages can also be run and resumed explicitly:

```bash
./scripts/run_pipeline.sh gate
./scripts/run_pipeline.sh phase1
./scripts/run_pipeline.sh aggregate1
./scripts/run_pipeline.sh phase2
./scripts/run_pipeline.sh finalize
```

Both collectors write one atomic JSON file per sample and skip completed
files, so rerunning a failed stage with the same `GPU_IDS` count resumes it.
Use a fresh `OUTPUT_ROOT` if changing the manifest, model, configuration, or
number of shards. Never merge outputs from runs with different settings.

## 6. Monitor

```bash
./scripts/status.sh
```

Worker logs are written to `OUTPUT_ROOT/logs/` with process names
`gpp_gate`, `gpp_phase1_sN`, and `gpp_phase2_sN`.

Useful manual commands:

```bash
tail -f /path/to/output/logs/gpp_phase1_s0.log
ps -eo pid,stat,etime,%cpu,%mem,cmd | grep gpp_
nvidia-smi
```

The collectors catch sample-level exceptions and continue. Therefore a worker
process exiting successfully is not sufficient evidence of validity. The
`finalize` stage must complete and the final audit decision must be:

```text
pass_final_phase1_phase2_integrity_audit
```

## Expected Final Outputs

The canonical result is expected to contain:

| Quantity | Expected |
|---|---:|
| Samples | 10,000 |
| Phase-1 final masks | 100,000 |
| Phase-1 trace rows | 2,800,000 |
| Phase-1 unique evaluated masks | 2,751,889 |
| Phase-2 requests | 371,864 |
| Phase-2 reused Phase-1 requests | 30,069 |
| Phase-2 new evaluations | 341,795 |
| Combined unique evaluations | 3,093,684 |

Exact candidate totals should reproduce under the frozen model, code,
manifest, and generation environment. The final files are written under:

```text
OUTPUT_ROOT/final_phase1_phase2/
```

The most important files are:

- `evaluated_mask_candidates.jsonl`;
- `phase1_permutation_final_masks.jsonl`;
- `phase2_route_requests.jsonl`;
- `sample_index.jsonl`;
- `summary.json`;
- `checksums.sha256`;
- `audit_summary.json`.

## Reproduction Acceptance Criteria

A run is accepted only if all of the following hold:

1. Preflight decision is `pass_reproduction_preflight`.
2. Gate decision is `canonical_current_model_anchor_gate_pass`.
3. The final aggregate reports 10,000 exact manifest/Phase-1/Phase-2 UIDs and
   zero raw errors.
4. The independent final audit reports
   `pass_final_phase1_phase2_integrity_audit`.
5. Model revision, processor mode, SDPA, generation settings, per-row image
   caps, and scoring helpers match this package.

If the gate fails, do not disable its source-score or generated-ID checks just
to continue. A failure means the run is not the same experiment; diagnose the
model snapshot, Transformers version, processor mode, image relocation,
prompt, max-pixel policy, and scoring code first.
