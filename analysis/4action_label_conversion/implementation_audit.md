# Four-Action Label Conversion Implementation Audit

Date: 2026-08-25 (Asia/Seoul)

Approved specification: `plans/4way_labeling.md`

Git commit at audit: `a3c6a41115490992b4f0cebb40d7e67d857c9286` (dirty research worktree; implementation hashes below bind the untracked/modified code)

## Decision and scope

This conversion reuses existing positive binary MCTS routes. It does not run binary MCTS, four-action MCTS, or use `datasets/mcts_v2`.

- GQA, TextVQA, and ChartQA use the frozen training-authoritative max-50 predictor manifest.
- WeMath Standard and Pro have no later selected/max-50 view under their authoritative math-label roots, so every declared evaluator-correct route in each available terminal cache record is used.
- Every source route will be replayed under the current unified executor. A replay failure is retained as an explicit exclusion and is never replaced with another route.
- Current unified FULL determines W2C versus C2C. Only W2C receives purification and monotone READ/WRITE refinement. C2C remains the mechanical `ON -> FULL`, `OFF -> IGNORE` conversion.

## Frozen source inventory

The checksum-bound normalized inventory is:

- Manifest: `datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl`
- Manifest SHA-256: `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7`
- Summary: `datasets/mcts_labels_4action/source_inventory_v1/source_inventory_summary_v1.json`

| Dataset | Positive samples | Positive binary routes | ALL-OFF routes |
|---|---:|---:|---:|
| GQA | 3,386 | 132,127 | 1,006 |
| TextVQA | 1,746 | 58,789 | 104 |
| ChartQA | 1,785 | 46,886 | 214 |
| WeMath2.0 Standard | 3,095 | 200,058 | 1,098 |
| WeMath2.0 Pro | 2,266 | 107,671 | 575 |
| **Total** | **12,278** | **545,531** | **2,997** |

Exact source artifacts:

1. GQA/TextVQA/ChartQA: `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl`, SHA-256 `3620a347a3498d16853463a6f9f8b842fecbab7b442cb869f1fb11bc9ab8aa52`.
2. WeMath Standard: the machine-local `datasets/math_labels/wemath20_standard_mcts_max400_latest` symlink resolves to the largest transferred same-contract max-400 cache. Its contract SHA-256 is `ad0a9805c09e817f552ff4f50757f25eea5fc6fc8ce493167fd16a3281818201`; its frozen 5,843-row source manifest SHA-256 is `49faf54bb9340c0d67453aa79bab17855269b8499f158da196d84b1b7d67267c`.
3. WeMath Pro: `datasets/math_labels/wemath20_pro_mcts_max400_v2`, with cap-400-v5 contract SHA-256 `03c6ce42b20d2ecaa35d788924ccf24259980737b29523aa435cd67606bce217`.

The Standard source manifest has 5,843 dataset rows, while the largest transferred terminal cache has 5,381 records. The remaining 462 rows have no terminal binary-label record on this server and therefore no existing positive binary route to convert. They are reported as unavailable terminal records—not inferred negative samples, converted labels, or regenerated routes. The 5,381 available records contain 3,095 positive samples and all 200,058 of their declared positive routes.

The previously considered combined v31 math preference bundle is excluded: it declares its Standard snapshot partial and is a different derived/runtime lineage. The older project-linked Standard copy is also excluded because the `latest` link is a strict same-contract superset (79 additional records and 2,329 additional positive routes).

## Assets and relocation

- Dataset root: `datasets -> /data/research/datasets/dynamic_mllm`.
- Standard latest-cache link: `datasets/math_labels/wemath20_standard_mcts_max400_latest`.
- Standard source/image link: `datasets/math_labels/wemath20_standard_source_v1`.
- VQA images: `datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images`.
- Pro images: `datasets/WeMath2Pro/pro_images_v1`.
- Qwen snapshot: `eval/reference/shared_prefix_eval_20260812/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

The source freeze verifies relocated file existence and byte size for VQA, and content SHA-256 for every distinct Standard/Pro image that supplies a positive label.

## Executor and conversion implementation

The existing unified executor now exposes complete heterogeneous-route capture through `capture_four_action_route`. Every layer action uses the same validated `four_action_layer` implementation, route-specific K/V cache, materialized prompt attention path, deterministic cached decoding, and batch size one.

Conversion semantics:

- suppression cost: `FULL=0`, `READ_ONLY=1`, `WRITE_ONLY=1`, `IGNORE=2`;
- W2C purification: deterministic early-to-late and late-to-early fixed-point FULL restoration, then select by cost, margin, stable tie-break;
- W2C refinement: monotone jointly evaluated beam search of width 8;
- C2C: no purification or refinement;
- exact route evaluations are cached across every source route for one sample;
- final labels are complete-route evaluator-correct and deduplicated with all source route IDs retained.
- every full-run sample also records current unified ALL-OFF generation and
  correctness so W2C ALL-OFF-correct and ALL-OFF-wrong populations remain
  analytically separate;
- full-run workers claim samples through a launch-scoped atomic shared queue.
  The queue is ordered by the observed pilot W2C cost multiplier, while a new
  Slurm launch token makes unfinished claims retryable without recomputing
  atomic completed records.

Implementation hashes:

| Artifact | SHA-256 |
|---|---|
| `binary_policy/executor/four_action.py` | `e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868` |
| `label_regeneration/runtime.py` | `1739b9d0f696ee3da3f601849f54c4d3f2077aa3cb72dea544f11b2ff796f201` |
| `tools/research_analysis/four_action/label_conversion.py` | `f0357be79774e26379d16cd01d616af4e6ad3e64e40102aefaac20bfd1b828b8` |
| `tools/research_analysis/four_action/label_sources.py` | `16add696dbba7e78549f9d8d5b1d9696c472d9baeb255c862943c6e7c0662e3d` |
| `tools/research_analysis/four_action/label_runtime.py` | `05a41875c70f03ec4b0782837a650f023f327641ffd618ef9d401db826ce42ab` |
| `tools/research_analysis/four_action/label_jobs.py` | `af269945553877ceffd0d1d177dc47bda98798a51e4f2e1430c8e44f9f5f47b6` |
| `experiments/run_four_action_label_conversion.py` | `633ae0a212cb302b7e8d320a20f9cf93eda663a8b83d662747d6f74475f56781` |
| `configs/four_action_label_conversion.yaml` | `893c31e8e84feac500c99e4f4583e4dd35b9337d9178d09c47630e9719a33d30` |

Pilot job 1598 loaded the earlier converter/runner/job hashes recorded at
submission (`1dec1d...`, `f95859...`, and `d328cf...`). The later changes add
joint-composition counters, current ALL-OFF output capture, execution-contract
hashing, and dynamic full-run claiming; they do not change four-action layer
semantics, purification selection, beam transitions, or canonical objectives.
The resume gate uses the current code, and every full record will carry the
complete current execution contract.

The active project `tests/` suite passes 380/380 tests. It covers complete-route
execution/cache semantics, four-action layer semantics, purification, beam
refinement, W2C/C2C and ALL-OFF separation, source normalization, exact-route
caching, atomic dynamic claiming, deduplication, canonical selection, pilot
selection, execution-contract hashing, and 16-worker/8-GPU layout.

Job 1604 exposed one server-specific Pillow safety-limit failure while opening
an otherwise checksum-matched frozen WeMath Standard image. A complete header
scan found 16 such images affecting 26 positive samples; none had a header
failure. The authoritative binary record had processed the failing image at its
native 19,831 x 11,651 resolution, and an exact local replay reproduced its
16,334 prompt tokens and 65,072 pixel rows. The runtime now retries only
`Image.DecompressionBombError`, only after verifying the frozen content SHA-256,
with the Pillow limit disabled for that open and immediately restored. The two
new regression tests and the complete active suite pass. Job 1604's 87 partial
records and one failure are preserved under
`full_pre_image_limit_fix_job1604/` as provenance but excluded from the clean
run, because the runtime patch changes the execution contract. The replacement
contract SHA-256 is
`32f6c3a5076e65ab15d7432ad296e16053528ebdb96b21fb1c695fda42fc2858`.

## Environment and live compute

- Project environment: `.venv` managed locally with `uv`.
- PyTorch `2.6.0+cu124`; Transformers `5.3.0`; Accelerate `1.6.0`; Pillow `11.1.0`; PyYAML `6.0.2`.
- Model dtype BF16; attention implementation SDPA; deterministic algorithms enabled; TF32 disabled.
- Live Slurm topology at 2026-08-25: one server, 192 CPUs, 1,869,259 MiB
  scheduler-visible RAM (about 1.78 TiB), and eight H100 80GB GPUs. A repeated
  live `scontrol` check during job 1598 reported 180G allocated and about
  1,467,881 MiB free.
- The machine template's 512G eight-GPU default fits this topology. The pilot
  intentionally requested only 180G because 16 loaded workers measured under
  50G aggregate RSS; this is resource right-sizing, not a live-memory limit.

## Pilot freeze and launch gate

- Pilot manifest: `datasets/mcts_labels_4action/conversion_v1/pilot/pilot_manifest_v1.jsonl`.
- Pilot SHA-256: `890ddbf933396267251d5023e04828b91f80ecc36c6fb80147478970aeb6dfc9`.
- Samples: 56 (GQA 12; each other dataset 11).
- Coverage proxies: 26 source-FULL-wrong, 30 source-FULL-correct, 17 ALL-OFF W2C, and 49 multi-route samples.
- Source routes exercised: 4,026; static estimated conversion cost: 59,651 route/OFF units.
- Pilot execution: 16 workers, two model replicas on each of all eight GPUs,
  one sample per worker at a time, deterministic LPT cost balancing,
  append-only progress, atomic per-sample completion, and exact resume. The
  measured fixed-bin tail motivated the shared atomic queue used only for the
  subsequent full run.

The pilot validation gate passed. Jobs 1598 and 1599 completed with exit code 0;
the exact-resume job exited in 20 seconds without overwriting records. The formal
audit at `analysis/4action_label_conversion/pilot_audit_v1.json` confirms:

- 56/56 sample records and exact accounting for all 4,026 source routes;
- 27 W2C and 29 C2C current-runtime semantics;
- exact generated-token/answer/correctness parity for the checked old-binary
  semantic baseline on all 56 samples;
- 3,905 replay-valid routes and 121 explicit replay failures, with no replacement;
- all 2,310 unique final routes jointly evaluator-correct;
- all 16 workers started and completed, with two replicas on every one of the
  eight GPUs and zero worker failure artifacts;
- peak measured memory 41,979 MiB and stable active-replica execution.

The saved full projection is
`analysis/4action_label_conversion/full_compute_estimate_v1.json`. Its central
estimate is 139.1 wall-hours at 80% scheduling efficiency, with a deliberately
wide stress-pilot range of 57.1--375.1 hours. Full mode therefore uses the tested
atomic dynamic queue and exact resume rather than reducing scientific scope.
