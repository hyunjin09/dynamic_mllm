# Dataset Inventory

Updated: 2026-08-11

## We-Math 2.0 benchmark downloads

- Requested datasets: `We-Math/We-Math2.0-Standard` and
  `We-Math/We-Math2.0-Pro`.
- Required destination: the approved Hugging Face dataset cache at
  `/data/dataset/huggingface/datasets`.
- Pre-download inventory: neither dataset is present under `/data/dataset` or
  `/home/hyunjin/.cache/huggingface/datasets`.
- Execution policy: CPU-only Slurm, four allocated CPUs per dataset. The
  required `infra/download_hf_dataset.py` enforces `num_proc <= 3`, so each
  four-CPU job uses `--num-proc 3`.
- Repository policy requires one dataset download at a time. Standard is
  downloaded first; Pro follows immediately after Standard reaches a terminal
  successful state.
- Standard result: downloaded by CPU-only Slurm job `99824`; 5,843 rows in
  split `standard`, materialized at
  `/data/dataset/huggingface/datasets/We-Math___we-math2.0-standard` (583 MB).
- Pro result: downloaded after Standard by CPU-only Slurm job `99825`; 4,552
  rows in split `pro`, materialized at
  `/data/dataset/huggingface/datasets/We-Math___we-math2.0-pro` (327 MB).
- Download reports:
  `outputs/dataset_downloads/wemath20_standard_v1.json` and
  `outputs/dataset_downloads/wemath20_pro_v1.json`.
- Status: both downloads completed successfully. The dataset builder reduced
  Standard to one worker for its one shard and Pro to two workers for its two
  shards; both Slurm jobs retained the requested four-CPU allocations.

## v3 independent null-calibration pool

- GQA source: pinned balanced train instructions and Visual Genome images.
- TextVQA source: pinned official train Arrow shards; selected images are
  materialized under
  `/data/dataset/dynamic_mllm/v3_null_redesign/calibration_images_v1`.
- Deterministic seed `2026080701`; one question per unique image.
- The entire GQA/TextVQA validation universes and all explicit inspected images
  were excluded. Manifests contain no answer fields.
- Initial pool: 1,000 unique images per dataset. The single authorized
  enlargement: 2,000 per dataset, with an image-disjoint 1,000-per-dataset
  delta.
- Evidence: `outputs/v3_null_redesign/calibration_pool_manifest.json` and
  `outputs/v3_null_redesign/calibration_pool_manifest_v2.json`.
- Status: geometry-only calibration completed; no dataset download was needed,
  no held-out answer was scored, and the null redesign failed its validity
  gates.

## v3 grounding-control annotations

- GQA grounding uses the official GQA v1.1 scene graphs. They are present at
  `/data/dataset/GQA/sceneGraphs_v1.1/` after CPU-only Slurm download
  `v3_gqa_scenegraphs_download_20260806`.
- TextVQA grounding requires word-level boxes. The local TextVQA Arrow cache
  contains OCR strings but no OCR coordinates. The official TextOCR v0.1
  annotations use the same image IDs and splits and provide word polygons and
  boxes. They are present at `/data/dataset/TextOCR/annotations_v0.1/` after
  CPU-only Slurm download `v3_textocr_annotations_download_20260806`.
- These annotation-only downloads are for prospective geometry/grounding
  eligibility. They will not be used to inspect intervention outcomes.
- Outcome-blind audit result: 123 GQA and 130 TextVQA proposed held-out records
  meet the frozen unambiguous-target and matched-control rules; evidence is
  `outputs/v3_preflight/grounding_eligibility_audit_v1.json`.

## v3 held-out candidate pools

- GQA source: pinned `lmms-lab/gqa` balanced validation metadata revision
  `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`, with images resolved under
  `/data/dataset/VG/VG_100K{,_2}`. All 10,234 unique metadata images are
  available and readable; no Stage A/B GQA image ID overlaps the validation
  pool.
- TextVQA source: the pinned official validation split below. After frozen
  invalid-record rules and all inspected-image exclusions, 3,605 records over
  2,362 images remain. Canonical RGB hashing confirms zero Stage A/B image
  overlap; all 800 v2 Stage C images and their co-image questions are excluded.
- Proposed v3 Stage C capacity: 800 unique images per dataset. TextVQA can use
  800 of 1,119 remaining singleton images, preserving all 1,243 remaining
  two-question images.
- Separate Stage C2 reserve capacity: at least 800 multi-question images per
  dataset after the proposed Stage C allocation.
- Evidence: `outputs/v3_preflight/candidate_pool_audit.json` and
  `outputs/v3_preflight/stage_c2_reserved_pool_audit.json`.
- Status: candidate identity previews only; no final v3 confirmation manifest
  is frozen and no held-out intervention outcome has been opened.

## TextVQA official validation split

- Dataset ID: `lmms-lab/textvqa`, split `validation` (5,000 records).
- Revision: `9c0699cd19768ac5ab97568f6b3cbac4c0062884`; source fingerprint
  `475bf9de899d571b`.
- Required use: outcome-blind Stage C candidate pool.
- Cache check order: absent from `/data/dataset`; no identifiable TextVQA cache
  was found under `/home/hyunjin/.cache/huggingface/datasets`.
- Status: downloaded under
  `/data/dataset/huggingface/datasets/lmms-lab___textvqa` by CPU-only Slurm job
  `stage_c_textvqa_validation_download_20260805`. The frozen eligibility audit
  retained 4,991 records across 3,162 unique images and selected 800 unique
  images with no Stage B record/image overlap.
- Frozen manifest: `outputs/stage_c/manifest/stage_c_manifest_v1.jsonl`, SHA-256
  `e3e9e08329fa626bc75706fba6623357f9ca05140bae1f138c98b9cd26e45357`.
- Frozen image copies:
  `/data/dataset/dynamic_mllm/TextVQA/stage_c_validation_images_v1`.
- Selection rationale: the locally available `easy_hard_5k` remainder is
  train-derived and selected by inherited model correctness. It remains a
  discovery/calibration source and is not used for the Stage C population mean.
- Official dataset evidence: the Hugging Face dataset card reports 5,000
  validation rows with image IDs, question IDs, questions, images, and ten
  accepted answers per record.

## easy_hard_5k

- Path: `/data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k`
- Status: present; no download required
- Inspected pool: `complete_correct_wrong_pools_20260713`
- Contents: 10,000 train-derived records: 5,000 `score == 1.0`
  (`complete_correct`) and 5,000 `score == 0.0` (`complete_wrong`).
- Benchmarks per bucket: GQA 2,000; ChartQA 1,000; DocVQA 1,000;
  TextVQA 1,000.
- Images: the pool audit reports 10,000 copied images and zero missing after
  consolidation; targeted samples were confirmed under local
  `images/{benchmark}/` directories.
- Intended current use: completed Stage A validation plus the user-approved
  400-candidate Stage B discovery source (GQA/TextVQA, 100 per inherited
  easy/hard cell). Candidate manifest:
  `data_manifests/stage_b_discovery_candidates_400.jsonl`.
- Restrictions/caveats:
  - manifests do not identify the exact model checkpoint/revision used to
    create the correct/wrong buckets;
  - bucket membership must be reproduced with pinned Qwen2.5-VL-7B-Instruct
    revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` before analytical use;
  - records are open-ended and contain no distractor option sets, so they do
    not directly supply the approved `M_max`/`M_lse` option-margin estimand;
  - manifest `image_path` values refer to the source machine and must be mapped
    to the local consolidated image tree without modifying the source pool;
  - discovery/confirmation/mechanism split assignments are not frozen.
  - Stage B candidate selection excludes all requested Stage A IDs and uses 400
    unique effective image assets; inherited buckets remain sampling metadata
    until pinned-model FULL relabeling is complete.
- Evidence: `complete_correct_wrong_audit.json`,
  `image_consolidation_audit.json`, per-benchmark JSONL manifests, and targeted
  local image existence checks.

### Label-regeneration 8K view

- User-specified source:
  `datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713`
  inside the approved project root; present locally, no download required.
- Frozen selection: 4,000 GQA, 2,000 TextVQA, and 2,000 ChartQA records;
  DocVQA is excluded from this phase.
- Historical balance: 4,000 historical ALL-ON-correct and 4,000 historical
  ALL-ON-wrong records. These are metadata only; P3 recomputes authoritative
  ALL-ON correctness.
- Canonical regenerated-label bundle:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/` (23 GB), containing 4,000 GQA,
  2,000 TextVQA, and 2,000 ChartQA records plus the raw route cache and all
  post-generation supervision/audit artifacts.
- Frozen source manifest:
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/source_manifest_v1.jsonl`
  (8,000 rows), with passing SHA-256
  `6abad68ad6c3a9ca2b1bfc1f5502ea2c61ca0e81d0e42f841bc9e257de5f236a`.
- Compatibility path: `outputs/label_regeneration/v1` is a symlink to the
  canonical dataset-root bundle, so existing training and audit commands remain
  valid.
- Image paths resolve to the consolidated local `images/{benchmark}/` tree;
  native Qwen processor defaults will be used without a custom visual-token
  cap.

### WeMath2.0-Pro MCTS labels

- Canonical hard-cap-400 bundle:
  `datasets/math_labels/wemath20_pro_mcts_max400_v2/` (14 GB), containing all
  4,544 technically eligible WeMath2.0-Pro records and their raw route caches.
- Frozen manifest:
  `datasets/math_labels/wemath20_pro_mcts_max400_v2/manifest/wemath2pro_valid_mcts_v1.jsonl`
  (4,544 rows), SHA-256
  `f3a3d8d11c48c508451d819c467f5ed3c91ff369e1931196f43cc7d334920946`.
- Compatibility path: `outputs/label_regeneration/wemath2pro_cap400_v2` is a
  symlink to the canonical dataset-root bundle. Active greedy-recovery jobs
  continued through this path after the atomic relocation.
- The incomplete `wemath2pro_cap400_v1` predecessor and older
  `wemath2pro_v1` search lineage remain under `outputs/label_regeneration/` as
  provenance and are not canonical training-label inputs.

## Binary visual-mask MCTS v2 labels

- User-authorized read-only source:
  `/home/hyemin/data/dataset/dynamic_mllm/mcts_v2`.
- Intended use: supervised binary visual-token policy training under
  `plans/dynamic_mllm_binary_visual_polar_plan_v1.md`; no new MCTS generation.
- Inventory: 4,000 records, four shards, ChartQA/DocVQA/GQA/TextVQA, with 500
  easy and 500 hard records per benchmark; normally 202 evaluated 28-bit masks
  per record.
- Positive-route availability: 3,408 records have at least one successful
  mask; 592 do not and remain evaluation-only under the proposed POLAR
  adapter.
- Source audit: passed 4,000/4,000, zero missing, eight raw errors recovered,
  zero unrecovered; SHA-256
  `725f4755e7ff683d4d7dcf21bf9e4bad400f97de7fcea27fbc1215f1114cd183`.
- Runtime provenance: pinned Qwen2.5-VL snapshot `cc594898...`, PyTorch 2.6.0,
  Transformers 5.3.0, SDPA, slow processor, repetition penalty 1.05.
- Current status: BP-0 parsed all 4,000 records with zero invalid records and
  reconciled all eight cells to the source audit. It found 3,824 unique image
  groups, zero cross-split groups, and a minimum prospective cell/split count
  of 52. The direct representation gate failed; no compact manifest was
  frozen. Audit evidence is
  `/data/dataset/dynamic_mllm/binary_polar_v1/binary_polar_label_geometry_audit_v1.json`
  with SHA-256
  `48f36c9057b2a8e977720a8667b9ed854c8d95772d366cb6ea1154c66adecc63`.

## Four-action label-conversion sources (current H100 server)

- Allowed external root: `/data/research/datasets`; project compatibility link:
  `datasets -> /data/research/datasets/dynamic_mllm`.
- Frozen five-dataset source inventory:
  `datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl`,
  SHA-256 `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7`.
- Population: 12,278 samples with at least one authoritative positive binary
  label and 545,531 positive routes across GQA, TextVQA, ChartQA, WeMath2.0
  Standard, and WeMath2.0 Pro.
- VQA authority: current regenerated max-50 predictor view under
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/`; `datasets/mcts_v2` is
  explicitly excluded from this conversion.
- WeMath Standard terminal source:
  `datasets/math_labels/wemath20_standard_mcts_max400_latest`, a machine-local
  link to `/data/research/datasets/Sparse_Visual_Contextualization/Qwen2.5-VL-7B-Instruct/04_MCTS/reasoning/wemath20_standard_max400_partial`.
  It contains 5,381/5,843 same-contract terminal records. The other 462 source
  rows have no terminal binary-label record on this server and therefore no
  existing label to convert; they are not inferred negative or regenerated.
- A targeted directory search of `/data/research/datasets` on 2026-08-25 found
  no larger same-contract Standard cache. The available combined v31 bundle is
  a different derived/runtime lineage and remains excluded.
- WeMath Standard source/images resolve through
  `datasets/math_labels/wemath20_standard_source_v1`; WeMath Pro labels resolve
  through `datasets/math_labels/wemath20_pro_mcts_max400_v2` and images through
  `datasets/WeMath2Pro/pro_images_v1`.
- No dataset download was required for the conversion.
