#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 SHARD_INDEX" >&2
  exit 2
fi

ROOT=/home/aix7101/hyemin/0618_visual_on
PY="$ROOT/dvr_qwen/.venv/bin/python"
SCRIPT="$ROOT/baseline_relative_visual_router/scripts/extract_input_admission_features.py"
OUT_ROOT=/mnt/hyemin/10k_dataset_mask/baseline_relative_visual_router
SHARD="$1"
NUM_SHARDS=3

export PYTHONPATH="$ROOT/baseline_relative_visual_router/src:$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

run_source() {
  local source="$1"
  local manifest="$2"
  local policy="$3"
  local data_root="$4"
  local output="$5"
  "$PY" -u "$SCRIPT" \
    --manifest-jsonl "$manifest" \
    --policy-rows-jsonl "$policy" \
    --data-root "$data_root" \
    --output-dir "$output" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD" \
    --chunk-size 64 \
    --first-gpu-max-memory-gb 20 \
    --other-gpu-max-memory-gb 20 \
    --min-free-gb 20 \
    --process-name "brvr-${source}-s${SHARD}"
}

run_source \
  natural \
  /mnt/hyemin/10k_dataset_mask/heldout_lmms_recommended_plus_pope_seed_lite_v1/samples.jsonl \
  /mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval/sw31_bt_leg_s41_heldout_plus_v1/merged_final/heldout_generation_rows.jsonl \
  /mnt/hyemin/10k_dataset_mask/heldout_lmms_recommended_plus_pope_seed_lite_v1 \
  "$OUT_ROOT/input_features_natural_canonical_v1"

run_source \
  external \
  /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2/samples.jsonl \
  /mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval/sw31_bt_leg_s41_mmstar_mmmu_final_v2/merged_final/heldout_generation_rows.jsonl \
  /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2 \
  "$OUT_ROOT/input_features_external_mmstar_mmmu_v1"
