#!/usr/bin/env bash
set -euo pipefail

BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
NUM_GPUS=${NUM_GPUS:-4}
FIRST_GPU_MAX_MEMORY_GB=${FIRST_GPU_MAX_MEMORY_GB:-40}
MIN_FREE_GB=${MIN_FREE_GB:-20}
USE_CACHED_BASELINE=${USE_CACHED_BASELINE:-1}
RUN_ID=${RUN_ID:-reproduced_sw31_pope}
OUT_ROOT="$BUNDLE/results/pope_regeneration"

export PYTHONPATH="$BUNDLE/code"
export HF_HOME="$BUNDLE/model"
export HF_HUB_CACHE="$BUNDLE/model"
export TRANSFORMERS_CACHE="$BUNDLE/model"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$BUNDLE/state/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$BUNDLE/results/logs"

MODEL="$BUNDLE/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA="$BUNDLE/data/heldout_pope_v1"
ROUTER="$BUNDLE/checkpoints/sw31/router_epoch_001.pt"
BASELINE="$BUNDLE/baseline/pope_all_on_generation_rows.jsonl"

"$PYTHON" "$BUNDLE/scripts/verify_bundle.py"
baseline_args=()
if [[ "$USE_CACHED_BASELINE" == "1" ]]; then
  baseline_args=(--baseline-rows-jsonl "$BASELINE")
fi

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      "$BUNDLE/code/dvr_qwen/scripts/evaluate_heldout_online_visual_router_generation.py" \
      --checkpoint "$ROUTER" \
      --heldout-dir "$DATA" \
      --model-source "$MODEL" \
      --hf-hub-cache "$BUNDLE/model" \
      --out-root "$OUT_ROOT" \
      --run-id "$RUN_ID" \
      --benchmarks pope_adversarial,pope_popular,pope_random \
      --num-shards "$NUM_GPUS" \
      --shard-index "$gpu" \
      --bootstrap-repetitions 1000 \
      --bootstrap-seed 20260723 \
      --processor-use-fast false \
      --attn-implementation sdpa \
      --device-map auto \
      --first-gpu-max-memory-gb "$FIRST_GPU_MAX_MEMORY_GB" \
      --other-gpu-max-memory-gb "$FIRST_GPU_MAX_MEMORY_GB" \
      --min-free-gb "$MIN_FREE_GB" \
      --process-name "repro-pope-s${gpu}" \
      "${baseline_args[@]}" 2>&1 | tee "$BUNDLE/results/logs/pope_shard_${gpu}.log"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" \
  "$BUNDLE/code/dvr_qwen/scripts/merge_heldout_router_eval_shards.py" \
  --run-id "$RUN_ID" \
  --out-root "$OUT_ROOT" \
  --num-shards "$NUM_GPUS" \
  --bootstrap-repetitions 1000 \
  --bootstrap-seed 20260723

echo "[done] $OUT_ROOT/$RUN_ID/merged_final"
