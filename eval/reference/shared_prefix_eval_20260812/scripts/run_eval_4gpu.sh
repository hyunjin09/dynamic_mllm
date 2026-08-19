#!/usr/bin/env bash
set -euo pipefail

BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
NUM_GPUS=${NUM_GPUS:-4}
FIRST_GPU_MAX_MEMORY_GB=${FIRST_GPU_MAX_MEMORY_GB:-40}
MIN_FREE_GB=${MIN_FREE_GB:-20}
RUN_RUNTIME_AUDIT=${RUN_RUNTIME_AUDIT:-1}

export PYTHONPATH="$BUNDLE/code/baseline_relative_visual_router/src:$BUNDLE/code"
export HF_HOME="$BUNDLE/model"
export HF_HUB_CACHE="$BUNDLE/model"
export TRANSFORMERS_CACHE="$BUNDLE/model"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$BUNDLE/state/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$BUNDLE/results/logs"

MODEL="$BUNDLE/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA="$BUNDLE/data/heldout_mmstar_mmmu_final_v2"
MANIFEST="$DATA/samples.jsonl"
BASELINE="$BUNDLE/baseline/all_on_generation_rows.jsonl"
ROUTER="$BUNDLE/checkpoints/sw31/router_epoch_001.pt"
GATE="$BUNDLE/checkpoints/prefix_admission/prefix_admission_selection.pt"
SELECTION_SUMMARY="$BUNDLE/checkpoints/prefix_admission/summary.json"
HYBRID="$BUNDLE/results/reproduced_prefix_hybrid"
FINAL="$BUNDLE/results/reproduced_prefix_admission_eval"

"$PYTHON" "$BUNDLE/scripts/verify_bundle.py"
PREFIX=$("$PYTHON" -c "import torch; p=torch.load('$GATE',map_location='cpu',weights_only=False); print(int(p['selected_accuracy_prefix_layers']))")
echo "[eval] selected accuracy prefix K=$PREFIX; shards=$NUM_GPUS"

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  (
    set -o pipefail
    cmd=(
      "$PYTHON" -u
      "$BUNDLE/code/baseline_relative_visual_router/scripts/generate_prefix_hybrid_outcomes.py"
      --manifest-jsonl "$MANIFEST"
      --baseline-rows-jsonl "$BASELINE"
      --data-root "$DATA"
      --output-dir "$HYBRID"
      --checkpoint "$ROUTER"
      --model-source "$MODEL"
      --hf-hub-cache "$BUNDLE/model"
      --prefix-layers "$PREFIX"
      --num-shards "$NUM_GPUS"
      --shard-index "$gpu"
      --chunk-size 32
      --processor-use-fast false
      --attn-implementation sdpa
      --device-map auto
      --first-gpu-max-memory-gb "$FIRST_GPU_MAX_MEMORY_GB"
      --other-gpu-max-memory-gb "$FIRST_GPU_MAX_MEMORY_GB"
      --min-free-gb "$MIN_FREE_GB"
      --process-name "repro-pfx-s${gpu}"
    )
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 |
      tee "$BUNDLE/results/logs/hybrid_shard_${gpu}.log"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" -c "from pathlib import Path; from baseline_relative_visual_router.input_admission import load_prefix_feature_cache; _,r=load_prefix_feature_cache(Path('$HYBRID')/f'prefix_{int($PREFIX):02d}',expected_prefix_layers=int($PREFIX)); assert len(r)==5807, len(r); print('hybrid UID audit passed:',len(r))"

mkdir -p "$FINAL"
score_cmd=(
  "$PYTHON" -u
  "$BUNDLE/code/baseline_relative_visual_router/scripts/evaluate_prefix_admission_external.py"
  --selection-checkpoint "$GATE"
  --external-feature-root "$HYBRID"
  --output-dir "$FINAL"
  --expected-count 5807
  --bootstrap-repetitions 5000
  --random-repetitions 5000
  --seed 20260812
  --device cuda
)
CUDA_VISIBLE_DEVICES=0 "${score_cmd[@]}" 2>&1 | tee "$BUNDLE/results/logs/scoring.log"

report_cmd=(
  "$PYTHON"
  "$BUNDLE/code/baseline_relative_visual_router/scripts/report_prefix_admission.py"
  --selection-summary "$SELECTION_SUMMARY"
  --external-summary "$FINAL/summary.json"
  --output-report "$FINAL/report.md"
  --output-figure "$FINAL/result.png"
)
"${report_cmd[@]}"

if [[ "$RUN_RUNTIME_AUDIT" == "1" ]]; then
  audit_cmd=(
    "$PYTHON" -u
    "$BUNDLE/code/baseline_relative_visual_router/scripts/validate_prefix_runtime_equivalence.py"
    --manifest-jsonl "$MANIFEST"
    --baseline-rows-jsonl "$BASELINE"
    --hybrid-feature-root "$HYBRID"
    --selection-checkpoint "$GATE"
    --external-predictions-jsonl "$FINAL/external_predictions.jsonl"
    --data-root "$DATA"
    --checkpoint "$ROUTER"
    --model-source "$MODEL"
    --hf-hub-cache "$BUNDLE/model"
    --output-json "$FINAL/runtime_equivalence.json"
    --samples-per-benchmark 16
    --first-gpu-max-memory-gb "$FIRST_GPU_MAX_MEMORY_GB"
    --min-free-gb "$MIN_FREE_GB"
    --process-name repro-prefix-runtime-audit
  )
  CUDA_VISIBLE_DEVICES=0 "${audit_cmd[@]}" 2>&1 |
    tee "$BUNDLE/results/logs/runtime_equivalence.log"
fi

echo "[done] $FINAL/report.md"
