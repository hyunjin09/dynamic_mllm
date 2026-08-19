#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aix7101/hyemin/0618_visual_on
PY="$ROOT/dvr_qwen/.venv/bin/python"
PACKAGE="$ROOT/baseline_relative_visual_router"
DATA_ROOT=/mnt/hyemin/10k_dataset_mask/baseline_relative_visual_router
SELECTION="$DATA_ROOT/prefix_admission_selection_v1"
EXTERNAL="$DATA_ROOT/prefix_hybrid_external_selected_v1"
FINAL="$DATA_ROOT/prefix_admission_external_eval_v1"
LOG_ROOT="$PACKAGE/logs/prefix_pipeline_v1"
export PYTHONPATH="$PACKAGE/src:$ROOT"

while [[ ! -f "$FINAL/summary.json" ]]; do
  echo "[$(date -Is)] waiting for external prefix admission summary"
  sleep 60
done

if [[ ! -f "$ROOT/reports/shared_prefix_actual_policy_admission_20260812.md" ]]; then
  "$PY" "$PACKAGE/scripts/report_prefix_admission.py" \
    --selection-summary "$SELECTION/summary.json" \
    --external-summary "$FINAL/summary.json" \
    --output-report "$ROOT/reports/shared_prefix_actual_policy_admission_20260812.md" \
    --output-figure "$PACKAGE/experiments/shared_prefix_actual_policy_admission_20260812.png"
fi

if [[ ! -f "$FINAL/runtime_equivalence.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "$PY" -u "$PACKAGE/scripts/validate_prefix_runtime_equivalence.py" \
    --manifest-jsonl /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2/samples.jsonl \
    --baseline-rows-jsonl /mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval/sw31_bt_leg_s41_mmstar_mmmu_final_v2/merged_final/heldout_generation_rows.jsonl \
    --hybrid-feature-root "$EXTERNAL" \
    --selection-checkpoint "$SELECTION/prefix_admission_selection.pt" \
    --external-predictions-jsonl "$FINAL/external_predictions.jsonl" \
    --data-root /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2 \
    --output-json "$FINAL/runtime_equivalence.json" \
    --samples-per-benchmark 16 \
    --process-name brvr-prefix-runtime-audit \
    2>&1 | tee "$LOG_ROOT/runtime_equivalence.log"
fi

echo "[$(date -Is)] prefix post-processing complete"
