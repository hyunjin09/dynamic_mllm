#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aix7101/hyemin/0618_visual_on
PY="$ROOT/dvr_qwen/.venv/bin/python"
PACKAGE="$ROOT/baseline_relative_visual_router"
DATA_ROOT=/mnt/hyemin/10k_dataset_mask/baseline_relative_visual_router
CANONICAL="$DATA_ROOT/prefix_hybrid_canonical_v1"
SELECTION="$DATA_ROOT/prefix_admission_selection_v1"
EXTERNAL="$DATA_ROOT/prefix_hybrid_external_selected_v1"
FINAL="$DATA_ROOT/prefix_admission_external_eval_v1"
LOG_ROOT="$PACKAGE/logs/prefix_pipeline_v1"
mkdir -p "$LOG_ROOT"

export PYTHONPATH="$PACKAGE/src:$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

while true; do
  ready=1
  for shard in 00 01 02 03; do
    [[ -f "$CANONICAL/summary_shard_${shard}_of_04.json" ]] || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  echo "[$(date -Is)] waiting for four canonical shared-prefix summaries"
  sleep 60
done

"$PY" - "$CANONICAL" <<'PY'
import sys
from pathlib import Path
from baseline_relative_visual_router.input_admission import load_prefix_feature_cache

root = Path(sys.argv[1])
uid_sets = []
for prefix in (2, 4, 8):
    _, rows = load_prefix_feature_cache(root / f"prefix_{prefix:02d}", expected_prefix_layers=prefix)
    if len(rows) != 22349:
        raise RuntimeError(f"K={prefix}: expected 22349 rows, found {len(rows)}")
    uid_sets.append([str(row["uid"]) for row in rows])
if not all(values == uid_sets[0] for values in uid_sets[1:]):
    raise RuntimeError("canonical prefix depths do not have identical UID order")
print("canonical shared-prefix audit passed: 3 x 22349")
PY

mkdir -p "$SELECTION"
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$PACKAGE/scripts/train_prefix_actual_policy_admission.py" \
  --feature-root "$CANONICAL" \
  --output-dir "$SELECTION" \
  --prefix-layers 2,4,8 \
  --expected-count 22349 \
  --device cuda \
  --process-name brvr-prefix-select \
  2>&1 | tee "$LOG_ROOT/selection.log"

PREFIX=$("$PY" - "$SELECTION/summary.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selected_accuracy_prefix_layers"])
PY
)
echo "[$(date -Is)] calibration selected K=$PREFIX; starting external generation"

pids=()
for gpu in 0 1 2 3; do
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$PACKAGE/scripts/generate_prefix_hybrid_outcomes.py" \
      --manifest-jsonl /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2/samples.jsonl \
      --baseline-rows-jsonl /mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval/sw31_bt_leg_s41_mmstar_mmmu_final_v2/merged_final/heldout_generation_rows.jsonl \
      --data-root /mnt/hyemin/10k_dataset_mask/heldout_mmstar_mmmu_final_v2 \
      --output-dir "$EXTERNAL" \
      --prefix-layers "$PREFIX" \
      --num-shards 4 \
      --shard-index "$gpu" \
      --chunk-size 32 \
      --first-gpu-max-memory-gb 20 \
      --min-free-gb 20 \
      --process-name "brvr-pfx-ext-s${gpu}" \
      2>&1 | tee "$LOG_ROOT/external_shard${gpu}.log"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" - "$EXTERNAL" "$PREFIX" <<'PY'
import sys
from pathlib import Path
from baseline_relative_visual_router.input_admission import load_prefix_feature_cache

root, prefix = Path(sys.argv[1]), int(sys.argv[2])
_, rows = load_prefix_feature_cache(root / f"prefix_{prefix:02d}", expected_prefix_layers=prefix)
if len(rows) != 5807:
    raise RuntimeError(f"external K={prefix}: expected 5807 rows, found {len(rows)}")
print(f"external shared-prefix audit passed: K={prefix}, n=5807")
PY

mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$PACKAGE/scripts/evaluate_prefix_admission_external.py" \
  --selection-checkpoint "$SELECTION/prefix_admission_selection.pt" \
  --external-feature-root "$EXTERNAL" \
  --output-dir "$FINAL" \
  --expected-count 5807 \
  --device cuda \
  2>&1 | tee "$LOG_ROOT/external_evaluation.log"

"$PY" "$PACKAGE/scripts/report_prefix_admission.py" \
  --selection-summary "$SELECTION/summary.json" \
  --external-summary "$FINAL/summary.json" \
  --output-report "$ROOT/reports/shared_prefix_actual_policy_admission_20260812.md" \
  --output-figure "$PACKAGE/experiments/shared_prefix_actual_policy_admission_20260812.png"

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

echo "[$(date -Is)] shared-prefix pipeline complete"
