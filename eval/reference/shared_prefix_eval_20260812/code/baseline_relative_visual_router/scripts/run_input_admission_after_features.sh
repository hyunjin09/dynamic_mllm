#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aix7101/hyemin/0618_visual_on
PY="$ROOT/dvr_qwen/.venv/bin/python"
PACKAGE="$ROOT/baseline_relative_visual_router"
FEATURE_ROOT=/mnt/hyemin/10k_dataset_mask/baseline_relative_visual_router
TRAIN_FEATURES="$FEATURE_ROOT/input_features_natural_canonical_v1"
EXTERNAL_FEATURES="$FEATURE_ROOT/input_features_external_mmstar_mmmu_v1"
OUTPUT="$FEATURE_ROOT/input_actual_policy_gate_v1"

export PYTHONPATH="$PACKAGE/src:$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

while true; do
  ready=1
  for shard in 00 01 02; do
    [[ -f "$TRAIN_FEATURES/summary_shard_${shard}_of_03.json" ]] || ready=0
    [[ -f "$EXTERNAL_FEATURES/summary_shard_${shard}_of_03.json" ]] || ready=0
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  echo "[$(date -Is)] waiting for six input-feature shard summaries"
  sleep 30
done

"$PY" - "$TRAIN_FEATURES" "$EXTERNAL_FEATURES" <<'PY'
import sys
from pathlib import Path
from baseline_relative_visual_router.input_admission import load_input_feature_cache

expected = [(Path(sys.argv[1]), 22349), (Path(sys.argv[2]), 5807)]
uid_sets = []
for path, count in expected:
    tensors, metadata = load_input_feature_cache(path)
    if len(metadata) != count:
        raise RuntimeError(f"{path}: expected {count} features, found {len(metadata)}")
    uid_sets.append({str(row["uid"]) for row in metadata})
if uid_sets[0] & uid_sets[1]:
    raise RuntimeError("canonical and external feature UIDs overlap")
print("input feature audit passed: canonical=22349 external=5807")
PY

mkdir -p "$OUTPUT"
"$PY" -u "$PACKAGE/scripts/train_input_actual_policy_admission.py" \
  --train-feature-dir "$TRAIN_FEATURES" \
  --external-feature-dir "$EXTERNAL_FEATURES" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --epochs 80 \
  --patience 10 \
  --batch-size 512 \
  --epsilon 0.002 \
  --bootstrap-repetitions 5000 \
  --random-repetitions 5000 \
  --cpu-threads 4 \
  --process-name brvr-input-admission
