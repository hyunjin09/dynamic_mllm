#!/usr/bin/env bash
set -euo pipefail

BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
HYBRID=${HYBRID:-$BUNDLE/results/reproduced_prefix_hybrid}
FINAL=${FINAL:-$BUNDLE/results/reproduced_prefix_admission_eval}
GATE="$BUNDLE/checkpoints/prefix_admission/prefix_admission_selection.pt"

export PYTHONPATH="$BUNDLE/code/baseline_relative_visual_router/src:$BUNDLE/code"
export TMPDIR="$BUNDLE/state/tmp"
mkdir -p "$TMPDIR" "$FINAL"

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
  --device cpu
)
"${score_cmd[@]}"

report_cmd=(
  "$PYTHON"
  "$BUNDLE/code/baseline_relative_visual_router/scripts/report_prefix_admission.py"
  --selection-summary "$BUNDLE/checkpoints/prefix_admission/summary.json"
  --external-summary "$FINAL/summary.json"
  --output-report "$FINAL/report.md"
  --output-figure "$FINAL/result.png"
)
"${report_cmd[@]}"
