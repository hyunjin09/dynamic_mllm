#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -f /etc/profile.d/research-modules.sh ]]; then
    source /etc/profile.d/research-modules.sh
fi
module purge
module load cuda/12.8
source .venv/bin/activate

PYTHONPATH=. python experiments/build_sequential_four_action_smoke.py --resume

telemetry_pid=""
cleanup() {
    if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}"
        wait "${telemetry_pid}" || true
    fi
}
trap cleanup EXIT

PYTHONPATH=. python experiments/monitor_route_conditioned_gpu.py \
    --output analysis/4action_sequential_label_conversion/smoke_gpu_telemetry.csv \
    --interval-seconds 5 \
    --resume &
telemetry_pid=$!

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. \
OMP_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=8 \
    experiments/run_sequential_four_action_label_conversion.py \
    --mode smoke \
    --resume

PYTHONPATH=. python experiments/verify_sequential_four_action_smoke_resume.py \
    --mode snapshot

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. \
OMP_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=8 \
    experiments/run_sequential_four_action_label_conversion.py \
    --mode smoke \
    --resume

PYTHONPATH=. python experiments/verify_sequential_four_action_smoke_resume.py \
    --mode verify
