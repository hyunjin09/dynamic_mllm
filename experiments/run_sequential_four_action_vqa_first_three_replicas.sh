#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 SMOKE_JOB_ID" >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is required" >&2
    exit 64
fi

readonly smoke_job_id="$1"
readonly config="configs/sequential_four_action_label_conversion_three_replicas.yaml"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -f /etc/profile.d/research-modules.sh ]]; then
    source /etc/profile.d/research-modules.sh
fi
module purge
module load cuda/12.8
source .venv/bin/activate

PYTHONPATH=. python experiments/audit_sequential_four_action_smoke.py \
    --job-ids "${smoke_job_id}" \
    --resume

PYTHONPATH=. python experiments/prepare_sequential_dataset_priority_claims.py \
    --config "${config}" \
    --defer wemath20_standard \
    --defer wemath2pro \
    --resume

telemetry_pid=""
cleanup() {
    if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}"
        wait "${telemetry_pid}" || true
    fi
}
trap cleanup EXIT

PYTHONPATH=. python experiments/monitor_route_conditioned_gpu.py \
    --output "analysis/4action_sequential_label_conversion_three_replicas/vqa_first_${SLURM_JOB_ID}_gpu_telemetry.csv" \
    --interval-seconds 5 \
    --resume &
telemetry_pid=$!

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. \
OMP_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=24 \
    experiments/run_sequential_four_action_label_conversion_three_replicas.py \
    --config "${config}" \
    --mode full \
    --resume

# WeMath execution and finalization are submitted only after the 24-worker
# throughput gate beats the preserved 16-worker baseline.
