#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 CALIBRATION_PILOT_JOB_ID" >&2
    exit 64
fi

readonly pilot_job_id="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -f /etc/profile.d/research-modules.sh ]]; then
    source /etc/profile.d/research-modules.sh
fi
module purge
module load cuda/12.8
source .venv/bin/activate

# This job has an afterok dependency on the calibration/pilot allocation.  The
# scientific gate is still checked explicitly: a completed allocation is not
# evidence that the new labeling semantics passed their real-data audit.
PYTHONPATH=. python experiments/audit_three_action_label_pilot.py \
    --job-ids "${pilot_job_id}" \
    --resume

PYTHONPATH=. python experiments/estimate_three_action_label_conversion.py --resume

telemetry_pid=""
cleanup() {
    if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}"
        wait "${telemetry_pid}" || true
    fi
}
trap cleanup EXIT

PYTHONPATH=. python experiments/monitor_route_conditioned_gpu.py \
    --output analysis/three_action_answer_aligned_label_conversion/full_gpu_telemetry.csv \
    --interval-seconds 5 \
    --resume &
telemetry_pid=$!

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. \
OMP_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=16 \
    experiments/run_three_action_label_conversion.py \
    --mode full \
    --resume
