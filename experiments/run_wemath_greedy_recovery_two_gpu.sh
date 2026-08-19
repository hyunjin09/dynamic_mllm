#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <phase1|phase2> <first-global-shard>" >&2
  exit 2
fi

phase="$1"
first_shard="$2"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "missing project interpreter: .venv/bin/python" >&2
  exit 1
fi

visible="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a devices <<< "$visible"
if [[ ${#devices[@]} -lt 2 ]]; then
  echo "two allocated GPUs are required; CUDA_VISIBLE_DEVICES=$visible" >&2
  exit 1
fi

mkdir -p runs/wemath_greedy_recovery
export PYTHONPATH="$project_root"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

pids=()
for local_index in 0 1; do
  global_shard=$((first_shard + local_index))
  log="runs/wemath_greedy_recovery/${phase}_shard_${global_shard}.log"
  CUDA_VISIBLE_DEVICES="${devices[$local_index]}" \
    .venv/bin/python experiments/run_wemath_greedy_recovery.py \
      --mode "$phase" \
      --num-shards 4 \
      --shard-index "$global_shard" \
      >"$log" 2>&1 &
  pids+=("$!")
  echo "started $phase global shard $global_shard on allocation device ${devices[$local_index]} pid ${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
