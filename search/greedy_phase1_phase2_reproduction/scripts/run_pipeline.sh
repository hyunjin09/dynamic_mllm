#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PACKAGE_ROOT}/config/paths.env}"
STAGE="${1:-all}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy config/paths.env.example to config/paths.env." >&2
  exit 2
fi
source "${ENV_FILE}"

CONFIG="${PACKAGE_ROOT}/config/collection_config.json"
CORE="${PACKAGE_ROOT}/scripts/core"
LOG_DIR="${OUTPUT_ROOT}/logs"
STATE_DIR="${OUTPUT_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${TMPDIR}"

export VISUAL_INJECTION_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TMPDIR
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

run_preflight() {
  (cd "${PACKAGE_ROOT}" && sha256sum -c reference/CHECKSUMS.sha256)
  local verify=()
  if [[ "${VERIFY_IMAGE_HASHES:-0}" == "1" ]]; then
    verify=(--verify-image-hashes)
  fi
  local strict=()
  if [[ "${STRICT_RUNTIME:-1}" == "1" ]]; then
    strict=(--strict-runtime)
  fi
  "${PYTHON_BIN}" "${PACKAGE_ROOT}/scripts/preflight.py" \
    --project-root "${PROJECT_ROOT}" \
    --model-source "${MODEL_SOURCE}" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --expected-samples "${EXPECTED_SAMPLES}" \
    --expected-revision "${EXPECTED_MODEL_REVISION}" \
    --expected-semantic-sha256 "${EXPECTED_MANIFEST_SEMANTIC_SHA256}" \
    --expected-config-sha256 "${EXPECTED_CONFIG_SHA256}" \
    "${verify[@]}" "${strict[@]}" | tee "${STATE_DIR}/preflight.json"
  "${PYTHON_BIN}" "${CORE}/collect_phase1_candidates.py" --self-test
  "${PYTHON_BIN}" "${CORE}/collect_phase2_candidates.py" --self-test
}

assert_shard_lock() {
  local phase="$1"
  local lock="${STATE_DIR}/${phase}_num_shards.txt"
  local count="${#GPUS[@]}"
  if [[ -f "${lock}" ]] && [[ "$(<"${lock}")" != "${count}" ]]; then
    echo "${phase} was started with $(<"${lock}") shards; refusing to resume with ${count}." >&2
    exit 2
  fi
  printf '%s\n' "${count}" > "${lock}"
}

run_shards() {
  local phase="$1"
  shift
  assert_shard_lock "${phase}"
  local pids=()
  local index gpu name log
  for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    name="gpp_${phase}_s${index}"
    log="${LOG_DIR}/${name}.log"
    echo "Launching ${name} on physical GPU ${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c 'exec -a "$1" "${@:2}"' _ "${name}" \
      "${PYTHON_BIN}" "$@" --num-shards "${#GPUS[@]}" --shard-index "${index}" > "${log}" 2>&1 &
    pids+=("$!")
  done
  printf '%s\n' "${pids[@]}" > "${STATE_DIR}/${phase}_pids.txt"
  local failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "${phase} shard ${index} failed; inspect ${LOG_DIR}/gpp_${phase}_s${index}.log" >&2
      failed=1
    fi
  done
  [[ "${failed}" == "0" ]]
}

run_gate() {
  local gpu="${GPUS[0]}"
  local log="${LOG_DIR}/gpp_gate.log"
  CUDA_VISIBLE_DEVICES="${gpu}" bash -c 'exec -a "$1" "${@:2}"' _ gpp_gate \
    "${PYTHON_BIN}" "${CORE}/collect_phase1_candidates.py" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_ROOT}/gate" \
    --mode gate \
    --gate-summary "${OUTPUT_ROOT}/gate/summary.json" \
    --gate-per-cell 2 \
    --data-splits "${DATA_SPLITS}" \
    --benchmarks "${BENCHMARKS}" \
    --model-source "${MODEL_SOURCE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --processor-use-fast "${PROCESSOR_USE_FAST}" \
    --first-gpu-max-memory-gb "${FIRST_GPU_MAX_MEMORY_GB}" \
    --other-gpu-max-memory-gb "${OTHER_GPU_MAX_MEMORY_GB}" \
    --cpu-max-memory-gb "${CPU_MAX_MEMORY_GB}" > "${log}" 2>&1
  "${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p,indent=2)); assert p["decision"]=="canonical_current_model_anchor_gate_pass"' \
    "${OUTPUT_ROOT}/gate/summary.json"
}

run_phase1() {
  run_shards phase1 "${CORE}/collect_phase1_candidates.py" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_ROOT}/raw/phase1" \
    --mode search \
    --gate-summary "${OUTPUT_ROOT}/gate/summary.json" \
    --data-splits "${DATA_SPLITS}" \
    --benchmarks "${BENCHMARKS}" \
    --model-source "${MODEL_SOURCE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --processor-use-fast "${PROCESSOR_USE_FAST}" \
    --first-gpu-max-memory-gb "${FIRST_GPU_MAX_MEMORY_GB}" \
    --other-gpu-max-memory-gb "${OTHER_GPU_MAX_MEMORY_GB}" \
    --cpu-max-memory-gb "${CPU_MAX_MEMORY_GB}"
}

aggregate_phase1() {
  "${PYTHON_BIN}" "${CORE}/aggregate_phase1.py" \
    --input-dir "${OUTPUT_ROOT}/raw/phase1" \
    --output-dir "${OUTPUT_ROOT}/phase1" | tee "${LOG_DIR}/aggregate_phase1.log"
}

run_phase2() {
  run_shards phase2 "${CORE}/collect_phase2_candidates.py" \
    --phase1-dir "${OUTPUT_ROOT}/raw/phase1" \
    --budget-statistics "${OUTPUT_ROOT}/phase1/benchmark_budget_statistics.json" \
    --gate-summary "${OUTPUT_ROOT}/gate/summary.json" \
    --output-dir "${OUTPUT_ROOT}/raw/phase2" \
    --benchmarks "${BENCHMARKS}" \
    --random-per-budget "${RANDOM_PER_BUDGET}" \
    --local-per-operation "${LOCAL_PER_OPERATION}" \
    --seed "${PHASE2_SEED}" \
    --model-source "${MODEL_SOURCE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --processor-use-fast "${PROCESSOR_USE_FAST}" \
    --first-gpu-max-memory-gb "${FIRST_GPU_MAX_MEMORY_GB}" \
    --other-gpu-max-memory-gb "${OTHER_GPU_MAX_MEMORY_GB}" \
    --cpu-max-memory-gb "${CPU_MAX_MEMORY_GB}"
}

finalize() {
  "${PYTHON_BIN}" "${CORE}/aggregate_phase1_phase2.py" \
    --manifest "${MANIFEST}" \
    --phase1-dir "${OUTPUT_ROOT}/raw/phase1" \
    --phase2-dir "${OUTPUT_ROOT}/raw/phase2" \
    --config "${CONFIG}" \
    --gate-summary "${OUTPUT_ROOT}/gate/summary.json" \
    --output-dir "${OUTPUT_ROOT}/final_phase1_phase2" \
    --expected-samples "${EXPECTED_SAMPLES}" | tee "${LOG_DIR}/aggregate_final.log"
  "${PYTHON_BIN}" "${CORE}/audit_final_phase1_phase2.py" \
    --input-dir "${OUTPUT_ROOT}/final_phase1_phase2" | tee "${LOG_DIR}/audit_final.log"
}

case "${STAGE}" in
  preflight) run_preflight ;;
  gate) run_preflight; run_gate ;;
  phase1) run_phase1 ;;
  aggregate1) aggregate_phase1 ;;
  phase2) run_phase2 ;;
  finalize) finalize ;;
  all)
    run_preflight
    run_gate
    run_phase1
    aggregate_phase1
    run_phase2
    finalize
    ;;
  *)
    echo "Usage: $0 {preflight|gate|phase1|aggregate1|phase2|finalize|all}" >&2
    exit 2
    ;;
esac
