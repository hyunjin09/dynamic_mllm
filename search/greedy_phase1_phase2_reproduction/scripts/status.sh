#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PACKAGE_ROOT}/config/paths.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy config/paths.env.example to config/paths.env." >&2
  exit 2
fi
source "${ENV_FILE}"

echo "Output: ${OUTPUT_ROOT}"
ps -eo pid,stat,etime,%cpu,%mem,cmd | grep -E 'gpp_(gate|p1|p2)' | grep -v grep || true

for phase in phase1 phase2; do
  root="${OUTPUT_ROOT}/raw/${phase}"
  if [[ -d "${root}" ]]; then
    samples="$(find "${root}" -type f -path '*/samples/*.json' | wc -l)"
    errors="$(find "${root}" -type f -name errors.jsonl -size +0c | wc -l)"
    summaries="$(find "${root}" -type f -name summary.json | wc -l)"
    echo "${phase}: samples=${samples}, shard_summaries=${summaries}, nonempty_error_files=${errors}"
  else
    echo "${phase}: not started"
  fi
done

for file in \
  "${OUTPUT_ROOT}/gate/summary.json" \
  "${OUTPUT_ROOT}/phase1/summary.json" \
  "${OUTPUT_ROOT}/final_phase1_phase2/summary.json" \
  "${OUTPUT_ROOT}/final_phase1_phase2/audit_summary.json"; do
  [[ -f "${file}" ]] && echo "exists: ${file}"
done
