#!/usr/bin/env bash
#SBATCH --job-name=3b-v2-dense-report
#SBATCH --account=research
#SBATCH --partition=cpu
#SBATCH --qos=cpu-light
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/dense-report-%j.out
#SBATCH --error=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/dense-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source analysis/3b_greedy_route_corpus_replay_audit_v2/code/environment.sh
.venv/bin/python analysis/3b_greedy_route_corpus_replay_audit_v2/code/dense_report.py
