#!/usr/bin/env bash
#SBATCH --job-name=3b-v2-repaired-gate
#SBATCH --account=research
#SBATCH --partition=gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/repaired-gate-%j.out
#SBATCH --error=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/repaired-gate-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source analysis/3b_greedy_route_corpus_replay_audit_v2/code/environment.sh
module load cuda/12.8
srun --ntasks=1 --kill-on-bad-exit=1 .venv/bin/python \
    analysis/3b_greedy_route_corpus_replay_audit_v2/code/repair_gate.py
