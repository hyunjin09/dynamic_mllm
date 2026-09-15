#!/usr/bin/env bash
#SBATCH --job-name=3b-route-replay-audit
#SBATCH --account=research
#SBATCH --partition=gpu
#SBATCH --qos=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=analysis/3b_greedy_route_corpus_replay_audit/logs/slurm-%j.out
#SBATCH --error=analysis/3b_greedy_route_corpus_replay_audit/logs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source analysis/3b_greedy_route_corpus_replay_audit/code/environment.sh
module load cuda/12.8
test -f analysis/3b_greedy_route_corpus_replay_audit/source_inventory/prepared_complete.json
# One allocation, one model replica per GPU. Full replay fails closed at the gate.
srun --ntasks=1 --kill-on-bad-exit=1 \
    .venv/bin/torchrun --standalone --nproc-per-node=8 \
    analysis/3b_greedy_route_corpus_replay_audit/code/replay.py
# CPU aggregation is a separate afterok job, so the GPUs are released promptly.
