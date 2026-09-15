#!/usr/bin/env bash
#SBATCH --job-name=3b-v2-routes
#SBATCH --account=research
#SBATCH --partition=gpu
#SBATCH --qos=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/routes-%j.out
#SBATCH --error=analysis/3b_greedy_route_corpus_replay_audit_v2/logs/routes-%j.err
#SBATCH --kill-on-invalid-dep=yes
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source analysis/3b_greedy_route_corpus_replay_audit_v2/code/environment.sh
module load cuda/12.8
srun --ntasks=1 --kill-on-bad-exit=1 .venv/bin/torchrun --standalone --nproc-per-node=8 \
    analysis/3b_greedy_route_corpus_replay_audit_v2/code/replay.py --stage routes
