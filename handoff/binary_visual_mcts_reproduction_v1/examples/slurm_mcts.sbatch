#!/usr/bin/env bash
# Edit only the scheduler resources and exported paths for the target server.
# The scientific search settings are fixed inside the bundled runner/MCTS.
#SBATCH --job-name=binary-mcts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --output=runs/mcts_%j.log

set -euo pipefail

: "${BUNDLE_ROOT:?export BUNDLE_ROOT}"
: "${MODEL_PATH:?export MODEL_PATH}"
: "${MANIFEST:?export MANIFEST}"
: "${CONTRACT_SHA256:?export CONTRACT_SHA256}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"
: "${DATASET_VERSION:?export DATASET_VERSION}"

cd "$BUNDLE_ROOT"
mkdir -p runs "$OUTPUT_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

srun .venv/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/run_label_regeneration.py \
  --mode mcts \
  --manifest "$MANIFEST" \
  --model-path "$MODEL_PATH" \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --contract-sha256 "$CONTRACT_SHA256" \
  --dataset-version "$DATASET_VERSION" \
  --output-root "$OUTPUT_ROOT" \
  --seed 20260810
