#!/usr/bin/env bash
# Source from the project root; all writable caches stay in the audit root.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$PWD/analysis/3b_greedy_route_corpus_replay_audit/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$PWD/analysis/3b_greedy_route_corpus_replay_audit/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$PWD/analysis/3b_greedy_route_corpus_replay_audit/tmp"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export UV_PYTHON_DOWNLOADS=never
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$XDG_CACHE_HOME"
