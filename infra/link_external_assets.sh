#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-/data/dataset/dynamic_mllm}"

if [[ ! -d "$data_root" ]]; then
  echo "Dataset root does not exist: $data_root" >&2
  exit 1
fi

cd "$project_root"

if [[ -L datasets ]]; then
  current_target="$(readlink -f datasets)"
  requested_target="$(readlink -f "$data_root")"
  if [[ "$current_target" != "$requested_target" ]]; then
    echo "datasets already points to $current_target, not $requested_target" >&2
    exit 1
  fi
elif [[ -e datasets ]]; then
  echo "datasets exists and is not a symlink; leaving it unchanged"
else
  ln -s "$data_root" datasets
fi

mkdir -p outputs/label_regeneration

ensure_relative_link() {
  local link_path=$1
  local link_target=$2
  local expected_path=$3

  if [[ ! -d "$expected_path" ]]; then
    echo "Required label directory is missing: $expected_path" >&2
    return 1
  fi

  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink "$link_path")" != "$link_target" ]]; then
      echo "$link_path exists with an unexpected target: $(readlink "$link_path")" >&2
      return 1
    fi
  elif [[ -e "$link_path" ]]; then
    echo "$link_path exists and is not a symlink" >&2
    return 1
  else
    ln -s "$link_target" "$link_path"
  fi
}

ensure_relative_link \
  outputs/label_regeneration/v1 \
  ../../datasets/mcts_labels/gqa_textvqa_chartqa_v1 \
  datasets/mcts_labels/gqa_textvqa_chartqa_v1

ensure_relative_link \
  outputs/label_regeneration/wemath2pro_cap400_v2 \
  ../../datasets/math_labels/wemath20_pro_mcts_max400_v2 \
  datasets/math_labels/wemath20_pro_mcts_max400_v2

echo "External dataset and canonical MCTS-label links are ready."
