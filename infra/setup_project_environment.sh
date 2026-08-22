#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export UV_CACHE_DIR="$project_root/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_root/.uv-python"
export UV_LINK_MODE=copy

python_version="3.12.7"
backup_root="$project_root/workspace/setup_backups"

if [[ -e .venv ]]; then
  backup_path="$backup_root/venv_before_environment_repair_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$backup_root"
  mv .venv "$backup_path"
  echo "Preserved prior environment at $backup_path"
fi

uv python install "$python_version"
uv venv .venv --python "$python_version"
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip check --python .venv/bin/python

diff -u requirements-lock.txt <(uv pip freeze --python .venv/bin/python)

.venv/bin/python - <<'PY'
import importlib.metadata
import platform

expected = {
    "accelerate": "1.6.0",
    "av": "17.0.1",
    "datasets": "4.0.0",
    "mathruler": "0.1.0",
    "pillow": "11.1.0",
    "pylatexenc": "2.10",
    "pytest": "9.1.1",
    "pyyaml": "6.0.2",
    "qwen-vl-utils": "0.0.14",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "5.3.0",
}

assert platform.python_version() == "3.12.7", platform.python_version()
for package, version in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == version, f"{package}: expected {version}, found {actual}"

import accelerate  # noqa: F401
import av  # noqa: F401
import datasets  # noqa: F401
import mathruler  # noqa: F401
import PIL  # noqa: F401
import qwen_vl_utils  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import transformers  # noqa: F401
import yaml  # noqa: F401

print("Pinned environment import verification passed.")
PY

.venv/bin/python -m pytest -q \
  tests/test_stage_a_utils.py \
  tests/test_binary_policy.py
