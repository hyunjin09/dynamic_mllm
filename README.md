# Dynamic MLLM

Research code, protocols, tests, and reports for dynamic visual computation in
Qwen2.5-VL. The repository intentionally separates version-controlled source
from machine-local datasets, model weights, checkpoints, and experiment
artifacts.

## What Git contains

- model instrumentation, binary visual-routing, MCTS, scoring, and evaluation
  code;
- predictor training and analysis implementations;
- frozen plans, manifests, reports, and compact research state;
- Slurm scheduling utilities and environment specifications;
- project-local agent instructions and the research-control skill;
- reference implementations used to validate execution and training semantics.

Git does **not** contain `.venv`, datasets/labels, model weights, checkpoints,
raw outputs, residual tensors, archives, or Slurm logs. See
[`workspace/external_assets.md`](workspace/external_assets.md) for the manual
transfer contract.

## Clone and environment setup

```bash
git clone git@github.com:hyunjin09/dynamic_mllm.git
cd dynamic_mllm

uv venv .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

The validated environment uses Python 3.12, PyTorch 2.6.0 with CUDA 12.4, and
Transformers 5.3.0. `requirements.txt` is the concise direct-dependency list;
`requirements-lock.txt` captures the complete validated environment.

## External data setup

After transferring datasets and labels to the second server, create the local
dataset link and compatibility label links:

```bash
bash infra/link_external_assets.sh /data/dataset/dynamic_mllm
```

The expected canonical label directories are:

```text
datasets/mcts_labels/gqa_textvqa_chartqa_v1/
datasets/math_labels/wemath20_pro_mcts_max400_v2/
```

Evaluation model/data/checkpoint payloads are also transferred manually when
needed. Their expected paths and verified sizes are documented in
[`workspace/external_assets.md`](workspace/external_assets.md).

## Project navigation

- `AGENTS.md`, `ACCESS_POLICY.md`: execution and access rules
- `plans/`, `workspace/`: approved protocols and compact research state
- `binary_policy/`, `label_regeneration/`, `interventions/`, `scoring/`:
  primary implementation
- `experiments/`, `tools/research_analysis/`: experiment entry points and
  versioned analysis code
- `eval/`: evaluation code and portable protocol metadata
- `tests/`: deterministic unit and contract tests
- `reports/`: scientific and engineering conclusions
