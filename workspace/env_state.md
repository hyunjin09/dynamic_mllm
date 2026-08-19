# Environment State

- Updated: 2026-08-13
- Environment: project-local `.venv/`
- Package cache: project-local `.uv-cache/`
- Environment tool: `uv 0.10.7`
- Python: 3.12.7
- PyTorch: 2.6.0+cu124
- PyTorch CUDA runtime: 12.4
- Transformers: 5.3.0
- Torchvision: 0.21.0
- Accelerate: 1.6.0
- Pillow: 11.1.0
- PyYAML: 6.0.2
- Pytest: 9.1.1
- Datasets: 4.0.0
- PyArrow: 25.0.0
- Pandas: 3.0.5
- Fsspec: 2025.3.0
- Hugging Face Hub: 1.27.0
- Tokenizers: 0.22.2
- Direct dependency specification: `requirements.txt`
- Setup command: `UV_CACHE_DIR=.uv-cache UV_LINK_MODE=copy uv venv .venv && UV_CACHE_DIR=.uv-cache UV_LINK_MODE=copy uv pip install --python .venv/bin/python -r requirements.txt`
- Slurm job: `97995` (`stage_a_env_localcache_20260804`, CPU-only on `a4000`/node05)
- Verification: pinned imports succeeded and `tests.test_stage_a_utils` passed 6 tests.
- Scheduling constraint amended 2026-08-18: `node04` is allowed again by
  explicit user instruction. `node03` remains prohibited. CPU jobs target
  `node05` first, then an approved GPU node if `node05` is unavailable.
- Stage C dataset extension job: `98372`
  (`stage_c_datasets_env_20260805`, CPU-only on `a4000`/node05). It installed
  the pinned `datasets==4.0.0` dependency in the same project-local `.venv/`;
  `uv pip check` reports all 59 packages compatible.
- Note: initial job `97987` was cancelled before completion because its default
  `uv` cache location was outside the project write boundary. Its partial
  project-local environment was preserved at
  `workspace/setup_backups/stage_a_env_partial_20260804_1506/` and is not used.
- Transformers migration job: `99717`
  (`binary_env_tf53_20260809`, CPU-only on `a4000`/node05). It migrated the
  project-local `.venv` from Transformers 4.51.3 to 5.3.0 to match the MCTS v2
  label runtime. `uv pip check` reports all 71 packages compatible; 10 Stage A
  utility tests and 12 binary-policy contract tests passed.
- Migration evidence:
  `outputs/env_migrations/transformers_5_3_0_v1.json`.
- Rollback pin:
  `workspace/env_migrations/requirements_transformers_4_51_3.txt`.
- Label-regeneration contract verification: project-local `.venv` reports
  Python 3.12.7, PyTorch 2.6.0+cu124, Transformers 5.3.0, Accelerate 1.6.0,
  and Pillow 11.1.0. These versions are frozen in
  `outputs/label_regeneration/v1/frozen_execution_contract.json`.
- We-Math2.0-Pro scorer extension (2026-08-11): added
  `mathruler==0.1.0` and `pylatexenc==2.10` to the same project-local `.venv`
  through CPU-only Slurm job `99832` on node05. Package installation
  completed; the job itself exited nonzero only because its final verification
  mistakenly invoked the intentionally absent `.venv/bin/python -m pip`.
  The authoritative `uv pip check --python .venv/bin/python` passed with 73
  compatible packages, and focused MathRuler equivalence tests passed.
- P11 test-runner addition (2026-08-13): added only `pytest==9.1.1` with
  `uv pip install --python .venv/bin/python pytest`; no runtime dependency or
  model package was changed.
- External evaluation dependency repair (2026-08-13): the first direct-router
  preflight stopped before processing a sample because `qwen_vl_utils` was not
  installed. The reference bundle pins `qwen-vl-utils==0.0.14` and
  `av==17.0.1`; these are now part of `requirements.txt` and were added to the
  same project-local `.venv` by CPU-only node05 Slurm job `100779`. `uv pip
  check` passed all 78 installed packages and both imports passed. No model,
  manifest, checkpoint, prompt, or evaluation result changed.
