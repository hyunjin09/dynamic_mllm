# LMMS-Eval scoring contract

- Package: `lmms-eval==0.7.3`
- Upstream tag: `v0.7.3` (`ac86d61af84f0e188422188c6b875226988c8988`)
- GQA: official `exact_match` with `ignore_case=true` and `ignore_punctuation=true`; binary iff score is 1.0.
- ChartQA: official `chartqa_process_results` / `relaxed_overall`; binary iff score is 1.0.
- TextVQA: official `textvqa_process_results` EvalAI leave-one-out consensus; raw fractional score is preserved; the repository's existing 8K convention defines correct as score >= 0.5.
- No historical correctness field participates in scoring.

## Installed source hashes

- `chartqa_process_results`: `5e336957fada294cc7cc493136e0177103efcf04e1d7bc7b4900fa7ab8c75e0e` (`/home/aix7101/hyemin/0830/dynamic_mllm/.venv/lib/python3.12/site-packages/lmms_eval/tasks/chartqa/utils.py`)
- `gqa_exact_match`: `eaf48195af80a46d22e545a03a1bd7e4cad4fcbadeb3d2d817ac7d148390d8f4` (`/home/aix7101/hyemin/0830/dynamic_mllm/.venv/lib/python3.12/site-packages/lmms_eval/api/metrics.py`)
- `textvqa_process_results`: `4d1e10d2a72136e2fb11486827f29d955f4782b51e0cafe7228b1d53da14d405` (`/home/aix7101/hyemin/0830/dynamic_mllm/.venv/lib/python3.12/site-packages/lmms_eval/tasks/textvqa/utils.py`)
