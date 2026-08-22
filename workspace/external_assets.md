# External Assets for a Second Server

These payloads are intentionally excluded from Git. Transfer only the subsets
needed for the next action.

## Required for binary predictor training

| Payload | Canonical project-relative path | Current size |
|---|---|---:|
| GQA/TextVQA/ChartQA regenerated MCTS labels | `datasets/mcts_labels/gqa_textvqa_chartqa_v1/` | 23 GB |
| WeMath2.0-Pro cap-400 MCTS labels | `datasets/math_labels/wemath20_pro_mcts_max400_v2/` | 14 GB |

The source clone does not contain `datasets`. Place these directories under a
server-local data root and run:

```bash
bash infra/link_external_assets.sh /path/to/dynamic_mllm_data_root
```

## Required only for the bundled external evaluation

Under `eval/reference/shared_prefix_eval_20260812/`, Git retains the protocol,
code, scripts, environment files, inventory, and checksums. Transfer these
payload directories separately when running the evaluation:

| Directory | Current size | Contents |
|---|---:|---|
| `model/` | 16 GB | pinned Qwen2.5-VL-7B-Instruct snapshot |
| `data/` | 12 GB | held-out benchmark images/manifests |
| `checkpoints/` | 241 MB | router checkpoints and predictions |
| `baseline/` | 14 MB | frozen ALL-ON generations |
| `results/` | 31 MB | generated evaluation results |

`baseline/` and `results/` are evidence, not prerequisites for a fresh run.
If the exact pinned model revision already exists in an allowed machine-local
model store, it may be linked instead of duplicated:

```bash
mkdir -p eval/reference/shared_prefix_eval_20260812/model
ln -s \
  /path/to/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  eval/reference/shared_prefix_eval_20260812/model/Qwen2.5-VL-7B-Instruct_cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Do not create substitute links for `data/` or `checkpoints/` unless their
bundle inventories and checksums match.

## Historical and regenerable artifacts

The following roots are excluded because they are large or machine-local:

| Root | Current size | Policy |
|---|---:|---|
| `artifacts/` | 144 GB | transfer only for old v3 null-analysis reproduction |
| `outputs/` | 32 GB | transfer selected checkpoints/results when resuming a run |
| `archives/` | 118 MB | historical frozen bundles; optional |
| `runs/` | 38 MB | Slurm logs; optional |
| `.venv/` | 5.7 GB | recreate from `requirements-lock.txt` |
| `.uv-cache/` | 5.7 GB | recreate automatically; never transfer |

Live scheduler queue files such as `state/slurm_job_queue.json` and the legacy
`state/gpu_experiment_queue.json` are excluded. They are machine-specific and
must not be replayed on another server. Compact completed-run records under
`state/runs/` remain in Git as provenance.

Machine-relocated copies matching
`search/greedy_phase1_phase2_reproduction/manifests/*.current_server.jsonl`
are excluded as well. Git retains the single canonical manifest; use the
packaged relocation script to adapt its image paths on the destination server.
