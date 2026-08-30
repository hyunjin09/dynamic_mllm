# Four-Action Label Executor Provenance Audit

Date: 2026-08-30 KST
Scope: provenance and scientific contract of the executor that produced
`datasets/mcts_labels_4action/sequential_branching_v1/`
Constraint: no current executor change, label change, regeneration, or W2C
repair was performed.

## Audit conclusion

The contract-bound historical source is exactly reconstructable. All 16 source
files and the YAML configuration reproduce the SHA-256 values embedded in the
authoritative label records. The recorded Git `HEAD` alone is not that source:
the label jobs ran from a dirty worktree at commit
`a3c6a41115490992b4f0cebb40d7e67d857c9286`.

The historical layer implementation does implement the requested scientific
truth table:

- `FULL`: READ on, WRITE on;
- `READ_ONLY`: READ on, WRITE off;
- `WRITE_ONLY`: READ off, WRITE on;
- `IGNORE`: READ off, WRITE off.

Here READ means the retained text/control computation has direct visual K/V at
that layer. WRITE means the retained visual state is the output of that decoder
layer. Same-layer READ always uses the **pre-layer** visual state; it never uses
the visual state after the same layer's WRITE.

At the source and scientific-semantics level, this is an exact recovery and
would be classification A. At the stricter end-to-end execution-parity level,
the final audit classification is **C**: the requested recovered-source H100
replay was canceled before allocation at the user's instruction, so the audit
cannot claim that the recovered runtime restores the cached generated tokens.
This fail-closed classification does not mean the source bytes are missing or
that the action semantics are invalid. The distinction is reported in
`replay_parity_report.md`.

## 1. Authoritative execution contract

The full label contract is
`datasets/mcts_labels_4action/sequential_branching_v1/full/execution_contract_v1.json`.
Its central fields are:

| Field | Frozen value |
|---|---|
| Schema | `exact_sequential_four_action_execution_contract_v1` |
| Contract SHA-256 | `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb` |
| Recorded Git `HEAD` | `a3c6a41115490992b4f0cebb40d7e67d857c9286` |
| Commit subject/date | `polar`, 2026-08-24 23:09:07 +09:00 |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Model revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Resolved snapshot | `/data/research/models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Model dtype | BF16 |
| Attention implementation | `sdpa` |
| Torch | `2.6.0+cu124` |
| Transformers | `5.3.0` |
| Layer count | 28 |
| Seed | `20260825` |
| Full topology | 8 H100s, 16 workers, 2 workers/GPU |
| Smoke topology | 8 H100s, 8 workers, 1 worker/GPU |
| Full source manifest SHA-256 | `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7` |
| Frozen config SHA-256 | `d3d10f2453516be31dd75ec0434186e0241bab3288aa1afc7c00744afcfa6a9a` |

The contract builder enumerates the 16 source paths and hashes
`project_root / relative_path` immediately in the runner
(`tools/research_analysis/four_action/sequential_label_jobs.py`, historical
lines 12-29 and 53-98). Every resumed record was accepted only if its embedded
contract equaled the launch contract
(`experiments/run_sequential_four_action_label_conversion.py`, historical
lines 307-358). The code hashes therefore bind the actual project-root files,
not merely a later report's description of them.

The smoke contract has the same source hashes, config, model, and package
versions; only its manifest/topology and derived contract hash differ. Its
contract SHA-256 is
`386d22d0664aeba80e57abf3ed107b9399990ccce0ab7ccbd1f6cab2677d4cc1`.

## 2. Git and dirty-worktree provenance

The recorded `HEAD` is real, but the worktree was definitely dirty. At
`a3c6a411...`, the contract-bound state was:

| Status relative to `a3c6a411...` | Contract-bound paths |
|---|---|
| Exact match at `HEAD` | `binary_policy/actions.py`; executor `cache.py`, `generation.py`, `inputs.py`, `layers.py`, `masks.py`, `model.py`; `reference/dvr_qwen/eval_metrics.py` |
| Modified tracked file | `binary_policy/executor/__init__.py` |
| Modified tracked file | `label_regeneration/runtime.py` |
| Untracked at `HEAD` | `binary_policy/executor/four_action.py` |
| Untracked at `HEAD` | `experiments/run_sequential_four_action_label_conversion.py` |
| Untracked at `HEAD` | `tools/research_analysis/four_action/label_runtime.py` |
| Untracked at `HEAD` | `tools/research_analysis/four_action/sequential_label_conversion.py` |
| Untracked at `HEAD` | `tools/research_analysis/four_action/sequential_label_jobs.py` |
| Untracked at `HEAD` | `tools/research_analysis/four_action/targets.py` |

The frozen YAML config and the smoke/full wrapper scripts were also absent from
the `a3c6a411...` tree. Their use is evidenced by the execution contract,
experiment log, and subsequent commit, but the wrappers themselves were not in
the 16-file hash set.

The exact contract-bound dirty contents are recoverable. A complete unrelated
`git status` from 2026-08-25 was not logged, so this audit cannot truthfully
enumerate unrelated dirty files that may also have existed.

The two tracked modifications relative to `a3c6a411...` are recoverable in
detail:

- `binary_policy/executor/__init__.py` imported and exported the complete
  frozen four-action API (`FOUR_ACTIONS`, the fixed/full/local/route-conditioned
  forward functions, generation and scoring helpers, and
  `unified_target_four_action_layer`). Its exact historical bytes are identified
  by SHA-256 `ad6f7c...` below.
- `label_regeneration/runtime.py` imported `hashlib.sha256`, added
  `_open_frozen_image`, and changed image opening to allow an oversized image
  only after its bytes match `sample["image_content_sha256"]`; it then restores
  Pillow's global decompression guard. No generation or evaluator branch was
  changed by this modification.

The six files listed as untracked above are also recovered byte-for-byte, not
approximated from a conceptual description. The authoritative per-file hashes
are:

| Contract-bound path | Historical SHA-256 |
|---|---|
| `binary_policy/actions.py` | `892a7d844f239e6ca777791ef230293457f463c4094d0d9b827f073960899696` |
| `binary_policy/executor/__init__.py` | `ad6f7c5300a28a8d884e01fb75da0ea46996e008efa04f7091edaf093ea228f1` |
| `binary_policy/executor/cache.py` | `363faeb4e76932816f4e6784bf4acf803c5f3d46d72561587b0afe0482609b93` |
| `binary_policy/executor/four_action.py` | `e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868` |
| `binary_policy/executor/generation.py` | `8fa73ab035dc3655dae396c48674f5e2be052a9d41999b92d9fd075ebb7dfe96` |
| `binary_policy/executor/inputs.py` | `0e22848f56aaaec1c510958eee37e407d0dc51726dc60e5a69a3eea54091d465` |
| `binary_policy/executor/layers.py` | `4aba5c8d61f76925b758889b738e665b019d3324d5380808c625ee45de8f1b71` |
| `binary_policy/executor/masks.py` | `e0059b2e6755384cbb9cf019f4fdb21555d5a6d10c9ba6d8c9edc361fe99c8f8` |
| `binary_policy/executor/model.py` | `432d4898ef438db1150521d1dd330ed324c9ae96898934793ff151f80c82318a` |
| `experiments/run_sequential_four_action_label_conversion.py` | `82366689c00220690ed6022548c2312269effbf229ae11d796f7522217c33fc7` |
| `label_regeneration/runtime.py` | `1739b9d0f696ee3da3f601849f54c4d3f2077aa3cb72dea544f11b2ff796f201` |
| `reference/dvr_qwen/eval_metrics.py` | `c2ce4974fb03110841d55af61d26307a1b8cde31cd214906e2b5bafc5ed5373e` |
| `tools/research_analysis/four_action/label_runtime.py` | `785163efe775e7488ffccbe92dee41b7f84b464fb0d61e97f7240859132305b1` |
| `tools/research_analysis/four_action/sequential_label_conversion.py` | `7715c3e9e50a0d24ffc8001ec362e8bfd25febce33e849b4596234035a49cba7` |
| `tools/research_analysis/four_action/sequential_label_jobs.py` | `3b6ef3653fbbe723f23cbf4fda3bca9fa504354544e922c880491281eef0dc34` |
| `tools/research_analysis/four_action/targets.py` | `04bcf1a4d135e528bdf979f693b36b49b7fbf5075a364d537d34530175a3863d` |

### Exact reconstruction

Commit `838c8527e0d976c04893df5cb28af5d6376be65c` (`feat: add four-action
research pipelines`, 2026-08-28 10:27:37 +09:00) contains 14/16 contract files
byte-for-byte. The remaining two are recovered by reversing one later API-only
addition:

1. From `binary_policy/executor/four_action.py`, remove the `Callable` import
   and the complete `capture_online_four_action_route` definition.
2. From `binary_policy/executor/__init__.py`, remove the import and `__all__`
   entry for `capture_online_four_action_route`.

Those two reconstructed files hash to:

- `four_action.py`:
  `e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868`;
- executor `__init__.py`:
  `ad6f7c5300a28a8d884e01fb75da0ea46996e008efa04f7091edaf093ea228f1`.

All 16/16 reconstructed files and the frozen YAML match the authoritative
contract. No exact historical `four_action.py` blob exists in reachable or
unreachable Git objects; the reconstruction is nevertheless exact because the
resulting bytes match the recorded cryptographic hash. There is no stash,
alternate local branch, tag, linked worktree, or unreachable object containing
a competing implementation.

## 3. Entry point, launch command, and import resolution

The Python entry point was:

```text
experiments/run_sequential_four_action_label_conversion.py
```

The frozen full wrapper executed:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. \
OMP_NUM_THREADS=2 \
torchrun --standalone --nproc_per_node=16 \
  experiments/run_sequential_four_action_label_conversion.py \
  --mode full --resume
```

The smoke used the same command with `--nproc_per_node=8 --mode smoke
--resume`, then ran the command a second time to prove exact resume. The full
wrapper first audited smoke job `1611`, then full job `1612` began inference.
Later jobs reused the same scientific entry point and unchanged contract. The
experiment log records job 1611 completing in 95 seconds and job 1612 starting
on eight H100s. The original `sbatch` submit line itself is no longer available
from live Slurm accounting, so only the wrapper command and recorded resource
request—not an invented scheduler line—are reported as exact.

The entry point explicitly inserts its own project root at `sys.path[0]` when
run as a script (historical lines 17-18). The wrapper also sets `PYTHONPATH=.`
after changing directory to the root. Its relevant imports resolve to:

| Imported name | Historical project path |
|---|---|
| `BinaryQwen25VL`, executor API | `binary_policy/executor/__init__.py` |
| fixed-route executor | `binary_policy/executor/four_action.py` |
| full/compacted layer primitives | `binary_policy/executor/layers.py` |
| stream construction | `binary_policy/executor/inputs.py` |
| heterogeneous K/V cache | `binary_policy/executor/cache.py` |
| masks | `binary_policy/executor/masks.py` |
| preprocessing/determinism/evaluator wrapper | `label_regeneration/runtime.py` |
| sample runtime | `tools/research_analysis/four_action/label_runtime.py` |
| branching policy | `tools/research_analysis/four_action/sequential_label_conversion.py` |
| queue/contract builder | `tools/research_analysis/four_action/sequential_label_jobs.py` |
| target construction | `tools/research_analysis/four_action/targets.py` |
| evaluator | `reference/dvr_qwen/eval_metrics.py` |
| native decoder implementation | `.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |

The project `.venv` contains no project editable-install `.pth` or `.egg-link`;
its only `.pth` files are virtualenv/setuptools bootstrapping. Current import
inspection resolves the same packages from the repository root, and the
environment was created on 2026-08-22 before the 2026-08-25 label jobs. The
84-package lock includes Torch 2.6.0, Transformers 5.3.0, Pillow 11.1.0,
PyYAML 6.0.2, qwen-vl-utils 0.0.14, and MathRuler 0.1.0. The label contract
itself cryptographically records only the two core package versions, not the
complete installed-package set or GPU driver version.

There is no evidence of an alternate checkout or copied Python module winning
import precedence. More strongly, the direct-script `sys.path[0]` insertion
places this project root before site-packages, and the contract builder hashes
the same root. A preloaded `sitecustomize`-style substitution was not logged;
none exists in the preserved environment.

## 4. Exact action semantics inside one decoder layer

The complete-route label runtime calls `capture_four_action_route`, which loops
over the 28 actions and calls `four_action_layer` with `native_causal=False`
(`binary_policy/executor/four_action.py`, historical lines 386-438).

The terms “executes” and “is retained” must be separated for partial actions.
READ_ONLY computes a full visual output and discards it; WRITE_ONLY computes a
full text output and discards it.

| Property | FULL | READ_ONLY | WRITE_ONLY | IGNORE |
|---|---|---|---|---|
| Materialized decoder calls | one full-row | one full-row | one cache-free full-row, then one compact text/control-row | one compact text/control-row |
| Visual rows execute attention | yes | yes, but output is discarded | yes | no |
| Visual rows execute MLP | yes | yes, but output is discarded | yes | no |
| Visual residual/MLP update retained | yes | no; incoming visual state is carried | yes | no; incoming visual state is carried |
| Text/control rows execute | full-row result retained | full-row result retained | full-row result discarded; compact result retained | compact result retained |
| Retained text has direct visual K/V | yes | yes | no | no |
| Visual rows included in retained layer cache | yes | yes | no | no |
| Visual K/V version for same-layer READ | input-layer-normalized pre-layer visual state | input-layer-normalized pre-layer visual state | none in retained text path | none |
| Next-layer visual state | full layer output | exact incoming visual state | full layer output | exact incoming visual state |

### FULL

`visual_on_layer` scatters the incoming text/control and visual streams back to
their original sequence positions, constructs the full causal mask and 3-D
position embeddings, and calls the unmodified Qwen decoder layer once
(`layers.py`, historical lines 60-100). Both text and visual outputs are split
and retained. Prompt K/V contains all valid prompt rows.

### READ_ONLY

The exact same full-row call is made. Its text/control output and full-row K/V
cache are retained, so READ is on. The visual output—including its attention,
residual, MLP, and second residual—is discarded, and the incoming visual state
is carried (`four_action.py`, historical lines 192-205). WRITE is therefore off
at the state-transition boundary even though visual computation was
materialized.

### WRITE_ONLY

First, a cache-free full-row native call starts from the incoming text and
visual states. Its visual output is retained and its text output is discarded.
Second, a compacted text/control-only call starts from the **same incoming
states**; its text output and text-only K/V cache are retained
(`four_action.py`, historical lines 206-230). The retained text path never has
direct visual K/V. The full call is deliberately cache-free so discarded
visual K/V cannot leak into generation.

### IGNORE

Only compacted text/control rows enter the layer. Visual rows execute neither
attention nor MLP and are carried exactly to the next layer
(`layers.py`, historical lines 103-137). The retained K/V cache is text/control
only. This is the same primitive as old binary VISUAL_OFF.

### Token partition and masks

`mm_token_type_ids == 1` (image) or `== 2` (video) defines visual rows. Every
other valid row—including system/user/assistant delimiters and vision boundary
tokens—is a text/control row. Image placeholder embeddings are replaced with
vision-encoder features, then the two streams retain their original indices
(`inputs.py`, historical lines 49-55, 77-100, and 131-163).

Full-row calls use an explicit additive causal mask over the full valid prompt.
Compacted calls use the original full-sequence indices, allowing a compact key
only when its original index is not later than the query. Masked values are
`torch.finfo(dtype).min` (`masks.py`, historical lines 8-26). Thus compaction
removes visual keys/values without changing text/control causal ordering.

## 5. Same-layer READ/WRITE ordering

The historical FULL implementation is not “WRITE visual rows, then let text
read the updated rows.” Qwen's decoder layer performs:

1. input RMSNorm of the shared pre-layer hidden states;
2. Q, K, and V projections for all rows from that normalized pre-layer tensor;
3. causal self-attention;
4. attention residual update;
5. post-attention RMSNorm, MLP, and second residual update.

The installed Transformers 5.3 implementation shows Q/K/V construction before
the attention output (modeling source lines 704-754) and both residuals/MLP
after it (lines 808-837). Therefore FULL is a simultaneous standard transformer
operation. Of the question's proposed choices, it is closest to A only in the
precise sense that text READ uses pre-layer visual K/V and the updated visual
state becomes available after the layer. It is not two separately ordered READ
then WRITE subroutines.

- **FULL:** text/control Q attends to permitted K/V projected from the
  pre-layer visual rows; both text and visual post-layer states are retained.
- **READ_ONLY:** identical full call and identical pre-layer visual K/V for
  text READ; the post-layer visual output is discarded.
- **WRITE_ONLY:** a full call creates the visual update from the shared
  pre-layer state, while an independent compact call creates retained text from
  that same pre-layer state. There is no retained same-layer READ.
- **IGNORE:** only the compact call exists; there is neither READ nor WRITE.

## 6. K/V cache and generation contract

One `BinaryRouteCache` is created per complete prompt route. Each layer stores a
different prompt length when necessary:

- FULL and READ_ONLY store full prompt K/V at that layer;
- WRITE_ONLY and IGNORE store compact text/control-only K/V;
- cache entries append on the sequence axis and are not shared between route
  evaluations (`cache.py`, historical lines 8-32).

After prompt execution, generation clones the complete heterogeneous cache.
Each generated token is a single text row passed through all 28 native decoder
layers, with each layer attending to its own prompt cache plus prior generated
tokens (`four_action.py`, historical lines 649-676 and 705-743). Greedy decoding
uses `argmax` after repetition penalty. No sampling path is invoked.

| Generation/evaluation item | Historical value |
|---|---|
| Model and processor source | same frozen snapshot/revision `cc594898...`; `local_files_only=True` |
| Processor | `AutoProcessor`, `use_fast=False` |
| Chat message | one user turn containing image first, then the sample's frozen `prompt` |
| Chat template | `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`; snapshot `chat_template.json` SHA-256 `ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96` |
| Image loading | provenance-checked path, Pillow open, RGB conversion |
| Image preprocessing | processor-native defaults; no custom visual-token cap; padding on; `return_mm_token_type_ids=True` |
| Processor config | min pixels 3,136; max pixels 12,845,056; patch 14; temporal patch 2; merge 2 |
| Decoding | deterministic greedy manual `argmax`; no sampling |
| `max_new_tokens` | sample field: 16 for GQA/TextVQA/ChartQA, 128 for WeMath2.0 Standard, 96 for WeMath2.0 Pro |
| EOS IDs | generation-config default `[151645, 151643]` |
| Repetition penalty | generation-config default `1.05` |
| Temperature/top-p | not used by manual greedy generation |
| Prompt/decode cache | on; heterogeneous per-layer cache cloned before each generation |
| Attention/dtype | SDPA, BF16 |
| Model state | `.eval()`, gradients disabled |
| Determinism | `torch.use_deterministic_algorithms(True)`, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed `20260825 + rank` |
| Decode text | skip special tokens, no cleanup, strip surrounding whitespace |
| Correctness timeout | 5 seconds; timeout scores zero |

The model artifact configuration hashes currently present at the frozen
snapshot include:

- `config.json`: `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`;
- `generation_config.json`:
  `0a3aea82869fe29f20dc95ccf3e2bcff380eca1f5ad6447a4a4b37110b08e43e`;
- `preprocessor_config.json`:
  `f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0`;
- `tokenizer_config.json`:
  `4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56`.

These artifact hashes were not stored in the original label contract; the
contract stored the resolved snapshot and revision. They are reported as a
current verification of the same persistent snapshot, not misrepresented as
historically logged hashes.

### Evaluator and normalization

The source record provides `metric_name`, answer/reference set, and threshold.
The historical scorer dispatches:

- GQA: case/punctuation-insensitive exact match, threshold 1.0;
- ChartQA: relaxed numeric/string accuracy, threshold 1.0;
- TextVQA: EvalAI-style consensus normalization, threshold 0.5;
- WeMath2.0 Pro: MathRuler 0.1.0, threshold 1.0.

The 3,095 WeMath2.0 Standard rows declare
`wemath20_mathruler_accuracy`, but the historical `score_prediction` dispatcher
has no branch for that exact string. It therefore falls through to the default
case/punctuation-insensitive exact-match evaluator. This is a verified
historical evaluator behavior, not a repair made by this audit. It does not
change the READ/WRITE truth table, but any use of WeMath2.0 Standard labels must
retain or explicitly revise that evaluator contract rather than assume
MathRuler was used.

## 7. Scientific-contract decision

The historical state transition matches the intended four actions. In
particular:

- WRITE off means the visual state crossing the layer boundary is unchanged,
  even though READ_ONLY materializes and discards a visual computation;
- READ off means the retained text/control result and retained prompt cache
  contain no visual K/V, even though WRITE_ONLY materializes and discards a
  full-row text computation;
- READ uses pre-layer visual K/V, consistent with the native transformer layer;
- no action supplies post-WRITE visual K/V to text in that same layer.

One qualification is essential: complete-route label generation selected among
one-call and two-call branches in `four_action_layer`. It did **not** use
`unified_target_four_action_layer`, which materializes full then compact calls
for all four actions. The latter was the stricter numerical-control machinery
for local factorial analysis. Thus the historical labels are scientifically
valid as discrete READ/WRITE routes, but their four actions are not a
numerically unified factorial comparison path.

The original eight-sample label smoke also compared the FULL/IGNORE mapping
against the old binary executor for one source route per sample: generated
token IDs, decoded answers, and correctness matched 8/8. This is supporting
semantic evidence, not a substitute for the recovered-source replay reported
separately.
