# Reproducing Binary Visual-Route MCTS on Another Dataset

## 1. Purpose and non-negotiable semantics

This runbook is written for another execution agent. Follow the order below.
Do not begin the full search until the manifest, evaluator, execution contract,
and exact ALL-ON parity smoke have passed.

The search produces route labels for a frozen Qwen2.5-VL-7B-Instruct model.
A route is a complete 28-bit mask:

```text
m = (m_0, ..., m_27),  m_l in {0, 1}
```

- `1` / ON: text/control and visual rows execute the native decoder layer.
- `0` / OFF: text/control rows execute the layer; visual hidden-state rows
  bypass that layer unchanged.

Every completed mask is evaluated by actual route-conditioned deterministic
greedy generation and the benchmark's frozen answer metric. The result is not
estimated from logits, a proxy model, or a predictor.

The search is not POLAR MCTS. It uses a project/DVR-style transposition graph:

- root mask: ALL-ON;
- additional anchor: ALL-OFF;
- a search action chooses `(any undecided layer, ON/OFF)`;
- no early-to-late ordering;
- no contiguous-segment restriction;
- random completion of undecided bits during rollout;
- binary correctness reward;
- all unique positive and negative evaluated masks are retained.

Do not add a custom `max_image_tokens` cap. Do not silently alter prompts,
answer normalization, correctness thresholds, generation length, layer count,
or model revision to make parity or search easier.

## 2. What is in the bundle

```text
binary_policy/actions.py                 mask validation
binary_policy/executor/                  verified Qwen binary executor
label_regeneration/mcts.py               unrestricted graph MCTS
label_regeneration/runtime.py            native processor, generation, scoring
reference/dvr_qwen/eval_metrics.py       bundled benchmark metrics
scripts/run_label_regeneration.py        smoke and sharded MCTS entry point
scripts/validate_manifest.py             static manifest/image audit
scripts/make_smoke_manifest.py           outcome-blind smoke selection
scripts/freeze_contract.py               immutable runtime/search contract
scripts/audit_cache.py                   terminal-cache completion audit
scripts/download_model.py                pinned snapshot downloader
examples/build_manifest_from_jsonl.py    simple dataset adapter example
examples/slurm_mcts.sbatch               Slurm launch template
tests/test_bundle.py                     CPU-only structural tests
docs/MODEL_AND_LABEL_GENERATION.md        original DVR/Qwen semantics reference
```

The bundle does not include the model or dataset. It also cannot know a new
benchmark's official evaluator. Adapting and validating that evaluator is the
one dataset-specific engineering step that must be completed before freezing
the contract.

## 3. Verify the transferred files

After copying the directory to the target server:

```bash
cd /path/to/binary_visual_mcts_reproduction_v1
sha256sum --check BUNDLE_SHA256SUMS
```

All entries must report `OK`. Preserve the entire folder with the raw output so
the exact executor and evaluator used for a cache remain recoverable.

If files must be intentionally changed for a new benchmark, make the smallest
change, test it, then regenerate the bundle inventory:

```bash
.venv/bin/python scripts/write_bundle_checksums.py
```

This intentionally invalidates the original transfer inventory. Record the
new `BUNDLE_SHA256SUMS` and `bundle_manifest.json` with the run.

## 4. Create the exact environment

The reference environment uses Python 3.12 and a project-local `uv` virtual
environment. Do not install into system Python or a global Conda environment.

```bash
cd /path/to/binary_visual_mcts_reproduction_v1
uv venv --python 3.12 .venv
UV_LINK_MODE=copy uv pip install --python .venv/bin/python -r requirements.txt
```

Run the CPU-only structural tests. These tests do not load Qwen:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected result: all tests pass. Also record installed versions:

```bash
.venv/bin/python - <<'PY'
import PIL, accelerate, torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("pillow", PIL.__version__)
PY
```

The key versions are `torch==2.6.0` and `transformers==5.3.0`. A different
CUDA driver may be usable, but deterministic BF16 tokens must still pass the
smoke on the actual target GPU type.

## 5. Obtain the pinned model snapshot

The required Hugging Face snapshot is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Download it once on the target server if it is not already present:

```bash
.venv/bin/python scripts/download_model.py --cache-dir /path/to/hf_model_cache
```

The script prints the local snapshot path. Set that exact directory as
`MODEL_PATH`. The runtime loads with `local_files_only=True`, so it will not
silently retrieve a different revision during search.

```bash
export MODEL_PATH=/absolute/path/to/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
test -f "$MODEL_PATH/config.json"
```

Do not substitute another Qwen size, instruction variant, snapshot, tokenizer,
processor, or chat template while claiming reproduction of this search.

## 6. Build the new dataset manifest

### 6.1 Required JSONL schema

Create one JSON object per image-query pair. Every record requires:

| Field | Meaning |
|---|---|
| `uid` | globally unique, stable record identifier |
| `sample_id` | benchmark's question/annotation identifier |
| `benchmark` | fixed benchmark name |
| `question` | literal human-readable question |
| `prompt` | exact text sent after the image content |
| `answer` | primary reference answer as a nonempty string |
| `all_answer_norms` | optional list of accepted references, otherwise `null` |
| `metric_name` | name dispatched by the bundled evaluator |
| `correctness_threshold` | score threshold in `[0,1]` defining a valid route |
| `max_new_tokens` | deterministic generation bound for this record |
| `image_group_id` | stable image identity, shared by questions on one image |
| `local_image_path` | absolute readable path on the target server |
| `max_image_tokens` | must be `null` or absent |

Recommended provenance fields are `image_content_sha256`, `source_file`,
`source_index`, `source_row_sha256`, and `historical_all_on_status`. Historical
correctness is metadata only. The new run always recomputes authoritative
ALL-ON correctness.

See `examples/dataset_manifest.example.jsonl`. For a simple source JSONL, the
example adapter can be used as a starting point:

```bash
.venv/bin/python examples/build_manifest_from_jsonl.py \
  --input /absolute/path/raw_dataset.jsonl \
  --output manifests/new_dataset_v1.jsonl \
  --image-root /absolute/path/images \
  --benchmark new_dataset \
  --metric-name exact_match_ignore_case_punctuation \
  --correctness-threshold 1.0 \
  --max-new-tokens 16
```

Do not use that generic adapter when the official task requires a special
prompt, multiple-reference aggregation, answer extraction, or image lookup.
Write a small dataset-specific adapter instead and freeze its output.

### 6.2 Scoring contracts already bundled

`reference/dvr_qwen/eval_metrics.py` supports these evaluator families:

| Metric name/family | Intended use |
|---|---|
| `exact_match_ignore_case_punctuation` | GQA-style short exact answers |
| name containing `textvqa` or `consensus` | TextVQA EvalAI normalization/consensus |
| name containing `relaxed` | ChartQA relaxed numeric accuracy |
| name containing `anls` | DocVQA-style ANLS |
| `wemath2pro_mathruler_accuracy` | We-Math2.0-Pro MathRuler grading |
| `dynamath_float_accuracy` | DynaMath numeric answer |
| multiple-choice metric names | standalone answer-letter grading |
| `reasoning_strict_accuracy` | conservative short reasoning answer matching |

The dispatcher historically falls back to punctuation-insensitive exact match
for unknown names. Do not rely on that fallback. For a new benchmark:

1. identify its official prompt, normalization, answer extraction, score, and
   correctness threshold;
2. add a named metric branch if no existing one matches exactly;
3. add evaluator fixtures with known correct and incorrect predictions;
4. run those fixtures before freezing the contract;
5. record the evaluator source hash in the frozen contract.

The valid-route label is `score >= correctness_threshold`; MCTS receives only
that binary reward. The raw score is still saved for audit, but it is not used
as a continuous UCB reward.

### 6.3 Static manifest audit

Run the audit before any GPU work:

```bash
mkdir -p manifests outputs/new_dataset_v1
.venv/bin/python scripts/validate_manifest.py \
  --manifest manifests/new_dataset_v1.jsonl \
  --expected-count EXPECTED_RECORD_COUNT \
  --verify-image-hash \
  --report outputs/new_dataset_v1/manifest_audit_v1.json
```

It verifies required fields, unique UIDs, image readability, optional image
hashes, generation bounds, thresholds, accepted-answer types, and the absence
of a custom visual-token cap. Resolve every error before continuing.

## 7. Freeze the execution contract

Freeze after the manifest and evaluator are final, and before smoke outcomes
are inspected:

```bash
export MANIFEST=/absolute/path/to/manifests/new_dataset_v1.jsonl
export DATASET_VERSION=new_dataset_native_qwen_binary_mcts_v1
export OUTPUT_ROOT=/absolute/path/to/outputs/new_dataset_v1

.venv/bin/python scripts/freeze_contract.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL_PATH" \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --dataset-version "$DATASET_VERSION" \
  --seed 20260810 \
  --output "$OUTPUT_ROOT/frozen_execution_contract.json"
```

The command prints the scientific contract hash. Load it without editing the
contract file:

```bash
export CONTRACT_SHA256=$(.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["contract_sha256"])' \
  "$OUTPUT_ROOT/frozen_execution_contract.json")
```

The contract binds the full manifest hash, model metadata, exact revision,
packages, executor, MCTS, evaluator, runner sources, prompt/image policy,
search constants, and seed. Every terminal record stores this hash. If a bound
file or manifest changes, freeze a new contract and use a new output root.

## 8. Create and run the minimal smoke

Select five records per benchmark, outcome-blind. Up to the first three
benchmarks receive one representative mixed-mask check on one selected record
each (additional checks are filled deterministically if requested):

```bash
.venv/bin/python scripts/make_smoke_manifest.py \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_ROOT/smoke_manifest_v1.jsonl" \
  --per-benchmark 5 \
  --mixed-records 3 \
  --seed 20260810
```

For one benchmark this produces five records; for three benchmarks it produces
15. Run smoke on exactly one scheduled GPU:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/run_label_regeneration.py \
  --mode smoke \
  --manifest "$OUTPUT_ROOT/smoke_manifest_v1.jsonl" \
  --model-path "$MODEL_PATH" \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --contract-sha256 "$CONTRACT_SHA256" \
  --dataset-version "$DATASET_VERSION" \
  --output-root "$OUTPUT_ROOT" \
  --required-smoke-count 5 \
  --seed 20260810
```

Set `--required-smoke-count` to the exact generated smoke size. The smoke must
show:

- exact generated-token equality between native Qwen and binary ALL-ON on
  every record;
- exact repeated generated tokens and score for every mixed-mask check;
- `"passed": true` in `smoke_report_v1.json`.

This is a hard gate. If it fails, stop before MCTS and diagnose the concrete
executor, processor, prompt, model, environment, or evaluator mismatch. Do not
weaken token parity, remove failed smoke records, reduce image resolution, or
introduce a visual-token cap.

## 9. Exact MCTS algorithm and budgets

For each sample, the runner:

1. preprocesses the image and prompt once with native Qwen defaults;
2. computes vision features once and builds reusable binary inputs;
3. evaluates ALL-ON and ALL-OFF;
4. uses current ALL-ON correctness to choose the budget;
5. runs unordered graph MCTS over complete 28-bit masks;
6. calls real greedy generation for every newly encountered complete mask;
7. caches repeat mask evaluations within the sample;
8. atomically writes every evaluated positive and negative route.

Frozen budgets:

```text
current ALL-ON correct: 200 simulations
current ALL-ON wrong:   400 simulations
extension to 600:       only if no correct route exists after 400
```

Frozen MCTS constants:

```text
exploration_constant   = 1.8
length_penalty         = 3.0
random_probability     = 0.1
rollout_off_probability= 0.5
root                   = 28 ON bits, all undecided
transposition table    = enabled
stop on first success  = false
```

At selection time, UCB is:

```text
mean binary reward
+ 1.8 * sqrt(log(parent visits) / child visits)
- 3.0 * fraction of visual-ON layers in the child partial mask
```

At expansion, an undecided layer and action are sampled from all remaining
`(layer, 0/1)` pairs. Rollout fills every remaining bit independently with
OFF probability 0.5. A transposition key contains both the partial mask and the
decided-bit vector. The per-sample seed is:

```text
base seed + integer(first 8 hex characters of SHA-256(uid))
```

Consequently, mask search is stable across worker counts and resume layouts,
assuming the same contract and deterministic model outputs.

## 10. Launch the full search

### 10.1 One process per GPU

The implementation uses independent data parallelism: one full Qwen instance
and one Python process per visible GPU. Ranks receive records by
`manifest_index % world_size`. The model is not tensor-parallelized and the
workers do not exchange model tensors.

For `N` GPUs on one allocated node:

```bash
export N=8
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

.venv/bin/torchrun \
  --standalone \
  --nproc_per_node="$N" \
  scripts/run_label_regeneration.py \
  --mode mcts \
  --manifest "$MANIFEST" \
  --model-path "$MODEL_PATH" \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --contract-sha256 "$CONTRACT_SHA256" \
  --dataset-version "$DATASET_VERSION" \
  --output-root "$OUTPUT_ROOT" \
  --seed 20260810
```

Use `.venv/bin/torchrun`, not an unqualified global `torchrun`. Ensure the
scheduler exposes local CUDA devices as `0..N-1`. Each GPU must have enough
memory for a full BF16 7B model plus the sample's native visual token sequence.

Because the workers are process-independent, GPU P2P is not part of the search
algorithm. Server-specific NCCL/P2P workarounds should be added only if the
local launcher requires them, and recorded in operational metadata; they must
not change the model, input, route, or evaluator contract.

### 10.2 Slurm

`examples/slurm_mcts.sbatch` is an eight-GPU template. Edit its `#SBATCH`
resource lines for the target cluster. Export required paths before submission:

```bash
export BUNDLE_ROOT=/absolute/path/to/binary_visual_mcts_reproduction_v1
export MODEL_PATH=/absolute/path/to/the/pinned/snapshot
export MANIFEST=/absolute/path/to/manifests/new_dataset_v1.jsonl
export CONTRACT_SHA256=THE_FROZEN_HASH
export OUTPUT_ROOT=/absolute/path/to/outputs/new_dataset_v1
export DATASET_VERSION=new_dataset_native_qwen_binary_mcts_v1
sbatch examples/slurm_mcts.sbatch
```

CPU threads per process should normally be
`SLURM_CPUS_PER_TASK / number_of_GPU_workers`. Do not run the full search on a
login node.

## 11. Resume and cancellation behavior

Each terminal sample is first written to a PID-specific temporary file and
then atomically renamed. A record is considered resumable only when:

- its sample UID matches;
- its contract hash matches;
- `candidate_executions` is a list;
- completed simulations equal requested simulations.

On restart, the runner scans all `raw_route_cache/shard_*_of_*` layouts. This
allows resuming with a different number of workers without deleting completed
records. It is safe to reuse a record only under the same frozen contract.

Before restarting after cancellation, check for errors and zero-byte files.
Never fabricate an empty terminal JSON for unfinished work. Unfinished
in-memory searches simply restart for that sample.

Use the identical launch command and output root. Do not manually move sample
files between shards. The runner discovers valid earlier layouts itself.

## 12. Output layout and label contents

```text
OUTPUT_ROOT/
  frozen_execution_contract.json
  smoke_manifest_v1.jsonl
  smoke_report_v1.json
  raw_route_cache/
    shard_000_of_NNN/
      summary.json
      samples/*.json
      errors/*.json
```

Every successful sample JSON contains:

- immutable sample, prompt, image, metric, and runtime metadata;
- recomputed current ALL-ON prediction, raw score, and correctness;
- ALL-OFF result;
- all unique route executions from anchors and MCTS;
- for each route: 28 bits, generated token IDs, decoded answer, raw score,
  thresholded validity, visual ON/OFF counts, transition count, Hamming
  distance from ALL-ON, and actual text/visual/full prompt token counts;
- all successful masks;
- the minimum-visual-ON successful mask, with lexicographic tie break;
- MCTS simulations, graph nodes, UCB statistics, transposition hits, seed,
  extension reason, and search settings.

Both positive and negative evaluated masks are kept. There is no requirement
that every sample yield 20 positive routes, and the raw cache is never
truncated to 32 routes. A zero-positive record is valid evidence and must not
be discarded.

## 13. Monitor and audit completion

Shard summaries show per-rank completed, skipped, error, elapsed, and last UID.
For a quick operational count:

```bash
find "$OUTPUT_ROOT/raw_route_cache" -path '*/samples/*.json' -type f | wc -l
find "$OUTPUT_ROOT/raw_route_cache" -path '*/errors/*.json' -type f | wc -l
```

After the job ends, run the contract-aware audit:

```bash
.venv/bin/python scripts/audit_cache.py \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --report "$OUTPUT_ROOT/cache_completion_audit_v1.json"
```

Completion passes only when every manifest UID has exactly one valid terminal
record, no unexpected or invalid records exist, and no error JSON remains.
Do not begin post-hoc label analysis or predictor training from a partial or
contract-mixed cache.

## 14. Reproduction checklist

Archive these together:

- this entire bundle and its SHA-256 inventory;
- exact full manifest and checksum;
- dataset adapter and source annotation provenance;
- frozen execution contract and checksum;
- pinned snapshot ID plus model metadata hashes;
- benchmark evaluator fixtures and result;
- smoke manifest/report/checksums;
- scheduler script, submitted resources, node/GPU type, driver and CUDA info;
- raw route cache, shard summaries, error directory, and completion audit;
- stdout/stderr logs.

A cache is scientifically comparable only when the execution contract matches.
Generated-answer agreement is not enough to substitute a different executor
contract after the fact.

## 15. Troubleshooting and hard stops

### ALL-ON parity fails

Stop. Confirm, in order: pinned snapshot, Transformers 5.3.0, prompt bytes,
chat template, processor defaults, image bytes, BF16/SDPA, generation length,
and executor source hashes. Do not continue with approximate parity.

### Repeated mixed routes differ

Stop. Confirm `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms,
greedy decoding, identical GPU/runtime, and unchanged sample input. Do not
average or vote nondeterministic outputs into a label.

### CUDA out of memory

The runner treats OOM as fatal. Record image/token geometry if available and
request an explicit contract amendment. Do not silently resize images or add a
visual-token cap.

### Unknown or disputed benchmark metric

Stop before search. Reproduce the benchmark's official evaluator on known
fixtures and implement a named deterministic adapter. A route cache with an
incorrect correctness contract cannot be repaired from decoded outputs unless
all required official annotations were preserved and a rescoring amendment is
explicitly documented.

### Error JSON for individual records

Inspect the exact traceback. Fix only the concrete technical issue, preserve
the manifest population, freeze a new contract if bound code changes, and
rerun. Do not silently remove records based on route outcomes.

### Search is slow

Use more independent GPUs, not tensor-parallel execution or reduced search
budgets. Native visual-token count and answer generation length dominate route
cost. One process per GPU gives near-linear sample throughput when storage and
CPU preprocessing keep up. Search budgets are per sample and must not be
reduced to improve ETA while claiming the same reproduction.

## 16. Boundary after extraction

The raw MCTS cache is label-generation evidence only. Image-group-disjoint
train/validation/test splits, valid-set supervision, route ranking data, and
POLAR-style segment views may be derived later. This bundle does not authorize
or implement router/predictor training, and it does not turn unrestricted
binary MCTS into POLAR's tri-state search.
