# Binary Visual-Token Policy with POLAR-Style Supervision — Plan v1

Status: active implementation detour; pre-training code is prepared; no
training or model-scale validation has been run.

This plan is a new supervised policy direction. It does not reopen or revise
the closed v2–v4 causal claims. All earlier artifacts remain preserved as
historical discovery and negative-result evidence.

## 1. Scientific and engineering objective

Test whether a lightweight question-conditioned predictor can select a
28-layer binary visual-token route for frozen Qwen2.5-VL-7B-Instruct that:

1. preserves benchmark correctness relative to the dense all-visual-ON model;
2. uses materially less local visual-row computation; and
3. generalizes to image-disjoint held-out records rather than memorizing MCTS
   samples.

The action at decoder layer `l` is exactly:

- `VISUAL_ON=1`: scatter text/control and visual rows into native sequence
  order and execute the unmodified Qwen decoder layer on all rows;
- `VISUAL_OFF=0`: remove visual rows from that layer, execute the same native
  layer on compacted text/control rows with the corresponding causal mask and
  MRoPE positions, and carry the visual rows unchanged to the next layer.

OFF is layer-local. Visual rows bypass that layer and may re-enter at a later
ON layer. They are not permanently deleted from the sequence.

## 2. Fixed scope and non-goals

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Base MLLM, vision encoder, and POLAR input embedding model remain frozen.
- Policy output: one 28-bit `visual_on_mask`.
- Existing MCTS v2 labels only; do not generate new MCTS labels.
- Benchmarks: ChartQA, DocVQA, GQA, and TextVQA.
- Primary deployable evaluation is a single top-1 predicted route. Top-k is an
  offline/secondary diagnostic, not free inference.
- No READ/WRITE tri-state intervention, skip/execute/repeat decoder program,
  model fine-tuning, causal harmfulness claim, or acceleration claim from
  route counts alone.
- Do not infer broad benchmark prevalence from the curated easy/hard label
  pool.

## 3. Approved sources and provenance

| Source | Role | SHA-256 / status |
|---|---|---|
| `reference/polar/2606.06574v1.pdf` | POLAR paper | `759543e28cf7a2b9b1608d5335c66293aca374c1e777d440ca5316ef80faeadf` |
| `reference/polar/PoLar/polar/` | Predictor, multi-program data, training and decoding reference | inspected; `model.py` `802ce841...`, `data.py` `3e3fb0fe...` |
| `reference/binary_action_qwen/core/` | Binary visual-token executor reference | inspected; `binary_layer.py` `2f6e62fb...`, `binary_generate.py` `a089872d...` |
| `/home/hyemin/data/dataset/dynamic_mllm/mcts_v2` | Existing labels, user-authorized read-only source | final audit passed; audit hash `725f4755...` |

The reference directories are not modified. Project-owned adaptations live in
`binary_policy/`.

## 4. Existing label inventory

The source contains 4,000 records, four shards, 200 MCTS simulations per
record, and normally 202 unique masks including anchors. The final source audit
reports no missing samples and all eight raw errors recovered.

| Cell | Samples | Samples with at least one successful mask | Coverage | Mean sparsest successful ON layers |
|---|---:|---:|---:|---:|
| ChartQA easy | 500 | 500 | 1.000 | 9.828 |
| ChartQA hard | 500 | 352 | 0.704 | 12.276 |
| DocVQA easy | 500 | 500 | 1.000 | 10.554 |
| DocVQA hard | 500 | 421 | 0.842 | 11.884 |
| GQA easy | 500 | 500 | 1.000 | 5.168 |
| GQA hard | 500 | 300 | 0.600 | 9.420 |
| TextVQA easy | 500 | 500 | 1.000 | 9.884 |
| TextVQA hard | 500 | 335 | 0.670 | 11.746 |

Thus 3,408 records have at least one positive route and 592 do not. Records
without a successful MCTS mask are retained for held-out online evaluation but
cannot supply positive-program supervision under the POLAR rule.

The label runtime was PyTorch `2.6.0+cu124`, Transformers `5.3.0`, SDPA, slow
processor, repetition penalty `1.05`, and the pinned model snapshot. The
project environment was migrated to Transformers `5.3.0` by CPU-only Slurm job
`99717`; reproduction remains a validity gate rather than an assumed
consequence of version alignment.

## 5. POLAR-to-binary adaptation

| POLAR component | Original | Binary adaptation |
|---|---|---|
| Action | segment-level `skip/execute/repeat` | layer-level `VISUAL_OFF/VISUAL_ON` |
| Program object | executed decoder-layer path | 28-bit visual-row presence mask |
| Input encoder | frozen Qwen3-Embedding question tokens | retained exactly for the primary question-only predictor |
| Layer conditioning | learned layer queries cross-attend to question tokens | retained |
| Cross-layer model | Transformer encoder over layer representations | retained |
| Output head | boundary logits plus 3-way operation head | direct 28 binary logits |
| Multiple labels | each valid MCTS program is supervision | all deduplicated valid masks form the supervision set |
| Search | beam over segment operations | exact top-k beam under binary mask logits |

Repeat has no valid binary-token analogue and is removed. Canonical maximal
run-length encoding is deterministic and implemented as a structured baseline,
but it is not the primary representation because the actual executor exposes
one binary decision per layer and the observed labels include non-contiguous
masks.

The primary predictor uses question tokens only, matching POLAR's pre-action
input contract. Image-conditioned predictor features are not silently added;
they would be a separate approved model change.

## 6. Multi-valid-mask objective

For logits `z_i` and a binary mask `m`, the direct model defines

\[
p(m\mid x)=\prod_l \sigma(z_l)^{m_l}(1-\sigma(z_l))^{1-m_l}.
\]

For the deduplicated observed valid set `V_i`, train with

\[
\mathcal L_i=-\log\sum_{m\in V_i}w_{i,m}p(m\mid x).
\]

Weights sum to one within each sample. A valid all-ON route is assigned weight
`0.25` when a shorter valid route exists; remaining weights are normalized.
This prevents samples with many discovered routes from dominating and avoids
forcing contradictory masks into a single averaged hard target. The
POLAR-faithful duplicated-path BCE remains implemented as an audit baseline.

Because MCTS observed only about 202 masks out of `2^28`, the valid set is
incomplete. Offline mask coverage is therefore diagnostic; fresh execution
with the official evaluator is the scientific endpoint.

## 7. Pre-training stages and gates

### Stage BP-0 — label and representation audit

Run `tools/audit_binary_polar_labels.py` as a CPU-only Slurm job over all 4,000
records. Verify schema, 28-bit masks, reward threshold, duplicate removal,
image identifiers, valid-route counts, ON-count/run-transition distributions,
and empirical top-1/5/10 valid-set coverage for both direct-bit and canonical
run-length factorizations.

Gate:

- all 4,000 records parse with no unresolved invalidity;
- the eight cell counts match the source audit;
- direct top-5 oracle-marginal coverage is no more than 0.02 below canonical
  run-length coverage in the macro average and no more than 0.05 below it in
  any cell;
- image-group splitting is feasible without cross-split image hashes.

If the representation gate fails, stop and amend the predictor representation;
do not train the direct head merely because it is implemented.

### Stage BP-1 — executor validity and label reproduction

Freeze 16 technical fixtures, two per benchmark/difficulty cell, without using
policy predictions. Run `experiments/binary_executor_preflight.py` through the
GPU scheduler.

Required checks:

1. split/scatter identity is exact;
2. all-ON reproduces the native dense logits within `5e-3` max absolute BF16
   tolerance and the exact greedy token IDs;
3. OFF equals a native decoder-layer call on the same compacted text/control
   rows and carries visual rows exactly;
4. repeated arbitrary-mask execution is deterministic;
5. cache lengths equal full prompt length at ON layers and compacted text
   length at OFF layers;
6. all-ON, all-OFF, and cached best-mask generated token IDs reproduce the
   existing label record under the pinned generation settings;
7. prompt construction, visual token positions, MRoPE, and processor template
   match label provenance.

Any cached-label mismatch blocks training. The project runtime now matches the
label's Transformers `5.3.0` version, so a remaining mismatch must be recorded
as an executor, prompt/template, kernel, or still-unknown provenance issue.
Labels must not be silently accepted under a non-reproducing executor.

### Stage BP-2 — compact manifest and split freeze

After BP-0 and BP-1 pass, run `tools/prepare_binary_polar_data.py` as a CPU-only
Slurm job and write derived data under
`/data/dataset/dynamic_mllm/binary_polar_v1/`.

Rules:

- valid route means existing reward greater than or equal to the record's
  frozen correctness threshold;
- deduplicate masks exactly;
- cap at 50 masks per sample only after the full representation audit;
- always retain the sparsest valid mask and the valid all-ON anchor;
- fill remaining cap slots by deterministic hash with seed `20260809`;
- group by image-content SHA-256, otherwise source asset ID;
- deterministic 75%/12.5%/12.5% train/validation/test hash split;
- require at least 40 records in every benchmark/difficulty/split cell and
  zero image overlap before freezing;
- retain no-success records in the manifest as evaluation-only records;
- checksum the manifest and audit.

### Stage BP-3 — bounded predictor training (not executed now)

Train only the lightweight predictor and projection layers. The
Qwen3-Embedding-0.6B encoder and Qwen2.5-VL base remain frozen. Initial model:
`d_model=256`, four heads, two cross-layer blocks. Use at most six validation
runs from POLAR's published learning-rate/batch ranges; select by validation
set-NLL followed by top-1/top-5 route coverage. Do not tune on the test split.

Training must use the Slurm GPU scheduler. No MCTS search or base-model forward
pass is part of predictor training.

### Stage BP-4 — frozen held-out online evaluation (not executed now)

Execute the top-1 predicted mask fresh on every test record, including records
with no successful cached mask. Compare with all-ON and all-OFF using the exact
official benchmark evaluator and pinned generation settings. Report separately
for all eight benchmark/difficulty cells and macro-average cells.

Primary joint success criterion:

- image-grouped 95% CI for macro correctness difference versus all-ON has a
  lower bound above `-0.01`;
- mean measured local visual FLOPs are at least 20% below all-ON;
- no benchmark easy cell regresses by more than 0.02 absolute accuracy;
- results are achieved by the top-1 route, not only top-k rescue.

Also report hard-cell improvements, ON-layer count, route frequencies,
per-layer actions, actual local FLOPs, wall-clock latency measured separately,
and MCTS-oracle upper bounds as non-deployable context.

## 8. Stop and pivot criteria

Stop before training if label parsing, direct-factorization coverage, image
grouping, native all-ON parity, compacted-text OFF identity, deterministic
generation, or cached-label reproduction fails.

Stop after evaluation if accuracy preservation and compute reduction do not
both pass, if gains occur only on the inspected/cached masks, or if top-k search
is required to hide a weak top-1 policy. Do not respond by generating more MCTS
labels, adding on-route teacher-forced hidden features, fine-tuning the base
MLLM, or expanding the benchmark suite without a new approved plan.

## 9. Implemented pre-training surface

- `binary_policy/executor/`: dual-API binary Qwen executor and cache.
- `binary_policy/labels.py`, `dataset.py`: validated MCTS adapter and grouped
  split.
- `binary_policy/predictor.py`: direct and canonical-run POLAR-style heads.
- `binary_policy/losses.py`, `decode.py`, `factorization_audit.py`: multi-valid
  set objective and deterministic decoding/audit.
- `experiments/binary_executor_preflight.py`: model-scale validity gate.
- `tools/audit_binary_polar_labels.py`: full label/factorization audit.
- `tools/prepare_binary_polar_data.py`: compact checksummed manifest builder.
- `experiments/train_binary_polar.py`: implemented but not run trainer.
- `configs/binary_polar_qwen2_5_vl_7b_v1.yaml`: proposed frozen configuration.
- `tests/test_binary_policy.py`, `tests/test_binary_executor.py`: lightweight
  contract tests.

## 10. Currently active action

Implementation is complete through the pre-training surface. The next bounded
action is BP-0, the CPU-only full label/representation audit. It has not been
run in this task. Predictor training remains unauthorized until BP-0, BP-1, and
BP-2 pass and the user explicitly authorizes training.
