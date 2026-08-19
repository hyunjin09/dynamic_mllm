# Binary MCTS Label–Executor Mismatch Analysis

Date: 2026-08-09

## Bottom line

The current evidence does **not** show that the direct binary head or exact
valid-set NLL is defective. It shows that the cached MCTS masks are
**executor-outcome labels tied to the exact label-generation runtime**, while
BP-1 replay did not initially reproduce that runtime contract.

The mismatch has two levels:

1. **Supported input-contract mismatch.** BP-1 ignored record-specific
   `max_image_tokens`. All four selected DocVQA fixtures were replayed with
   4,800–5,248 visual rows rather than the cached 1,989–2,040 rows. This changes
   both the vision input and the MRoPE positions of later text tokens. Replaying
   with the label image budget restored cached geometry for three of the four
   failing DocVQA records and repaired important route outputs.
2. **Residual runtime sensitivity with exact subcause unresolved.** ChartQA and
   TextVQA mismatches remain even after enabling the recorded TF32 policy and
   using the recorded software/model configuration. The source records do not
   pin the GPU model, driver/CUDA kernel details, `qwen_vl_utils` version, or a
   deterministic-kernel contract. MCTS evaluates each mask once and stores a
   thresholded answer outcome, so a numerically fragile greedy branch can
   become a nominally valid route.

Diagnosis: **supported** for the DocVQA preprocessing mismatch; **supported**
that residual cached outcomes are runtime-sensitive/incompletely reproducible;
the particular remaining kernel or hardware cause is **unknown**.

Training remains blocked under the unchanged exact-token BP-1 gate.

## What the label-producing model actually executes

For each of 28 language layers, a route bit controls only visual-row
participation:

- `ON`: scatter current text and visual rows into original token order and run
  the native Qwen decoder layer over the full sequence;
- `OFF`: run the same decoder layer on compacted text/control rows, omit visual
  K/V and visual MHSA/FFN computation, and carry the visual hidden rows forward
  unchanged.

Visual rows can re-enter at a later ON layer. Each layer creates a
route-dependent prefill cache: full multimodal K/V for ON and text-only K/V for
OFF. Autoregressive answer tokens then pass through every decoder layer while
reading those heterogeneous per-layer caches. Consequently, a mask changes
both prefill state and every later decoding step.

The MCTS runtime:

- starts from all ON and also evaluates all OFF;
- explores 200 additional binary masks per record;
- evaluates each mask by deterministic greedy generation;
- applies repetition penalty `1.05`;
- scores the generated text with the benchmark evaluator;
- retains every mask whose score reaches the record threshold;
- names the successful mask with the fewest ON layers as `best_mask`.

These labels mean “this exact executor produced a correct string on this
evaluation.” They are not architecture-only annotations and are not proofs
that a mask is robustly correct under every numerically equivalent runtime.

## Provenance distinction missed by the first BP-1 analysis

The new source description states that MCTS inserted its packaged source tree
first on `sys.path` and imported `dvr_qwen` from that package. The earlier BP-1
trace instead loaded `reference/binary_action_qwen` as the reference.

Static comparison shows that the core ON/OFF functions in
`reference/dvr_qwen/binary_layer.py` and
`reference/binary_action_qwen/core/binary_layer.py` are identical apart from
package imports and Accelerate meta-device handling. Split/scatter also differs
only in imports and later router-only instruction-mask metadata. Therefore,
this reference-selection mistake weakens the provenance proof but does not by
itself explain the remaining static-route output differences.

The supplied `reference/dvr_qwen` directory is not independently executable:
support modules such as `cache.py`, `masks.py`, and `generate.py` are absent.
Thus a byte-for-byte replay of the original package is not currently possible
from this reference folder alone.

## Confirmed preprocessing mismatch

The label runtime builds image content with the record's image path and, when
set, `max_pixels = max_image_tokens * 28 * 28`. BP-1 instead opened the original
image and passed it directly to the processor without applying the record
budget.

| Fixture | Cached text/visual/full rows | Original BP-1 rows | Geometry match |
|---|---:|---:|---|
| DocVQA `ba41...` | 33 / 2,028 / 2,061 | 33 / 4,800 / 4,833 | no |
| DocVQA `07032...` | 35 / 2,028 / 2,063 | 35 / 4,800 / 4,835 | no |
| DocVQA `7c497...` | 37 / 1,989 / 2,026 | 37 / 4,819 / 4,856 | no |
| DocVQA `9492...` | 41 / 2,040 / 2,081 | 41 / 5,248 / 5,289 | no |

The fourth row happened to pass cached token checks despite the wrong geometry;
that is output coincidence, not contract equivalence.

A CPU reconstruction of the label runtime's documented portable fallback
exactly produced 2,028 visual rows for DocVQA `07032...`, versus 4,800 in BP-1.
This confirms that the missing image-budget application is causal for the input
geometry discrepancy.

## Bounded five-fixture replay

The replay used:

- the five previously failing fixtures only;
- Qwen2.5-VL-7B snapshot
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- PyTorch `2.6.0+cu124`;
- Transformers `5.3.0`;
- BF16 and SDPA;
- slow processor;
- repetition penalty `1.05` and the recorded EOS IDs;
- TF32 enabled and no forced deterministic-algorithm mode;
- record-specific image budgets.

It did not train, search masks, or inspect predictor results.

| Fixture/route that failed original BP-1 | Recorded-contract replay | Correctness status |
|---|---|---|
| ChartQA mixed best mask | still `Germany` vs cached `India` | cached valid → replay invalid |
| DocVQA `ba41...` best mask | exact cached tokens restored | valid → valid |
| DocVQA `07032...` all OFF | exact cached tokens restored | invalid → invalid |
| DocVQA `07032...` best mask | `JOB ASSIGNMENT DOCUMENT.` vs cached `JOB ASSIGNMENT FORM.` | both remain above ANLS threshold |
| DocVQA `7c497...` best mask | still differs; fallback produced 2,040 rather than cached 1,989 visual rows | cached valid → replay invalid; exact preprocessing unresolved |
| TextVQA all OFF | still `123456789` vs cached `10` | invalid → invalid |

The replay also produced one punctuation-only all-ON difference on DocVQA
`ba41...`; both outputs remained correct. This reinforces that exact generated
IDs are more sensitive than benchmark correctness.

The `7c497...` fallback mismatch indicates that the label run likely took the
`qwen_vl_utils.process_vision_info` branch rather than the portable manual
resize branch. The exact installed utility version is not recorded or present
in the project environment.

## Why successful labels can stop working

### 1. The label is conditional on preprocessing

The mask is applied after the vision tower, but the initial visual-token count,
visual features, and multimodal positions are determined by preprocessing.
Changing image resolution changes the function being labeled. For all-OFF,
visual hidden rows never enter a decoder layer, but the question tokens still
receive MRoPE positions derived from the multimodal layout; therefore image
geometry can affect even an all-OFF route.

### 2. Mixed routes amplify small numerical changes

An OFF layer preserves visual rows while advancing text rows. A later ON layer
combines representations that have experienced different computation
histories. Small attention/kernel differences can therefore be amplified when
the streams are recombined and then propagated through route-specific decode
caches. Greedy argmax finally turns a small logit-order change into a discrete
answer change.

### 3. MCTS stores a one-run threshold event

The label generator does not require repeated-route stability, a probability
margin, or agreement across hardware. A route is positive after one generated
answer crosses the benchmark threshold. The sparsest successful route is then
favored even though sparsity can select a fragile boundary solution.

This is especially important for DocVQA `7c497...`, which has only one cached
successful mask among 202 evaluated candidates. ChartQA's failing fixture has
75 cached successful masks, so one unstable best mask does not imply its whole
valid set is unusable. Exact set NLL can exploit multiple masks, but it still
assumes that a meaningful portion of the cached valid set remains valid under
the deployment executor.

### 4. Exact-token mismatch and label invalidity are not equivalent

The five-fixture evidence includes:

- valid → invalid changes, which are substantive label drift;
- valid → different-but-valid changes;
- invalid → different-invalid changes;
- punctuation-only token changes preserving correctness.

The frozen BP-1 gate deliberately requires exact IDs, so all of these still
block training. Scientifically, however, they should not be interpreted as the
same failure mode.

## Explanations ruled out or weakened

- **Binary head/objective failure:** ruled out as an explanation because no
  predictor participated in label generation or BP-1 replay.
- **Exact valid-set NLL:** ruled out for the same reason.
- **Wrong model revision, Transformers version, PyTorch version, dtype, or
  attention backend:** ruled out for the recorded fields; they match.
- **Current port versus the earlier binary reference:** ruled out on the traced
  mixed fixture by layer-by-layer bit-exact equality.
- **Different core ON/OFF semantics between the two supplied references:**
  weakened strongly by static equality of the relevant layer functions.
- **TF32 setting alone:** ruled out as a complete explanation; enabling it did
  not repair ChartQA or TextVQA.
- **All cached masks are invalid:** not supported. Eleven of sixteen original
  fixtures passed, and recorded preprocessing repaired additional behavioral
  mismatches.

## Remaining unknowns

- GPU model and exact source node used for each label shard;
- CUDA driver, cuDNN, and SDPA kernel provenance;
- `qwen_vl_utils` version and exact smart-resize branch;
- whether each cached successful mask is stable over repeated executions;
- the fraction of the complete 184,785-mask cache that remains behaviorally
  valid under the target executor.

## Consequence for the planned predictor

Exact valid-set NLL correctly optimizes the probability of complete cached
masks. It cannot repair an executor-domain shift: if the source valid set is
partly invalid under the target executor, the loss assigns probability to
stale positives. This would appear downstream as cache incompleteness or label
noise, not necessarily factorization or optimization failure.

The smallest defensible next step would be an explicitly approved BP-1
contract repair:

1. make fixture preprocessing consume the recorded image budget and exact
   vision utility path;
2. separate native all-ON parity from label-executor all-ON replay, because the
   native maskless kernel and the label executor's explicit causal mask can
   produce different BF16 token sequences;
3. rerun the unchanged fixtures while reporting both exact-token and
   correctness-preservation status;
4. if substantive valid→invalid drift remains, decide prospectively whether to
   revalidate the cached valid sets under the target executor or abandon these
   labels.

That is a protocol decision, not performed here. Predictor training remains
blocked.

## Evidence

- Label-generation contract:
  `reference/dvr_qwen/MODEL_AND_LABEL_GENERATION.md`
- Label runtime:
  `reference/dvr_qwen/runtime.py`
- Original BP-1 result:
  `outputs/binary_polar/preflight/executor_preflight_v2.json`
- Bounded replay:
  `outputs/binary_polar/preflight/label_runtime_contract_v1.json`
- Bounded replay checksum:
  `outputs/binary_polar/preflight/label_runtime_contract_v1.json.sha256`
- Diagnostic implementation:
  `experiments/diagnose_binary_label_runtime_contract.py`
- Slurm job: `99730`, A6000 node03

