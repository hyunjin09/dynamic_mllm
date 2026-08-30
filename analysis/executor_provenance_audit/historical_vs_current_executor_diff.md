# Historical vs Current Four-Action Executor

Date: 2026-08-30 KST

## Bottom line

There is no semantic difference in the fixed complete-route READ/WRITE executor
that can explain the Phase-42 token mismatch. The historical and current
`capture_four_action_route -> four_action_layer` path is byte-identical after
removing a later, unused online-selection API from the current file. The only
current input-path change moves `mm_token_type_ids` to the already-selected
embedding device; Phase-42 preprocessing had already moved every tensor to the
GPU, so that line is a no-op for the replay inputs.

The strongest remaining candidate is an execution-environment difference:
historical labels were generated with BF16 SDPA on eight H100s, whereas the
failing Phase-42 replay ran on four RTX 6000 Ada GPUs. That remains an inference,
not a verified cause. The recovered-source H100 replay was canceled before it
received an allocation at the user's instruction, so this audit does not
promote the hardware/kernel explanation to a supported diagnosis.

## Source-level differences

Of the 16 contract-bound files, 13 are byte-identical at current `HEAD`
`5aa6ac6641879aa699d3c053203ea5fd0b933017`. The three differing paths are:

| Path | Historical SHA-256 | Current SHA-256 | Exact difference | Fixed-route effect |
|---|---|---|---|---|
| `binary_policy/executor/four_action.py` | `e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868` | `7f8a076289ef3cc0d09dea09e5c8a5a2606bc758f3ce60eea75928b810759037` | current adds `Callable` and `capture_online_four_action_route` | none; fixed-route functions are unchanged |
| `binary_policy/executor/__init__.py` | `ad6f7c5300a28a8d884e01fb75da0ea46996e008efa04f7091edaf093ea228f1` | `b7eabebfeea13efd3c5aede60fa21753777f19cec2a31dd31e0d4076981e56db` | current exports that online API | none |
| `binary_policy/executor/inputs.py` | `0e22848f56aaaec1c510958eee37e407d0dc51726dc60e5a69a3eea54091d465` | `b0fa74da3ab134e5e08ae3e36877b64d59d42ff5b50f01cf01914f2daa137716` | current moves `mm_token_type_ids` to `full_embeddings.device` before 3-D position construction | can prevent a cross-device runtime error; no value change when already colocated |

The Phase-42 `prepare_sample` path calls `build_native_processor_inputs`, which
moves every tensor in the processor batch to the selected device before
`build_binary_inputs`. Therefore the `inputs.py` change does not change the
Phase-42 smoke values.

## Component comparison

| Component | Historical label executor | Current fixed-route executor | Same? | Can change fixed-route output? |
|---|---|---|---|---|
| FULL | one full-row `visual_on_layer`; retain text, visual, full K/V | same | yes | no source difference |
| READ_ONLY | one full-row call; retain text/full K/V, carry incoming visual | same | yes | no source difference |
| WRITE_ONLY | cache-free full-row call for retained visual, compact call for retained text/text K/V | same | yes | no source difference |
| IGNORE | compact text/control call; carry visual | same | yes | no source difference |
| Visual-row bypass | exact incoming state when WRITE off | same | yes | no |
| Retained text visual-K/V access | FULL/READ_ONLY only | same | yes | no |
| Same-layer ordering | READ sees pre-layer visual K/V | same | yes | no |
| Residual/MLP handling | full native layer for retained WRITE; discarded full output in READ_ONLY | same | yes | no |
| Full mask | materialized additive causal mask for complete routes | same | yes | no |
| Compact mask | causal mask using original text/control indices | same | yes | no |
| Prompt K/V | heterogeneous full vs compact cache by action | same | yes | no |
| WRITE_ONLY cache protection | full call cache-free; compact cache retained | same | yes | no |
| Generation core | clone prompt cache, repetition penalty, greedy argmax, text-only decode | same contract-bound function | yes | no source difference |
| Complete route entry | `capture_four_action_route` | same | yes | no |
| Online route entry | absent | added later | no | not called by Phase-42 fixed replay |
| Input token-type placement | pass processor tensor as supplied | explicitly `.to(full_embeddings.device)` | no | error-prevention only for the replay's already-GPU tensor |

## Label route executor vs numerically unified local-factorial executor

Two functions coexist and must not be conflated:

| Function | Intended use | Calls per target action |
|---|---|---|
| `four_action_layer` | complete multi-layer label routes | FULL 1, READ_ONLY 1, WRITE_ONLY 2, IGNORE 1 |
| `unified_target_four_action_layer` | local single-layer factorial analysis | all actions execute full-row then compact calls; action selects retained text, visual, and cache |

The historical labels used the first. This still gives exact state-level
READ/WRITE semantics, but it does not make continuous scores across the four
actions share an identical target-layer kernel path. The stricter second
function was introduced for within-unified causal margins and existed in the
historical source; it was not called by complete-route label generation.

## Generation/evaluation wrapper differences in Phase 42

The historical label runtime lets the model generation configuration supply
EOS IDs `[151645, 151643]` and repetition penalty 1.05. Phase 42 explicitly
passed EOS `[151645]` and the same repetition penalty. This difference is real
but cannot explain the observed 312-route smoke set:

- all 312 historical token sequences end in 151645;
- none contains 151643;
- all 312 Phase-42 sequences also end in 151645.

The prompt hashes were frozen per sample and reproduced in Phase 42. The model
revision was unchanged. Evaluator differences cannot produce a generated-token
mismatch, although they can change correctness after tokens differ.

## Likely cause of replay mismatch

### Directly verified facts

1. Phase 42 replayed 312 cached-correct routes; 37 were incorrect under the
   current runtime, affecting 10/12 samples across GQA, ChartQA, and TextVQA.
2. Four representative failures were each repeated twice. All four current
   pairs were exact; none matched the original generated IDs.
3. The fixed-route four-action source is semantically and bytewise unchanged.
4. The input-source change is a no-op for already-device-moved processor
   tensors.
5. Historical labels ran on H100s with Torch 2.6.0+cu124, Transformers 5.3.0,
   BF16, and SDPA. Phase 42 ran on RTX 6000 Ada GPUs through direct `torchrun`.
6. The narrower EOS list is not encountered by any of the 312 sequences.

### Likely inference

The smallest remaining contract difference capable of changing generated
tokens is hardware/kernel execution: BF16 SDPA and downstream BF16 GEMMs need
not be bitwise identical between H100 and Ada. Small logit changes can alter a
greedy argmax at a close token boundary and then change the full continuation.

This is intentionally described as an inference until same-H100 replay evidence
tests it. “The current executor source changed semantics” is not supported.

### Unresolved items

- Phase 42 did not freeze a complete package/driver/kernel manifest in the
  tracked artifacts. Its GPU model is recorded, but exact CUDA driver and every
  installed dependency are not.
- The historical contract records the model snapshot/revision but not
  per-weight-file hashes or the CUDA driver.
- A full unrelated `git status` from label-generation time is unavailable;
  only the contract-bound dirty source is exactly reconstructed.
- If recovered-source H100 replay is not 312/312, an additional unrecorded
  runtime component remains, and the exact historical execution environment
  cannot be claimed recovered.

## Final decision

The source-only conclusion is exact recovery with scientifically valid action
semantics. The stricter final execution-contract decision is **C: exact
end-to-end parity could not be established**, because the only discriminating
H100 replay was canceled before execution at the user's instruction.

This is not decision B: the line-level audit does not contradict the intended
READ/WRITE state semantics. It is also not evidence that the recovered source
would fail parity. The result is simply unmeasured. A future 312/312 exact
cached-token, answer, and correctness replay would be sufficient to upgrade the
end-to-end decision to A for the recorded recovered-source H100 environment.
