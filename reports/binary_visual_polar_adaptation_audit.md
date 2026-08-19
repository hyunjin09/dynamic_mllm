# Binary Visual Policy / POLAR Adaptation Audit

## Outcome

The two reference implementations can be combined coherently, but not by
copying POLAR's tri-state labels. The binary Qwen reference changes token
presence inside each decoder layer, while POLAR changes decoder-layer program
execution. The valid adaptation retains POLAR's lightweight layer-conditioned
predictor and multi-program supervision while replacing its action/output
space with the actual 28-bit visual-token mask.

No training or model-scale inference was run in this task.

## Confirmed implementation facts

### Binary Qwen reference

`reference/binary_action_qwen/core/binary_layer.py` defines the relevant
counterfactual:

- ON reconstructs the full original row order and calls the native Qwen layer;
- OFF calls the native layer on text/control rows only and returns the incoming
  visual state unchanged;
- OFF is not permanent deletion because carried visual rows can be scattered
  back at a later ON layer;
- per-layer prompt K/V lengths differ, so generation needs a per-layer cache;
- the provided high-level generator is validated only for batch size one.

The older `modeling_dvr_qwen2_5_vl.py` implements text-to-visual READ gating,
not the requested token-presence action, and is not used as the primary path.

The reference `router_teacher_forcing` path is also not a valid POLAR policy:
it depends on a missing `phase5b.router_features` module, consumes features
produced along the labeled route, and does not implement free-running predicted
routing. The adaptation therefore uses a single pre-action prediction.

### POLAR paper and code

The paper and `reference/polar/PoLar/polar/` use:

- frozen Qwen3-Embedding-0.6B question-token embeddings;
- learned per-layer queries and question cross-attention;
- a Transformer encoder across layer representations;
- segmentation BCE plus masked three-way operation CE;
- multiple successful MCTS programs as supervision;
- downweighting the original dense path when a shorter valid path exists;
- beam decoding of top-k programs.

POLAR's repeat operator and decoder-path parser have no direct counterpart in
binary visual-row presence. Canonical maximal runs can encode every binary mask
and are retained as a baseline, but a direct bit head is the smaller primary
interface.

## Label evidence

The existing source audit passed 4,000/4,000 records, four balanced benchmarks,
and easy/hard cells of 500 records each. There are 3,408 records with one or
more successful masks. Success coverage is 1.0 for every easy cell and ranges
from 0.60 to 0.842 on hard cells. Records contain 28-bit masks, binary official
task rewards, cached generated IDs, image hashes, source asset IDs, prompt/token
counts, and cache-length diagnostics.

The source itself is approximately 9 GB. Full parsing and representation
analysis must therefore use a CPU-only Slurm job; only a single schema fixture
was parsed locally. That fixture passed the adapter and produced normalized
route weights summing to one.

## Adaptation decision and challenge

The provisional ranking was direct binary head, segmented binary POLAR, then
the existing sequential hidden-state router. An independent research review
agreed with this ordering but objected that independent logits may assign high
probability to invalid combinations when the valid mask set is correlated or
multimodal.

The implementation was revised in two ways:

1. the primary objective is now valid-set likelihood rather than an averaged
   single hard mask;
2. the full label audit compares empirical top-k valid-route coverage for the
   direct and canonical-run representations before the direct head may be
   trained.

The review also identified a missing OFF oracle. Tests now verify that OFF is
exactly the compacted-text native-layer result and that visual rows bypass the
layer unchanged.

The strongest remaining objection is that a question-only predictor may lack
image-specific information. It is retained for the first bounded attempt
because it is POLAR's actual pre-action input contract and avoids introducing a
new multimodal feature architecture. A multimodal predictor would be a later
explicit amendment, not an automatic response to failure.

## Implemented components

| Component | Path | Status |
|---|---|---|
| Action contract and decoding | `binary_policy/actions.py`, `decode.py` | implemented, lightweight tests pass |
| Multi-valid losses | `binary_policy/losses.py` | implemented, lightweight tests pass |
| POLAR-style predictor | `binary_policy/predictor.py` | direct and segmented heads implemented |
| MCTS label adapter | `binary_policy/labels.py`, `dataset.py` | implemented; one real fixture parsed |
| Factorization audit | `binary_policy/factorization_audit.py`, `tools/audit_binary_polar_labels.py` | implemented, full job not run |
| Binary executor | `binary_policy/executor/` | implemented with Transformers 4.51/5.x layer-call compatibility; model-scale validation pending |
| Executor preflight | `experiments/binary_executor_preflight.py` | implemented, GPU job not run |
| Manifest builder | `tools/prepare_binary_polar_data.py` | implemented, CPU job not run |
| Trainer | `experiments/train_binary_polar.py` | implemented, deliberately not run |
| Configuration | `configs/binary_polar_qwen2_5_vl_7b_v1.yaml` | proposed; gated on audit/parity |
| Unit contracts | `tests/test_binary_policy.py`, `tests/test_binary_executor.py` | 12 direct assertion tests pass |

## Unresolved blockers before training

1. The full 4,000-record label/run-geometry audit has not run.
2. Direct-bit factorization has not yet passed the prospective coverage gate
   against canonical runs.
3. All-ON native parity and exact cached generated-ID reproduction have not run
   on the pinned 7B model.
4. The final image-group train/validation/test manifest is not frozen.
5. Formal pytest is unavailable in the current `.venv`; the twelve new tests were
   executed directly as pure assertion functions. No package was installed or
   environment changed in this task.

The environment was subsequently migrated to Transformers 5.3.0 by CPU-only
Slurm job `99717`; dependency checks and the lightweight regressions passed.
Training is blocked until items 1–4 pass. The absence of pytest is an
environment-maintenance issue, not evidence about model validity.

## Verification performed

- Python byte-compilation passed for the new package and entrypoints.
- Twelve lightweight behavior tests passed: action normalization/run encoding,
  top-k decoding, set likelihood, both predictor output contracts, label
  deduplication/capping/weighting, split determinism, split/scatter identity,
  visual bypass, native full-row execution using a fake layer, and protection
  against accidentally unwrapping a native Hugging Face conditional LM, and
  the prospective direct-representation gate.
- One real ChartQA MCTS record parsed successfully with normalized route
  weights.
- No secrets or credential-like strings were introduced.
- The workspace is not a Git worktree, so no branch or atomic commit could be
  created; changes remain additive and confined to the requested detour plus
  compact project state files.

## Next bounded action

Run the complete label/factorization audit through Slurm with zero GPUs. Do not
start training. If the direct representation passes, run the 16-fixture GPU
executor/label reproduction preflight. If either gate fails, stop at that
failure rather than changing labels or silently migrating runtimes.
