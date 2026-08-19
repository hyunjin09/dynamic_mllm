# Phase 11: Binary Visual-Token POLAR Memory

## Current Objective
Prepare a valid, bounded supervised-learning path that adapts POLAR to
layer-local binary visual-token ON/OFF actions for frozen Qwen2.5-VL, using the
existing MCTS v2 labels without starting training.

## Active Constraints
- Preserve all closed v2–v4 and query-refinement artifacts and conclusions.
- Base Qwen2.5-VL and the POLAR question encoder remain frozen.
- No new MCTS or base fine-tuning. The user now authorizes one frozen direct-head
  training run after the amended BP-0A, BP-1, and BP-2 gates pass.
- OFF removes visual rows only at the selected layer and carries them unchanged;
  it is not READ/WRITE gating or permanent deletion.
- External MCTS source is user-authorized read-only; derived bulk data must be
  written under `/data/dataset`, not the project.
- GPU and CPU-heavy validation must use `infra/gpu_scheduler.py`.

## Current State
- Done: inspected the POLAR paper/code, binary Qwen reference, MCTS schema and
  source audit; challenged the adaptation; wrote the plan; implemented the
  executor, data adapter, predictors, losses, decoding, audit/preflight/trainer
  entrypoints, config, and tests; migrated the project `.venv` to Transformers
  5.3.0 with a recorded 4.51.3 rollback pin.
- In progress: none; the authorized equivalence repair and BP-1 rerun are done.
- Blocked: BP-1 cached non-FULL reproduction still fails; BP-2, training, and
  online evaluation remain unopened.
- Most recent useful observation: BP-1 ignored the label records' image-token
  budgets. Recorded-budget replay repairs several failures but not all; the
  cached valid ChartQA best mask becomes invalid under the target replay.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| ON runs native full rows; OFF runs compacted text rows and bypasses vision | `reference/binary_action_qwen/core/binary_layer.py` | Fixes the binary causal/execution contract | confirmed |
| POLAR uses frozen question embeddings, layer queries, cross-layer encoding, and multiple valid programs | `reference/polar/2606.06574v1.pdf`, `reference/polar/PoLar/polar/` | Identifies reusable method components | confirmed |
| Source audit passed 4,000 records, eight 500-record cells | `/home/hyemin/data/dataset/dynamic_mllm/mcts_v2/final/audit_summary_full_v2.json` | Establishes available supervision | confirmed |
| 3,408 records have a successful mask; 592 do not | same audit | Requires an evaluation-only role for no-success records | confirmed |
| Existing masks can be non-contiguous and samples have many valid masks | `/data/dataset/dynamic_mllm/binary_polar_v1/binary_polar_label_geometry_audit_v1.json` | 184,785 deduplicated valid routes have mean transition counts around 13 per mask | confirmed |
| Empirical per-bit marginals lose route correlations | same BP-0 audit | Diagnostic against duplicated-mask BCE/marginal decoding; not a rejection of exact set-NLL | confirmed |
| Trainer computes weighted complete-mask log-sum-exp | `binary_policy/losses.py`, `binary_policy/training.py`, `outputs/binary_polar/preflight/bp0a_exact_set_nll_v1.json` | Formula and padded-route behavior match exactly; coherent-mode sanity passed | confirmed |
| Image-group splitting is feasible | same BP-0 audit | 3,824 groups, zero cross-split groups, minimum cell/split count 52 | confirmed |
| Project and label runtime now both use Transformers 5.3.0/SDPA provenance | sample `.runtime`, `workspace/env_state.md`, `outputs/env_migrations/transformers_5_3_0_v1.json` | Removes version skew but does not replace cached-label reproduction | confirmed |
| Twelve lightweight implementation tests pass | direct `.venv` assertion run over `tests/test_binary_policy.py` and `tests/test_binary_executor.py` | Supports pure logic and executor contracts, not 7B parity | confirmed |
| BP-1 split/scatter, OFF identity, and deterministic repeats pass 16/16 | `outputs/binary_polar/preflight/executor_preflight_v1.json` | Rules out several basic executor faults | confirmed |
| Initial BP-1 all-ON native logits and cached mixed masks failed | same plus `outputs/binary_polar/preflight/executor_diagnostic_v1.json` | Historical failure that motivated the equivalence repair | superseded in part |
| Current and reference mixed execution are bit-exact | `outputs/binary_polar/preflight/executor_equivalence_trace_v1.json` | Rules out the project port as the cached mixed-mask mismatch | confirmed |
| Native maskless SDPA fixes ALL-ON parity | `outputs/binary_polar/preflight/executor_preflight_v2.json` | Restores the strongest invariant at zero error without relaxing tolerance | confirmed |
| Original BP-1 used wrong DocVQA image geometry | `outputs/binary_polar/preflight/executor_preflight_v2.json`, source records | Four fixtures used 4,800–5,248 rather than cached 1,989–2,040 visual rows | confirmed |
| Recorded-contract five-failure replay | `outputs/binary_polar/preflight/label_runtime_contract_v1.json` | Repairs some cached routes but leaves substantive ChartQA drift and unresolved exact provenance | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Run new tests with pytest | `.venv/bin/python: No module named pytest` | supported | local command output | Execute pure assertion tests directly now; add/test pytest only through approved CPU environment maintenance | Do not install globally or mutate the environment on the login node |
| Treat the reference teacher-forced router as the training policy | It imports a missing feature module and has no online predicted rollout | supported | `reference/binary_action_qwen/core/binary_generate.py` | Use a single pre-action POLAR-style predictor | Do not train on labeled-route hidden features and evaluate free-running without an exposure-shift design |
| Reject the direct head using empirical marginal coverage | The audit optimized per-bit marginals, not the implemented exact valid-set NLL | supported | `binary_policy/factorization_audit.py`, `binary_policy/losses.py`, approved amendment 01 | Treat BP-0 as a label-structure diagnostic and add a loss-aligned sanity gate | Do not reuse marginal coverage as an exact-set-NLL rejection gate |
| BP-1 executor preflight | All 16 rows failed the full gate; 4/12 cached best masks disagreed and native logit parity exceeded tolerance | unknown for the executor mismatch; supported for a cache-comparator bug | `reports/binary_polar_bp1_executor_preflight.md` | Stop training; compare the port against the supplied reference before retrying BP-1 | Do not accept cached labels or relax parity because greedy all-ON happened to match |
| Repaired BP-1 rerun | ALL-ON parity became exact, but the same two ALL-OFF and four best-mask cached mismatches remained | supported for native mask-dispatch cause; unknown for cached provenance | `reports/binary_polar_bp1_executor_repair.md` | Training stays blocked; a different cache-validity action requires approval | Do not alter reference mixed semantics to chase stored tokens |
| Label-runtime contract replay | Image-budget repair restored three failing DocVQA geometries and several outputs, but not every token sequence | supported for preprocessing mismatch; remaining exact cause unknown | `reports/binary_mcts_label_mismatch_analysis.md` | Treat cached masks as source-runtime outcomes; reconstruct or prospectively revalidate before training | Do not attribute cache drift to the binary head or exact set NLL |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Direct 28-bit head with exact valid-set NLL | Scores coherent complete masks and matches top-1 deployment | Approved primary binary policy | medium | selected; gated by BP-0A/BP-1/BP-2 |
| Canonical maximal-run binary POLAR | Higher marginal-diagnostic coverage but not loss-aligned or yet trained | Structured fallback | medium | deferred unless diagnosed route-structure failure |
| Sequential on-route hidden-state router | Could condition later choices on earlier execution | Richer online state | high | rejected for first attempt due missing rollout and teacher-forcing exposure mismatch |
| Pre-action image+question predictor | Could capture image-specific route needs | Question-only information limitation | high | deferred; requires explicit amendment |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: validate and train the direct factorized head
  under its actual complete-mask objective, then use image-group-disjoint online
  correctness as the real gate.
- Relevant memory item used: the old 0.0771 marginal coverage and the exact
  `multi_valid_set_nll` implementation are different objectives.
- Confirmed observation: the direct trainer already evaluates every complete
  valid mask before weighted log-sum-exp; the current predictor input is
  question-only and must be minimally extended to satisfy the approved
  image/question-conditioned contract.
- Unverified interpretation: the shared predictor can select coherent valid
  masks on unseen image groups and preserve online correctness.
- Diagnosis: supported; the prior rejection was objective-misaligned.
- Evidence path if diagnosis is not unknown: amendment 01 and
  `binary_policy/factorization_audit.py` versus `binary_policy/losses.py`.
- Viable alternatives considered: direct exact set-NLL (selected), canonical
  runs (fallback), stop.
- Chosen action: completed the equivalence repair and unchanged BP-1 rerun;
  stop because cached non-FULL reproduction still fails.
- Strongest objection: exact set-NLL can select one valid training mode but does
  not ensure that a frozen image/question predictor generalizes.
- How this differs from failed attempts: the loss and gate now operate on
  complete masks; empirical bit marginals are diagnostic only.
- Automatic execution authorized: yes
- Authorization basis: explicit user approval for the BP-1 equivalence repair,
  unchanged BP-1 rerun, and training only if BP-1 passes.
- Stop condition: reached; do not train or switch to canonical segmentation.

### Label-generator contract audit (2026-08-09)

- Deliberation mode: deep
- Active objective and bottleneck: determine whether the five cached non-FULL
  mismatches reflect invalid labels or replay under a different execution
  contract.
- Confirmed observation / unverified interpretation: the MCTS runtime used its
  packaged `dvr_qwen`, record-specific `max_image_tokens`, and TF32 enabled;
  BP-1 used the earlier `binary_action_qwen` reference, ignored
  `max_image_tokens`, disabled TF32, and forced deterministic algorithms. The
  three failing DocVQA rows have cached visual geometry near 2,000 rows but
  BP-1 replay geometry near 4,800. TF32 sensitivity for the remaining
  same-geometry failures is not yet verified.
- Diagnosis: supported for DocVQA preprocessing-contract mismatch; suspected
  for numerical-execution mismatch on ChartQA/TextVQA. Evidence:
  `reference/dvr_qwen/MODEL_AND_LABEL_GENERATION.md`, failing source records,
  and `outputs/binary_polar/preflight/executor_preflight_v2.json`.
- Viable alternatives considered: exact recorded-contract replay (selected),
  discard/refresh labels without localization, or keep the current block.
- Chosen action and strongest objection: replay only the five failures with
  label-generation preprocessing and numerical settings; the project executor
  is not the complete packaged MCTS tree, although its core ON/OFF layer
  functions statically match and prior traces establish port equivalence.
- How this differs from failed attempts: it compares against the newly supplied
  label-producing runtime contract, not the earlier generic binary reference.
- Authorization and stop condition: user requested mismatch analysis; perform
  one bounded diagnostic, do not train, regenerate labels, or alter BP-1 gates.

## Latest Research-Action Result
- Action taken: audited the newly supplied label-producing `dvr_qwen` contract
  and replayed only the five known failures with recorded image budgets,
  generation policy, TF32 setting, and software/model pins.
- Result: the prior BP-1 loader ignored `max_image_tokens`; all four DocVQA
  fixtures had wrong visual geometry. Correcting the budget restored cached
  geometry for three failing DocVQA rows, made one cached best mask exact, and
  restored one cached all-OFF output. ChartQA's valid best mask and TextVQA's
  invalid all-OFF output still differ; TF32 alone is not the cause. One DocVQA
  record still needs the unavailable exact `qwen_vl_utils` resize path.
- Evidence saved: `outputs/binary_polar/preflight/label_runtime_contract_v1.json`
  and `reports/binary_mcts_label_mismatch_analysis.md`.
- Failure or issue: exact cached reproduction remains failed. Diagnosis is
  supported for preprocessing mismatch and incomplete runtime reproducibility;
  the remaining kernel/hardware/utility subcause is unknown.
- Lesson learned: MCTS masks are outcomes under a complete executor contract,
  not portable architecture-only labels. Exact-token drift includes both
  substantive valid-to-invalid changes and correctness-preserving changes.
- Next implication: do not train. A BP-1 contract amendment must first choose
  exact label-runtime reconstruction and, if drift remains, whether cached
  masks may be revalidated under the target executor.

### Training-gate decision after mismatch analysis (2026-08-10)

- Deliberation mode: standard
- Active objective and bottleneck: determine whether removing the known BP-1
  mismatches is sufficient to authorize predictor training.
- Confirmed observation: the known mismatches are technical-audit fixtures,
  not a prevalence sample; at least one cached positive mask changes from valid
  to invalid under the target executor, while other token mismatches preserve
  benchmark validity.
- Diagnosis: supported executor-domain label drift, with incomplete exact
  source-runtime provenance. Evidence: `reports/binary_mcts_label_mismatch_analysis.md`
  and `outputs/binary_polar/preflight/label_runtime_contract_v1.json`.
- Viable alternatives considered: delete only known failures (rejected as
  outcome-dependent filtering), repair the label-runtime contract and rerun the
  unchanged BP-1 suite (selected first), or uniformly revalidate every cached
  mask admitted to training under the target executor (valid fallback requiring
  a prospective amendment and regenerated valid sets/weights).
- Chosen action: do not start training merely by deleting mismatches. First
  restore the exact preprocessing/runtime contract and rerun unchanged BP-1.
  Training may proceed if the frozen gate passes; if valid-to-invalid drift
  remains, require uniform target-executor cache revalidation before BP-2.
- Strongest objection: exact source-runtime reconstruction may be impossible
  because the packaged vision utility and hardware/runtime provenance are
  incomplete; in that case uniform target-executor revalidation is more costly
  but is the smallest scientifically defensible repair.
- Authorization status: decision recorded; no repair, cache revalidation,
  BP-2 freeze, or training is authorized by this question alone.

### BP-1 input-contract repair result (2026-08-10)

- Deliberation mode: deep
- Active objective and bottleneck: repair the known preprocessing mismatch and
  determine whether the unchanged exact-cache BP-1 gate permits training.
- Confirmed observation / unverified interpretation: all 16 cached prompt
  geometries now match and complete row passes improved from 11/16 to 12/16,
  but four exact-token mismatches remain, including two cached-positive to
  target-invalid routes. The exact residual runtime cause is unverified.
- Diagnosis: supported for repaired preprocessing; unknown for remaining token
  drift. Evidence: `outputs/binary_polar/preflight/executor_preflight_v3.json`
  and `reports/binary_polar_bp1_input_contract_repair.md`.
- Viable alternatives considered: another exact source-runtime reconstruction,
  uniform prospective target-executor cache revalidation, or deleting only the
  known failures. The last is rejected as outcome-dependent filtering.
- Chosen action and strongest objection: stop training and request approval to
  precommit a stratified image-group-disjoint cohort plus coverage floor, then
  uniformly revalidate every cached mask admitted for that cohort. The cache
  remains incomplete even after revalidation, and the necessary cohort size is
  not yet frozen.
- How this differs from failed attempts: the rerun uses exact cached prompt
  geometry for every fixture, including the prior 1,989-row outlier, so image
  budget/layout mismatch no longer explains the remaining failures.
- Authorization and stop condition: the user authorized the repair and rerun;
  that action is complete. BP-1 failed, so do not open BP-2 or train.
- Independent review: `revise` with medium confidence. It rejected automatic
  full-pool revalidation as disproportionate and identified cohort-scoped
  uniform revalidation as the lower-cost form of the same validity repair.
