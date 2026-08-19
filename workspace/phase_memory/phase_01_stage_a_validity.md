# Phase 01: Stage A Validity Memory

## Current Objective

Establish clean, exact, deterministic READ/WRITE intervention validity on the
frozen Qwen2.5-VL-7B-Instruct primary model before any discovery work.

## Active Constraints

- The unchanged source is `plans/dynamic_mllm_read_write_causal_analysis_plan_v2.md`.
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`, frozen BF16.
- Primary READ uses fixed original softmax weights and subtracts only the visual
  value path; primary WRITE restores pre-layer visual rows while preserving
  current-layer text output.
- Every factorial state starts from a clone of the same cached pre-layer state
  and runs the unchanged suffix.
- Stage B is outside the completed action.

## Current State

- Done: Stage A architecture graph, token layouts, hook validation, FULL no-op
  parity, READ/WRITE reconstruction at hook and suffix, four-state stability,
  evaluator/FULL scoring reproduction, activation plausibility, and artifacts.
- In progress: None.
- Blocked: No Stage A gate condition. The 16,314-token stock-eager regime is not
  validated on A6000 and inherited correct/wrong labels are not yet valid for
  analytical use under the pinned revision.
- Most recent useful observation: All required Stage A checks passed on 23
  samples (20–50 required), with a maximum validated prompt length of 4,861.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| All required Stage A checks pass; Stage B entry gate is satisfied and Stage B was not run. | `outputs/stage_a/stage_a_summary.json` | Final gate classification. | confirmed |
| FULL path has exact layer/logit/option/generation parity. | `outputs/stage_a/no_op_parity.csv`, `outputs/stage_a/option_score_parity.csv`, `outputs/stage_a/benchmark_scoring_reproduction.csv` | Instrumentation does not alter the base model. | confirmed |
| READ and WRITE reconstruct at their hooks and through the unchanged suffix. | `outputs/stage_a/read_reconstruction.csv`, `outputs/stage_a/write_reconstruction.csv` | Satisfies the source plan's algebraic validity kill gate. | confirmed |
| All four states are finite, repeat-identical, and injected from the same cached prestate. | `outputs/stage_a/four_state_stability.csv` | Establishes deterministic factorial execution. | confirmed |
| Actual graph, token order, visual rows, causal mask, GQA/mRoPE, and hooks are recorded. | `outputs/stage_a/architecture_causal_graph.md`, `outputs/stage_a/token_layout.json`, `outputs/stage_a/runtime.json` | Grounds counterfactual semantics in the repository/runtime. | confirmed |
| One requested 16,314-token record is excluded with preserved reason/evidence. | `outputs/stage_a/stage_a_requested_samples.jsonl`, `outputs/stage_a/stage_a_resource_exclusions.json`, `outputs/stage_a_attempt_04_decoder_eager_oom/stage_a_failure.json` | Bounds the validated runtime domain transparently. | confirmed |
| Stored predictions reproduce stored evaluator scores 23/23 and instrumented FULL matches the pinned model 23/23; inherited bucket scores remain stable only 22/23. | `outputs/stage_a/benchmark_scoring_reproduction.csv` | Scoring gate passes, but pool bucket labels require revision-specific revalidation before analytical use. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial deterministic CUDA probe | cuBLAS deterministic configuration missing. | supported | `runs/stage_a_probe_20260804/slurm.log` | Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python. | Launching deterministic CUDA without it. |
| Ideal BF16 READ add-back | Hook error 0.015625 and suffix-logit error 3.59375. | supported | `outputs/stage_a_probe_cublas/read_reconstruction.csv` | Use exact representable `FULL-OFF` add-back and gate its adjustment by local half-ULP. | Adding an ideal residual to an already-rounded OFF state. |
| Stock eager on 16,314-token prompt | Decoder attention exceeded A6000 memory. | supported | `outputs/stage_a_attempt_04_decoder_eager_oom/stage_a_failure.json` | Validated domain excludes this preserved resource outlier. | Claiming validity for the 16k-token regime. |
| SDPA substitute | Prospective causal eager-reference suffix-logit RMS 0.03484 exceeded 0.0078125. | supported | `outputs/stage_a_sdpa_reference_probe_attempt_03_valid_sdpa_rejected/stage_a_summary.json` | SDPA is not a valid primary substitute for this intervention. | Relaxing frozen equivalence thresholds after seeing the result. |
| Query-chunked eager substitute | Against stock eager at 4,793 tokens, suffix-logit RMS was 0.12424. | supported | `outputs/stage_a_chunked_stock_equivalence_boundary/chunked_eager_equivalence.json` | Keep the actual model on Transformers stock eager. | Treating query-chunked eager as numerically identical to stock eager. |
| Fresh inherited-bucket check | `boy` became `boys`, changing one GQA score; bucket stability 22/23. | supported | `outputs/stage_a/benchmark_scoring_reproduction.csv` | Separate evaluator/FULL parity from unknown source-checkpoint provenance; revalidate labels before analysis. | Using inherited correct/wrong labels as pinned-revision evidence. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Begin a separately authorized Stage B only after approving option data and frozen splits. | Stage A entry gate passed. | Discovery protocol and data readiness. | high | requires user approval |
| Validate the 16,314-token regime on a faithful higher-memory stock-eager setup. | Would broaden runtime coverage. | Current resource-domain limitation. | unknown | optional, not required by Stage A sample-count gate |

## Next-Step Decision

- Deliberation mode: deep
- Active objective and bottleneck: Stage A is complete; later analytical use is
  bottlenecked by option-scoring data/splits and pinned-revision label validity.
- Relevant memory item used: Alternate attention runtimes failed prospective
  equivalence; stock eager is the only validated FULL path.
- Confirmed observation: Every required Stage A check passes on 23 samples.
- Unverified interpretation: None needed for the Stage A gate.
- Diagnosis: supported
- Evidence path if diagnosis is not unknown: `outputs/stage_a/stage_a_summary.json`
- Viable alternatives considered: Classify inherited bucket stability as a
  scoring gate, or correctly retain it as a provenance diagnostic; independent
  review ranked the latter first because the evaluator and FULL parity are 23/23.
- Chosen action: Mark Stage A passed within its explicit runtime domain and stop.
- Strongest objection: The excluded 16k sample leaves that token-length regime
  unvalidated; this limitation must travel with all later claims.
- How this differs from failed attempts: No alternate decoder runtime is used,
  and the resource exclusion and label drift are explicit artifacts.
- Automatic execution authorized: no further research action
- Authorization basis: The user required stopping after the Stage A decision.
- Stop condition: Reached for this action; Stage B was not executed.

## Latest Research-Action Result

- Action taken: Completed the Stage A validity gate on the faithful stock-eager
  runtime and classified the required checks using the approved source semantics.
- Result: Pass on 23 samples, layers 0/14/27, prompts up to 4,861 tokens; Stage B
  entry gate satisfied, Stage B not executed.
- Evidence saved: `outputs/stage_a/` and
  `runs/stage_a_stock_eager_validity_23_20260804/slurm.log`.
- Failure or issue: The 16,314-token resource regime is excluded; split BF16
  recomposition is not suffix-equivalent and inherited bucket stability is 22/23.
- Lesson learned: Preserve stock eager for the causal FULL path and treat pool
  labels as unverified until regenerated or revalidated under the pinned model.
- Next implication: Await explicit user direction before any Stage B work.
