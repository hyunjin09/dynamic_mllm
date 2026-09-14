# Phase 51: Shared Stage-1 Global Risk Gate Memory

## Current Objective
Determine whether one layer-conditioned shared failure predictor plus one trajectory-level raw-score threshold yields a more stable conservative Stage-1 admission gate than the frozen Phase-48/50 independent probes.

## Active Constraints
- Reuse the exact Phase-48 7,999-row population, 6,399/800/800 image-group-disjoint split, current LMMS labels, and `text_final`/`text_mean`/`visual_mean` summaries.
- Train exactly four configs: state-only All-28, layer-only All-28, state+layer All-28, and state+layer Random-4.
- Use no dataset, answer, route, W-to-C, or sparse-router inputs.
- Use one train-only normalization pooled over samples and layers, one shared score space, and validation-only checkpoint/threshold/window selection.
- State-only and layer-only are validation architecture controls. Only the two predeclared state+layer schemes may be newly scored on test, together in one aggregate pass after freeze.
- Do not use test to switch the validation-designated primary scheme/window.
- Do not connect treatment, run OOD, resume W-to-C/four-action work, run MCTS, or run external evaluation.

## Current State
- Done: Completed all four validation-trained variants, froze the validation-designated primary and windows/thresholds, scored the two predeclared state+layer test models once, generated all comparisons/figures, and passed the final integrity audit.
- In progress: None.
- Blocked: No.
- Most recent useful observation: Shared global calibration reduces 99%-point preservation drift to 0.0025, but dataset wrong-recall spread worsens to 0.7600 and fails the frozen readiness criterion.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-48 split is 6,399/800/800 with zero UID/image-group overlap and 128 hash-bound feature shards | `analysis/dense_failure_stage1/layerwise_failure_probe/` | Fixes the population and feature contract | confirmed |
| Independent test AUROC is 0.8421 at layer 0 and peaks at 0.8992 at layer 21 | Phase-48 analysis summary | Sets the predictor baseline | confirmed |
| Independent sequential test preservation fell to 0.9575 at the validation 99% point | Phase-50 decision summary | Sets the global-risk robustness baseline | confirmed |
| Four GPUs are idle; host has 1 TiB RAM | live machine audit, 2026-08-31 | Supports four concurrent small-head workers over cached features | confirmed |

## Open Candidates
| Candidate | What It Resolves | Cost | Status |
|---|---|---|---|
| One pooled train-only normalization; four configs matching the plan output; validation-only selection | Tests the intended common score space without extra variants | medium | selected after review |
| Reuse Phase-48 per-layer normalization | Easier reuse | low | rejected because it weakens the single raw-score-space claim |
| Train all three architectures under both schemes | Fully crosses an ambiguity | higher | rejected as redundant and beyond the four required configs |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: Test whether shared parameterization and trajectory-level calibration reduce Phase-50 preservation drift and dataset mismatch.
- Confirmed observation / unverified interpretation: Independent probes rank failure well, but their sequential union is not conservative on test; whether sharing repairs calibration is unverified.
- Diagnosis: unknown; Phase-50 establishes the failure but not whether representation, calibration, or independent parameterization causes it.
- Viable alternatives considered: pooled shared normalization/four configs; per-layer normalization; full six-config architecture-by-scheme crossing.
- Chosen action and strongest objection: Execute the plan with pooled normalization, projection 256, layer embedding 32, deterministic hidden width 256, FP32 AdamW for 10 epochs, and exact validation trajectory thresholds. Pooled normalization may leave depth shifts for the embedding to absorb.
- Independent review: `revise`; accepted. Architecture controls remain validation-only, and only the two required state+layer schemes, with validation-frozen windows, receive one new test pass.
- How this differs from failed attempts: It calibrates one shared raw score directly on complete sample trajectories instead of composing 28 layer-specific tails.
- Authorization and stop condition: Explicitly authorized by the user via `plans/shared_stage1_global_risk_gate_plan.md`; stop after the in-domain shared-predictor/global-gate comparison and integrity/state update.

## Latest Research-Action Result
- Action taken: Trained state-only All-28, layer-only All-28, state+layer All-28, and state+layer Random-4 on four direct GPUs; froze validation checkpoints, global 99/98/95 trajectory thresholds, and 0-27 versus 16-27 windows; then scored only both state+layer arms in one aggregate test pass.
- Result: Final contract `264a9407e5acb35e19bd4b53436ae8bf1175ad989f8c974cf7d8e859565e095b`. Validation selected Random-4/full 0-27 at 0.9900 preservation and 0.3950 recall. Test is 0.9875/0.4050 (precision 0.9701, median trigger layer 5). Shared All-28 is 0.9925/0.3925. Mean test layer AUROC is 0.8769 primary versus 0.8737 independent. The primary dataset preservation/recall spreads are 0.0300/0.7600, versus Phase 50's 0.0250/0.7250. Readiness is NO because dataset preservation spread did not improve.
- Evidence saved: `analysis/dense_failure_stage1/shared_global_gate/` contains four configs/checkpoints/histories, all validation/test trajectories, thresholds, fixed/independent baselines, calibration tables, five inspected figures, decision summary, and a 33-file passing hash manifest.
- Failure or issue: The first validation-freeze attempt stopped before emitting selected operating points because the Phase-50 fixed-layer source table also contains `sequential_layer_specific` rows. The import was restricted to `layer_14/21/27`; the entire old contract `6dd970d9...fa59f4` output is quarantined at `analysis/dense_failure_stage1/shared_global_gate.invalid_6dd970d9`. The action was refrozen and rerun from training; no checkpoint or score crossed contract hashes.
- Lesson learned: Parameter sharing and one global trajectory threshold repair aggregate validation-to-test preservation drift without repairing task-dependent score semantics. Layer identity adds negligible ranking benefit over state-only, and sequential gating does not clearly beat the validation-selected shared fixed L27.
- Next implication: Stop this action and do not connect treatment. Any stronger calibration/global-risk formulation is a separately authorized research action.
