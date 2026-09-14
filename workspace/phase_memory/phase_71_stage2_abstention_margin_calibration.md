# Phase 71: Stage-2 Abstention-Margin Calibration Memory

## Current Objective
Test whether one global positive best-non-FULL-minus-FULL logit margin can improve the frozen Experiment-A Stage-2 router's development net correction under a 99.5% C-preservation constraint, then conditionally perform one locked external rerun only if a nonzero margin is selected.

## Active Constraints
- Follow `plans/stage2_abstention_margin_calibration_plan.md` as one bounded research action.
- Freeze Robust ALL-source Stage-1 at strict P90, Stage-2 Experiment A checkpoint, Qwen2.5-VL, four-action semantics, generation, and LMMS-compatible scoring.
- Freeze one global candidate grid from training-side states before reading per-margin development outcomes; no dataset/action-specific margin.
- Re-execute actual sequential development rollouts for every finite margin because abstention changes downstream states.
- Select on Historical-800 only with C→C preservation >=99.5%; do not use Phase-69/70 external answer changes for calibration.
- Run one locked external four-family comparison if and only if full-development selects a nonzero margin; never retune externally.
- Do not retrain, change Stage 1, add search/MCTS/on-policy data, or add another confidence mechanism.

## Current State
- Done: Froze contract `87debbca...900b2`; collected 3,907 final-epoch routed-state margins over 1,048 draws; froze 11 global candidates before development outcomes; completed 1,210 four-GPU sequential rollouts over all 121 P90-triggered Historical-800 samples; verified exact δ=0 Phase-66 parity; completed five group-disjoint cross-fit, diagnostics, figures, and artifact audit.
- In progress: None; the authorized action is complete and stopped.
- Blocked: No.
- Most recent useful observation: every positive margin reduced rescue before it improved Net. δ=0 remains best at W→C/C→W/Net `4/1/+3`; q10 is `3/1/+2`, q25 `2/1/+1`, q40 `2/0/+2`, q50/q60 `1/0/+1`, and q70 onward `0/0/0`. All five cross-fit training folds selected δ=0.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Experiment A P90 gives Historical-800 W→C/C→W/net `4/1/+3` with 400 C and 400 W | `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/rollout/P90/overall_metrics.json` | Defines the δ=0 development baseline and selection population | confirmed |
| Experiment-A final checkpoint SHA-256 is `48536bfb...9bbc7f` | `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/training/selected_checkpoint.json` | Freezes Stage-2 weights | confirmed |
| Phase-70 external accounting is `3/119` treated-W rescues versus `19/92` treated-C regressions | `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/funnel/pooled_funnel.csv` | Motivates abstention but is forbidden for calibration | confirmed |
| All four RTX 6000 Ada GPUs are currently idle | live `nvidia-smi` on 2026-09-05 | Allows direct four-worker execution | transient confirmed |
| Frozen grid contains 400 positive margins from 3,907 final-epoch selected states; q10..q95 = 0.1563..3.8189 | `analysis/dense_failure_stage2/abstention_margin/margin_grid/` | Establishes development-independent candidates | confirmed |
| Historical-800 selects δ=0; all five cross-fit folds also select δ=0, pooled held-out `4/1/+3` | `analysis/dense_failure_stage2/abstention_margin/development/`, `crossfit/` | Rejects the global positive-margin candidate | confirmed |
| Rescue and regression maximum-margin medians are 0.6747 and 0.7210 | `analysis/dense_failure_stage2/abstention_margin/diagnostics/margin_by_transition.csv` | Router margin does not descriptively rank rescues above regressions | confirmed, sparse events |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Use compact Phase-65 state shards directly for margin logits | Shards contain only text-final/text-mean/visual-mean vectors, while the router attends over full text/visual token sequences | supported representation mismatch | feature schema and `dense_failure_stage2/v1_router.py` | Re-evaluate the frozen training schedule through Qwen and the final router | Do not approximate router margins from pooled vectors |
| First contract freeze model-integrity check | Runtime-only inventory helper omitted README and `.gitattributes`, falsely reporting a snapshot mismatch | supported validator-shape mismatch; all 16 physical hashes matched | archived `analysis/dense_failure_stage2/abstention_margin_failed_d5632750/` and direct SHA check | Compare the full Phase-66 inventory like-for-like | Do not compare differently scoped inventories |
| First grid-binding validation | Self-hash helper excluded `contract_sha256` rather than `binding_sha256` | supported implementation bug | four ranks failed before model load; regression test now passes | Use artifact-specific self-hash fields; re-freeze provenance | Do not reuse incompatible failed-freeze rows |
| First aggregation plot write | Expected `figures/` directory was absent | supported output-path bug | all rollout/selection rows complete; rerun succeeded after creating only the expected directory | Output-only repair did not alter scientific results | Do not interpret as experiment failure |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Final-checkpoint margins on the frozen final training epoch | Matches the actual A training sampler and is outcome-independent | Freezes a compact global margin grid | medium | completed |
| All unique Phase-65 oracle states | Broader state coverage | Whether sampler weighting changes grid quantiles | high; token sequences are not stored | rejected as disproportionate unless review finds a validity issue |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: complete; one global positive Stage-2 abstention margin is not supported on Historical-800.
- Relevant memory item used: Phase 70's preservation-dominated external loss motivated the direct confidence-separability test.
- Confirmed observation: positive margins monotonically suppress interventions, but they discard rescues before producing any Net improvement over δ=0; all folds select zero.
- Unverified interpretation: whether a richer Stage-2 representation or different training signal can separate treatment benefit remains unknown.
- Diagnosis: supported negative result for the one-dimensional global current-router margin; broader Stage-2 cause remains unknown.
- Evidence path: `analysis/dense_failure_stage2/abstention_margin/development/per_margin_rollout_summary.csv`, `crossfit/stability_summary.json`, and `diagnostics/margin_by_transition.csv`.
- Viable alternatives considered: no follow-on action was authorized. The plan-prescribed future direction is to revisit representation/training signal rather than retune this margin.
- Chosen action: stop at δ*=0; do not run the conditional external benchmark.
- Strongest objection: the sampler-weighted training grid may be coarse, but q10 already loses rescue without reducing the sole regression and q40 removes the regression only after losing two rescues; development outcomes cannot be used to add thresholds post hoc.
- Automatic execution authorized: exhausted.
- Authorization basis: explicit user request to perform this single plan.
- Stop condition: met; external `NOT_RUN.md` records the prohibited branch.

## Latest Research-Action Result
- Action taken: Calibrated the frozen Experiment-A best-non-FULL-minus-FULL margin on Historical-800 using a training-only grid and five-fold group cross-fit.
- Result: δ*=0. Baseline is `4/1/+3`; no positive candidate improves Net. All five folds select zero and pooled held-out results reproduce `4/1/+3`.
- Evidence saved: `analysis/dense_failure_stage2/abstention_margin/` under contract `87debbca51a6508ff9d7c149285047696648cb6f94e6ece23dd92dddff6900b2`.
- Failure or issue: two fail-closed implementation checks and one output-directory bug were repaired without development leakage; the first incompatible freeze is archived. Scientific coverage is complete.
- Lesson learned: the current router's global raw margin measures intervention willingness, but does not rank the four rescues above the one regression well enough to improve Net with a positive threshold.
- Next implication: stop. A separately authorized action should revisit the Stage-2 representation or training signal; do not retune on Historical-800 or run external evaluation for δ=0.
