# Phase 58: Stage-2 V1 Training Revised Memory

## Current Objective
Train the first shared post-trigger four-action Stage-2 router on the frozen 39 Corpus-A preservation and 698 Corpus-B single-fixable train samples, then evaluate one selected final checkpoint by validation-only free rollout. Stop before V1.5 or test evaluation.

## Active Constraints
- Follow `plans/stage2_v1_training_plan_revised_698_first.md` exactly: frozen Qwen/Stage-1/trigger map/executor, Corpus A+B only, plain CE, no layer embedding, Stage-1 latent, class weighting, focal loss, MCTS Corpus C, or dataset-specific router.
- READ and WRITE must use the actual current routed visual-token sequence. Phase-56 compact `visual_mean` tensors are provenance evidence only and cannot substitute for the full sequence.
- Training must be W-sample-balanced with one uniformly sampled successful route/UID and positive-anchored Random-4 states; C:W draws are approximately 1:2.
- Validation free rollout is primary. Do not open test, retune Stage-1, or automatically run V1.5.
- This direct-execution host has four occupied RTX 6000 Ada GPUs. The user explicitly permits sharing; every launch must record live headroom and pass the prospective one-rank peak-memory smoke first.

## Current State
- Done: Read the revised plan, Phase-57 handoff, promoted lessons, environment/GPU policies, and current executor paths.
- Done: Confirmed `capture_four_action_route` exposes actual pre-layer text and visual-token states, and `capture_online_four_action_route` selects from actual routed states; a native-full-row option is required for current-dense parity.
- Done: Independent research review returned `revise`: keep online exact replay, but freeze update/optimizer/seed/final-checkpoint rules and require a worst-case one-rank replay+backward memory/time smoke.
- Done: Implemented and tested the shared READ/WRITE router, exact routed-token replay sampler, native-full-row online rollout, fixed 3,144-update schedule, final-update-only selection, and finite-value guards.
- Done: Passed implementation and 48-W/24-C overfit gates, trained the complete 39-C/698-W V1, and opened the frozen 800-sample validation split once.
- Done/stopped: Validation completed for all 235 triggered UIDs and all 800 aggregate UIDs. No V1.5, test evaluation, Corpus C, Stage-1 change, or external evaluation ran.
- Blocked: No.
- Most recent useful observation: V1 yields 5 W→C and 0 C→W (+0.625 accuracy points) but remains 98.397% FULL post-trigger, acts immediately at the trigger for 96.30% of intervened samples, and never intervenes on validation GQA.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 70.3% of W samples are delayed-only and naive B is 96.17% FULL | `analysis/dense_failure_stage2/single_label_audit/` | Requires post-trigger FULL timing states and sample-balanced sampling | confirmed |
| Phase-56 feature schema contains only three compact means | `analysis/dense_failure_stage2/full_corrective_labels/states/feature_schema.json` | Rules out the one-token `visual_mean` shortcut for cross-attention | confirmed |
| Exact route replay retains full pre-layer text/visual token tensors | `binary_policy/executor/four_action.py` | Enables online exact-state training without regenerating labels | confirmed |
| Validation trigger map contains 800 rows, including 212 triggered W and 23 triggered C | `analysis/dense_failure_stage1/trigger_map/manifests/trigger_map_val.jsonl` | Defines the held-out free-rollout population | confirmed |
| Four GPUs each showed about 31 GiB live headroom despite existing jobs | live `nvidia-smi` at 2026-09-01 11:53 KST | Shared execution was feasible and was confirmed by the accepted one-rank memory smoke | confirmed |
| Worst-case 1,792-token replay/backward peaks at 16,778 MiB reserved and retains 13,849 MiB live free | `analysis/dense_failure_stage2/v1_training_revised/smoke/implementation_smoke.json` | Validates shared-GPU execution and exact native dense parity | confirmed |
| Overfit final non-FULL recall 0.7292; RO/WO/IGNORE recall 0.50/1.00/0.6875 | `analysis/dense_failure_stage2/v1_training_revised/smoke/overfit_gate.json` | Shows the router/feature path can learn all actions locally | confirmed |
| Validation transitions W→C/W→W/C→C/C→W = 5/395/400/0 | `analysis/dense_failure_stage2/v1_training_revised/free_rollout/overall_metrics.json` | Establishes small positive net correction with perfect preservation | confirmed |
| Post-trigger actions are 98.397% FULL; 96.30% of intervened samples act immediately; GQA has zero interventions | `free_rollout/action_behavior.csv`, `diagnostics/collapse_check.json`, `free_rollout/dataset_breakdown.csv` | Qualifies the positive result as narrow and timing-collapsed | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| First full run used BF16 autocast inside the small router | Aggregate epoch-5 loss became NaN at global update 1,310; no checkpoint emitted | suspected router numerical instability; exact source unknown | `analysis/dense_failure_stage2/v1_training_revised_failed_bf16_router_83d647b2/training/failure_report.*` and all 698 Phase-56 routed summary shards finite | Run router in FP32 with state/logit/loss/gradient/parameter finite guards; keep Qwen BF16 and scientific contract unchanged | Do not use the failed trajectory or accept non-finite aggregate logs |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Online exact route replay per training draw | Preserves actual full token states and epoch route resampling | Directly tests the specified router | high | provisional winner, gated by memory smoke |
| Exact finite-schedule full-token cache | Preserves state semantics if storage is sufficient | Fallback if repeated online replay is infeasible | high storage | fallback only; not automatic |
| Treat `visual_mean` as a one-token sequence | Existing compact features are cheap | Avoids Qwen replay | low | rejected: query-independent degenerate READ attention |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: completed. The simplest shared router produces a small positive net correction, but broad action recall, wait timing, and GQA coverage remain unresolved.
- Relevant memory item used: Phase 57 showed delayed correction and severe route/FULL duplication, selecting sample-balanced timing-aware Random-4.
- Confirmed observation: exact full visual-token states are available only by replaying the frozen executor; compact means cannot implement the READ contract.
- Confirmed interpretation: 698 single-fixable W samples are sufficient for a proof-of-direction result (+5 net, zero C harm), but not for broad corrective coverage under this V1 sampler/loss/router.
- Diagnosis: supported for near-FULL and immediate-intervention collapse; the relative roles of data diversity, label ambiguity, sampler/loss imbalance, and exposure shift remain unknown.
- Evidence path if diagnosis is not unknown: Phase-56 feature schema and `binary_policy/executor/four_action.py`.
- Viable alternatives considered: online exact replay; exact finite-schedule cache; invalid compact-mean shortcut.
- Chosen action: stop after the completed V1. Preserve the positive-but-narrow result and do not automatically add Corpus C or change architecture/loss/timing.
- Strongest objection: repeated Qwen replay is expensive and final-checkpoint selection gives a less sharp teacher-forced generalization diagnosis; a finite exact cache is the runner-up only if the mandatory smoke shows online execution infeasible.
- How this differs from failed attempts: prior routers used compact/upfront summaries or collapsed supervision; this action uses actual routed token sequences, sample-balanced route resampling, and a prospective fixed final checkpoint.
- Automatic execution authorized: exhausted; a new plan and explicit user request are required for any follow-up.
- Authorization basis: explicit user request to perform the revised plan; explicit user permission to share all four occupied GPUs.
- Stop condition: implementation and overfit gates, one fixed full A+B training run, validation-only free rollout, collapse/data-sufficiency diagnosis, all required artifacts, and compact state update are complete; no V1.5 or test.

## Latest Research-Action Result
- Action taken: Trained the frozen shared Stage-2 V1 on Corpus A+B and executed validation-only free rollout.
- Result: Implementation/overfit gates passed. Final train diagnostic remained near-FULL (0.0315 non-FULL recall). Validation accuracy rose from 0.5000 to 0.50625 through 5 W→C and 0 C→W; C→C preservation is 1.0.
- Evidence saved: `analysis/dense_failure_stage2/v1_training_revised/` under contract `8f090f7d40faa1f791eda9ae0ac579365c3b7ea0282ac7c402e9831ce52bdb14` and checkpoint SHA-256 `6cb7638af1dbc81dc59c4dfb0d53e0f711d5ac01385efa95416fb78935621a24`.
- Failure or issue: The accepted run is numerically clean, but 98.397% of post-trigger actions are FULL, 96.30% of intervened samples act at the trigger, WRITE_ONLY is never selected, and GQA receives no non-FULL decision. An earlier BF16-router attempt was quarantined after NaN loss.
- Lesson learned: Exact routed full-token inputs plus local overfit learnability do not guarantee broad population action separation or learned waiting. Positive net correction can coexist with severe behavioral collapse.
- Next implication: Stop. V1.5 is not automatic. Any next plan must prospectively discriminate data diversity from sampling/loss/timing or exposure-shift limitations and preserve the no-test/no-Corpus-C boundary until authorized.
