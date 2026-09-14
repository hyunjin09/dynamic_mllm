# Phase 76: Closed-Loop Trajectory-Set Stage-2 Memory

## Current Objective
Train the frozen SharedReadWriteRouter as a genuinely closed-loop Stage-2 policy with one exact trajectory-set marginal loss per P90-triggered UID, then evaluate the full-refit checkpoint on the established 19,960-row external protocol.

## Active Constraints
- Freeze Qwen2.5-VL-7B snapshot `cc594898...cfb5`, Robust ALL-source Stage-1 P90, trigger/action/executor semantics, LMMS scorers, and the exact ChartQA/TextVQA/MMMU-Pro/POPE manifests.
- Use all 569 eligible UIDs and all 4,948 unique replay-valid programs; Dense-C receives only all-FULL preservation, Dense-W retains every valid corrective route.
- Keep the existing depth-agnostic READ/WRITE representation; do not add layer/source IDs, Stage-1 score, history, preference heuristics, a program decoder, or external-set tuning.
- Exact route-state replay, exact marginal-loss numerical/gradient parity, group-disjoint schedule selection, full refit, and complete four-family evaluation are mandatory.
- Use direct execution and all four GPUs where the workload is parallel; place the large routed-state cache under the allowed `/mnt/hyemin/qwen_train_eval` root and expose it through a narrow output-root symlink.

## Current State
- Done/stopped: frozen protocol, test-first implementation, exact 4-GPU replay/cache construction, internal group-disjoint schedule selection, full-corpus refit, complete 19,960-row external evaluation, and artifact audit.
- Result: Closed-loop W-to-C/C-to-W/net is `0/8/-8` at accuracy `0.780160`, below Dense (`0.780561`) and Open-loop Program (`0.780311`, `3/8/-5`), while improving preservation versus Sequential-A (`3/19/-16`).
- Blocked: none.
- Most recent useful observation: exact routed-state feedback plus set-valued trajectory supervision did not produce any external rescue; the fixed diagnostic rule identifies on-policy state-distribution shift as the most consistent next bottleneck.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-74 full program corpus has 569 UIDs, 4,948 routes, and exact replay parity | `analysis/dense_failure_stage2/polar_suffix_program/` | Supplies the complete frozen trajectory identities and expected outcomes | confirmed |
| Open-loop Program achieved W-to-C/C-to-W/net 3/8/-5 versus Sequential-A 3/19/-16 | `analysis/dense_failure_stage2/polar_suffix_program/metrics/full_benchmark_summary.csv` | Establishes the baselines and preservation/rescue bottleneck | confirmed |
| Phase-75 W-to-C at rank 1/8 is 3/33; 30 ranking versus 463 generation failures | `analysis/dense_failure_stage2/program_beam_oracle_audit/metrics/benchmark_breakdown.csv` | Makes a fixed-trigger open-loop retry indefensible but leaves closed-loop feedback unresolved | confirmed |
| Current executor exposes actual pre-action text/visual states to an online selector | `binary_policy/executor/four_action.py` | The intended closed-loop inference contract is implementable without changing action semantics | confirmed |
| Four RTX 6000 Ada GPUs are idle; repository filesystem has only 18 GiB free while `/mnt` has 39 TiB | current machine audit | Requires an external cache target but does not constrain compute | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Sequential-A local CE | 3 rescues and 19 regressions | supported: local single-label imitation is poorly aligned | Phase-69/70 evidence | Preserve route coherence and test actual state feedback | another unchanged local-CE refit |
| Open-loop suffix program | 3 rescues and 8 regressions | supported for this method: preservation improved, corrective transfer did not | Phase-74/75 evidence | Do not infer closed-loop failure from a trigger-only decoder | another fixed-trigger program rerank/retrain |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Closed-loop trajectory-set policy | Recomputes actions on actual routed states while optimizing probability mass on one coherent successful route | Whether feedback plus set supervision repairs treatment transfer | high | completed negative |
| Minimal representation enrichment | Phase-75 failures predominantly lacked any correct open-loop beam candidate | Whether trigger-state representation limits support | high | deferred; not this authorized action |
| On-policy state relabeling diagnostic | Offline successful states may differ from policy free-run states | Whether exposure shift dominates after strong internal fit | medium/high | recommended but not authorized |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: test whether actual routed-state feedback plus a per-UID trajectory-set target transfers corrective behavior without losing Dense-C preservation.
- Relevant memory item used: Phase-75's 463/493 open-loop generation failures and Phase-74's improved preservation but unchanged rescue count.
- Confirmed observation: the full corpus and exact online executor are present and four GPUs are available.
- Unverified interpretation: updating the same representation on each actual routed state may recover programs that a single trigger snapshot cannot generate.
- Diagnosis: unknown
- Viable alternatives considered: requested closed-loop trajectory-set training; minimal representation enrichment; on-policy state-distribution diagnosis.
- Chosen action: execute the user-specified closed-loop trajectory-set full train/refit/evaluation as one combined-formulation test.
- Strongest objection: feedback and loss change together, and Phase 75 makes a representation limitation plausible; a negative result cannot isolate those components.
- How this differs from failed attempts: the policy observes each state produced by its prior action, and the loss marginalizes complete successful trajectories per UID instead of imitating routes independently or decoding a suffix once.
- Automatic execution authorized: yes
- Authorization basis: user explicitly requested execution of `plans/closed_loop_trajectory_set_stage2_full_train_eval_plan.md`; the required independent review returned `stable` with medium confidence.
- Stop condition: all 4,948 routes pass cache/replay validation, objective tests pass, internal schedule and full 569-UID refit complete, all 19,960 external rows complete, metrics/diagnostics and exactly one unexecuted recommendation are saved; or a mandatory validity gate fails.

## Latest Research-Action Result
- Action taken: executed the complete frozen closed-loop trajectory-set replay, exact marginal-loss training/refit, and full four-family external evaluation.
- Result: contract `00639e7e...5f6a`; 569 UIDs, 4,948 programs, 35,565 unique exact-prefix states, and 69,178 state references passed with zero quarantines. Internal epoch 1 was selected from 20 (`0.425405` dev loss) and the full 569-UID one-epoch refit completed. All 19,960 evaluation rows completed with exact Dense/Stage-1 and state-feedback parity. Closed-loop accuracy was `0.780160` versus Dense `0.780561`, with `0/8/-8` W-to-C/C-to-W/net. ChartQA and TextVQA each contributed four regressions; MMMU-Pro and POPE were unchanged.
- Evidence saved: `analysis/dense_failure_stage2/closed_loop_trajectory_set/`, especially `summaries/closed_loop_trajectory_set_full_eval_summary.md`, `metrics/full_benchmark_summary.csv`, `metrics/stage1_stage2_funnel.csv`, `training/frozen_epoch_selection.json`, and `artifact_manifest.json`.
- Failure or issue: this exact formulation is not a deployment winner. Only 39/496 triggered Dense-W samples received any non-FULL action and none rescued, despite median best-route geometric action probability `0.7269`; internal trajectory likelihood did not transfer to free rollout.
- Lesson learned: actual state feedback plus per-UID successful-trajectory marginalization improves preservation over Sequential-A but does not repair corrective transfer. Under the plan's fixed gate, on-policy state-distribution shift is the most consistent next bottleneck; the combined experiment does not isolate feedback, representation, and offline-state support causally.
- Next implication: stop Phase 76. The single unexecuted recommendation is one separately authorized on-policy state-distribution diagnostic/relabeling experiment with architecture and Stage-1 fixed.
