# Phase 82: Counterfactual Effect Identifiability Memory

## Current Objective
Execute `plans/counterfactual_effect_identifiability_plan.md` to test whether frozen READ/WRITE full-suffix utility becomes identifiable from exactly one layer of action execution, comparing pre-state, individual post-state, paired post-state, and within-state delta representations.

## Active Constraints
- Use all 1,413 UIDs / 1,385 image groups / 15,185 exact Step-A dense states and the inherited Step-B five-fold image-group-disjoint registry.
- Keep `U_READ = q_FULL - q_WRITE_ONLY` and `U_WRITE = q_FULL - q_READ_ONLY`; do not regenerate or redesign labels.
- New predictor inputs stop immediately after the current decoder layer and exclude suffix states, final-answer logits, final correctness, dataset/source IDs, routes, and future actions.
- Require same-prestate branching, exact action semantics, deterministic replay, live/cached pre-state identity, and exact FULL-to-canonical post-state parity before bulk extraction.
- Run the 35,565 exact routed states only as the plan-defined secondary selection-biased analysis and preserve exact prefix identity.
- Use four direct RTX 6000 Ada GPUs; no Slurm. Store large tensor artifacts under `/mnt/hyemin`.
- Stop after the required identifiability evidence and exactly one next-method recommendation. Do not train or evaluate a deployment router.

## Current State
- Done: froze contract `a70e921a...2705`; the stratified dense/routed smoke passed deterministic repeatability, same-prestate/action-bit checks, and exact FULL-to-canonical next-state parity, including layer 27.
- Done: extracted and globally validated 15,185 dense states / 45,555 branches and 35,565 routed states / 106,695 branches with no missing or duplicate state.
- Done: completed all 570 primary OOF tasks, 60 routed-secondary OOF tasks, and 60 Dense-to-routed transfer tasks on four direct GPUs under the inherited five-fold image-group-disjoint registry.
- Done: all 96 declared repository artifacts verify; final artifact manifest `3a0eb918...143e`; 52 focused tests and bound-module compilation pass.
- Done: READ, WRITE, and joint conclusions are frozen as Case D. The plan's only unexecuted recommendation is a two-layer / short-horizon counterfactual-identifiability audit.
- Blocked: none.
- Most recent useful observation: one-step local action effects remain weak predictors of full-suffix utility. Dense OOF best READ/WRITE Spearman is 0.0773/0.0421; routed token OOF is 0.1290/0.0726, and Dense-to-routed token transfer is 0.1141/0.0699.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Complete dense/routed state and utility censuses with exact replay parity | `analysis/predictability_generalization/stepA_measurement/` | Fixes the populations and targets without rerunning suffix searches | confirmed |
| Five-fold group-disjoint registry and frozen training contract | `analysis/predictability_generalization/stepB_id_learnability/` | Prevents a new split or capacity explanation | confirmed |
| Immediate branch post-states are absent from Step A | Step-A state payload/schema audit | Requires fresh one-layer execution; pre-state reuse alone cannot answer the question | confirmed |
| Existing executor exposes unified action-isolated target-layer execution from one captured baseline | `binary_policy/executor/four_action.py` | Supports a narrow stop-after-one-layer implementation | confirmed |
| Independent review verdict is stable | `/root/phase82_identifiability_review` | Confirms full plan is the discriminating action if parity is a hard preflight gate | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial smoke with only GPU 0 visible | Frozen runtime expected four visible devices and aborted | supported configuration mismatch | smoke launcher log | Keep all four GPUs visible and select the local device by rank | Do not hide devices under the four-device frozen contract |
| First one-step helper path | Native-causal baseline and full compact-state parity were not handled exactly | supported implementation defect | focused executor tests and pre-final smoke | Reuse the exact Step-A target-layer action path and validate FULL against the canonical post-state | Do not substitute a materialized/unified approximation |
| Direct-path analysis CLI invocation | `dense_failure_stage2` import failed before artifact access | supported invocation-path issue | terminal traceback | Invoke the unchanged bound script as a Python module; do not modify bound code after freezing | Do not run the analysis file directly under this contract |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Full dense plus routed one-layer extraction and fixed OOF ladder | Directly matches the authorized plan | Whether post/counterfactual effects identify utility and whether it survives routed states | high | completed: Case D |
| Dense-only analysis | Answers the primary question more cheaply | Omits the required routed robustness check | medium | rejected by plan |
| Reuse pre-state and suffix-output artifacts only | Avoids new GPU extraction | Cannot construct leakage-free immediate post-state inputs | low | rejected |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: the authorized one-step identifiability audit is complete; the remaining bottleneck is that even exact one-step alternatives do not expose stable full-suffix READ/WRITE utility.
- Relevant memory item used: Phase 81 showed that another pre-state local-utility head is not defensible; Step B fixed the weak in-domain reference and the fold/training contract.
- Confirmed observation: READ delta improves only nominally over PRE (0.0773 versus 0.0626; 95% group-bootstrap difference CI includes zero), WRITE pair remains 0.0421, token comparisons remain weak, and strong correctness-flip ranking is below chance.
- Unverified interpretation: a bounded longer horizon may expose downstream utility that one layer cannot; this was not tested.
- Diagnosis: supported Case D for one-step identifiability; the cause of weak identifiability remains unknown.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage2/counterfactual_effect_identifiability/`.
- Viable alternatives considered: no further action; two-layer / short-horizon audit; deployment probe-and-route controller.
- Chosen action: stop Phase 82 and retain exactly one unexecuted recommendation: a two-layer / short-horizon counterfactual-identifiability audit.
- Strongest objection: routed OOF READ token Spearman reaches 0.1290, but it remains weak, selection-biased, and does not establish dense-primary or transfer identifiability.
- How this differs from failed attempts: it would add only bounded horizon, while preserving targets, folds, representations/controls, and the no-deployment boundary.
- Automatic execution authorized: no
- Authorization basis: the current plan's stop rule is satisfied; any follow-up is a separate research action.
- Stop condition: satisfied; no deployment, external evaluation, router retraining, MCTS, or longer-horizon experiment was run.

## Latest Research-Action Result
- Action taken: complete one-step READ/WRITE counterfactual-effect identifiability audit on the full dense population, then the required routed-state OOF and Dense-to-routed transfer analyses.
- Result: Case D for READ, WRITE, and jointly. Dense OOF MLP READ PRE/FULL/OFF/PAIR/DELTA/PAIR+DELTA is `0.0626/0.0533/0.0552/0.0523/0.0773/0.0568`; token is `0.0694`. WRITE is `0.0354/0.0364/0.0399/0.0421/0.0380/0.0399`; token is `0.0383`. All bootstrap gains over the relevant baselines include zero.
- Evidence saved: `analysis/dense_failure_stage2/counterfactual_effect_identifiability/`, contract `a70e921a...2705`, artifact manifest `3a0eb918...143e`.
- Failure or issue: no scientific execution failure. Three implementation/invocation defects were fail-closed and repaired or avoided before the accepted run.
- Lesson learned: exact one-layer post-states and within-state action differences do not make full-suffix READ/WRITE utility reliably identifiable under this representation and capacity ladder.
- Next implication: do not build a one-layer speculative probe-and-route controller. If separately authorized, the smallest remaining discriminator is the documented two-layer / short-horizon audit.
