# Phase 84: READ Short-Horizon Counterfactual Propagation Memory

## Current Objective
Execute `plans/read_short_horizon_counterfactual_propagation_plan.md` once to determine whether the frozen READ-harm target becomes identifiable after the exact READ ON/OFF difference propagates through H=1/2/4/8 common FULL layers.

## Active Constraints
- READ only; target remains `H_R = q_WRITE_ONLY - q_FULL` over the exact 15,185 dense states / 1,413 UIDs / 1,385 image groups.
- The two branches differ only at intervention layer `l`; every later executed layer is FULL. Do not fabricate states after layer 27.
- Primary emergence comparison uses H=8 common support; native support is descriptive. Reuse inherited Step-B five-fold image-group-disjoint roles, UID-balanced Huber training, and identical capacity/hyperparameters across horizons.
- Freeze pooled ON/OFF/PAIR/DELTA/PAIR+DELTA, text/visual delta ablations, and one identical token comparator at every horizon before labels are inspected.
- Require exact H=1 Phase-82 parity, exact ON-to-dense parity at every reached horizon, identical continuation traces, fresh-cache repeatability, and swapped ON/OFF execution-order parity before bulk extraction.
- Random-pair and permuted-target controls are mandatory. Routed evidence is secondary and conditional under the plan.
- Direct execution only. Use free GPUs after live inspection; GPU 0 is currently occupied and may not be oversubscribed under machine-local policy. Store large tensors under `/mnt/hyemin`.
- Stop after the H-READ-A/B/C/D decision and exactly one unexecuted next recommendation. Do not start WRITE, routing, search, deployment, or external evaluation.

## Current State
- Done: read the plan and relevant Phase-82/83 evidence; audited reusable executors, H=1 caches, population/fold manifests, environment, and live GPUs.
- Done: research-control selected the explicitly authorized fixed action in standard mode.
- Done: independent review returned `revise`; accepted fresh-cache swapped-order H=8 smoke and mandatory frozen token comparator.
- Done: froze contract `3a0d2251e23b43082227b9b05befd9a37d9b9ecc6bb40a570861ebd7affebb6a`; all smoke, parity, trace, census, and readback gates passed.
- Done: extracted 47,133 horizon state-pairs / 94,266 ON/OFF branches across all 15,185 states, and completed all 2,115 five-fold/three-seed training tasks.
- Done: aggregated fixed controls, 5,000-draw image-group bootstraps, subgroup audits, seven figures, and the H-READ-A/B/C/D decision.
- Stopped: **H-READ-D**; no tested H=1/2/4/8 representation passes the complete prospective emergence gate, and no useful high-precision subset exists.
- Blocked: none.
- Most recent useful observation: H8 DELTA reaches rho 0.1097 / AUROC 0.5483, but gains over H1 are only +0.0340/+0.0167 and both group-bootstrap lower bounds remain below zero.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Exact dense population and READ target | `analysis/predictability_generalization/stepA_measurement/` | Fixes states and `H_R` without relabeling | confirmed |
| Exact one-step branches and token caches | `analysis/dense_failure_stage2/counterfactual_effect_identifiability/` | Supplies H=1 parity oracle and reusable trainer | confirmed |
| Weak local READ structure/mechanism models | `analysis/read_harm_structure_learnability/` | Justifies the bounded propagation question | confirmed |
| Image-group-disjoint roles | `analysis/predictability_generalization/stepB_id_learnability/splits/` | Prevents state/image leakage | confirmed |
| Independent validity review | `/root/phase84_short_horizon_review` | Adds H>1 cache/order isolation and freezes token comparator | confirmed |
| Complete short-horizon result | `analysis/read_harm_short_horizon_propagation/` | Establishes H-READ-D under the frozen population, representations, horizons, and gates | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Direct analyzer invocation | Repository package was absent from the script import path | supported: module invocation resolved imports without code change | preparation traceback | Invoke the analyzer with `python -m experiments.analyze_read_short_horizon_propagation` | Do not use the direct file invocation |
| Initial aggregation | Mapping-only `canonical_hash` received an ordered state-ID list | supported: traceback at prediction-manifest hashing; test reproduced the import/implementation gap | aggregation traceback and `summaries/aggregation_provenance_note.md` | Use the dedicated canonical JSON sequence hash; scientific metrics/configuration were unchanged | Do not pass sequences to the mapping-only helper |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Bounded longer-horizon READ planning/search study | Short horizons are non-material despite descriptive effect growth | Whether useful READ-harm information requires longer planning | high | unapproved strategic follow-up |
| Stop this branch | H-READ-D rejects another one-state or short-horizon controller under this family | Prevents positive-result chasing | none | selected |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: the authorized short-horizon discriminator is complete; the unresolved question lies beyond the tested H<=8 information family.
- Relevant memory item used: Phase 82/83 local signals were weak; Phase 84 now shows that bounded H<=8 propagation does not meet prospective materiality.
- Confirmed observation: DELTA rho is non-monotone (`0.0756/0.0688/0.0646/0.1097`), H8 gain confidence intervals include zero, token H8 remains 0.0854, and precision@10% improves prevalence by only 0.0536.
- Unverified interpretation: longer planning or search may expose stable READ-harm information.
- Diagnosis: supported absence of short-horizon identifiability under this frozen family; cause beyond that is unknown.
- Viable alternatives considered: optional routed confirmation under selection bias; stop and preserve the dense-primary negative result.
- Chosen action: stop with H-READ-D. Routed confirmation is not promoted because primary dense emergence is non-material and the plan makes it optional for D.
- Strongest objection: H8 point estimates improve modestly; this is insufficient because absolute-gain, bootstrap, pair-over-single, monotonicity, and high-precision requirements do not jointly pass.
- How this differs from failed attempts: it closes the pre-registered H=1/2/4/8 propagation question rather than adding another local representation.
- Automatic execution authorized: no follow-on action.
- Authorization basis: the plan's stop rule and H-READ-D decision branch.
- Stop condition: met; no routed, WRITE, search, deployment, or external-evaluation action ran.

## Latest Research-Action Result
- Action taken: exact H=1/2/4/8 READ ON/OFF propagation extraction, fixed OOF learnability/control grid, and prospective category decision.
- Result: **H-READ-D**. H8 effect magnitude grows descriptively (median pooled norm ratio 2.268x H1), but predictive performance remains weak and non-monotone; no horizon passes the complete gate.
- Evidence saved: `analysis/read_harm_short_horizon_propagation/`, artifact manifest `1f9005bf11cf48e00b795e205aaec80370c218b4145d47f7548f68a67ad35961`.
- Failure or issue: aggregation initially called a mapping-only hash helper on a list; the exact canonical-list hash compatibility fix completed aggregation and now has a permanent regression test (64 focused tests pass).
- Lesson learned: downstream READ effects can grow in norm without becoming reliably predictable from the tested paired state representations.
- Next implication: do not train a one-state or H<=8 READ controller from this family. A bounded longer-horizon planning/search study is the sole unexecuted recommendation and requires separate authorization.
