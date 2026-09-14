# Phase 55: Trigger-Conditioned Corrective Search Pilot Memory

## Current Objective
Measure bounded corrective support for a prospectively frozen 120-sample diagnostic subset of the Phase-54 train triggered Dense-W cohort, using exhaustive single interventions followed by sequential trajectory-conditioned MCTS only for unresolved samples.

## Active Constraints
- Use only the 1,881 Phase-54 train triggered Dense-W handoff rows; no validation/test labels or 39 triggered-C search.
- Preserve the selected Shared Random-4 Stage-1 gate, strict trigger threshold, current LMMS targets, Qwen snapshot, and four-action executor semantics.
- Use all four direct RTX 6000 Ada GPUs while live-idle; no Slurm exists.
- Stop after the 120-sample pilot, successful-route state replay, metrics, compute projection, and decision summary.
- Do not run the remaining 1,761 samples, train Stage 2, change Stage 1, resume W-to-C repair, or run external evaluation.

## Current State
- Done: Read the active plan, Phase-54 memory/contract/handoff, relevant Phase-53 executor evidence, machine-local runtime policy, and live GPU state.
- Done: Confirmed the exact 1,881-row source population and its dataset×trigger-depth cells; all 12 cells can support the prospective allocation with distinct image groups.
- Done: Completed the required independent review and revised the prospective rollout protocol before execution.
- Done: Froze contract `6b4eb6de4573a0f06397304721483348523b78a1ce905c5076e49763f15afa2c` after 46 focused/regression tests and a passing independent code/protocol review.
- Done: Four direct GPU workers passed the 12-row smoke and completed all 120 frozen pilot UIDs exactly once with no failures or partial-rank success.
- Done: Aggregated all required single/MCTS metrics, five figures, compute projection, 360 exact successful-route replays, 7,776 compact routed state rows, and a 106-file SHA-256 audit.
- Stopped: No remaining 1,761-row search, triggered-C search, Stage-2 training, Stage-1 change, validation/test search, or external evaluation ran.
- Blocked: No.
- Most recent useful observation: 35/120 samples were single-fixable and another 22/120 were MCTS-only fixable; Fixable@100/200/300 was 52/56/57, so the frozen 0.01 absolute-gain rule selects a 200-iteration cap.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Train handoff has 1,881 triggered Dense-W and 39 triggered Dense-C rows | `analysis/dense_failure_stage1/trigger_map/decision_summary.md` | Fixes eligible search population and excluded preservation cohort | confirmed |
| Population cells are GQA 3/57/112/256, ChartQA 579/70/43/29, TextVQA 399/191/84/58 over L0/L1-8/L9-18/L19-27 | Phase-54 train triggered-W manifest audit | Supports 40/dataset diagnostic allocation but requires weighted population reporting | confirmed |
| Phase 53 passed native dense/unified-FULL and cached-prefix/full-route token parity | `analysis/dense_failure_stage1/treatment_correctability/decision_summary.md` | Supports exact pre-trigger dense prefix reuse | confirmed |
| Independent review rejected i.i.d. FULL=.70 rollouts due depth-dependent route cardinality | read-only `research_reviewer` packet, 2026-08-31 | Makes fixed-cardinality rollout coverage preferable | confirmed |
| Balanced-pilot single/MCTS-only/total support is 0.2917/0.1833/0.4750; Phase-54 cell-weighted values are 0.3980/0.1330/0.5309 | `analysis/dense_failure_stage2/corrective_search_pilot/metrics/overall_correctability.csv` | Shows sequential multi-action search adds material discovered support beyond exhaustive singles | confirmed |
| Fixable@100/200/300 is 0.4333/0.4667/0.4750; 200→300 gain is 0.0083 | `analysis/dense_failure_stage2/corrective_search_pilot/metrics/budget_saturation.csv` | Selects 200 under the prospective 0.01 materiality rule | confirmed |
| Preferred successful routes use median one non-FULL action (IQR 1–3), while 14/57 fixable samples expose multiple successful trigger actions | `metrics/route_complexity.csv`; `metrics/first_action_distribution.csv` under the Phase-55 root | Preserves single-route simplicity while showing action-label multiplicity | confirmed |
| Projected full-cohort cost is about 362,007 terminal routes, 31.15 GPU-hours, or 7.79 four-GPU wall-hours | `analysis/dense_failure_stage2/corrective_search_pilot/metrics/compute_summary.csv` | Quantifies the cost of a separately authorized scale-up | confirmed |
| All 106 artifact hashes, 57 feature shards, 360 route identities, and 7,776 state rows independently validate | `analysis/dense_failure_stage2/corrective_search_pilot/artifact_manifest.json`; post-aggregation verifier | Establishes accepted output/provenance completeness | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Prospective i.i.d. rollout proposal | Expected non-FULL count scales with suffix length (8.4 at L0 under FULL=.70) | supported protocol confound | independent review arithmetic | Cycle deterministic 2/3/4 total-intervention rollouts and report route complexity explicitly | Do not freeze an i.i.d. per-layer rollout prior for this pilot |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Exhaustive singles then cardinality-stratified sequential MCTS | Separates simple rescue from actual trajectory-conditioned multi-action rescue | Need for sequential Stage-2 policy and 200-vs-300 cap | high | completed |
| Exhaustive singles only | Cheapest baseline | Single-intervention support | medium | completed as Phase A |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: determine whether triggered Dense-W correction usually needs sequential multi-layer action search and whether a 200- or 300-iteration cap is defensible.
- Relevant memory item used: Phase 54 froze the exact train trigger cohort; Phase 53 established executor parity but found only bounded single/pair support.
- Confirmed observation: The 12 population cells are highly imbalanced, and trigger depth changes suffix length.
- Unverified interpretation: The chosen 2/3/4 intervention coverage is sufficient to reveal useful multi-action support within 300 iterations.
- Diagnosis: supported
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage1/trigger_map/metrics/trigger_by_depth_bin.csv`; independent review packet.
- Viable alternatives considered: i.i.d. FULL=.70 rollouts, uniform i.i.d. rollouts, and deterministic total-cardinality-stratified rollouts.
- Chosen action: freeze 120 image-group-unique rows (40/dataset; prospective per-depth allocation), exhaust every single non-FULL suffix intervention, then run prefix-tree UCB1 MCTS on unresolved rows with deterministic 2/3/4 total-non-FULL rollouts and binary current LMMS reward.
- Strongest objection: Limiting rollouts to low-cardinality completions may under-detect corrections requiring many interventions; results must remain bounded-search lower estimates.
- How this differs from failed attempts: Search enters at the actual Stage-1 trigger state, uses ordered trajectory-conditioned suffix execution, separates all single successes, and removes suffix-length-induced rollout cardinality.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform `plans/trigger_conditioned_corrective_search_pilot_plan.md`.
- Stop condition: all 120 rows have complete single/MCTS classifications, successful routes are replayed into compact pre-layer state shards, required metrics/figures/hashes are audited, and the decision summary answers the 12 plan questions.

## Latest Research-Action Result
- Action taken: Froze 120 image-group-unique train triggered Dense-W rows, exhaustively searched every single suffix intervention, ran ordered cardinality-stratified MCTS only on single-unresolved rows, and replayed all retained correct routes into compact pre-layer state shards.
- Result: SINGLE-FIXABLE 35, MCTS-FIXABLE 22, UNRESOLVED 63. Balanced total bounded correctability is 0.4750 and Phase-54 cell-weighted total is 0.5309. Fixable@100/200/300 is 0.4333/0.4667/0.4750; the prospective rule selects 200. GQA/ChartQA/TextVQA total support is 0.400/0.475/0.550 and L0/L1-8/L9-18/L19-27 support is 0.652/0.438/0.562/0.303.
- Evidence saved: `analysis/dense_failure_stage2/corrective_search_pilot/decision_summary.md`, all required single/MCTS metrics and figures, 57 routed feature shards, `artifact_manifest.json`, and the frozen contract/protocol.
- Failure or issue: No scientific or execution failure. The bounded 2/3/4-cardinality rollout can still miss higher-cardinality corrections, and the 120-row balanced diagnostic allocation is not a direct random population estimate.
- Lesson learned: Sequential multi-action search discovers nontrivial additional correct routes beyond exhaustive singles, but almost all discovered support saturates by iteration 200 and correctability remains materially dataset/depth dependent.
- Next implication: If the user separately authorizes full train-label generation, use the exact Phase-54 1,881-row manifest and a 200-iteration MCTS cap, preserve single-vs-MCTS provenance and triggered-C FULL supervision separation, and do not infer learned-policy generalization from oracle search support.
