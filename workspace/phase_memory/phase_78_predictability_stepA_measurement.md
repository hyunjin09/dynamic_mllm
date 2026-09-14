# Phase 78: Predictability Step-A Measurement Memory

## Current Objective
Construct and freeze the full Stage-1 dense-failure census and the primary dense plus secondary routed Stage-2 four-branch utility measurements required by `plans/predictability_phase_stepA_measurement_construction_plan.md`. Stop after measurement/readiness reporting without training a predictor or router.

## Active Constraints
- Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 28 layers, current dense prompts/greedy decoding/LMMS correctness, robust ALL-source head, strict P90 threshold `0.9061332901863008`, and current same-layer four-action semantics are frozen.
- Internal population is the complete 6,399 Historical-train plus 4,000 Canonical population; Stage-2 dense includes every P90-triggered UID and every layer from first trigger through 27.
- No outcome filtering, utility thresholding, probe/router training, MCTS, suffix search, OOD evaluation, or predictability claim is authorized.
- Use four direct GPUs after live occupancy checks; store large raw state caches under `/mnt/hyemin` and compact manifests/reports in the repository.
- Preserve the dirty user-owned worktree; touch only new Phase-78 files and narrowly required shared code/tests.

## Current State
- Done: froze contract `6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d`; constructed the complete 10,399-sample / 291,172-layer-state Stage-1 census; measured all 15,185 planned dense-origin states and 35,565 unique routed states under all four actions; finalized every required audit and report.
- Done: all readiness gates passed, all 41 compact artifacts match their recorded hashes, and all 1,413 external dense-state files remain present.
- In progress: none. Step A is complete and stopped at its required boundary.
- Blocked: none.
- Most recent useful observation: controlled utility has broad signed support in both censuses, but Step A contains no evidence yet that hidden states predict that utility.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Historical train contains 6,399 complete UIDs; Canonical contains 4,000; no overlap | `analysis/dense_failure_stage1/layerwise_failure_probe/split_manifest.jsonl`; current dense and data-scale manifests | Fixes the 10,399-sample Stage-1 population and permits exact artifact reuse | confirmed |
| P90 has 1,413 triggered UIDs and 15,185 post-trigger dense states | Phase-64 P90 rows in `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/trigger_maps/` | Fixes the primary Stage-2 census before outcomes | confirmed |
| Routed corpus has 569 UIDs, 4,948 programs, 35,565 exact states and passed exact replay | `analysis/dense_failure_stage2/closed_loop_trajectory_set/corpus/corpus_summary.csv`; `work/replay_completion.json` | Fixes the secondary routed-state population and its exact identity | confirmed |
| Existing likelihood implementation frequency-weights TextVQA accepted answers | `scoring/reference_likelihood.py` | Provides an established, prospectively reusable continuous-q aggregation | confirmed |
| ChartQA correctness accepts a numeric tolerance not representable as a finite answer-string set | `scoring/benchmark_metrics.py` | Requires explicit limitation: continuous q uses annotated gold while LMMS correctness retains relaxed semantics | confirmed |
| Primary dense four-branch census completed 15,185/15,185 states and 60,740/60,740 branches | `analysis/predictability_generalization/stepA_measurement/stage2_dense/` | Establishes complete controlled local utility labels for every planned P90 post-trigger dense state | confirmed |
| Secondary routed census completed 35,565 unique states / 142,260 branches after deduplicating 69,178 occurrences | `analysis/predictability_generalization/stepA_measurement/stage2_routed/` | Establishes the separate on-route utility census with exact state and anchor parity | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Proposed max-over-unique-reference q | Discards TextVQA answer multiplicity and therefore diverges from its EvalAI consensus structure | supported | independent Phase-78 research review plus `scoring/reference_likelihood.py` | Use one weighted-logsumexp-over-token-mean rule: singleton weight for GQA/ChartQA, empirical normalized-reference weights for TextVQA | Do not silently use max-over-unique references |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Hash-validated reuse plus fresh stratified dense replay | Existing artifacts cover every internal UID under the same runtime; fresh replay tests the live artifact/runtime link | Stage-1 census without redundant full regeneration | low | promising |
| Full Stage-1 regeneration | Literal re-execution would independently recreate every dense feature | Avoids artifact reuse assumption | high | rejected as dominated if replay parity passes |
| Full dense and routed four-branch measurement | Directly implements the approved controlled utility design | Creates S2-Dense and S2-Routed | high | completed |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: build a provenance-safe, evaluator-grounded measurement pipeline at full scale; the main bottlenecks are cross-task continuous-q semantics and exact routed-prefix reconstruction.
- Relevant memory item used: Phase 77 established that routed exposure and geometric route likelihood did not imply deployment recall, so this phase measures controlled local utility without assuming predictability.
- Confirmed observation: exact dense artifacts and exact Phase-76 routed-state identities exist for the complete planned populations.
- Unverified interpretation: continuous local utility will have enough nonzero/sign diversity for a later predictor.
- Diagnosis: unknown
- Viable alternatives considered: full dense regeneration; validated dense reuse plus live replay; narrowing to prior successful UIDs.
- Chosen action: validate/reuse exact dense artifacts with a fresh stratified replay, then execute every planned dense and routed four-action branch with FULL suffix under one frozen contract.
- Strongest objection: token likelihood cannot exactly encode ChartQA's infinite numeric tolerance set, so annotated-gold q may disagree with relaxed correctness for near-numeric answers.
- How this differs from failed attempts: it constructs controlled per-state outcome measurements and makes no claim that prior searched routes, labels, or trained routers identify utility.
- Automatic execution authorized: yes
- Authorization basis: user explicitly requested execution of the complete Step-A plan.
- Stop condition: all required censuses, parity/reproducibility/algebra audits, reports, artifact hashes, and `READY_FOR_STEP_B` decision are frozen; no Step-B training starts.

## Latest Research-Action Result
- Action taken: constructed and validated the complete Stage-1 census plus primary dense-origin and secondary exact-routed four-branch Stage-2 outcome measurements.
- Result: PASS. Stage 1 contains 10,399 UIDs, 9,982 image groups, and 291,172 layer states. Strict P90 triggers 1,413 UIDs. Primary Stage 2 contains 15,185 states / 60,740 branches; secondary Stage 2 contains 35,565 unique states / 142,260 branches from 69,178 route occurrences.
- Evidence saved: `analysis/predictability_generalization/stepA_measurement/`, especially `summaries/stepA_measurement_summary.md`, `summaries/stepB_readiness.md`, and `artifact_manifest.json`.
- Failure or issue: none affecting validity. The retained limitation is that ChartQA's relaxed numeric correctness interval cannot be represented by a finite answer-string likelihood, so continuous q uses the literal annotation while discrete correctness remains LMMS-exact.
- Lesson learned: controlled READ, WRITE, and interaction utility is non-degenerate and signed across both dense-origin and routed states; routed-state and dense-state distributions must remain separate. This is measurement evidence only, not a predictability result.
- Next implication: `READY_FOR_STEP_B = true`, but Step B has not started and requires a separate authorized action.
