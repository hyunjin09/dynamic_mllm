# Phase 70: Full-Benchmark Exhaustive Regression/Rescue Audit Memory

## Current Objective
Decompose the frozen Phase-69 full-benchmark regression into Stage-1 admission, Stage-2 intervention, treatment-quality, preservation, timing, and benchmark-specific components using only the completed paired artifacts.

## Active Constraints
- Follow `plans/full_benchmark_exhaustive_regression_rescue_audit_plan.md` as one bounded analysis action.
- Keep the Phase-69 Stage-1 head, strict P90 threshold, Stage-2 Experiment A checkpoint, executor, generation contract, samples, and LMMS-compatible correctness fixed.
- Audit ChartQA, TextVQA, MMMU-Pro, and POPE only; exhaustively include all 3 W→C and 19 C→W rows.
- Do not train, retune thresholds, run corrective search, or rerun inference unless a required diagnostic field is absent from stored evidence.
- Stop after the complete funnel, answer-change audits, bottleneck classification, and one unexecuted recommendation.

## Current State
- Done: Completed the deterministic offline audit over all 19,960 Phase-69 paired rows; generated and hash-verified every required table, all 22 answer-change records, four benchmark audits, six figures, the final summary, and one unexecuted recommendation.
- In progress: None; Phase 70 is complete and stopped.
- Blocked: No.
- Most recent useful observation: pooled non-FULL activation is similar for triggered W and C (`119/496=0.2399` versus `92/405=0.2272`), but conditional outcomes are sharply asymmetric: `3/119=0.0252` rescue success versus `19/92=0.2065` regression risk.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase 69 completed 19,960 paired rows with pooled W→C/C→W/net `3/19/-16` | `analysis/dense_failure_stage2/full_benchmark_eval/paired/all_paired.jsonl` | Defines the complete frozen audit population | confirmed |
| The paired rows expose the full Stage-1 trajectory and Stage-2 action trace | same paired file, schema `full_benchmark_eval_paired_row_v1` | Makes the requested audit reconstructible without GPU inference | confirmed |
| Development P90 expected pooled W recall `0.3572` with Historical/Canonical C preservation `0.9500/0.9003` | `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/thresholds/named_operating_points.csv` | Provides the requested calibration context without retuning | confirmed |
| Pooled Stage-2 treatment rescues `3/119` treated W but regresses `19/92` treated C | `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/funnel/pooled_funnel.csv` | Makes preservation the dominant pooled net-loss accounting | confirmed |
| TextVQA accounts for 12/19 regressions and -10/-16 net; MMMU-Pro treats 82 W with zero rescue; POPE scores never reach P90 | benchmark audits and `metrics/stage1_score_distribution.csv` | Establishes a mixed family picture rather than one universal failure stage | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| None in Phase 70 | — | unknown | — | Use the already complete paired evidence | Do not launch redundant inference |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Offline exhaustive funnel and answer-change audit | Explicitly authorized and all fields are stored | Which frozen pipeline stage dominates the negative net correction | low | completed |
| Preservation-calibrated Stage-2 abstention margin | Directly targets the 19 regressions without changing Stage 1 | Whether existing Stage-2 confidence separates harmful from useful non-FULL actions | low/medium | recommended, not authorized |
| New action/timing supervision | Could address MMMU-Pro's 0/82 W treatment success | Whether the checkpoint lacks a generalizable corrective policy | high | runner-up, not authorized |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: completed; the frozen candidate loses net correctness primarily through poor preservation after Stage-2 intervention, with MMMU-Pro treatment failure and POPE inactivity as separate family modes.
- Relevant memory item used: Phase 69 established a valid regression-dominated result and cautioned that the cause of individual regressions remained unknown.
- Confirmed observation: the current candidate yields 3 rescues and 19 regressions; TextVQA contributes 12 regressions, MMMU-Pro 4, ChartQA 3, and POPE never triggers.
- Unverified interpretation: a margin on the existing FULL-versus-non-FULL confidence may separate harmful interventions from the three rescues.
- Diagnosis: supported for descriptive pooled preservation limitation and family heterogeneity; causal action-level diagnosis remains unknown.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/`.
- Viable alternatives considered: preservation-calibrated Stage-2 abstention; new action/timing supervision; broader Stage-1 admission. The first directly targets the largest net-loss term at the lowest cost; action/timing is the runner-up if existing confidence is not discriminative; Stage-1 expansion is unsupported while treated-W success is 2.52%.
- Chosen action: completed the audit and retained exactly one unexecuted recommendation: a development-calibrated Stage-2 abstention margin under a C-preservation constraint.
- Strongest objection: existing Stage-2 confidence may not distinguish the three rescues from the 19 regressions; if so, action/timing supervision is the better next experiment.
- How this differs from failed attempts: it introduces no new model run or post-hoc threshold; it decomposes the first full external evaluation using stored traces.
- Automatic execution authorized: no further action.
- Authorization basis: the user explicitly requested execution of the named plan.
- Stop condition: all required tables, figures, benchmark audits, manifest, final summary, and one unexecuted recommendation are verified.

## Latest Research-Action Result
- Action taken: fail-closed offline audit of every Phase-69 paired trace, including the full C/W→trigger→non-FULL→outcome funnel and all 22 correctness changes.
- Result: pooled W admission is `496/4,380`, with `119/496` receiving non-FULL and only `3/119` rescued; pooled C false admission is `405/15,580`, with `92/405` receiving non-FULL and `19/92` regressed. ChartQA/TextVQA classify preservation-limited, MMMU-Pro treatment-quality-limited, and POPE inactive. WRITE_ONLY is the first action in 13/19 regressions and IGNORE in 6/19; no changed trajectory uses READ_ONLY.
- Evidence saved: `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/`; 28 recorded artifact hashes, all 22 changed UID traces, and six figures verified.
- Failure or issue: no execution failure. Interpretation remains descriptive: the traces do not causally isolate an individual action.
- Lesson learned: low external Stage-1 W admission is real (`0.1132` versus development pooled `0.3572`), but increasing admission is not defensible while current treated-W success is 2.52% and treated-C regression risk is 20.65%.
- Next implication: if separately authorized, test one preservation-calibrated Stage-2 abstention margin selected only on development data; do not use these 22 external answer changes for threshold selection.
