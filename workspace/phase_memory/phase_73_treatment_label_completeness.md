# Phase 73: Treatment Label Completeness Memory

## Current Objective
Audit whether Phase-72 exact-state treatment labels are materially incomplete under a frozen bounded search, then recheck the unchanged lightweight probe families on audited clean labels.

## Active Constraints
- Freeze exactly 1,200 exact states: 500 old KEEP, 500 old INTERVENE, 200 old MIXED; maximum four states per UID.
- Sampling is outcome-blind apart from the old label and uses only dataset/source/layer-bin/route-source/UID-group metadata.
- Each unobserved first action receives direct FULL suffix, exhaustive one-later-nonFULL search, then MCTS capped at 200 from the actual post-action state.
- Exact replay is required for discoveries; results remain bounded-search labels.
- Use four GPUs for search. Do not retrain Stage 2, alter Stage 1, or run external evaluation.

## Current State
- Done: all 1,200 frozen states completed on four GPUs, aggregation and replay validation passed, audited labels and reports were produced.
- Done with an estimability limitation: the frozen five-fold audited binary probe cannot be computed because only four clean KEEP UID/groups remain after excluding MIXED.
- In progress: none.
- Blocked: none.
- Most recent useful observation: 496/500 old KEEP and 271/500 old INTERVENE labels became MIXED; only 4 AUDITED_KEEP, 229 AUDITED_INTERVENE, and 967 AUDITED_MIXED states remain.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| 21,071 exact states: 18,438 KEEP, 1,657 INTERVENE, 976 MIXED | `analysis/dense_failure_stage2/treatment_selectivity_separability/dataset/exact_state_manifest.jsonl` | Defines the frozen audit census | confirmed |
| RW OOF AUROC 0.5620 with weak high-precision support | `analysis/dense_failure_stage2/treatment_selectivity_separability/metrics/probe_summary.csv` | Baseline for the audited-label probe recheck | confirmed |
| 2,519 replay-valid successful routes and exact routed-state provenance | `analysis/dense_failure_stage2/robust_gate_corrective_search/routes/global_route_store.jsonl` | Supplies exact successful continuations and current evaluator provenance | confirmed |
| KEEP/INTERVENE invalidation 99.2%/54.2%, zero quarantines | `analysis/dense_failure_stage2/treatment_label_completeness/metrics/overall_completeness.csv` | Directly establishes material old-label incompleteness | confirmed |
| 2,504 new actions replayed exactly; all 1,200 old successes also replayed | `analysis/dense_failure_stage2/treatment_label_completeness/search/replay_validation.jsonl` | Validates every action-set expansion used in audited labels | confirmed |
| Audited clean support is 4 KEEP / 229 INTERVENE | `analysis/dense_failure_stage2/treatment_label_completeness/probe_recheck/probe_estimability_note.md` | Makes the frozen five-fold binary probe non-estimable | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-72 separability diagnostic | Weak treatment-label discrimination | unknown; label incompleteness remained a live alternative | `workspace/phase_memory/phase_72_stage2_treatment_selectivity_separability.md` | Audit labels before selecting a representation pivot | Do not call the weak probe a representation failure without this audit |
| First Phase-73 smoke contract `66243490…` | 8/8 states quarantined at exact representation parity | supported: the check recomputed one layer while Phase 72 batched all selected route layers; MHA batching changed reductions by at most `2.7e-5` | `analysis/dense_failure_stage2/treatment_label_completeness_failed_contract_66243490/` and the one-state batch diagnostic | Reconstruct the original extraction batch for bit-exact feature parity, then refreeze | Do not weaken to an unexplained tolerance or reuse the failed contract |
| Second preflight contract `cc0185ec…` | All eight states passed, but only forced FULL and READ_ONLY were represented | supported: smoke action census | `analysis/dense_failure_stage2/treatment_label_completeness_preflight_incomplete_cc0185ec/` | Require all four forced actions before full launch | Do not treat partial action coverage as a complete executor smoke |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Frozen 1,200-state completeness audit | Directly tests missing successful first actions | Label-limited versus representation-limited interpretation | high | completed |
| Skip audit and accept representation limitation | Cheapest | Nothing about the known label-miss limitation | low | rejected |
| Prospectively specified set-valued Stage-2 target | Matches supervision to replay-validated successful-action sets | Whether target misspecification is the primary learned-policy bottleneck | medium | proposed; requires explicit approval |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: determine whether bounded search incompleteness explains weak Stage-2 treatment separability.
- Relevant memory item used: Phase-72 Case D interpretation explicitly listed incomplete observed-valid sets as the strongest objection.
- Confirmed observation: all 21,071 Phase-72 states and their frozen representations are present and hash-bound.
- Unverified interpretation: broader bounded search will materially invalidate old labels and improve probe separability.
- Diagnosis: unknown
- Viable alternatives considered: execute the approved audit; skip it and accept representation limitation; alter sample/budget (out of approved scope).
- Chosen action: completed the exact 1,200-state audit after freezing numeric decision criteria and passing a forced-action replay smoke.
- Strongest objection: the plan left “large,” “material,” and “saturated” undefined; they are now prospectively numeric in the config before outcomes.
- How this differs from failed attempts: it expands action completeness at the exact routed state rather than fitting another head to unchanged labels.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform `plans/large_scale_treatment_label_completeness_audit_plan.md`.
- Stop condition: met. Audit/search/replay and audited labels are complete; the five-fold probe was correctly recorded as non-estimable rather than altered post hoc; one recommendation was recorded.

## Latest Research-Action Result
- Action taken: four-GPU bounded completeness audit of 3,194 unobserved first-action branches across 1,200 frozen exact states.
- Result: zero quarantines; 1,087/1,200 states gained an action. KEEP invalidation was 496/500 and INTERVENE invalidation was 271/500. Final audited counts are 4 KEEP, 229 INTERVENE, and 967 MIXED. MCTS saturation passed the frozen rule.
- Evidence saved: `analysis/dense_failure_stage2/treatment_label_completeness/`, contract `3c371c2f4f8372b36eb5965a97344adc6d4f1a76ae906009a93c72e63bff08cc`.
- Failure or issue: the audited five-fold binary probe is not statistically estimable because four clean KEEP UID/groups cannot cover five group-disjoint test folds. This is an outcome-induced support limitation, not a runtime failure.
- Lesson learned: the old observed single-label KEEP/INTERVENE target was severely overconfident; Phase-72 weak separability cannot be interpreted as a representation failure using these labels.
- Next implication: stop. The only proposed next direction is a prospectively specified set-valued treatment target; it requires explicit user authorization.
