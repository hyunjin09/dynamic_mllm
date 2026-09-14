# Phase 57: Stage-2 Single-Label Distribution Audit Memory

## Current Objective
Audit the exact Phase-56 Corpus A preservation and Corpus B successful-single supervision, quantify sample/route/state balance and ambiguity, and recommend the simplest defensible Stage-2 V1 data-loading contract. Stop before training.

## Active Constraints
- Use only the authoritative Phase-56 train corpora under `analysis/dense_failure_stage2/full_corrective_labels/`.
- Do not run Qwen inference, corrective search, MCTS, Stage-2 training, validation/test labeling, threshold changes, rollout, or external evaluation.
- Keep sample-, route-, and state-level quantities separate; use sample weighting for the primary action interpretation.
- Treat successful actions as observed alternatives, not exhaustive validity.

## Current State
- Done: Reverified all 977 Phase-56 artifact hashes under source manifest SHA-256 `a26d31e1d118e811ef107152213edcef3cd48ca3ed83ec17b18447ff490ff4d4`.
- Done: Validated exactly 39 Corpus A samples/routes/400 states and 698 Corpus B samples/7,628 routes/199,193 states under source contract `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`.
- Done: Completed the prescribed distribution, delay, ambiguity, sampling, preservation-balance, dataset/depth, and exact feature-row redundancy audits under analysis contract `32261c07880a0b38a96ce5f25c5af1c5147bb7408f085ea197b2819ad53f7bc5`.
- Done: Selected sample-balanced uniform route sampling plus S1 local timing states and a 1:2 C:W sample mix as the simplest V1 loader recommendation; no model was trained.
- Blocked: No.
- Most recent useful observation: Naive Corpus B state expansion is 96.17% FULL and overweights samples by up to 60x; 70.3% of W samples are delayed-only, so a V1 loader must both cap FULL repetition and preserve post-trigger timing context.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Successful routes/sample have mean 10.93, median 7, IQR 2-15, range 1-60 | `analysis/dense_failure_stage2/single_label_audit/metrics/sample_route_multiplicity.csv` | Rules out naive route-balanced sample weighting | confirmed |
| Sample-weighted actions are RO/WO/IGNORE 0.2666/0.2649/0.4685 | `metrics/action_distribution_sample_weighted.csv` under the audit root | Establishes action mix without route-rich sample dominance | confirmed |
| 491/698 = 0.703 are delayed-only; median delay is 13 layers | `metrics/immediate_vs_delayed.csv`; `metrics/trigger_to_intervention_delay.csv` | FULL must remain a normal action after Stage-1 handoff | confirmed |
| 2,685/4,421 sample/layer positions have multiple observed non-FULL rescue actions | `metrics/same_layer_action_ambiguity.csv` | A single route is one valid trajectory, not a unique action oracle | confirmed |
| Naive Corpus B is 191,565 FULL versus 7,628 non-FULL (25.11:1) | `metrics/naive_state_class_balance.csv` | Naive all-state CE has an obvious FULL-collapse risk | confirmed |
| Exact audit finds 90,609 duplicate semantic rows and zero semantic IDs with inconsistent feature hashes | `metrics/state_redundancy.csv` | Quantifies route-expansion redundancy while validating deterministic state identity | confirmed |
| S3+S1 reduces expected W FULL share to 0.7847 while retaining local pre/post timing | `metrics/sampling_scheme_simulation.csv` | Supports the simplest timing-aware sample-balanced V1 sampler | confirmed |
| Natural C visibility is 0.0529; 1:2 C:W requires 8.95x per-C repetition | `metrics/preservation_balance_simulation.csv` | Makes the 39 preservation samples visible without a weighted loss | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| First direct audit CLI launch | Python stopped at import with `ModuleNotFoundError: dense_failure_stage2` before tensor access | supported: direct script execution lacked the repository-root path bootstrap | initial CLI traceback; regression test `test_audit_cli_is_directly_executable_from_repo_root` | Add one tested path bootstrap to the CLI; the accepted audit then completed once | Do not invoke a repository-package CLI without a direct-execution import test |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| S3 uniform route per W sample + S1 corrective/2-pre/2-post | Equal sample weight and explicit timing context | Reduces FULL and route-multiplicity collapse risks | low | recommended for V1 |
| S3 + S2 K=2 | Most action-balanced simple candidate | Reduces FULL further | low | runner-up; loses explicit pre/post timing |
| Naive all-route/all-state | Uses every stored row directly | Simplest implementation only | low | rejected: 96.17% FULL and up to 60x sample weighting |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: choose a minimal Stage-2 V1 loader that respects delayed correction and does not let route multiplicity or FULL suffixes dominate.
- Relevant memory item used: Phase 56 explicitly warned that route multiplicity must not silently determine future training weight.
- Confirmed observation: 70.3% of single-fixable samples are delayed-only, while naive all-state Corpus B is 96.17% FULL.
- Unverified interpretation: S3+S1 and a 1:2 C:W mix will improve learned free-rollout preservation/correction.
- Diagnosis: supported for the data-geometry risks; unknown for learned-policy performance.
- Evidence path if diagnosis is not unknown: audit metrics and summaries under `analysis/dense_failure_stage2/single_label_audit/`.
- Viable alternatives considered: naive S0, sample-balanced S2 K=2/4/6, and sample-balanced S1.
- Chosen action: recommend one uniformly sampled successful route per W sample, one corrective plus up to two pre- and two post-FULL states, and C:W 1:2 sample mixing; retain alternative-action metadata but defer multi-label loss.
- Strongest objection: S1 is still 78.47% FULL and repeats exact states with alternative labels; S2 K=2 is more balanced but drops the timing context demanded by the high delayed-only rate.
- How this differs from failed attempts: the accepted audit ran only after the direct-execution import path was regression-tested; no scientific contract or source population changed.
- Automatic execution authorized: yes for this audit only.
- Authorization basis: explicit user request to read and perform `plans/stage2_single_label_distribution_audit_plan.md`.
- Stop condition: all required metrics/figures/summaries, a hash-verified artifact manifest, and a simple V1 data recommendation are complete.

## Latest Research-Action Result
- Action taken: Performed the frozen Corpus A+B single-label distribution and exact routed-state redundancy audit.
- Result: The corpus is conditionally suitable for a simple shared V1 only with sample-balanced route sampling and explicit preservation mixing; it is not suitable for naive all-route/all-state CE.
- Evidence saved: `analysis/dense_failure_stage2/single_label_audit/`, manifest SHA-256 `6da39063379032250eff55fd3b354e52885c8d7cd537dcc8569c24c5aa91d9f8`, 28 verified artifact hashes, and 18 passing focused/inherited tests.
- Failure or issue: One implementation-only import failure occurred before data access; it was regression-tested and fixed without changing scientific inputs or outputs.
- Lesson learned: The dominant Corpus B loader hazards are route-derived sample weighting, repeated FULL states, delayed correction, and legitimate alternative labels at identical entering states.
- Next implication: Stop. A separately authorized Stage-2 V1 training pilot may implement the recommended loader; do not add Corpus C or train automatically.
