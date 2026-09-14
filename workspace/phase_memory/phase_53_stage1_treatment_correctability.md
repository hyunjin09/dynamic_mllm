# Phase 53: Stage-1 Gate Treatment Correctability Memory

## Current Objective
Measure whether samples admitted by the frozen Stage-1 gates are actually correctable by current-runtime four-action visual interventions, and compare dynamic suffix treatment with fixed-L27 detect-and-replay without training Stage 2.

## Active Constraints
- Follow `plans/stage1_gate_treatment_correctability_plan.md` and stop after oracle-feasibility analysis.
- Use the current 800-validation/800-test split, current dense labels, Qwen2.5-VL snapshot, and LMMS-Eval task scorers.
- Do not retrain Stage 1, tune thresholds on treatment outcomes, reuse historical route correctness as authority, run broad MCTS, train Stage 2, resume W2C repair, or run external evaluation.
- Dynamic treatment starts from the cached all-FULL state at the frozen first-trigger layer; fixed L27 performs a second treatment pass from layer 0.
- Use all four direct GPUs only while they remain idle; no Slurm exists on this server.

## Current State
- Done: Read the plan, Phase-52 memory, promoted lessons, current executor/search code, runtime contract, split/score sources, and live compute state.
- Done: Froze contract `2f935dd01763840d4e0420267b21c11980e45b8d9929054821a18f2d9c956f1e` and passed the 12-record exact-token/LMMS smoke.
- Done: Executed all 443 validation and 433 test execution samples on four direct RTX 6000 Ada GPUs, producing 2,196 regime records and 59,239 unique current-runtime route records with no final missing, duplicate, or failed record.
- Done: Validation selected and froze `shared_random4` before test treatment results were read; held-out test did not change the selection.
- Done: Final artifacts, 22 required SHA-256 values, 35 focused/regression tests, compilation, and whitespace checks pass. All four GPUs are idle.
- Stopped: No Stage-2 training, broad W2C repair/MCTS, routing training, or external evaluation ran.
- Blocked: No.

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: distinguish failure detection from visual-treatment responsiveness under a fair, current-runtime bounded intervention contract.
- Confirmed observation / unverified interpretation: selected gate trigger sets contain roughly 200 wrong and 20 correct samples per gate per split; old route correctness is non-authoritative and the prior bounded beam was unstable. Whether local current-runtime interventions rescue these failures is unmeasured.
- Diagnosis: unknown; failure prediction does not identify treatment correctability.
- Viable alternatives considered: adaptive four-action beam/MCTS; historical correct routes; deterministic all-single plus fixed pair-panel search. Beam/MCTS is broader but unstable and much more expensive; historical routes violate the current-runtime authority; the deterministic panel supplies a reproducible lower bound.
- Chosen action and strongest objection: evaluate every permitted one-layer intervention plus exactly 12 seeded distinct-layer pairs (or all pairs if fewer), early-stopping at the first LMMS-correct route. This can miss corrections requiring three or more changed layers, so results are bounded lower estimates rather than exhaustive oracle rates.
- How this differs from failed attempts: it does not generate training labels, does not rely on old cached correctness, and does not use an outcome-adaptive unstable beam. It directly evaluates current LMMS correctness from the gate-implied treatment boundary.
- Authorization and stop condition: explicitly authorized by the user through the named plan; stop after validation/test correctability, enrichment, timing/compute, figures, reports, and state update.
- Independent review: required because of cost and comparison validity. The reviewer revised fill-to-96 pair sampling to a fixed 12-pair panel, preventing later triggers from receiving disproportionate pair-search effort.

## Latest Research-Action Result
- Action taken: Ran current-runtime LMMS-Eval bounded treatment feasibility for the frozen shared Random-4, independent sequential, and fixed-L27 detect-and-replay gates. Every permitted suffix single-layer action plus a fixed seeded 12-pair panel was evaluated with first-correct early stop; all-wrong enrichment used the same layer-0 replay panel.
- Result: Validation selected `shared_random4`: population rescue 0.2025 and triggered-wrong correctability 0.3821, versus 0.1825/0.3544 for independent and 0.1975/0.3657 for fixed L27. On held-out test these were 0.2100/0.4118, 0.1975/0.3709, and 0.2200/0.4171. Shared-gate full-replay enrichment was 1.1211 validation and 1.1132 test. All triggered-correct treatment cohorts were bounded-search preservable, while pre-treatment correct preservation remained 0.94-0.95 because the gates themselves false-trigger some correct samples.
- Evidence saved: `analysis/dense_failure_stage1/treatment_correctability/decision_summary.md`, the five `metrics/*.csv` tables, five required figures, `stage2_future_manifest.jsonl`, `stage2_future_features.pt`, and `artifact_manifest.json`.
- Failure or issue: The first frozen smoke exposed a missing regime-level `passed` field; a regression guard fixed it. The next validation attempt exposed native-versus-materialized BF16 SDPA drift on one ChartQA dense baseline (`375` native versus `389` materialized, both wrong). The executor now preserves native causal dispatch for full-row calls and branches suffix treatment from the exact native dense prefix. Both invalid contracts and raw evidence are quarantined under `treatment_correctability_invalid_50da72f7/` and `treatment_correctability_invalid_dd532bbc/`; neither contributed final records.
- Lesson learned: Frozen failure gates modestly enrich treatment-correctable errors, and current-runtime visual interventions rescue about one fifth of all dense-wrong samples under this bounded search. Early shared triggers have higher conditional correctability than later triggers, but the longer intervention horizon/search opportunity is inseparable here, and fixed-L27 replay slightly exceeds dynamic population rescue on test.
- Next implication: A learned Stage-2 action head is provisionally supportable from the 463 frozen selected-gate states, but fixed-L27 replay remains a serious fallback and GQA treatment potential is weak. Any Stage-2 training is a separate research action requiring explicit authorization.
