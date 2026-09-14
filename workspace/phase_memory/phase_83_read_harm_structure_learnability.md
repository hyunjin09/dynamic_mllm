# Phase 83: READ-Harm Structure and Learnability Memory

## Current Objective
Execute `plans/read_harm_structure_learnability_plan.md` to determine whether harmful READ has persistent within-UID structure, a repeatable READ-operation signature, and image-group-disjoint out-of-fold learnability.

## Active Constraints
- READ only: `H_R = q_WRITE_ONLY - q_FULL`; positive is harmful and negative is beneficial.
- Primary population is the 15,185 exact dense states over 1,413 UIDs; the 35,565 routed states are secondary and selection-qualified.
- Preserve the inherited Step-B folds and Step-C semantic/source/LODO definitions. No state-random split or OOD tuning.
- F1-F7 use only the current READ operation; no suffix state, final answer, dataset/source ID, route identity, or future-action input.
- Require exact Phase-82 state/post-state parity and operation-level SDPA reconstruction before bulk extraction.
- Select the routed feature/model only by the prospectively frozen dense OOF rule.
- Use four direct RTX 6000 Ada GPUs; no Slurm. Store large work products under `/mnt/hyemin`.
- Stop after structure, mechanism, learnability, generalization, routed-secondary evidence, and one unexecuted next-method recommendation. Do not start WRITE or a deployment router.

## Current State
- Done: audited the Phase-78/79/82 population, target, state-cache, fold, and baseline contracts.
- Done: implemented fail-closed population preparation, READ-operation features, exact smoke validation, extraction census checks, structure analysis, matched comparisons, and the primary training/transfer scaffolding.
- Done: froze contract `eaaef860...e4d6`, passed the 24-state exact parity/operation smoke, and extracted all 15,185 dense and 35,565 routed READ states with complete F1-F7 censuses on four GPUs.
- Done: completed structure, matched-mechanism, image-group-disjoint OOF learnability, semantic/source/LODO transfer, and selection-qualified routed-secondary analyses.
- Done: verified all 75 final artifact hashes under manifest `35524bf3...05ab` after reporting-only summary corrections.
- In progress: none; Phase 83 is completed and stopped.
- Blocked: none.
- Most recent useful observation: explicit local READ-operation features do not materially rescue full-suffix READ-harm prediction; the dense winner reaches Spearman `0.0697` and harmful AUROC `0.5338`.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Full dense and routed exact state/utility censuses | `analysis/predictability_generalization/stepA_measurement/` | Fixes identities and `H_R` without new search | confirmed |
| Exact one-step branch/state caches and parity | `analysis/dense_failure_stage2/counterfactual_effect_identifiability/` | Provides a strong cross-check for current READ extraction | confirmed |
| Inherited five-fold image-group registry | `analysis/predictability_generalization/stepB_id_learnability/` | Prevents leakage and split selection | confirmed |
| Step-C semantic/source/LODO registries | `analysis/predictability_generalization/stepC_generalization/` | Fixes the transfer tests | confirmed |
| Independent review requested two validity gates | `/root/phase82_identifiability_review` | Requires actual-SDPA validation and pre-result routed winner selection | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial analysis-script draft | One malformed holdout expression and one indentation artifact failed compilation | supported implementation defect | `py_compile` traceback before execution | Repair and compile/test before freezing | Do not freeze or launch the draft |
| Initial routed structure grouping | Treating all routed prefix states for one UID as one chain caused duplicate-layer/noncontiguous sequences | supported representation defect | fail-closed structure traceback plus routed prefix inspection | Use the exact prefix DAG and root-to-leaf adjacency, with UID-consistent node shuffling | Do not collapse branched routed prefixes into a UID chain |
| Initial SDPA max-error gate | All exact parity checks passed, but a `0.03125` maximum difference failed the `0.015625` cap despite cosine `0.9999990` and mean error `0.000805` | supported BF16/kernel accumulation tolerance defect | emitted operation-smoke diagnostic | Use BF16-aware max `0.0625`, mean `0.002`, and cosine `0.9999` jointly | Do not require a sub-ULP max bound between different SDPA query shapes |
| First matched-probe training pass | Dense/fusion tasks completed, but all ranks stopped when fold 1's 52-state matched subset had zero rows in the inherited inner calibration role | supported sparse-subset split defect | rank logs plus per-fold role/class census | Preserve inherited outer folds and derive a frozen image-group-disjoint calibration subset within each matched training fold | Do not require a sparse diagnostic subset to populate a parent inner role by chance |
| Second different-shape SDPA check | A 1,149-token TextVQA state reached max/mean error `0.125/0.002862` while cosine remained `0.9999992` and every state/branch/feature parity check passed | supported kernel-shape confound | emitted operation-smoke diagnostic | Reconstruct the full-query causal SDPA tensor, matching the executor shape, and inspect its final row | Do not tune tolerances around one-query versus full-query kernels |
| Initial exact-layer summary selection | A support-one early-layer row with undefined Spearman was selected as the apparent best exact layer | supported reporting defect; no fit, prediction, label, or feature changed | `validation/reporting_correction.json` | Report the prospectively fixed Early/Middle/Late aggregates and retain exact-layer rows as descriptive | Do not rank sparse exact-layer cells by an undefined metric |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Full authorized READ structure/feature/OOF action | Directly tests the unresolved local-mechanism hypothesis | R-STRUCT, R-MECH, and R-LEARN categories | high | completed |
| Reuse only Phase-82 generic one-step results | Cheap | Cannot answer explicit attention/update structure | low | rejected |

## Next-Step Decision
- Deliberation mode: fast
- Active objective and bottleneck: execute the fixed Phase-83 plan; the immediate bottleneck is validating that reconstructed attention statistics match the actual READ operation.
- Relevant memory item used: Phase 82's exact caches and Case-D result define the baseline and validation source.
- Confirmed observation: all fixed parent populations, labels, folds, and exact state caches are present.
- Unverified interpretation: explicit READ-operation mechanics may expose signal hidden by generic state representations.
- Diagnosis: unknown
- Viable alternatives considered: full authorized action; reuse generic evidence only.
- Chosen action: completed the narrow implementation, exact validation, and one frozen four-GPU execution.
- Strongest objection: reconstructed q/k/v statistics could describe the weights without faithfully representing the executor's actual READ operation; the smoke makes this a fail-closed gate.
- How this differs from failed attempts: it measures preregistered READ-specific operation features and preserves the fixed target/folds rather than adding another generic representation.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform `plans/read_harm_structure_learnability_plan.md`.
- Stop condition: all plan-required READ artifacts, routed-secondary analysis, decision categories, and exactly one recommendation are complete; no follow-on action runs.

## Latest Research-Action Result
- Action taken: completed the fixed READ-harm structure, mechanism, learnability, generalization, and routed-secondary audit.
- Result: **R-STRUCT-B / R-MECH-B / R-LEARN-C**. Harmful READ is common but mostly isolated; only 1/34 matched feature effects excludes zero, the matched probe is below chance, and the best dense local predictor is weak (`rho=0.0697`, AUROC `0.5338`).
- Evidence saved: `analysis/read_harm_structure_learnability/`, contract `eaaef860...e4d6`, manifest `35524bf3...05ab`.
- Failure or issue: no scientific execution failure remains; implementation/representation/sparse-calibration/reporting defects were repaired without changing the scientific contract.
- Lesson learned: explicit attention/update mechanics do not make full-suffix READ harm robustly locally identifiable under this fixed representation and capacity ladder.
- Next implication: stop. The sole unexecuted recommendation is one short-horizon READ effect-propagation/planning audit; it requires separate authorization.
