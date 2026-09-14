# Phase 81: Predictability Step-D External Transfer Memory

## Current Objective
Execute the frozen full external-transfer study in `plans/predictability_phase_stepD_full_external_transfer_plan.md`: full-internal Stage-1/Stage-2 refits, zero-retuning transfer to ChartQA, TextVQA, MMMU-Pro, and POPE, and controlled local READ/WRITE utility measurement only in the existing robust-P90 Stage-2 domain.

## Active Constraints
- Reuse the exact Phase-78 targets and Phase-79 model, feature, loss, seed, and optimization families; do not redesign the method, target, architecture, or hyperparameters.
- External labels may construct outcomes and final metrics only. They may not choose models, layers, seeds, thresholds, normalization, q aggregation, k, or subsets.
- Use exactly the established 19,960-sample external population and the Qwen2.5-VL snapshot, prompts, generation, image, and LMMS-compatible scoring contracts from Phase 69 / `shared_prefix_eval_20260812`.
- The external Stage-2 domain is fixed by the existing robust ALL-source strict P90 threshold `score > 0.9061332901863008`; do not force states for non-triggered samples.
- Hash and freeze Stage-2 predictions before executing/scoring external four-branch labels. Stop on unexplained dense, feature, trigger, FULL-branch, or q-adapter parity failures.
- Use four direct RTX 6000 Ada GPUs after checking live occupancy; no Slurm. Keep large state/checkpoint artifacts under `/mnt/hyemin`.
- Stop after the complete Step-D evidence, final A-to-D ladder, and exactly one method implication. Do not begin a new routing-method phase.

## Current State
- Done: froze final contract `efa342f0a...e571cf` after exact parent, code, runtime, model, robust-head, population, and evaluation-contract checks.
- Done: independent review returned `stable`; Phase-69 was used only as a hash-bound parity oracle while all Step-D states, predictions, and utility labels were newly produced.
- Done: completed all 15 full-internal Stage-1/Stage-2 refits and the bound 10-row smoke across all seven task variants and all three triggered families.
- Done: recomputed all 19,960 external dense trajectories: 558,880 layer states, exact Phase-69 dense/trigger parity, 901 strict-P90 triggers, 8,442 post-trigger states, and zero POPE triggers.
- Done: froze 558,880 Stage-1 and 8,442+8,442 Stage-2 READ/WRITE predictions before external utility labels, then measured all 33,768 four-action branches with complete state/FULL/q/algebra parity.
- Done: completed all pre-registered tables, figures, 5,000-draw image-group bootstraps, categories `D1-C` / `D2-A`, final A-to-D summary, and one method implication. All 72 final artifact hashes verify; 51 focused/inherited tests pass.
- Blocked: none.
- Most recent useful observation: Stage-1 external all-state M3 AUROC is 0.4998/0.6794/0.5399/0.5582 for ChartQA/TextVQA/MMMU-Pro/POPE (macro 0.5693), while Stage-2 READ/WRITE macro Spearman is only 0.0730/0.0232. The fixed categories are `D1-C` and `D2-A`.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Step-C Stage-1 survives semantic Q1 and cluster OOD but collapses under source/LODO; Stage-2 remains weak | `analysis/predictability_generalization/stepC_generalization/summaries/stepC_generalization_summary.md` | Fixes the transfer question and rules out semantic novelty as the only shift axis | confirmed |
| Exact 19,960 external manifests and Dense outputs exist; prior robust P90 produced about 901 triggers and zero POPE triggers | `analysis/dense_failure_stage2/full_benchmark_eval/` | Provides a parity oracle, not the missing Step-D state/utility measurements | confirmed |
| Evaluation snapshot, prompts, generation, and scorers are frozen | `eval/reference/shared_prefix_eval_20260812/EVAL_PROTOCOL.md` | Prevents benchmark or evaluator drift | confirmed |
| Phase-69 rows have scores/traces but no all-layer feature census or per-state four-branch q labels | Phase-69 artifact/schema audit and independent review | Requires live feature/state extraction and counterfactual measurement | confirmed |
| Complete external Dense/state census and prediction-label firewall | `analysis/predictability_generalization/stepD_external_transfer/external_manifests/`, `predictions_frozen_before_labels/`, and `stage2_measurement/parity_report.json` | Establishes that the negative transfer result is not a partial-run or parity artifact | confirmed |
| Stage-1 external M3 macro AUROC 0.5693; only TextVQA is clearly useful relative to controls | `stage1/benchmark_metrics.csv` and `statistics/group_bootstrap_ci.csv` | Rejects broad and named-family-only transfer under the fixed representation | confirmed |
| Stage-2 READ/WRITE macro rho 0.0730/0.0232; MMMU-Pro and WRITE CIs generally include zero | `stage2_predictability/*_benchmark_metrics.csv` and `statistics/group_bootstrap_ci.csv` | Confirms weak current-state local-utility prediction externally | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial parent-manifest verification | Generic code assumed one historical manifest self-hash convention | supported implementation mismatch | preparation traceback; all declared artifact file hashes were valid | Verify declared files and bind the complete historical manifest bytes rather than reinterpret its legacy self-hash | Do not impose one self-hash convention retroactively |
| First refit launch | Per-task thread-pool setup failed after parallel work began | supported implementation defect | rank logs | Configure the PyTorch worker runtime once at process entry | Do not reconfigure inter-op threads inside each task |
| First refit finalization | Mixed classification/regression rows exceeded first-row CSV columns | supported serialization defect | finalizer traceback and regression test | CSV writers use a deterministic union of row fields | Do not infer a homogeneous schema for combined evidence tables |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Execute Step D with mandatory contract/parity gates | It is explicitly authorized and completes the planned generalization ladder | External Stage-1 transfer and Stage-2 local-utility transfer | high | selected |
| Stop at a failed preflight gate | Preserves validity if required state/adapters cannot be reproduced | Prevents invalid external claims | low | fallback only |
| Redesign or subset | Could reduce cost or respond to collapse | Changes the scientific contract | high | rejected; unapproved pivot |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: complete external transfer without using external outcomes for adaptation; the bottleneck is missing external state/utility evidence, not missing external dense answers.
- Relevant memory item used: Phase 80 established source/dataset specificity and warned against global threshold deployment.
- Confirmed observation: Phase 69 supplies exact external parity oracles but lacks the Step-D feature census and local counterfactual labels.
- Unverified interpretation: external Stage-1 may transfer within the named ChartQA/TextVQA families while collapsing on MMMU-Pro/POPE; this is a pre-registered result category, not an assumption.
- Diagnosis: supported source/dataset specificity for the current internal Stage-1 contract and weak current-state Stage-2 utility prediction; external behavior remains unknown.
- Evidence path if diagnosis is not unknown: `analysis/predictability_generalization/stepC_generalization/`.
- Viable alternatives considered: gated Step-D execution; stop on gate failure; redesign/subset (disallowed).
- Chosen action: execute the full Step-D plan, reusing only hash/parity-verified Phase-69 identity and output oracles and recomputing missing states/labels.
- Strongest objection: dense-answer parity does not establish hidden-state/feature parity or benchmark-specific q validity.
- How this differs from failed attempts: predictions, thresholds, adapters, and aggregation are frozen before any external utility labels, and state/trigger/FULL parity is checked directly rather than inferred.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform the Step-D plan.
- Stop condition: all required Step-D artifacts and summaries are complete, or the first unexplained contract/parity failure is preserved and reported.

## Latest Research-Action Result
- Action taken: completed the full frozen Step-D external-transfer study.
- Result: complete negative/mixed transfer evidence. Stage 1 is `D1-C`: all-state M3 AUROC is ChartQA 0.4998, TextVQA 0.6794, MMMU-Pro 0.5399, POPE 0.5582, macro 0.5693 (pooled 0.6980, which is not evidence of uniform benchmark transfer). Stage 2 is `D2-A`: READ rho is 0.0848/0.1095/0.0246 and WRITE rho is 0.0335/0.0158/0.0204 on ChartQA/TextVQA/MMMU-Pro; POPE has no frozen-P90 states.
- Evidence saved: `analysis/predictability_generalization/stepD_external_transfer/`; contract `efa342f0a...e571cf`; artifact manifest `bf22455a...f0d9` (72/72 hashes verified).
- Failure or issue: no scientific validity failure. Three pre-execution/finalization implementation defects were repaired prospectively; each code change invalidated and regenerated the contract before accepted refits. The final run has exact 19,960-UID, 8,442-state, and 33,768-branch completeness.
- Lesson learned: the current dense-failure representation does not transfer robustly across the four external families, and the current-state local READ/WRITE utility target remains weak despite complete counterfactual labels. High pooled Stage-1 AUROC coexists with weak per-benchmark transfer and must not be treated as deployment robustness.
- Next implication: first distinguish source-calibration failure from representation failure for Stage 1, while treating nonlocal/history/counterfactual information as the unresolved Stage-2 direction. This is a recommendation only; no next experiment is authorized.
