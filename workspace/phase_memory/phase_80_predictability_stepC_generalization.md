# Phase 80: Predictability Step-C Semantic/Source Generalization Memory

## Current Objective
Execute the frozen semantic-similarity, question-cluster, source-regime, and dataset generalization study in `plans/predictability_phase_stepC_semantic_source_generalization_plan.md`. Stop after internal Step-C evidence classification and procedural Step-D readiness; do not run Step D or external evaluation.

## Active Constraints
- Reuse the exact Phase-78 targets and Phase-79 M0/M1/M3 model and optimization contracts; do not redesign labels, features, losses, or architectures.
- Use one prospectively frozen, label-blind question encoder; all clustering and split balancing must ignore correctness and utility targets.
- Preserve the Phase-79 five image-group-disjoint folds for similarity analysis. For OOD fits, test groups may not contribute to fitting, preprocessing, early stopping, or threshold calibration.
- Question kNN is fixed at exact cosine `k=5`; dense Stage-2 neighbors must match absolute layer. Under-supported exact-layer cells are non-estimable, not repaired with a smaller k.
- Use three fixed M3 seeds, four direct RTX 6000 Ada GPUs, and 5,000 image-group bootstrap draws for primary uncertainty.
- Stop before external ChartQA/TextVQA/MMMU-Pro/POPE evaluation, Step-D refit, target redesign, or another Stage-2 method.

## Current State
- Done: encoder contract `d574a8af...30a9` freezes cached `Qwen/Qwen3-Embedding-0.6B` snapshot `c54f2e6e...03418`, one shared symmetric semantic-similarity instruction, NFKC/whitespace normalization, left padding, last-token pooling, BF16 SDPA inference, float32 L2 output, and exact-repeat smoke.
- Done: experiment contract `fafd6442...07ecd` binds 10,399 UID embeddings, 9,982 group embeddings, deterministic K=100 spherical clusters, 22 group-disjoint OOD role maps, six prospectively unsupported dense Stage-2 source-transfer cells, parent contracts/code/runtime, and 300 training tasks.
- Done: four GPUs completed all 300/300 M0/M1/M3 fits. Exact kNN controls, five cluster folds, bidirectional pooled/within-dataset source transfer, three LODO tests, six pairwise transfers, layer/trigger breakdowns, train-calibrated operating points, and uncertainty are complete.
- Done: final artifact manifest `ca340f26...da875` verifies the compact output set. Aggregation patches `9c13519a...1ea85` and `89580e7f...f5ed9` are provenance-bound and changed no model, target, split, hyperparameter, checkpoint, or fitted prediction.
- Blocked: none.
- Most recent useful observation: Stage-1 survives semantic dissimilarity and K=100 cluster OOD but collapses under source and dataset shift; Stage-2 remains weak throughout. The plan-defined categories are S1-C source-specific signal and S2-A weak everywhere.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Q1/Q5 M3 AUROC is 0.7828/0.8012; delta +0.0183, 95% group-bootstrap CI [-0.0050, 0.0423] | `analysis/predictability_generalization/stepC_generalization/statistics/similarity_q1_q5_bootstrap.csv` | Does not support a material similar-question dependence | confirmed |
| Concatenated K=100 cluster-OOD M3 AUROC is 0.7725 versus ID 0.7869 | `question_cluster_ood/stage1_metrics.csv` | Unseen semantic clusters preserve most in-domain Stage-1 ranking | confirmed |
| Historical→Canonical and reverse Stage-1 M3 AUROC is 0.4342/0.5563 | `source_transfer/*_stage1.csv` | Direct source shift destroys the pooled Stage-1 signal | confirmed |
| LODO Stage-1 M3 AUROC is 0.6043 ChartQA, 0.5508 GQA, 0.6006 TextVQA | `dataset_lodo/lodo_stage1.csv` | Failure signal does not transfer reliably across dataset families | confirmed |
| LODO 95%-calibrated C preservation is 0.142 ChartQA, 0.060 GQA, 0.941 TextVQA | `dataset_lodo/lodo_stage1.csv` | Threshold calibration can fail catastrophically under dataset shift | confirmed |
| Cluster-OOD READ/WRITE M3 Spearman is 0.0453/0.0164; LODO correlations remain near zero | `question_cluster_ood/stage2_*_metrics.csv`; `dataset_lodo/lodo_stage2_*.csv` | No useful Stage-2 utility signal emerges under shift | confirmed |
| Exact-layer kNN is estimable for 15,181/15,185 OOF states per target | `controls/question_knn_state_support.csv` | Four early-layer states lack five training references; no smaller-k fallback was used | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| First Stage-2 bounded smoke | Reused trainer raised `KeyError: targets` | supported implementation omission | `work/smoke_rank3.log` | Copy the unchanged Phase-79 target-scaling block into the Step-C contract before refreezing | Do not treat a partial smoke as authorization for full fits |
| First aggregation | Four OOF early-layer states had fewer than five exact-layer training references | supported sparse-support condition | `controls/question_knn_state_support.csv` | Preserve k=5 and mark only those controls non-estimable | Do not silently lower k or impute a default |
| First OOD kNN retry | Empty exact-layer reference index defaulted to float and failed before support handling | supported implementation defect | `work/aggregate_patch.log` | Bind an aggregation-only patch with explicit integer empty indices and non-estimability | Do not refit models for reporting-only defects |
| Initial automatic evidence label | Report selected S1-B despite Q1≈Q5 and cluster≈ID, followed by source/LODO collapse | supported reporting-rule defect | final primary tables and `summaries/final_audit.md` | Use the plan-defined taxonomy: S1-C source-specific signal | Do not infer semantic dependence from nuisance strength in Q1 |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Stop at Step-C boundary | All authorized internal shift tests and controls are complete | Preserves negative OOD evidence without an unapproved pivot | low | selected |
| Step-D external transfer | Procedurally ready under the plan | Tests whether the source-specific internal result extends to the four external families | high | not authorized |
| Redesign source-robust Stage-1 or Stage-2 | Source/LODO collapse and weak utilities motivate reconsideration | Could address the observed bottlenecks | high | strategic pivot; not authorized |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: Step C is complete; the bottleneck is source/dataset robustness, not semantic-cluster novelty.
- Relevant memory item used: Phase 79 warned that in-domain Stage-1 AUROC did not establish template/source independence.
- Confirmed observation: Q1/Q5 and cluster OOD remain near ID, but bidirectional source transfer, LODO, and pairwise transfer are weak; Stage-2 stays near chance.
- Unverified interpretation: a redesigned source-robust model could recover generalization. Step C does not test that.
- Diagnosis: supported source/dataset specificity for the current Stage-1 contract; weak current-state local utility prediction remains supported for Stage 2.
- Evidence path if diagnosis is not unknown: `analysis/predictability_generalization/stepC_generalization/summaries/stepC_generalization_summary.md` and primary tables.
- Viable alternatives considered: stop; run Step D; redesign the Stage-1/Stage-2 method.
- Chosen action: stop at the authorized Step-C boundary.
- Strongest objection: K=100 cluster OOD and Q1 performance show genuine semantic robustness, so “source-specific” must not be overstated as total absence of transferable hidden-state information.
- How this differs from failed attempts: this conclusion comes from explicit label-blind semantic, source, and dataset holdouts rather than same-population OOF or historical/canonical mixture splits.
- Automatic execution authorized: no further research action.
- Authorization basis: the user authorized Step C only; its stop rule is satisfied.
- Stop condition: all required semantic, kNN, cluster, source, LODO, pairwise, uncertainty, summary, readiness, and provenance artifacts are complete.

## Latest Research-Action Result
- Action taken: completed the full Step-C internal generalization ladder with a frozen label-blind question encoder and unchanged Phase-79 model/target/optimization contracts.
- Result: S1-C / S2-A. Stage-1 is insensitive to nearest-question quintile and retains AUROC 0.7725 on unseen semantic clusters, but falls to 0.4342/0.5563 under pooled source transfer and 0.5508-0.6043 under LODO. Stage-2 READ/WRITE remains near chance across semantic, source, and dataset shifts.
- Evidence saved: `analysis/predictability_generalization/stepC_generalization/`; final manifest `ca340f2635d67e41c566fea4783b8a3813fd6127c3cd040b18db9aaec9bda875`.
- Failure or issue: one smoke omission, two exact-layer kNN aggregation defects, and one automatic taxonomy defect were repaired under provenance-bound contracts without changing scientific fits or results.
- Lesson learned: the current Stage-1 failure signal is broader than semantic-neighbor memorization but not robust to source/dataset distribution shift; the Stage-2 local utility signal remains unusably weak.
- Next implication: stop. `READY_FOR_STEP_D = true` is procedural only; Step D, external evaluation, and any redesign require separate authorization.
