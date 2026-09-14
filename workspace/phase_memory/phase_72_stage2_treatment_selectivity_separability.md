# Phase 72: Stage-2 Treatment-Selectivity Separability Memory

## Current Objective
Determine whether the frozen Stage-2 Experiment-A READ/WRITE representations contain group-generalizable information separating exact routed states that require intervention from states that require keeping FULL.

## Active Constraints
- Use only the frozen 569-UID replay-valid training-side union and exact prefix-state identity.
- Freeze Qwen2.5-VL, robust Stage-1 maps, Experiment-A router/checkpoint, route store, evaluator semantics, and state timing.
- Train diagnostic probes only; no deployment gate, Stage-2 retraining, new search, external-label fitting, or external evaluation.
- Use five-fold UID/image-group-disjoint evaluation, UID-balanced training, fold-local normalization, and matched nuisance sensitivity.

## Current State
- Done: Completed the frozen exact-state dataset, four-GPU representation extraction, five-fold probes, nuisance/matched controls, figures, summaries, and 89-file artifact audit.
- In progress: None.
- Blocked: None.
- Most recent useful observation: RW is weak (OOF AUROC 0.5620, AUPRC 0.1086) and has no useful high-precision subset; the optional MLP is no better, nuisance-only AUROC is 0.8150, and matched RW remains weak at 0.5763.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Exact-prefix valid-set reconstruction yields 21,071 unique states from 34,253 occurrences | `analysis/dense_failure_stage2/observed_valid_set_loss/valid_sets/exact_state_index.jsonl` | Provides exact entering-state action sets without treating search misses as KEEP | confirmed |
| Experiment-A global nonFULL-minus-FULL margin did not select rescue over regression | `analysis/dense_failure_stage2/abstention_margin/summaries/decision_summary.md` | Motivates testing the full logits and branch representations rather than another scalar margin | confirmed |
| Frozen router uses separate READ and WRITE attention summaries, concatenated into one four-action head | `dense_failure_stage2/v1_router.py` | Defines z_R and z_W to extract without changing the model | confirmed |
| M0/M1/R/W/RW AUROC is 0.4632/0.4640/0.5370/0.5219/0.5620; optional RW MLP is 0.5583 | `analysis/dense_failure_stage2/treatment_selectivity_separability/metrics/probe_summary.csv` | No frozen representation meets the prospective strong/high-precision gate | confirmed |
| Nuisance-only AUROC is 0.8150; matched RW AUROC is 0.5763 on 18,936 raw supported states/463 UIDs | `analysis/dense_failure_stage2/treatment_selectivity_separability/metrics/nuisance_controls.csv`; `metrics/matched_sensitivity.csv` | Obvious priors are strong, while RW stays weak after reducing them | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Global nonFULL-minus-FULL abstention margin | All five validation folds selected delta 0; positive margins were not rescue-selective | supported: scalar readout lacks selectivity on held-out external outcomes | `analysis/dense_failure_stage2/abstention_margin_calibration/` | Test frozen internal representations under clean replay-valid supervision | Do not tune another global margin here |
| Initial fold allocator | Absolute post-placement cost packed the real skewed groups into only three folds | supported: allocator objective error | Pre-output prepare traceback; regression test `test_group_folds_do_not_pack_homogeneous_groups_into_only_near_target_folds` | Use incremental global imbalance cost; refreeze only after all five folds are populated | Do not compare absolute per-fold post-placement costs |
| First frozen diagnostic run | Probe fitting completed but heterogeneous operating-point dictionaries failed CSV serialization | supported: output-schema mismatch | Preserved root `analysis/dense_failure_stage2/treatment_selectivity_separability_failed_contract_92228ac4/` | Rectangularize operating-point rows, regression-test, and rerun extraction under a new contract | Do not promote the failed contract or rebind its shards |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| M0 scalar margin | Existing behavior baseline | Whether ranking signal exists in current margin | low | rejected: AUROC 0.4632 |
| M1 four logits | Scalar collapse may discard joint logit structure | Whether selectivity survives near the action output | low | rejected: AUROC 0.4640 |
| R, W, and RW linear probes | Branch summaries may encode treatment need before the four-way head | Head/objective versus representation distinction | medium | rejected as actionable: 0.5370/0.5219/0.5620 |
| One hidden-layer RW probe | Only warranted if RW is weak but above random | Whether simple nonlinearity exposes latent signal | low | rejected: AUROC 0.5583 |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: The frozen router intervenes without external selectivity; determine whether treatment-need information exists in its representations.
- Relevant memory item used: Phase 71 found the global action margin non-selective and stopped without modifying Stage 2.
- Confirmed observation: The clean exact-state population is highly imbalanced but has 1,657 intervention-required states across 463 UIDs.
- Unverified interpretation: The existing four-way head/objective, rather than the READ/WRITE representations, may be the main bottleneck.
- Diagnosis: supported at the scoped level: the current frozen READ/WRITE summaries do not linearly or with one small MLP expose a generalizable high-precision treatment-need subset under observed-valid supervision.
- Viable alternatives considered: Another scalar threshold, frozen-representation probes, or new representation/training. The fixed plan selects the smallest discriminating diagnostic.
- Chosen action: Extract exact Experiment-A logits/z_R/z_W for every unique replay-valid state, run five-fold group-disjoint probes and matched nuisance controls, and make one evidence-conditioned recommendation.
- Strongest objection: Successful-route observation is incomplete, so KEEP_REQUIRED means uniquely observed FULL rather than proof that all interventions fail; conclusions must remain scoped to observed-valid supervision.
- How this differs from failed attempts: It evaluates full internal representations with exact-state clean labels and group-disjoint OOF predictions rather than calibrating the failed scalar deployment margin on external outcomes.
- Automatic execution authorized: yes
- Authorization basis: The user explicitly requested execution of `plans/stage2_treatment_selectivity_separability_plan.md`.
- Stop condition: Dataset/provenance audit, representation extraction, five-fold probes, high-precision and matched analyses, artifacts, and exactly one unexecuted recommendation are complete.

## Latest Research-Action Result
- Action taken: Reconstructed exact clean/mixed labels, extracted frozen Experiment-A z_R/z_W/logits for every unique state on four GPUs, and evaluated M0/M1/R/W/RW plus the prospectively triggered RW MLP under five group-disjoint folds and matched controls.
- Result: Decision Case D. RW AUROC/AUPRC is 0.5620/0.1086; top-5% precision is 0.1234, recall at 90% precision is 0.0012, and the MLP is 0.5583. Nuisance AUROC is 0.8150 and matched RW is 0.5763.
- Evidence saved: Authoritative contract `082c9f459da798c419b50b637c0afb1c1b388b403ea2b51c73092b1df3e44edd` and 89 verified files under `analysis/dense_failure_stage2/treatment_selectivity_separability/`.
- Failure or issue: Two implementation defects were caught before promotion. The first failed before output creation; the second frozen partial root is preserved as `treatment_selectivity_separability_failed_contract_92228ac4` and is explicitly non-authoritative. The repaired run was refrozen and repeated end to end.
- Lesson learned: The current four-way logits and READ/WRITE summaries do not support another lightweight treatment gate; strong nuisance predictability requires explicit matched controls.
- Next implication: Do not add another head. If separately authorized, diagnose representation/training-state diversity with one bounded discriminator; do not execute it automatically.
