# Phase 66: Stage-2 Shared-Union Training Memory

## Current Objective
Train one shared Stage-2 router on the deduplicated robust-threshold union, compare the same checkpoint at P98/P95/P90, and add MCTS supervision only if the single-only condition is technically healthy.

## Active Constraints
- Follow `plans/stage2_shared_union_threshold_comparison_plan.md` as one bounded research action.
- Freeze Qwen2.5-VL, the Phase-63 five-head Stage-1 ensemble, P98/P95/P90, four-action execution, LMMS scoring, the existing Stage-2 architecture, plain CE, optimizer/update/checkpoint rules, and C:W mixture.
- Use only exact-replay-valid Phase-65 routes; train threshold-agnostically and sample W bases uniformly rather than routes.
- The only leakage-free Stage-2 validation population physically available is the frozen 800-row Historical validation split. Canonical OOF rows overlap the union-training population and must not be presented as held-out Stage-2 evidence.
- Stop after Experiment A, Experiment B only if A is healthy, three-threshold validation rollout, and a validation operating-point decision. Do not open test or change Stage 1.

## Current State
- Done: Read the plan, research-control instructions, Phase-58 and Phase-65 memory, compute/environment policy, prior runner/config/results, Phase-63 gate, and Phase-65 corpus schemas.
- Done: Independent review returned `revise`: preserve Historical-800 as primary validation, mark Canonical validation unavailable, and avoid an all-source deployment claim.
- Done: Audited union versus Historical validation: zero UID, image-group, and image-SHA overlap.
- Done: Frozen contract `b17a81d...40f8`; data-loader audit, native-dense implementation smokes, and four-action overfit gates passed for both experiments.
- Done: Experiment A and B each completed the fixed 12-epoch/3,144-update four-GPU fit, followed by complete P98/P95/P90 free rollouts over all 800 Historical validation UIDs.
- Done: Experiment A rescued 4 W and regressed 1 C at P90 (net +3); Experiment B rescued and regressed zero samples at every operating point.
- Done: Final aggregation and the 145-file hash manifest passed. The planned final condition selects P98 by conservative tie-break, but remains Decision C because its routed accuracy is unchanged from dense.
- Blocked: No.
- Most recent useful observation: adding MCTS supervision reduced final teacher-forced non-FULL recall from 0.1920 to 0.1687 and changed the only positive rollout cell (A/P90 net +3) to B/P90 net 0; high MCTS oracle-label yield did not translate into learned rollout benefit under this shared router.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-58 V1 used exact online routed-token replay, 3,144 updates, final-update selection, and Historical-800 rollout | `workspace/phase_memory/phase_58_stage2_v1_training_revised.md` | Defines the frozen architecture/optimization/evaluator baseline | confirmed |
| Phase-65 retained 2,519 exact routes and 34,253 state rows | `analysis/dense_failure_stage2/robust_gate_corrective_search/` | Defines all authorized union supervision | confirmed |
| Union counts are 106 preservation, 270 single W, and 216 MCTS W UIDs | Phase-65 threshold corpora deduplication audit | Sets A/B population sizes | confirmed |
| Union versus Historical-800 overlap is UID/group/SHA = 0/0/0 | read-only audit on Phase-65 work manifest and Phase-58 validation manifest | Validates the available primary Stage-2 holdout | confirmed |
| Canonical OOF rows are among Phase-65 Stage-2 label sources | Phase-59 memory and Phase-65 work manifest | Rules out Canonical-4000 as held-out Stage-2 rollout | confirmed |
| Experiment A P98/P95/P90 net correction is 0/0/+3 | `analysis/dense_failure_stage2/shared_union_training/experiment_A_single/metrics/threshold_comparison.csv` | Single-only supervision has a small, threshold-specific Historical-validation signal | confirmed |
| Experiment B P98/P95/P90 net correction is 0/0/0 | `analysis/dense_failure_stage2/shared_union_training/experiment_B_single_plus_mcts/metrics/threshold_comparison.csv` | MCTS supervision does not improve the learned routed policy | confirmed |
| A/B final non-FULL recall is 0.1920/0.1687 | experiment summaries and teacher-forced metrics | More oracle supervision did not increase non-FULL policy coverage | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat Canonical OOF as Stage-2 validation | Canonical UIDs supply union training routes | supported Stage-2 evaluation leakage | `analysis/dense_failure_stage2/robust_gate_corrective_search/work/manifests/full_work_manifest.jsonl` | Report Canonical validation as unavailable and keep conclusions Historical-only | Do not call Stage-1 OOF status a Stage-2 holdout |
| Final report aggregation | First finalizer invocation could not save plots because `figures/` was absent | supported mechanical output-directory omission | finalizer traceback; rerun after creating the directory | Create the output directory and rerun the unchanged hash-bound finalizer; training/rollouts were unaffected | Do not interpret this as a scientific failure or rerun training |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Experiment A: preservation + single | Directly tests whether expanded single-route support generalizes | Single-supervision value and threshold trade-off | high | selected first |
| Experiment B: A + MCTS | MCTS supplied most newly resolved Phase-65 pairs | Incremental value of richer trajectories | high | conditional on A health |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: Execute the frozen shared-router comparison without threshold/model confounding; the main validity bottleneck is a leakage-free validation definition.
- Relevant memory item used: Phase 58 showed positive but narrow behavior with severe near-FULL and immediate-action collapse; Phase 65 substantially broadened exact corrective support, especially via MCTS.
- Confirmed observation: the Historical-800 validation split is disjoint from all union supervision by UID, image group, and image SHA.
- Unverified interpretation: broader union supervision may improve rollout rescue without increasing C regression.
- Diagnosis: supported for the Canonical validation mismatch; unknown for whether single or MCTS supervision will generalize.
- Evidence path if diagnosis is not unknown: Phase-58 validation manifest, Phase-65 work manifest, and the overlap audit recorded above.
- Viable alternatives considered: Historical-only primary rollout; unauthorized new Canonical holdout; invalid Canonical-4000 training-overlap rollout.
- Chosen action: Run the plan on the frozen Historical-800 holdout, emit explicit N/A Canonical source cells, and limit the final operating-point statement to this validation population.
- Strongest objection: Section 26 requests Canonical source metrics, which cannot be supplied honestly from the transferred/frozen populations without changing the split or leaking training UIDs.
- How this differs from failed attempts: one shared router is trained on deduplicated cross-threshold supervision, while evaluation uses the robust five-head Stage-1 trigger at each threshold and a zero-overlap holdout.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform the named plan.
- Stop condition: corpus audit; A training and three rollouts; B only if A is technically healthy; matched comparison and Historical-validation threshold choice; no test.

## Latest Research-Action Result
- Action taken: trained matched shared Stage-2 routers with preservation+single supervision (A) and preservation+single+MCTS supervision (B), then evaluated each at the frozen P98/P95/P90 Stage-1 operating points.
- Result: A gave net corrections 0/0/+3 and B gave 0/0/0 on the leakage-free Historical-800 validation set. B selects P98 only by the prospective conservative tie-break among three equal-accuracy points; this is not a positive deployment result. Decision C applies.
- Evidence saved: `analysis/dense_failure_stage2/shared_union_training/`, including the contract, corpora, checkpoints, complete rollouts, comparisons, figures, summaries, and 145-file artifact manifest.
- Failure or issue: Canonical held-out Stage-2 validation is absent and was reported as unavailable; B under-generalized despite passing implementation and overfit gates. The cause is unknown. A mechanical missing-figure-directory error was repaired without changing code or scientific artifacts.
- Lesson learned: oracle corrective-route abundance, including MCTS routes, is not sufficient evidence that the current single-label shared router will learn useful free-rollout corrections. MCTS supervision reduced rather than expanded deployed non-FULL behavior here.
- Next implication: stop. Do not run the held-out test or choose a follow-on repair without a new prospective plan and explicit authorization.
