# Phase 79: Predictability Step-B In-Domain Learnability Memory

## Current Objective
Execute the frozen five-fold group-disjoint in-domain learnability study in `plans/predictability_phase_stepB_full_id_learnability_plan.md` for Stage-1 dense failure and Stage-2 READ/WRITE utility. Stop after the Step-B evidence classification and Step-C readiness report.

## Active Constraints
- Reuse Step-A contract `6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d` and its labels/features exactly; never regenerate or change targets.
- Main estimates are five-fold image-group-disjoint OOF with a single shared base-group registry, UID-equal total training weight, fold-local preprocessing/target scaling, and three fixed M2/M3 seeds.
- M0 is nuisance-only; M1/M2 receive no dataset/source/layer/trigger IDs; Stage-1 M3 retains a layer embedding only because it is structural in the frozen current architecture. Stage-2 M3 uses the current READ/WRITE cross-modal topology trained from scratch inside each fold.
- Keep primary dense and secondary selection-biased routed results separate. No nearest-question analysis, clustering, LODO, external four-benchmark evaluation, deployment evaluation, target redesign, or Step-C execution is authorized.
- Use four direct RTX 6000 Ada GPUs after a fresh occupancy check. Put large derived caches/checkpoints under `/mnt/hyemin`; keep only compact artifacts in the repository.
- Preserve the existing dirty worktree and do not commit or modify unrelated prior-phase files.

## Current State
- Completed/stopped: contract `792cc760bb41ce8c9b11776f21f43a88e54409ef1543471709000f83ec21bba4` froze the shared five-fold registry, exact Step-A sources, current model families, three seeds, runtime, and all implementation hashes. All four direct GPUs completed 600/600 OOF training tasks and 140/140 state-regime transfer tasks.
- Completed/stopped: 10,399 Stage-1 UIDs/9,982 image groups/291,172 states and 1,413 dense Stage-2 UIDs/1,385 groups/15,185 states were evaluated OOF with zero UID/group overlap. The 35,565-state routed study remained separate and selection-qualified.
- Completed/stopped: Stage-1 M0/M1/M2/M3 AUROC is 0.6814/0.7665/0.7874/0.7869; M3 peaks at layer 20 with AUROC 0.8257. Dense Stage-2 M3 joint READ/WRITE Spearman is only 0.0416/0.0347, with harmful AUROC 0.5193/0.5115 and top-10% precision 0.4937/0.4775 versus natural prevalence 0.480/0.472.
- Completed/stopped: all eight preregistered group-bootstrap intervals have 5,000 valid draws; all 79 final compact artifact hashes independently verify. Evidence category is Case D, Stage-1 strong / Stage-2 weak. `READY_FOR_STEP_C = true` is procedural only; Step C was not run.
- Blocked: none.
- Most recent useful observation: current state carries substantial in-domain eventual-failure signal above nuisance, but immediate READ/WRITE utility is near chance, provides no branch specialization, and yields no useful conservative harmful subset under the complete frozen capacity ladder.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Step A is complete and `READY_FOR_STEP_B = true` | `analysis/predictability_generalization/stepA_measurement/summaries/stepB_readiness.md` | Authorizes the fixed learnability question without measurement repair | confirmed |
| Complete populations: 10,399 Stage-1 UIDs, 1,413 dense Stage-2 UIDs, 569 routed UIDs | `analysis/predictability_generalization/stepA_measurement/artifact_manifest.json` | Fixes OOF populations and expected completeness | confirmed |
| Primary utilities have broad positive and negative support | `analysis/predictability_generalization/stepA_measurement/stage2_dense/utility_distribution_summary.csv` | Makes regression and harmful-ranking evaluation estimable | confirmed |
| Current Stage-1 and Stage-2 architecture definitions already exist | `dense_failure_stage1/shared_global_gate.py`; `dense_failure_stage2/v1_router.py` | Prevents an unapproved model-family change | confirmed |
| Four RTX 6000 Ada GPUs are currently idle | live `nvidia-smi` on 2026-09-08 | Supports the plan's three-seed full ladder | confirmed but transient |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Import a conventional sklearn/scipy analysis stack | Both packages are absent from `.venv` | supported | project-local import audit | Use existing Numpy/PyTorch metric and optimizer patterns; do not add a dependency that does not reduce the streaming problem | Do not modify the environment merely for familiar APIs |
| Initial packed-state training loader | Row-wise memmap collation and unconstrained CPU threading left GPUs underfed | supported | batch timing and live utilization; exact packed/prefetch parity tests | Direct pinned allocation, one-batch prefetch, and bounded worker threads reduced measured batch time about 2.6x without changing order/content | Do not restore row-wise packed collation or CPU oversubscription |
| First final aggregation | `_stage2_outputs` returned `oof_read/oof_write`, while all consumers required `read/write` | supported | deterministic `KeyError: 'read'` after all fits completed | Preserve the parent aggregator hash and use a provenance-chained output-local alias wrapper | Do not refreeze/relabel/retrain for a reporting-only key mismatch |
| First 5,000-draw bootstrap pass | Frozen implementation rescanned up to 291,172 tied-score groups in Python for every draw | supported | live stack trace and 1.49-second reference draw benchmark | Cache immutable stable score orders/tie groups and use algebraically equivalent vectorized reductions; randomized and full-size checks had zero numerical difference | Do not run the per-row Python tie loop inside every bootstrap draw |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Pure-PyTorch full ladder with cache-aware loaders and three seeds | Matches existing model contracts and available environment | Complete Step-B question without dependency or target changes | high | completed |
| Install sklearn/scipy for M0/M1, retain PyTorch M2/M3 | Standard linear baselines | Only implementation convenience; does not solve state streaming | medium | rejected as dominated |
| One seed or reduced routed ladder | Cheaper | Partial learnability estimate | medium | rejected because compute is available and the plan specifies the full ladder |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: Step B is complete; any next research action would ask whether the observed signal survives stronger semantic/source distribution shift, but no Step-C execution is authorized.
- Relevant memory item used: Phase 78 established valid signed utilities but explicitly made no predictability claim.
- Confirmed observation: Stage-1 state models materially exceed nuisance and rise with depth; every dense Stage-2 state model remains weak, READ nuisance exceeds the state models, and z_R/z_W specialization is absent.
- Unverified interpretation: stronger semantic/source-disjoint testing will preserve the in-domain Stage-1 advantage; Step B cannot answer this.
- Diagnosis: supported Case D for the frozen in-domain ladder; causal reason for weak Stage-2 utility prediction remains unknown.
- Viable alternatives considered: stop at the authorized boundary; proceed to the plan-defined Step C under separate authorization; redesign Stage-2 representation/target (strategic pivot, not authorized).
- Chosen action: stop with the complete Step-B artifacts and procedural Step-C readiness report.
- Strongest objection: pooled in-domain Stage-1 performance may still rely on semantic/template structure shared across folds, so it must not be promoted as OOD robustness.
- How this differs from failed attempts: this evaluates controlled continuous utility directly on unseen image groups rather than fitting incomplete route/action labels or interpreting route recall as utility predictability.
- Automatic execution authorized: no further research action.
- Authorization basis: the user authorized Step B only, whose stop condition is now satisfied.
- Stop condition: all required OOF ladders, controls, routed analysis, bootstrap CIs, evidence category, hashes, and Step-C readiness are frozen; do not execute Step C.

## Latest Research-Action Result
- Action taken: completed the frozen five-fold, three-seed Stage-1/Stage-2 capacity ladders, controls, routed secondary study, bidirectional state-regime transfer, and 5,000-draw group bootstrap on four direct GPUs.
- Result: Case D. Stage-1 M3 AUROC/AUPRC is 0.7869/0.6860 (95% group-bootstrap AUROC CI [0.7789, 0.7946]); dense Stage-2 M3 READ/WRITE Spearman is 0.0416/0.0347 with CIs [0.0200, 0.0643]/[0.0124, 0.0571]. Harmful-ranking CIs approach/include chance and high-confidence precision is near natural prevalence. Routed-state and bidirectional transfer correlations remain weak and selection-qualified.
- Evidence saved: `analysis/predictability_generalization/stepB_id_learnability/`; final artifact manifest `6dc42193bac74aedf479cc70c39c51a9e06e861cc255f21a6c86db04db3a3b4c`; aggregation patch contract `fb2110750899eb93cbf34bbe1ca471ba883780380d12f843f8804d738135d3d1` chained to parent execution contract `792cc760bb41ce8c9b11776f21f43a88e54409ef1543471709000f83ec21bba4`.
- Failure or issue: two aggregation-only defects were repaired without changing or rerunning scientific fits; the original frozen aggregator remains byte-identical, and exact/reference numerical equivalence plus 26 focused tests pass.
- Lesson learned: eventual dense failure is substantially decodable in-domain from current hidden state, but current one-bit READ/WRITE utility is not readily predictable by the tested nuisance/linear/MLP/router ladder and does not exhibit the expected branch specialization.
- Next implication: stop. Step C is procedurally ready but requires a separately authorized plan; do not redesign the Stage-2 target/representation or infer OOD/deployment validity from Step B.
