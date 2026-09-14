# Phase 62: Stage-1 ALL-Source Robustness Memory

## Current Objective
Fit the unchanged Shared Random-4 Stage-1 head on source-balanced Historical
plus Canonical data, measure robustness on both source regimes, and run three
target-blind leave-one-dataset-out evaluations.

## Active Constraints
- Follow `plans/stage1_all_source_mixed_training_and_ood_plan_v2.md` as one
  bounded action.
- Keep the exact 10,752-dimensional feature definition, old global
  normalization, Shared Random-4 architecture, optimizer/scheduler, and
  current-runtime dense correctness target.
- Give Historical-C, Historical-W, Canonical-C, and Canonical-W exact equal
  epoch quotas; preserve the natural dataset mixture within each cell.
- Reuse the canonical five-fold group-disjoint OOF assignment and keep the
  historical validation/test sets untouched for the main evaluation.
- Exclude each LODO target dataset from training, internal validation,
  checkpoint selection, and layer selection.
- Do not calibrate thresholds, regenerate trigger maps or labels, change Stage
  2, run corrective search, or perform external evaluation.

## Current State
- Done: Read the complete plan and relevant Phase-49/Phase-61 evidence.
- Done: Verified 11,999 aligned Historical+Canonical rows, zero cross-source
  UID/image-group/image-SHA overlap, and finite BF16 features with exact shape
  `[11999,28,10752]`.
- Done: Implemented exact four-cell sampling, source/dataset/group-aware
  internal validation, target exclusion, deterministic training, ensemble and
  OOD aggregation, and fail-closed provenance checks.
- Done: 22 focused/inherited tests pass before contract freeze.
- Done: Froze contract `08dbf146...90a4fa`; trained five main folds and three
  LODO heads on four direct GPUs with exact source×correctness quotas.
- Done: Produced 4,000 unique Canonical OOF trajectories, five-fold ensemble
  scores for all 1,600 Historical validation/test rows, and complete full-target
  LODO scores.
- Done/stopped: Decision B under the prospective rule; all 85 output artifact
  hashes and all eight checkpoint provenance records validate, and 22 tests
  pass. No threshold or downstream action ran.
- In progress: none.
- Blocked: none.
- Most recent useful observation: source mixing yields aggregate Historical /
  Canonical AUROC 0.8394 / 0.7689, but max-trajectory LODO worst-source AUROC is
  only 0.5691 ChartQA, 0.6243 TextVQA, and 0.5931 GQA.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Historical specialist transfers at 0.4056 canonical AUROC; canonical specialist transfers at 0.5511 historical AUROC | Phase-61 source summary | Motivates a shared source-balanced fit | confirmed |
| Historical and Canonical populations have 0 cross-source UID/group/SHA overlap | preflight audit of Phase-48 and Phase-61 manifests | Supports clean source evaluation and mixed training | confirmed |
| Historical has 7,999 rows and Canonical has 4,000; all 11,999 features load finite as `[N,28,10752]` | full read-only feature preflight | Rules out schema/order incompatibility before fitting | confirmed |
| Phase-49 full-target Historical OOD references are 0.4502 ChartQA, 0.6160 GQA, 0.7236 TextVQA | `analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md` | Fixes the qualified historical comparison | confirmed |
| ALL-source Historical/Canonical AUROC is 0.8394/0.7689; worst-source 0.7689 | `main_all/metrics/source_summary.csv`; `robustness_metrics.csv` | One head maintains useful aggregate ranking across both observed source regimes | confirmed |
| Historical GQA/ChartQA/TextVQA is 0.7370/0.9487/0.9488; Canonical is 0.7538/0.5826/0.6111 | `main_all/metrics/dataset_source_breakdown.csv` | Aggregate source robustness masks weaker Canonical ChartQA/TextVQA cells | confirmed |
| LODO Historical/Canonical AUROC is 0.7016/0.5691 ChartQA, 0.6243/0.6259 TextVQA, and 0.6034/0.5931 GQA | `lodo/lodo_summary.csv` | Dataset transfer remains mixed and fails the frozen all-target 0.60 worst-source rule | confirmed |
| Post-hoc L26 has minimum AUROC 0.6270 across all six target×source cells | per-target `layerwise_auroc.csv` | Transferable late-layer signal may exist, but selecting L26 from target results would be post-hoc | confirmed diagnostic only |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Require one unique image group per combined UID | Preflight rejected Historical repeated-image groups | supported overstrict implementation guard | manifest audit: 11,999 UIDs / 11,476 groups, 513 multi-record groups | Require zero cross-partition/source overlap while preserving whole groups | Do not discard or split repeated Historical images |
| Interpret LODO Historical target as only original val/test | Phase-49 reference supports revealed full target populations (4,000/1,999/2,000) | supported protocol mismatch | Phase-49 layerwise support counts | Evaluate every row of the completely excluded target dataset | Do not compare a val/test-only LODO result to Phase 49 |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Exact source×correctness-balanced ALL fit | Directly suppresses the known source/outcome prior while changing only population | Whether one shared boundary survives source shift | medium | completed / source-robust |
| Three ALL-source LODO fits | Directly tests target-blind benchmark transfer under the same sampler | Whether source diversity also improves dataset OOD | medium | completed / mixed |
| In-scope ALL-mixture threshold calibration | Source robustness passed, while universal OOD did not | A conservative gate for only the declared three-benchmark mixture | medium | recommended by Decision B, not authorized |

## Next-Step Decision
- Deliberation mode: standard.
- Active objective and bottleneck: Phase 62 is complete; the remaining
  bottleneck is calibrating a conservative gate for the declared in-scope
  mixture without making a benchmark-universal claim.
- Relevant memory item used: Phase 61 showed same-head canonical learnability
  but poor bidirectional cross-source transfer.
- Confirmed observation: source-balanced ALL training repairs bidirectional
  source transfer in aggregate, but the frozen max-trajectory LODO metric is
  below 0.60 for Canonical ChartQA and Canonical GQA.
- Unverified interpretation: whether a prospectively selected common late layer
  could generalize beyond these observed OOD targets; L26 is post-hoc evidence.
- Diagnosis: supported source-regime repair with remaining benchmark-specific
  transfer limitation.
- Evidence path if diagnosis is not unknown:
  `analysis/dense_failure_stage1/canonical_refit_diagnostic/`.
- Viable alternatives considered: in-scope ALL-mixture threshold calibration;
  new layer selection; shared-head conflict diagnosis.
- Chosen action: stop under Decision B. Recommend only separately authorized
  in-scope ALL-mixture threshold calibration; do not select L26 post hoc or
  claim benchmark-general detection.
- Strongest objection: Canonical ChartQA is only 0.5826 even in the main OOF
  mixture and Canonical TextVQA has only 19 W, so aggregate source robustness is
  not uniform per-dataset robustness.
- How this differs from failed attempts: source/outcome quotas are equalized and
  both source directions plus target-blind dataset transfer are measured.
- Automatic execution authorized: yes.
- Authorization basis: the user explicitly requested this plan.
- Stop condition: reached after source robustness, all three LODO evaluations,
  artifact audit, and one robustness decision.

## Latest Research-Action Result
- Action taken: completed exact four-cell ALL-source five-fold training,
  Historical ensemble evaluation, Canonical OOF evaluation, and three
  target-blind full-population LODO fits.
- Result: Decision B. ALL-source average/worst source AUROC is 0.8042/0.7689;
  LODO is mixed and fails the frozen broad-transfer rule.
- Evidence saved:
  `analysis/dense_failure_stage1/all_source_robustness/`.
- Failure or issue: no execution failure. Canonical ChartQA is the weakest main
  cell (0.5826); LODO Canonical ChartQA and GQA are below 0.60, and Canonical
  TextVQA uncertainty is wide because W=19.
- Lesson learned: equal source/outcome exposure is sufficient for a useful
  shared boundary over the current mixture, but not evidence for a universal
  benchmark-agnostic detector.
- Next implication: await explicit authorization before any in-scope threshold
  calibration; do not generate a trigger map or touch Stage 2.
