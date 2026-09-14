# Phase 61: Stage-1 Canonical Refit Diagnostic Memory

## Current Objective
Test whether the exact historical Shared Random-4 Stage-1 architecture can
recover held-out canonical dense-failure ranking when only its weights are
refit on canonical current-runtime LMMS labels, retaining the frozen old
normalization.

## Active Constraints
- Follow `plans/stage1_canonical_refit_diagnostic_plan.md`, Arm A only.
- Keep the 10,752-dimensional text-final/text-mean/visual-mean input, shared
  Random-4 architecture, optimizer/scheduler, and old global normalization.
- Use five image-group-disjoint OOF folds; keep evaluation prevalence natural
  and balance C/W only during training.
- Do not calibrate a threshold, regenerate a trigger map, change Stage 2, run
  search, execute Arm B, or train on mixed historical/canonical sources.

## Current State
- Done: Froze contract `c1a0420...0c826`, five exact 800-record outer folds,
  and training-only group-disjoint internal-validation partitions.
- Done: Reproduced 36 stratified frozen-old-head score trajectories over all
  28 layers with maximum absolute error `6.22e-08`.
- Done: Trained five balanced canonical-label folds on four direct GPUs and
  produced exactly one OOF trajectory for every one of 4,000 canonical UIDs.
- Done: Fit one full-canonical head for historical cross-evaluation only and
  completed the paired bootstrap, layerwise, score-distribution, and
  cross-regime analyses.
- Done/stopped: 23 required output artifacts pass SHA-256 verification and 13
  focused/inherited tests pass. No downstream action ran.
- In progress: none.
- Blocked: none.
- Most recent useful observation: the same head recovers canonical ranking
  overall and on GQA/ChartQA, but TextVQA remains unresolved with only 19 W;
  the canonical-trained full fit does not preserve historical ChartQA/TextVQA
  ranking.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Canonical OOF max-score AUROC is 0.8178 versus 0.4056 for the frozen old head; paired gain +0.4123, 95% CI [+0.3862,+0.4376] | `metrics/canonical_oof_summary.csv`; `metrics/paired_bootstrap_difference.csv` | The unchanged feature/head form can relearn substantial canonical failure signal under old normalization | confirmed |
| GQA/ChartQA/TextVQA OOF AUROC is 0.7214/0.7021/0.4640 | `metrics/canonical_dataset_breakdown.csv` | ChartQA inversion disappears and GQA remains useful, but TextVQA is not demonstrated | confirmed |
| TextVQA has 19 W and refit-minus-old CI [-0.4226,+0.0131], with fold AUROCs highly variable | `metrics/paired_bootstrap_difference.csv`; fold OOF predictions | TextVQA cannot support a strong recovery claim | confirmed limitation |
| Canonical Dense-C old/refit max-score mean is 0.8056/0.4677 and p95 is 1.0000/0.9186 | `metrics/score_distribution_summary.csv` | Canonical refitting reduces the extreme high-risk behavior on canonical correct samples | confirmed descriptive |
| Full-canonical fit historical val+test AUROC is 0.5511 versus 0.8885 for the old head | `metrics/historical_cross_eval.csv` | One source-specific fit does not transfer robustly in the reverse direction | confirmed |
| Frozen-old reproduction error is at most 6.22e-08; every OOF UID and image group occurs exactly once | `training/historical_recipe_reproduction_check.json`; `artifact_manifest.json` | The gain is not explained by a changed feature schema, normalization, or OOF leakage | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Transfer a source-specific fitted boundary across historical and canonical populations | Old→canonical overall AUROC is 0.4056; canonical→historical is 0.5511 | supported population-regime dependence | `metrics/historical_cross_eval.csv` | A robust deployment head needs explicit cross-regime training/evaluation rather than another one-source fit | Do not deploy either source-specific checkpoint as a universal gate |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Source-balanced historical + canonical training | The architecture learns within canonical GQA/ChartQA, but both directional cross-regime transfers fail | Whether one unchanged head can rank failures across both regimes | medium | recommended, not authorized |
| Canonical train-fold normalization (Arm B) | Could still affect TextVQA or calibration | Old-normalization contribution beyond the now-recovered primary ranking | medium | deferred because Arm A met its primary rule |

## Next-Step Decision
- Deliberation mode: standard.
- Active objective and bottleneck: Phase 61 is complete; the next bottleneck is
  learning one source-robust Stage-1 boundary without sacrificing within-source
  signal.
- Relevant memory item used: Phase 60 found source accessibility and old
  fitted-geometry shift but could not determine whether the head remained
  canonically learnable.
- Confirmed observation: canonical OOF ranking recovers strongly overall and
  for GQA/ChartQA with the same architecture and old normalization; reverse
  historical transfer collapses and TextVQA is unresolved.
- Unverified interpretation: whether source-balanced mixed training can retain
  both regimes without exploiting their different label prevalences.
- Diagnosis: supported source-specific fitted-boundary/population-regime
  failure; TextVQA-specific learnability remains unknown.
- Evidence path if diagnosis is not unknown:
  `analysis/dense_failure_stage1/canonical_refit_diagnostic/`.
- Viable alternatives considered: source-balanced mixed training; Arm B
  canonical normalization; architecture change.
- Chosen action: stop under Decision A. Recommend a separately planned,
  source-balanced old+canonical robustness experiment before any Stage-2 label
  regeneration; defer Arm B and architecture changes.
- Strongest objection: overall AUROC benefits from different dataset failure
  prevalences, and TextVQA has only 19 W. The within-dataset GQA/ChartQA OOF
  results mitigate but do not remove that limitation.
- How this differs from failed attempts: the completed test changed only fitted
  weights and evaluated every canonical UID out of fold.
- Automatic execution authorized: no.
- Authorization basis: the user authorized only the Phase-61 plan.
- Stop condition: reached.

## Latest Research-Action Result
- Action taken: completed Arm A five-fold canonical refitting, paired frozen-old
  comparison, layerwise analysis, and full-canonical historical cross-evaluation.
- Result: Decision A under the prospective rule; overall OOF AUROC 0.8178,
  ChartQA inversion removed, but TextVQA unresolved and historical reverse
  transfer weak.
- Evidence saved:
  `analysis/dense_failure_stage1/canonical_refit_diagnostic/`.
- Failure or issue: no execution failure; scientific limitation is only 19
  canonical TextVQA-W samples and poor reverse source transfer.
- Lesson learned: the historical fitted boundary/population regime was a major
  failure source, while the architecture retains canonical failure signal for
  GQA and ChartQA. This is not evidence that one source-specific head is robust.
- Next implication: await explicit authorization for any source-balanced mixed
  Stage-1 experiment; do not change thresholds or downstream data yet.
