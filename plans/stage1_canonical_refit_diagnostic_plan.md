# Stage-1 Canonical Refit Diagnostic Plan
## Same Head + Frozen Old Normalization + Canonical Current-Runtime Labels

## 1. Objective

Run the smallest discriminating Stage-1 repair experiment after the historical-population / shortcut audit.

Primary question:

> If we keep the Stage-1 architecture, input features, and old normalization fixed, but refit the head on the new canonical-source current-runtime correctness labels, does held-out canonical failure ranking recover?

This experiment distinguishes:

- old fitted decision boundary / historical selection-regime failure
- from deeper Stage-1 feature or head limitation.

Do not change Stage-2.
Do not rerun corrective search.
Do not retune the Stage-1 trigger threshold in the primary analysis.
Do not introduce a new router architecture.

## 2. Frozen Reference

Current frozen old-head behavior:

- Historical test AUROC:
  - GQA: 0.747
  - ChartQA: 0.960
  - TextVQA: 0.972
- New canonical AUROC:
  - GQA: 0.745
  - ChartQA: 0.347
  - TextVQA: 0.685

Canonical 4K population:

- GQA: C=1264, W=736
- ChartQA: C=884, W=116
- TextVQA: C=981, W=19
- Overall: C=3129, W=871

The prior shortcut audit showed that source identity is strongly encoded for ChartQA/TextVQA, measurable nuisance variables create real shortcut opportunity, visual-token matching does not explain most old ranking, and new correct ChartQA/TextVQA states move toward the historical wrong geometry.

## 3. Primary Hypothesis

### H1 — Old-fit / population-regime hypothesis

The Stage-1 feature representation still contains useful canonical failure information, but the head trained on the historical quota-selected population learned a source-sensitive decision boundary.

Prediction:

same architecture + same old normalization + canonical current-runtime labels
→ held-out canonical AUROC recovers substantially.

If this occurs, the Stage-1 architecture remains viable.

## 4. Alternative Hypothesis

### H2 — Representation / normalization / head limitation

Even when trained directly on canonical current-runtime labels, the same head with frozen old normalization cannot rank canonical failures reliably.

If this occurs, the next diagnostic should recompute normalization from the canonical training fold before changing the head architecture.

## 5. Frozen Components

Use exactly the historical Stage-1 implementation.

Freeze:

- Qwen2.5-VL backbone
- Stage-1 input feature definition
- feature block concatenation
- Shared Random-4 head architecture
- layer-sharing pattern
- Random-4 training mechanism
- optimizer family
- scheduler family
- regularization recipe
- old normalization artifact

The only newly fitted parameters are the Stage-1 head weights.

Do not add source IDs, dataset IDs, visual-token count, explicit layer embeddings, larger MLPs, new pooling, or new normalization.

## 6. Canonical Labels

Use only the frozen canonical 4K current-runtime labels.

Target:

- y=1 if dense FULL answer is wrong
- y=0 if dense FULL answer is correct

Do not use W→C membership, single-fixable labels, MCTS labels, Stage-2 labels, or trigger labels.

## 7. Split Discipline

Because TextVQA has only 19 wrong samples, prefer 5-fold group-disjoint cross-validation rather than a single 80/20 split.

Stratify by dataset and dense correctness as far as group constraints allow.

Requirements:

- zero UID overlap
- zero image-content-group overlap
- zero SHA-256 image overlap where applicable

Every canonical sample should receive exactly one out-of-fold prediction.

Held-out folds must not contribute to training, normalization, or early-stopping decisions.

If the historical recipe requires an internal validation set, derive it only from the current training folds.

## 8. Why OOF Predictions

Primary output should be one prediction per canonical sample from a model that did not train on that sample.

This yields 4,000 out-of-fold canonical score trajectories and makes better use of the tiny TextVQA-W support.

Do not report in-sample canonical AUROC as evidence of recovery.

## 9. Training Sampling

Keep evaluation prevalence natural.

For training only, use simple C/W balancing so 871 W examples are not overwhelmed by 3,129 C examples.

Preferred first choice:

- balanced C/W sampling within each training fold

Do not rebalance held-out folds.

Do not create dataset-specific thresholds or heads.

Record the exact training sampling probability.

## 10. Dataset Imbalance Caveat

Support is highly nonuniform:

- GQA W=736
- ChartQA W=116
- TextVQA W=19

For TextVQA, report point estimate, bootstrap CI, and support counts. Avoid strong claims from the point estimate alone.

Overall canonical OOF ranking is the primary metric.

## 11. Primary Metrics

Using concatenated out-of-fold predictions, report:

- Overall AUROC
- GQA AUROC
- ChartQA AUROC
- TextVQA AUROC
- Overall and per-dataset AUPRC

Also report layerwise AUROC/AUPRC for L0...L27 and the deployed max/trajectory score.

Do not select a new threshold yet.

Ranking comes first.

## 12. Direct Comparison Against Frozen Old Head

Evaluate the frozen old head on the exact same 4K canonical samples.

Keep paired scores:

- frozen old head
- canonical-refit OOF head

Compute paired UID-level bootstrap intervals for AUROC differences overall and by dataset.

Primary question:

> Does refitting the same head on canonical labels restore failure ranking?

## 13. Ranking-Recovery Decision

Call the refit a ranking recovery if:

1. overall OOF AUROC is clearly above random;
2. paired improvement over the frozen old head on canonical data is positive and material;
3. ChartQA no longer shows the severe inversion seen in the old head;
4. no provenance or implementation issue explains the gain.

Do not require recovery to historical ~0.95 AUROC to conclude that the old fitted boundary was a major problem.

## 14. Layerwise Diagnostic

For each layer L0...L27 compare:

- old frozen canonical AUROC/AUPRC
- canonical-refit OOF AUROC/AUPRC

Questions:

- Does signal recover from L0 onward?
- Does the strongest layer move?
- Does canonical refitting mainly repair early-layer source inversion?
- Is the recovered signal broad or late-concentrated?

Do not change the deployed gate yet.

## 15. Minimal Source-Sensitivity Recheck

Using new OOF scores, compare historical vs canonical score distributions within Dense-C and Dense-W separately.

The purpose is only to see whether canonical refitting removes the extreme canonical-correct high-risk behavior.

Do not rerun the full shortcut audit.

## 16. Historical Cross-Evaluation

After OOF evaluation, train one final canonical head on the full canonical population using the same frozen recipe.

This full-canonical fit is not used for canonical performance claims.

Evaluate it on the frozen historical validation/test sets.

Report:

- canonical-trained head → historical held-out AUROC
- old historical head → historical held-out AUROC

Interpretation:

- canonical OOF high + historical held-out still high → canonical fit may be more robust
- canonical OOF high + historical held-out collapses → source-specific boundaries differ; mixed old+new training likely needed later

Do not use historical performance to select the canonical model.

## 17. Calibration Is Secondary

Do not optimize a deployment threshold until ranking recovery is established.

If ranking recovers, report only descriptive calibration:

- score distributions
- correct-tail quantiles
- wrong-tail distributions

Do not freeze a final threshold in this phase.

## 18. No Stage-2 / Search Changes Yet

Even if the canonical refit changes trigger behavior, do not immediately regenerate trigger maps, reuse labels, rerun search, or train Stage-2.

First decide whether Stage-1 itself is repaired.

## 19. Contingent Arm B — Only If Arm A Fails

Do not execute automatically unless separately authorized.

If same head + old normalization remains weak:

- same architecture
- same features
- canonical labels
- canonical train-fold normalization

For each fold, normalization statistics must come from the training fold only.

Compare Arm A vs Arm B.

Interpretation:

- B >> A → old normalization materially amplified source shift
- B still weak → deeper feature/head limitation becomes more plausible

Do not change architecture before this comparison.

## 20. Mixed Old+New Training Comes Later

Do not run mixed training in the primary diagnostic.

Only after canonical learnability is established should a later phase test source-balanced historical + canonical training.

That experiment asks whether one Stage-1 head can remain robust across both regimes.

## 21. Sanity Checks

Require:

1. exact historical feature schema
2. exact old normalization artifact in Arm A
3. no OOF leakage
4. canonical dense correctness matches frozen LMMS labels
5. zero fold group/image overlap
6. bounded reproduction check that the training code matches the historical recipe

## 22. Required Tables

### Canonical OOF ranking

| Head | Overall AUROC | GQA | ChartQA | TextVQA | Overall AUPRC |
|---|---:|---:|---:|---:|---:|
| Frozen old head | | | | | |
| Canonical refit, old norm | | | | | |

### Layerwise recovery

| Layer | Old canonical AUROC | Refit canonical OOF AUROC | Difference |
|---:|---:|---:|---:|
| 0 | | | |
| ... | | | |
| 27 | | | |

### Cross-regime evaluation

| Training regime | Historical held-out | Canonical held-out |
|---|---:|---:|
| Historical old head | | |
| Canonical refit | | |

## 23. Required Figures

Create:

- canonical_old_vs_refit_roc.png
- canonical_layerwise_auroc_recovery.png
- canonical_layerwise_auprc_recovery.png
- chartqa_score_distribution_old_vs_refit.png
- textvqa_score_distribution_old_vs_refit.png
- canonical_refit_cross_regime_comparison.png

## 24. Required Outputs

Use:

analysis/dense_failure_stage1/canonical_refit_diagnostic/

Create:

- protocol.md
- splits/fold_manifest.jsonl
- splits/fold_summary.csv
- training/fold_0 ... fold_4
- training/full_canonical_fit/
- training/checkpoint_manifest.json
- predictions/canonical_oof_scores.jsonl
- predictions/frozen_old_head_canonical_scores.jsonl
- predictions/full_canonical_head_historical_scores.jsonl
- metrics/canonical_oof_summary.csv
- metrics/canonical_dataset_breakdown.csv
- metrics/layerwise_auroc.csv
- metrics/layerwise_auprc.csv
- metrics/paired_bootstrap_difference.csv
- metrics/historical_cross_eval.csv
- metrics/score_distribution_summary.csv
- figures/*
- summaries/canonical_refit_summary.md
- summaries/stage1_repair_decision.md
- artifact_manifest.json

## 25. canonical_refit_summary.md Must Answer

1. Does the same Stage-1 architecture learn canonical dense-failure ranking when trained on canonical labels?
2. What is canonical OOF AUROC overall?
3. What are GQA / ChartQA / TextVQA AUROCs?
4. Does ChartQA inversion disappear?
5. Which layers recover most strongly?
6. Does canonical refitting reduce high scores on canonical Dense-C?
7. How large is the paired improvement over the frozen old head?
8. How uncertain is TextVQA because of only 19 canonical wrong samples?
9. How does the full canonical fit perform on old historical held-out data?
10. Is the evidence sufficient to blame the old fitted boundary/population regime rather than the head architecture?

## 26. stage1_repair_decision.md

Choose one:

### Decision A — Same architecture is adequate

Use when canonical OOF ranking clearly recovers.

Conclusion:

> The main deployment failure was the historical fitted boundary / population regime, not an inherent inability of the Stage-1 feature/head to represent canonical failure.

Next phase:

- source-balanced old + canonical robust Stage-1 training

Do not yet regenerate Stage-2 labels.

### Decision B — Old normalization is the next suspect

Use when canonical refit with old normalization remains weak.

Next phase:

- same head
- canonical train-fold normalization
- canonical labels

### Decision C — Deeper representation/head limitation

Only consider after both old-normalization and canonical-normalization refits fail.

## 27. Stop Rule

STOP after:

- same-head canonical refit with frozen old normalization
- canonical OOF ranking analysis
- historical cross-evaluation
- decision summary

Do not automatically run:

- Arm B normalization refit
- old+new mixed training
- new threshold calibration
- new trigger map
- corrective search
- Stage-2 training
- test deployment

## 28. Core Logic

OLD:

historical labels + old normalization + same head
→ excellent same-regime ranking
→ severe canonical failure

NOW:

canonical labels + old normalization + same head
→ ?

If ranking recovers:

representation/head can learn canonical failure
→ old fitted boundary / population regime was dominant

If ranking does not recover:

keep architecture fixed
→ change normalization next

This is the smallest clean test before reopening the rest of the Dynamic MLLM pipeline.
