# Stage-2 Abstention-Margin Calibration Plan
## Freeze the Router; Calibrate When a Non-FULL Action Is Confident Enough to Execute

## 1. Why This Experiment Matters

The full-benchmark audit identified the current dominant method-level failure:

- Overall W→C = 3
- Overall C→W = 19
- Net = -16

The key funnel result was:

- P(non-FULL | triggered W) ≈ 23.99%
- P(non-FULL | triggered C) ≈ 22.72%

while treatment outcomes were highly asymmetric:

- P(W→C | triggered W, non-FULL) = 3/119 ≈ 2.52%
- P(C→W | triggered C, non-FULL) = 19/92 ≈ 20.65%

Thus the current router does not sufficiently distinguish “this state truly deserves a corrective action” from “Stage-1 triggered, but Stage-2 should still abstain and keep FULL.”

The next experiment tests the smallest possible fix:

> Execute a non-FULL action only when it beats FULL by a sufficiently large Stage-2 confidence margin.

This directly targets treatment selectivity without changing Stage-1, Stage-2 weights, architecture, training data, search labels, or four-action semantics.

## 2. Core Research Question

At routed state s, let Stage-2 output logits:

- z_FULL
- z_READ_ONLY
- z_WRITE_ONLY
- z_IGNORE

Define:

a_nonfull* = argmax over {READ_ONLY, WRITE_ONLY, IGNORE}

margin(s) = z[a_nonfull*] - z[FULL]

Current behavior is approximately:

if margin > 0:
    choose a_nonfull*
else:
    choose FULL

Proposed rule:

if margin > δ:
    choose a_nonfull*
else:
    choose FULL

with δ >= 0.

Question:

> Does increasing the required non-FULL-vs-FULL margin suppress harmful interventions faster than beneficial interventions?

## 3. What This Experiment Can Establish

This phase can establish whether:

- Stage-2's own confidence margin contains information about intervention reliability;
- a simple conservative abstention rule reduces C→W;
- net correction can improve without retraining;
- current regression is partly a decision-threshold/selectivity problem.

## 4. What This Experiment Cannot Establish

A negative result must not be interpreted as evidence that:

- dynamic routing is impossible;
- Stage-2 representation is fundamentally unusable;
- READ/WRITE actions cannot help;
- more training cannot help.

It would establish only that the current Stage-2 non-FULL-vs-FULL confidence margin is insufficient by itself to separate beneficial from harmful intervention.

## 5. Frozen Model Contract

Freeze exactly:

- Robust ALL-source Stage-1 head
- P90 Stage-1 operating point
- Stage-2 Experiment A checkpoint
- Qwen2.5-VL backbone
- four-action executor
- LMMS-Eval correctness
- generation settings

Do not retrain anything.

The only changed variable is Stage-2 abstention margin δ.

## 6. Why Use Experiment A

Use the current single-supervision Stage-2 Experiment A because it is the only learned Stage-2 candidate that produced a positive development signal:

Historical-800 / P90:
- W→C = 4
- C→W = 1
- Net = +3

Do not use B or C in the primary margin experiment.

## 7. Development Calibration Set

Do not calibrate δ on the four-family external full-benchmark results.

The external 3 W→C / 19 C→W cases are motivation only.

Use the existing Stage-2 development population:

Historical-800 validation/development set.

No external benchmark result may influence δ.

## 8. Candidate Margin Grid

Construct the candidate margin grid before examining development W→C/C→W under each margin.

Preferred construction:

1. collect positive non-FULL-vs-FULL margins from training-side/oracle routed states already available;
2. freeze a small grid from their quantiles.

Example:

- δ0 = 0.0
- δ1 = training q10
- δ2 = training q25
- δ3 = training q40
- δ4 = training q50
- δ5 = training q60
- δ6 = training q70
- δ7 = training q80
- δ8 = training q90
- δ9 = training q95
- δ∞ = +∞

Remove duplicates.

Use one global δ. No dataset- or source-specific margins.

## 9. Why Include δ = +∞

δ = +∞ means Stage-1 may trigger but Stage-2 always chooses FULL.

This should reproduce the dense baseline exactly:

- W→C = 0
- C→W = 0
- ΔAccuracy = 0

Any mismatch indicates a runtime bug.

## 10. Development Rollout Must Be Re-Executed Per Margin

Do not estimate margin effects by masking stored actions.

Suppressing one intervention changes the downstream routed state.

Therefore for every candidate δ, run actual sequential free rollout on the full development set:

Stage-1 P90 frozen
→ Stage-2 A frozen
→ margin rule δ applied at every post-trigger layer
→ actual routed next state

Inference only. No training.

## 11. Development Metrics per Margin

For every δ report:

- Dense accuracy
- Routed accuracy
- ΔAccuracy
- W→C
- C→W
- Net = W→C - C→W
- C→C preservation
- W→C rate among Dense-W
- Stage-1 triggered count
- samples using >=1 non-FULL
- total non-FULL count
- post-trigger FULL fraction

Primary curve:

δ → W→C / C→W / Net

## 12. Core Selectivity Diagnostic

For every δ compute:

Rescue-to-regression ratio = W→C / max(C→W, 1)

and:

Net correction = W→C - C→W

The goal is not to maximize intervention count.

The goal is to retain reliable corrections while suppressing regressions.

## 13. Preservation Constraint

Define the development constraint before selection:

C→C preservation >= 99.5%

If Historical-800 still contains 400 Dense-C examples, this permits at most 2 C→W regressions.

Do not use the external 99.87% result to tune this constraint.

## 14. Margin Selection Rule

Select δ* using:

maximize Net = W→C - C→W

subject to:

C→C preservation >= 99.5%

Tie-break:

1. fewer C→W
2. higher W→C
3. fewer non-FULL interventions
4. larger δ

δ = 0 is a valid candidate.

If no positive-margin candidate improves over δ=0, keep δ*=0 and conclude that abstention-margin calibration is not supported on development.

Do not force a new margin.

## 15. Cross-Fit Stability Check

Because Historical-800 is small and prior answer changes were sparse, perform a 5-fold UID/group-level cross-fit stability analysis.

For each fold:

1. choose δ_fold on the other 4 folds using the frozen selection rule;
2. evaluate it on the held fold;
3. record held-fold Net and preservation.

Report:

- selected margin range
- median selected margin
- pooled held-fold W→C
- pooled held-fold C→W
- pooled held-fold Net

The final δ* is selected once on the full development set after this stability check.

## 16. Margin-Separation Diagnostic

On development only, inspect margins for samples where δ=0 produces:

- W→C
- C→W
- no correctness change

Use:

- first executed non-FULL margin
- maximum executed non-FULL margin

This is descriptive only.

## 17. Action-Type Diagnostic

For READ_ONLY / WRITE_ONLY / IGNORE, report development margin distributions.

Do not choose action-specific δ.

## 18. Development Decision Cases

### Case A — Margin improves Net by suppressing regressions

Proceed to one locked external re-evaluation.

### Case B — Margin suppresses both rescue and regression similarly

Confidence mostly measures willingness to intervene, not reliability.

### Case C — Positive margins collapse to all-FULL

Corrective logits are weakly separated from FULL.

### Case D — δ=0 remains best

A global Stage-2 margin is not supported on development.

Stop before external re-evaluation with a new margin.

## 19. Locked External Re-Evaluation

Only if development selects a nonzero δ*, freeze:

- Stage-1 = robust ALL-source P90
- Stage-2 = Experiment A
- Stage-2 margin = δ*

Then run one prospective external re-evaluation on:

- ChartQA
- TextVQA
- MMMU-Pro
- POPE

Use the exact same external evaluation contract as before.

Do not retune δ after seeing external results.

## 20. External Comparison

Compare original δ=0 against margin δ*:

| Family | Original W→C | Original C→W | Margin W→C | Margin C→W | ΔNet |
|---|---:|---:|---:|---:|---:|
| ChartQA | 1 | 3 | | | |
| TextVQA | 2 | 12 | | | |
| MMMU-Pro | 0 | 4 | | | |
| POPE | 0 | 0 | | | |
| Overall | 3 | 19 | | | |

Also report:

- Dense accuracy
- Routed accuracy
- ΔAccuracy
- C→C preservation
- Stage-2 non-FULL usage

## 21. External Interpretation

### Positive transfer

If C→W drops substantially while some W→C remains and Net improves, the margin is a meaningful low-complexity method improvement.

### Dev-only improvement

If development improves but external does not, do not retune on external. Conclude that selectivity calibration did not transfer sufficiently.

### External all-FULL behavior

If almost every intervention disappears, the margin is too conservative under the external score distribution.

Do not choose another margin from external results.

## 22. No Stage-1 Threshold Changes

Do not combine this experiment with P95/P98 or lower Stage-1 thresholds.

The tested question is specifically whether Stage-2 can abstain more reliably under frozen P90 admission.

## 23. No Retraining

Do not:

- fine-tune Stage-2;
- change loss;
- add MCTS;
- add valid-set loss;
- add on-policy states.

If margin calibration fails, that failure informs the next training experiment.

## 24. Required Outputs

Use:

analysis/dense_failure_stage2/abstention_margin/

Create:

- protocol.md
- margin_grid/training_margin_distribution.csv
- margin_grid/candidate_margins.csv
- development/per_margin_rollout_summary.csv
- development/per_margin_transition_counts.csv
- development/per_margin_action_behavior.csv
- development/per_sample_results.jsonl
- crossfit/fold_assignments.jsonl
- crossfit/selected_margin_by_fold.csv
- crossfit/heldout_fold_metrics.csv
- crossfit/stability_summary.json
- diagnostics/margin_by_transition.csv
- diagnostics/margin_by_action.csv
- diagnostics/selectivity_summary.csv
- selection/selected_margin.json
- selection/development_decision.md
- external/run_only_if_authorized/
- figures/net_vs_margin.png
- figures/rescue_regression_vs_margin.png
- figures/preservation_vs_margin.png
- figures/nonfull_rate_vs_margin.png
- figures/margin_by_transition.png
- figures/crossfit_margin_stability.png
- summaries/abstention_margin_summary.md
- summaries/next_improvement_recommendation.md
- artifact_manifest.json

## 25. `abstention_margin_summary.md` Must Answer

1. What candidate margins were frozen before outcome evaluation?
2. How do W→C and C→W change as margin increases?
3. Does C→W fall faster than W→C?
4. What is the best development Net under the preservation constraint?
5. What margin is selected?
6. Is it stable across folds?
7. Does it reduce non-FULL usage as expected?
8. Are rescue margins descriptively larger than regression margins?
9. Does one action type dominate low-confidence intervention?
10. Is a nonzero abstention margin supported?
11. What does a negative result not justify concluding?

## 26. `next_improvement_recommendation.md`

Recommend exactly one next step based on the result.

If margin succeeds on development and external:
- freeze it as a method component and then decide whether coverage should be increased.

If margin succeeds only on development:
- do not retune on external; diagnose selectivity generalization.

If margin fails on development:
- move away from post-hoc confidence gating and revisit Stage-2 representation/training signal.

## 27. Stop Rule

STOP after the development margin sweep, cross-fit stability analysis, and one margin selection decision.

If and only if a nonzero margin is selected prospectively, perform one locked external re-evaluation and then stop.

Do not:
- retune on external;
- change Stage-1 threshold;
- retrain Stage-2;
- run new search;
- add another confidence mechanism.

## 28. Core Principle

The full-benchmark audit showed that the current problem is not “we need more interventions.”

It is:

> **the interventions are not selective enough.**

Therefore the next smallest discriminating experiment is to require stronger evidence before overriding FULL:

best_nonFULL_logit - FULL_logit > δ

The router may correct fewer W samples.

That is acceptable if it reduces C→W even more.

The method objective remains:

W→C > C→W
