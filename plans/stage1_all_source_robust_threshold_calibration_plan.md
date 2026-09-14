# Stage-1 ALL-Source Robust Threshold Calibration Plan

## 1. Objective

Calibrate a single global Stage-1 trigger threshold for the validated ALL-source Shared Random-4 head.

The ALL-source head already showed:

- Historical held-out AUROC: 0.8394
- Canonical OOF AUROC: 0.7689
- Worst-source AUROC: 0.7689

The goal now is conservative admission:

> Keep dense-correct samples on FULL whenever possible, and send only sufficiently high-confidence failure cases to Stage-2.

This phase selects the Stage-1 operating point only.

Do not train Stage-2.
Do not regenerate corrective labels.
Do not rerun single search or MCTS.

## 2. Frozen Stage-1 Candidate

Freeze:

- Qwen2.5-VL backbone
- Shared Random-4 architecture
- ALL = Historical + Canonical training
- old normalization artifact
- Stage-1 feature definition
- layerwise score computation

Only the global threshold tau is selected.

## 3. Trigger Rule

For sample i, let the frozen head output scores:

s_i,0 ... s_i,27

Trigger iff:

score(layer) > tau

for any layer 0...27.

The first trigger layer is:

l_i* = first layer whose score exceeds tau.

Calibration must use this true sample-level any-layer rule. Do not calibrate individual layer scores independently.

## 4. Primary Principle

The policy is asymmetric:

- uncertain -> FULL
- high-confidence failure -> trigger Stage-2

Priority:

1. C->C preservation
2. useful W detection

Do not maximize W recall at the cost of large C false admission.

## 5. Calibration Populations

Historical:

- use Historical validation for threshold selection
- keep Historical test out of threshold selection

Canonical:

- use the existing 5-fold canonical OOF score trajectories
- never use in-sample canonical predictions

Because no untouched canonical test set exists, explicitly report cross-fold threshold stability rather than treating all canonical OOF rows as a pristine final test set.

## 6. Cross-Fitted Threshold Stability

For canonical fold k:

Calibration:
- Historical validation
- Canonical OOF folds except k

Check:
- Canonical OOF fold k

Repeat k=0...4.

Report:

- fold-selected threshold
- mean/median/IQR/range of thresholds
- held-out canonical-fold C preservation
- held-out canonical-fold W recall
- held-out trigger precision

The purpose is to verify that the chosen operating point is not driven by one canonical subset.

## 7. Threshold Sweep

Sweep tau over all useful score breakpoints or a sufficiently dense monotonic grid.

For each tau, run the actual any-layer sequential gate and compute sample-level metrics.

## 8. Metrics Per Threshold

Report separately for Historical, Canonical, and pooled ALL:

- C preservation = P(no trigger | Dense-C)
- C false admission = P(trigger | Dense-C)
- W recall = P(trigger | Dense-W)
- W miss rate
- trigger precision = P(Dense-W | trigger)
- overall trigger rate
- median/IQR first-trigger layer
- early/middle/late trigger fractions

## 9. Worst-Source Preservation

Define:

Worst-source C preservation
= min(Historical C preservation, Canonical C preservation)

This is the primary safety metric.

Never select tau from pooled C preservation alone.

A threshold giving 99% Historical preservation but 90% Canonical preservation is not robust.

## 10. Worst-Source W Recall

Also report:

Worst-source W recall
= min(Historical W recall, Canonical W recall)

and average-source W recall.

These are secondary to preservation.

## 11. Reference Operating Points

Explicitly identify the most permissive thresholds satisfying:

- >=99% worst-source C preservation
- >=98% worst-source C preservation
- >=95% worst-source C preservation

For each report:

- Historical C preservation
- Canonical C preservation
- Historical W recall
- Canonical W recall
- pooled ALL W recall
- trigger precision
- median first-trigger layer

## 12. Primary Selection Rule

Primary candidate:

Choose the threshold with the highest pooled ALL W recall subject to:

Worst-source C preservation >= 98%

Tie-break:

1. higher worst-source C preservation
2. higher worst-source W recall
3. higher trigger precision
4. later median trigger
5. higher/more conservative threshold

Do not use downstream Stage-2 outcomes for this threshold phase.

## 13. Fallback Rule

If the 98% constraint yields essentially zero useful W detection:

- do not silently relax the rule
- report the 99/98/95% operating points
- mark the 98% point as not practically useful
- require explicit review before using 95%

## 14. Why Not Use the Old Utility

Do not select tau by:

wrong triggered - correct triggered

as the primary criterion.

The current project preference is conservative C preservation.

Final end-to-end utility will later be measured as:

W->C - C->W

after Stage-2 is connected.

## 15. Dataset x Source Audit

For all candidate/reference thresholds report six cells:

- Historical GQA
- Historical ChartQA
- Historical TextVQA
- Canonical GQA
- Canonical ChartQA
- Canonical TextVQA

For each:

- N_C / N_W
- C preservation
- W recall
- trigger precision
- median trigger layer

Do not create dataset-specific thresholds.

Pay special attention to Canonical ChartQA because it is the weakest ALL-head dataset/source ranking cell.

Canonical TextVQA W estimates must include uncertainty because W support is only 19.

## 16. Catastrophic-Cell Warning

Flag any adequately supported dataset/source cell with:

C preservation < 90%

This is a warning, not a dataset-specific calibration rule.

If the chosen 98% worst-source operating point still creates a catastrophic cell, stop before freezing the gate.

## 17. Bootstrap Uncertainty

Use UID/group-level bootstrap for the primary and 99/98/95 reference points.

Estimate CIs for:

- Historical C preservation
- Canonical C preservation
- Historical W recall
- Canonical W recall
- pooled W recall
- trigger precision
- source differences

## 18. Trigger-Layer Analysis

Use depth bins:

- Early: 0-8
- Middle: 9-18
- Late: 19-27

Report first-trigger distributions separately for:

- true-trigger W
- false-trigger C
- Historical
- Canonical

Check whether conservative calibration removes the old L0 canonical false-trigger pathology while still leaving meaningful W trigger opportunities.

Do not optimize tau directly for trigger timing.

## 19. Score-Trajectory Sanity Check

For the selected/reference thresholds inspect random:

- triggered-C
- triggered-W
- near-threshold C
- near-threshold W

Verify:

- ALL-head provenance
- strict score > tau comparison
- first-trigger computation
- no stale Historical-head scores

## 20. Compare with the Old Gate

For context, compare against the old Shared Random-4 threshold (~0.92538).

Report old-gate and new-gate metrics on the same Historical and Canonical populations.

Do not keep the old threshold for backward compatibility.

## 21. Freeze Artifact

If the primary 98% rule is viable and passes the dataset/source audit, freeze:

- ALL-source head identity/hash
- normalization identity/hash
- global threshold tau*
- strict comparison rule
- first-trigger rule
- calibration population hashes
- selection rule and metrics

Write:

frozen/robust_stage1_gate.json

This becomes the authoritative Stage-1 gate for later phases.

## 22. Next Phase

After tau* is frozen, the next separately authorized phase is:

new robust Stage-1 trigger map
-> compare with old trigger map
-> Stage-2 corrective-label compatibility audit

Do not assume old trigger samples/layers remain valid.

Later classify prior labeled samples as:

- same sample + same trigger layer
- same sample + different trigger layer
- newly triggered
- no longer triggered

Only then decide which single/MCTS labels can be reused and which require re-search.

## 23. Required Figures

Create:

- preservation_vs_wrong_recall.png
- source_specific_pareto.png
- threshold_vs_source_preservation.png
- threshold_vs_source_wrong_recall.png
- threshold_vs_trigger_precision.png
- trigger_depth_by_source_and_outcome.png
- dataset_source_preservation_heatmap.png
- dataset_source_wrong_recall_heatmap.png
- cross_fold_threshold_stability.png

Main figure:

Worst-source C preservation vs source-specific/pool W recall.

## 24. Required Outputs

Use:

analysis/dense_failure_stage1/all_source_threshold_calibration/

Create:

protocol.md

inputs/
- head_manifest.json
- score_manifest.json
- calibration_population_manifest.json

crossfit/
- fold_0_threshold_sweep.csv
- fold_1_threshold_sweep.csv
- fold_2_threshold_sweep.csv
- fold_3_threshold_sweep.csv
- fold_4_threshold_sweep.csv
- threshold_stability.csv

metrics/
- full_threshold_sweep.csv
- reference_operating_points.csv
- source_breakdown.csv
- dataset_source_breakdown.csv
- bootstrap_intervals.csv
- trigger_depth_breakdown.csv
- old_vs_new_gate_comparison.csv

figures/
- preservation_vs_wrong_recall.png
- source_specific_pareto.png
- threshold_vs_source_preservation.png
- threshold_vs_source_wrong_recall.png
- threshold_vs_trigger_precision.png
- trigger_depth_by_source_and_outcome.png
- dataset_source_preservation_heatmap.png
- dataset_source_wrong_recall_heatmap.png
- cross_fold_threshold_stability.png

summaries/
- threshold_calibration_summary.md
- stage1_gate_freeze_decision.md

frozen/
- robust_stage1_gate.json

artifact_manifest.json

## 25. threshold_calibration_summary.md Must Answer

1. What is the preservation-vs-W-recall Pareto frontier?
2. Which thresholds satisfy 99%, 98%, and 95% worst-source C preservation?
3. At those thresholds, what are Historical and Canonical W recall?
4. What is the primary 98%-constraint threshold?
5. How stable is tau across canonical cross-fit checks?
6. How different are Historical vs Canonical preservation/recall?
7. Does any dataset/source cell show catastrophic false admission?
8. Is Canonical ChartQA acceptable?
9. Has the old L0 false-trigger pathology been substantially reduced?
10. How does the new gate compare with the historical gate?

## 26. stage1_gate_freeze_decision.md

Choose one:

### Decision A — Freeze robust Stage-1 gate

Use if:

- worst-source C preservation >=98%
- W recall is nontrivial
- cross-fold threshold is stable
- no catastrophic dataset/source cell

Next:

new trigger map + Stage-2 label compatibility audit

### Decision B — 98% gate too conservative

If 98% preservation gives essentially zero useful W detection, report 95% for explicit review but do not silently adopt it.

### Decision C — Calibration remains source/dataset unstable

If one source or dataset/source cell still has severe false admission, stop before trigger-map regeneration.

## 27. Stop Rule

STOP after:

- global threshold sweep
- cross-fit stability analysis
- 99/98/95 reference operating points
- primary operating-point decision
- optional robust gate freeze

Do not automatically:

- generate new trigger map
- reuse/research Stage-2 labels
- rerun corrective search
- train Stage-2
- evaluate final routed accuracy

## 28. Core Principle

The repaired Stage-1 gate should be:

one robust ALL-source head
+
one shared global threshold
+
explicit worst-source C-preservation constraint

The operating philosophy is:

> Intervene only when Stage-1 is sufficiently confident that dense FULL computation is heading toward failure, even if this sacrifices some W recall in order to protect C->C.
