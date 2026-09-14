# Frozen branch-critic diagnostic protocol

Execute only the user-selected plan, after safely stopping Phase85. All 15,185 exact dense states / 1,413 UIDs / 1,385 image groups are primary. No data subset, retraining, recalibration, threshold optimization, search, suffix generation or controller. Existing in-sample five-head gate reuse is an offline diagnostic, not held-out training generalization.

Hard gate: fresh deterministic 32-UID stratified parity, covering both sources, all datasets, Dense W/C, early/middle/late and layer27. At every included state verify exact live compact prestate, common branch prestate, action bits, ON canonical output, native Stage1 feature parity and both compact post-state hashes against Phase82. The full extraction repeats these same checks, without resampling. Missing OFF Stage1 representations (100%) require one-layer replay; native ON cached features are reused and checked exactly. No new branch suffix/answer evaluation.

Metrics: state-micro branch failure AUROC and average precision (AUPRC), Brier, 10 fixed equal-width ECE bins; per-UID macro descriptive checks; probability-difference Pearson/Spearman and strict sign agreement, with exact zero targets excluded and score ties counted as no strict preference. Fixed tiny-q sensitivity |H_R|>1e-4,1e-3,1e-2 is descriptive. Behavioral flips are the primary preference check; balanced accuracy is half the two class accuracies, discordant AUROC labels harmful=1. All key intervals use 5,000 paired image-group bootstrap draws, seed2026091301, resampling full groups; missing-class replicates omitted and counted. No subgroup cherry-picking or arbitrary new decision threshold.

Quadrants use the plan's >=tau rule. First-trigger policy chooses OFF iff ON>=tau and OFF<ON; exact ties choose ON. Fixed branches then FULL continuation use frozen q/correctness only. Report all 1,413 decisions, treated denominators and W→C minus C→W. Descriptive layer bins0–8/9–18/19–27; exact absolute and trigger-relative layers, Dense class and dataset/source cells.

Stop after full verification, BC-A/B/C/D evidence assessment and exactly one unexecuted next recommendation.
