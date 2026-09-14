# Stage-1 ALL-Source Robustness Protocol

- Frozen contract: `08dbf1464f5b1d2bdd4c9398ff11ce961a7d59aa1c52967e3505c7fac890a4fa`
- Exact historical Shared Random-4 feature/head architecture and old normalization.
- Main fold training: Historical train plus four Canonical folds; evaluation: the fifth Canonical fold and untouched Historical validation/test.
- Every training epoch has exact 25% quotas for Historical-C, Historical-W, Canonical-C, and Canonical-W. Sampling is uniform within each cell, preserving its dataset mixture in expectation; no dataset balancing is used.
- Internal validation is selected only from each run's allowed training pool and stratified by source, dataset, and current dense correctness.
- Historical primary scores average the five main-fold probabilities per layer before computing the max trajectory score.
- Each LODO head excludes its target dataset from training, internal validation, checkpoint selection, and layer selection.
- Source robustness requires both aggregate source AUROCs >= 0.70. Broad OOD requires every target's worst-source AUROC >= 0.60 and non-inferior mean Historical LODO AUROC versus the Phase-49 references.
- Ranking only: no threshold calibration, trigger map, Stage-2 change, search, or external evaluation.
