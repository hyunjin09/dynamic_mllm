# One post-result decision-changing check: hold the outcome target fixed

Purpose: distinguish an OFF-score generalization collapse (BC-B) from a relative-choice failure with retained absolute signal (BC-C). This check was selected after observing the primary ON/OFF AUROC gap; it is diagnostic and does not replace a frozen primary metric. No new model execution, labels, fit, threshold or population.

Compute the same empirical AUROC for each of the four `(target, score)` combinations on all 15,185 states, using `BinaryMetric(d["y_"+target], d["p_"+score])(ones(N))`. Targets and scores each take ON/OFF. All four output rows are in fixed_target_cross_score_diagnostic.csv.

For fixed ON targets: ON score 0.672822, OFF score 0.671044. For fixed OFF targets: ON score 0.594796, OFF score 0.594880. Thus essentially the entire original branch AUROC difference is also observed changing targets while holding scores fixed. The original branch-to-branch comparison cannot by itself establish an OFF representation/scoring collapse. This does not identify the cause of weak paired preference or prove that calibration can recover it.

Independent reviewer: stable, BC-C qualified above BC-B above BC-D, medium confidence. Strongest objection retained: OFF absolute signal is moderate and the corpus is selected/internal. No additional diagnostic is necessary for this decision.
