# Next Stage-1 Repair Recommendation

Do not execute a repair in this phase.

The smallest discriminating next experiment is **same head + frozen old normalization + canonical training labels**. Keep architecture, features, optimizer, split discipline, and evaluation fixed. If this recovers held-out canonical ranking, the old fitted boundary/population regime—not an inherent feature limitation—is sufficient to explain the deployment failure.

Only if that arm fails should a second arm recompute normalization from the canonical training fold. That follow-up distinguishes normalization amplification from a deeper representation/head limitation. A source-balanced old+new mixed fit should be later still; it tests mixture robustness but is not the smallest first diagnostic.

Why this is the smallest defensible action: source is strongly linearly available overall (strong for ChartQA/TextVQA, weak for GQA), exact token matching changes old test AUROC by 0.043 on average, and the frozen old score shifts sharply for canonical correct ChartQA/TextVQA. A single same-architecture fit directly tests old-boundary failure while preserving the Stage-1 target.
