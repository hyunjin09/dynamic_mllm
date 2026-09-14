# Stage-1 Repair Decision

## A — Same architecture is adequate

Evidence:

- Canonical refit OOF AUROC: 0.8178; chance-difference 95% CI [+0.3028, +0.3323].
- Refit-minus-old paired AUROC difference: +0.4123, 95% CI [+0.3862, +0.4376].
- ChartQA canonical OOF AUROC: 0.7021.

The same architecture is adequate for canonical failure ranking. The next separately authorized phase would test source-balanced historical + canonical robust Stage-1 training; do not regenerate Stage-2 labels yet.

This phase stops here. No threshold selection or downstream execution was performed.
