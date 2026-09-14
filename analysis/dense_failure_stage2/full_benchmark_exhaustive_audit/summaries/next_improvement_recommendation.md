# One next-improvement recommendation

## Recommendation

Test a **preservation-calibrated Stage-2 abstention rule** on frozen Stage-2 logits, allowing a non-FULL action only when its confidence advantage over FULL exceeds one prospectively selected margin. Keep Stage 1, P90, the Stage-2 checkpoint, action meanings, and evaluator fixed.

## Why it matters

The pooled method loses 16 net answers because 19 correct answers regress while only 3 wrong answers are rescued. Stage-2 activation itself is not W-selective: 24.0% of triggered W and 22.7% of triggered C receive non-FULL.

## Observed failure pattern

TextVQA supplies 12 regressions versus 2 rescues; MMMU-Pro supplies 4 regressions and no rescue. Across the full population, W treatment succeeds for 3/119 non-FULL cases while C treatment regresses 19/92. This makes harmful action admission—not simply low Stage-1 W recall—the largest direct contributor to net loss.

## Separately authorized test

On a development-only population, freeze one margin from Stage-2's existing FULL-versus-best-non-FULL scores under a prospective C-preservation constraint, then evaluate that single frozen margin on the untouched full-benchmark traces or a new paired run as technically required. Do not select the margin on these 22 answer changes.

## Interpretation

- Positive: substantially fewer C→W while retaining enough W→C to make net correction nonnegative would support action-level abstention as the next local improvement.
- Negative: if regressions and rescues cannot be separated by existing Stage-2 confidence, the present checkpoint lacks useful deployment selectivity and action/timing supervision becomes the next diagnosis.
- Not justified by one negative result: abandoning four-action routing, changing Stage 1, or claiming corrective interventions cannot work.

This is a recommendation only; no calibration or evaluation is executed in Phase 70.
