# Post-hoc frozen-gate shift diagnostic

Date: 2026-09-02. This is a read-only interpretation diagnostic over the
accepted Phase-59 outputs; it does not alter the frozen search artifacts or
their manifest.

## Validity checks

- The Phase-59 contract binds the same Shared Random-4 checkpoint,
  normalization, global threshold `0.9253839280601031`, strict `>` comparison,
  Qwen snapshot, feature definitions, and LMMS correctness implementation used
  by the prior trigger audit.
- All 4,000 Phase-59 dense rows and score trajectories completed, and the
  accepted trigger binding contains exactly 4,000 rows.
- Manual inspection of representative triggered-correct ChartQA and TextVQA
  rows showed exact generated-answer/GT agreement and the expected LMMS label.

## Score-distribution comparison

The score below is the maximum frozen-gate score over layers 0-27.

| Cohort | Dataset | Class | N | Trigger rate | Median max score | Median L0 score | First trigger at L0 / N |
|---|---|---:|---:|---:|---:|---:|---:|
| Old | ChartQA | C | 799 | 0.0088 | 0.2023 | 0.1712 | 0.0000 |
| New | ChartQA | C | 884 | 0.8654 | 1.0000 | 0.9801 | 0.7783 |
| Old | TextVQA | C | 800 | 0.0138 | 0.1097 | 0.0684 | 0.0037 |
| New | TextVQA | C | 981 | 0.8522 | 0.9984 | 0.9527 | 0.6137 |
| Old | GQA | C | 1,600 | 0.0131 | 0.5041 | 0.4075 | 0.0000 |
| New | GQA | C | 1,264 | 0.0712 | 0.6167 | 0.4505 | 0.0032 |

Maximum-score failure AUROC also shifts:

| Cohort | ChartQA | GQA | TextVQA | Overall |
|---|---:|---:|---:|---:|
| Old train | 0.9900 | 0.8906 | 0.9873 | 0.9529 |
| New canonical-source pool | 0.3469 | 0.7453 | 0.6846 | 0.4056 |

The gate is therefore not merely miscalibrated at the old threshold: ranking
itself reverses on new ChartQA and degrades materially on the other tasks.

## Observable source correlates

The new correct populations also move toward visual-token regimes associated
with old wrong rows. ChartQA median visual-token counts are 630 for old C, 580
for old W, and 580 for new C. TextVQA medians are 888 for old C, 962 for old W,
and 999 for new C. GQA remains 234 in all major cells and shows the smallest
false-trigger shift. Token count is not proven causal, but this pattern is
consistent with source/image-format statistics influencing the hidden-state
features used by the gate.

## Diagnosis

- Supported: the frozen Stage-1 gate has a severe canonical-source/OOD
  generalization failure, concentrated in ChartQA and TextVQA. The high
  `P(trigger|C)` is a real score-trajectory shift, not a counting artifact.
- Suspected: the gate learned source/image-format or representation shortcuts
  correlated with correctness in the historically balanced pool. Visual-token
  statistics provide indirect evidence but do not identify the exact causal
  feature.
- Unknown: whether benchmark-train familiarity or another source-specific
  property explains why many new samples are easy for dense Qwen while still
  appearing high-risk to Stage 1.

