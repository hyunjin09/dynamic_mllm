# READ-harm structure summary

## Population and signs

- The complete primary census contains **1,413 UIDs, 1,385 image groups, and 15,185 dense post-trigger states**.
- There are **7,285 harmful**, **7,900 beneficial**, and **0 exact-zero** READ states under `H_R = q_WRITE_ONLY - q_FULL`.
- The behavioral cohorts contain **625 READ-harmful correctness flips** and **41 READ-beneficial correctness flips**.

## Persistence, spans, and flip neighborhoods

- Harmful adjacent persistence is `0.5174`, below the UID-shuffled 97.5th percentile `0.5185`; this component gate fails. Beneficial persistence is `0.5512`.
- There are 3,842 harmful spans. Their mean length is `1.896`; `55.78%` are singletons, `23.74%` have length 2, `9.27%` have length 3, and the maximum is 13. The observed mean narrowly exceeds the shuffled 97.5th percentile `1.891`, although the fractions of harmful states in spans of at least 2 (`70.58%`) and at least 3 (`45.55%`) do not exceed their corresponding shuffled-null gates.
- At a harmful-flip layer, mean `H_R` is `0.1300`. The six neighboring offsets have harmful prevalence `0.4639-0.5293`; the two immediate neighbors average `0.4853`, only slightly above the population prevalence `0.4797`. Thus strong flips are not reliably embedded in broad harmful regions.

## Trigger depth, sample burden, and source variation

- Harmful prevalence is `0.4586` at the Stage-1 trigger, `0.4717` at +1-2 layers, `0.4830` at +3-5, and `0.4847` at +6 or later. Harmful-flip prevalence remains approximately `3.82-4.22%`, so READ harm is not confined to the trigger boundary.
- Sample-level burden is broad: median harmful-layer fraction is `0.500`, 90th percentile is `0.769`, and the maximum is `1.000`; `50.74%` of UIDs have at least half of post-trigger layers harmful. Dense-W UIDs have mean burden `0.474` (1,307 UIDs) and Dense-C UIDs `0.460` (106 UIDs). Some UIDs are globally READ-fragile, but that does not create strong population-level adjacency.
- Harmful prevalence across the six frozen dataset/source cells ranges from `0.3676` (canonical TextVQA, only 68 states) to `0.5108` (canonical GQA). Historical/canonical ChartQA is `0.4508/0.4454`, GQA `0.5008/0.5108`, and TextVQA `0.4431/0.3676`. These are descriptive prevalence differences, not mechanism claims.

## Decision

**R-STRUCT-B — mostly isolated.** The fixed rule requires adjacent persistence, span enrichment, and harmful-neighborhood enrichment together; the component gates are `False/True/True`. Harm is common and individual UIDs can be fragile, but the decisive population evidence does not show a robust contiguous harmful regime.
