# Stage C Results

## Integrity

All frozen integrity checks passed for 800 records and 800 unique images. The source plan, manifest, primary scorer, structured nulls, seeds, bootstrap, and success rules remained pinned. The approved real-residual caliper amendment is exactly `19/12`; every record was recomputed in fresh shards. The contextual-token amendment applied only to the secondary `Answer:` robustness condition.

## Primary endpoint

- Mean: `-0.07294332` nats/token
- Standard deviation: `0.88503527`
- Median: `0.00000097`
- 5% trimmed mean: `-0.00281866`
- 20% trimmed mean: `-0.00011059`
- Image-clustered 95% CI: `[-0.14127645, -0.01710262]`
- Fraction below zero: `0.496250`
- Fraction at or below -0.05: `0.130000` with clustered CI `[0.106250, 0.153750]`
- Primary gate: `PASS`

## Structured nulls

- Covariance real-minus-null mean: `-0.00757412`, CI `[-0.02833798, 0.01255147]`, `FAIL`
- Real-residual all-800 real-minus-null mean: `0.00168244`, CI `[-0.01228950, 0.01741056]`, `FAIL`
- Secondary original-1.5-supported 798-target sensitivity: mean `0.00182635`, CI `[-0.01203445, 0.01741453]`, `FAIL`. This does not replace the all-800 comparison.
- Coverage warning: `textvqa_validation_36174` had only three donors at 1.5; five selected donors enter at the amended boundary.

## Secondary outcomes

- FULL-wrong contrast eligible records: `189`; mean `Delta C = 0.41465425`,
  median `0.01744709`, clustered 95% CI `[0.17302903, 0.69316965]`;
  fraction above zero `0.560847`, CI `[0.492063, 0.629630]`.
- Greedy strict transitions:
  - FULL wrong to WRITE_ONLY correct: `22`
  - FULL correct to WRITE_ONLY wrong: `12`
  - wrong to different wrong: `48`
  - unchanged correct: `599`
  - unchanged wrong: `119`

These are secondary behavioral outcomes and do not establish a causal accuracy
improvement.

## Frozen robustness analyses

- Sequence-sum effect: mean `-0.13544451`, CI
  `[-0.27602624, -0.01992088]`; per-token directional agreement `0.87375`,
  Pearson `0.95644`, Spearman `0.84661`.
- Uniform accepted-answer aggregation: mean `-0.06833159`, CI
  `[-0.13572732, -0.01386681]`; sign agreement with primary `0.91875`.
- Contextual `Answer:` prefix: mean `-0.00485841`, CI
  `[-0.02332004, 0.01396219]`; sign agreement with primary `0.5725`. The paired
  prefix-minus-primary difference is `0.06808491`, CI
  `[0.01305588, 0.13494431]`. Raw per-token likelihood levels were not compared
  across tokenizations.
- The primary distribution is heavy-tailed: its median is approximately zero,
  while the 5% and 20% trimmed means are `-0.00281866` and `-0.00011059`.

The contextual-prefix result is an unresolved robustness risk and reinforces
the restriction against a confirmed answer-misaligned or harmful-mechanism
claim.

Full secondary and robustness outputs are under `outputs/stage_c/analysis_v1/`.
