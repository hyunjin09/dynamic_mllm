# Stage C Full-Sweep Failure Report

## Status

**STOPPED before merge, aggregation, or scientific conclusion.** The amended
prefix preflight passed, but the frozen same-layer real-residual null could not
be generated for every held-out record under its immutable settings.

No primary mean, confidence interval, null comparison, wrong-answer contrast,
greedy aggregate, robustness aggregate, or Outcome A/B/C was computed or
inspected. Stage D remains unauthorized.

## Exact failed condition

On shard 0, after 10 records completed, target
`textvqa:textvqa_validation_39543` had only **7** eligible Stage B
real-residual donors under the frozen composite matching caliper **1.5**. The
protocol requires eight closest donors without replacement.

```text
ValueError: Only 7 eligible donors for
textvqa:textvqa_validation_39543; 8 required
```

This is the explicit fail-closed rule in the frozen structured-null
specification. The other seven shards were cancelled immediately.

## Confirmed observations

- The source-plan, manifest, configuration, model, primary scorer, null
  artifacts, and seed checks passed at job startup.
- The contextual-prefix amendment passed all 800 records and 2,279 answer
  components before the sweep. Repeated-score maximum absolute difference was
  0.0.
- The model and official cached TextVQA validation data loaded successfully.
- Covariance-null construction and the first 10 complete shard-0 records did
  not trigger the failure.
- The real-donor selector returned seven candidates for the named record under
  the exact frozen exclusions and 1.5 cap.
- Partial shard files contain 93 completed records in total. They are preserved
  as raw execution evidence but were not merged or opened for outcome
  interpretation.
- `outputs/stage_c/stage_c_results_v1.jsonl` and
  `outputs/stage_c/analysis_v1/` do not exist.

## Diagnosis

Diagnosis: **supported**. The immediate cause is insufficient donor coverage
under the frozen real-residual matching rule, evidenced by
`outputs/stage_c/failures/stage_c_shard_00_failure.json` and
`runs/stage_c_full_v1_20260805/shard_0/slurm-98398.out`.

Ruled out as the immediate cause: the contextual-prefix repair, empty target
spans, dataset/model loading, frozen-file checksum mismatch, FULL-parity
failure, covariance-null generation, and a missing results directory.

No broader claim is made about why this held-out target's residual geometry has
sparse donor coverage; that diagnosis would not authorize changing a frozen
null setting.

## Stop status and viable follow-ups

The frozen Stage C operational stop condition has been reached. The source
plan's scientific stop/pivot condition has not been evaluated because there is
no valid Stage C aggregate result.

Smallest viable follow-ups, all requiring explicit protocol approval:

1. Run one outcome-blind, geometry-only coverage audit over all 800 frozen
   targets, without likelihood scoring, to determine whether a single minimal
   caliper can supply eight donors everywhere. Then decide whether amending the
   1.5 caliper is defensible.
2. Expand and refreeze the real-residual donor pool while keeping eight draws
   and the 1.5 cap. This is more expensive and changes a larger null component.
3. Reduce the draw count globally. This is cheaper but weakens the frozen null
   and cannot be chosen safely before manifest-wide coverage is known.
4. Stop Stage C as technically inconclusive under the frozen null protocol.

The recommended next action is option 1 only: a bounded outcome-blind geometry
audit. Its strongest objection is that it uses held-out activation geometry
after partial likelihood files were generated, even though those likelihood
outcomes remain unopened; any resulting amendment must therefore be explicit
and transparently reported.
