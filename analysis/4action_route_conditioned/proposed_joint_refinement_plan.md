# Proposed Bounded Joint READ/WRITE Refinement Pilot

Status: **PROPOSED ONLY — NOT AUTHORIZED OR LAUNCHED**

## Decision target

Test the one unresolved assumption left by the completed route-conditioned
decomposition: whether READ/WRITE component restorations that preserve a known
correction one position at a time can be combined while retaining that
correction.

The completed experiment found 5,781 individually necessary anchor-OFF
positions where at least one component can be restored on its own. This does
not establish that multiple restorations compose, improve compute, or support a
learned four-action router.

## Proposed bounded pilot

- Population: 64 validated-anchor samples with at least two component-relaxable
  positions, stratified by GQA/TextVQA, anchor OFF count, mechanism mix, and
  FULL-context agreement.
- Start state: the frozen current-correct binary anchor route.
- Executor: the same unified route-conditioned four-action executor and fixed
  correct/original-FULL-wrong target identities used in the completed study.
- Search: deterministic beam width 4, at most 8 anchor-OFF positions, and at
  most two partial restorations per expansion.
- Candidate actions:
  - READ-mediated position: restore WRITE with WRITE_ONLY while READ remains
    suppressed;
  - WRITE-mediated position: restore READ with READ_ONLY while WRITE remains
    suppressed;
  - either-removal-sufficient position: consider both partial restorations;
  - redundant position: FULL restoration may be considered as route cleanup,
    but must be reported separately from component-specific refinement.
- Selection objective: preserve evaluator correctness first; then maximize
  restored operations, improve the fixed answer margin, and reduce binary OFF
  use, with deterministic tie-breaks.
- No training, new MCTS, 4^K enumeration, dataset change, or router claim.

## Budget

The strict upper bound is:

```text
64 samples × beam 4 × depth 8 × 2 expansions = 4,096 evaluations
```

At the conservative matched-pilot throughput of 12.183885 valid cells/s, this
is approximately 0.09 eight-GPU wall-hours or 0.75 GPU-hours, excluding small
startup and merge overhead. A fresh pilot must still measure actual throughput.

## Required gates

- Every starting anchor remains current-correct.
- Every combined candidate preserves all unmodified anchor actions exactly.
- Generated answer, evaluator correctness, fixed scores, and action schedule
  are saved for every expansion.
- Resume coverage is unique and append-only; all eight GPUs are used for GPU
  inference, while merge/analysis runs locally.
- Report composition success separately from individual relaxability and from
  the earlier FULL-context local analysis.

## Decision outcomes

- High composability across both datasets: consider a separately specified
  full-cohort joint refinement, then decide whether a four-action search/router
  is warranted.
- Low composability: retain the current study as mechanism evidence and stop;
  do not infer router value from single-position relaxations.
- Strongly heterogeneous composability: report the strata and seek a separate
  decision before any targeted follow-up.

Do not launch this proposal without explicit user approval.
