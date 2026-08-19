# Baseline-Relative Visual Router Plan

## Decision

The first method checkpoint is not another actor sweep. We first test whether
the frozen proposer outputs contain a meaningful natural-distribution
accuracy-compute Pareto point under a perfect outcome-aware selector.

## Stages

1. **Completed:** audit exact UID, benchmark, and all-on baseline alignment
   across six proposers (`22,349/22,349`, zero mismatches).
2. **Completed:** measure single-proposer and proposer-union oracle upper bounds with paired
   bootstrap confidence intervals.
3. **Completed:** freeze SW31 and train admission directly on its actual
   preserve/harm/rescue/unsolved outcomes. The hierarchical diagnostic combines
   a harm-controlled efficiency region with a high-confidence rescue override.
4. **Completed as a feasibility diagnostic:** final dense-prefill features show
   treatment predictability, but still require a full dense prefill.
5. **Completed, negative:** the input-only gate does not transfer to the
   UID-disjoint MMStar/MMMU task holdout (`-0.36%p`, harm/rescue AUROC
   `0.530/0.515`). Layer-0 observables are insufficient.
6. **Running:** generate actual outcomes after shared dense prefixes
   `K={2,4,8}`. Select the prefix and admission threshold on the fixed canonical
   calibration subset, then evaluate once on MMStar/MMMU. The route continues
   from the same prefix state and its budget includes `K`.
7. **Pending after this design is fixed:** lock a new publication confirmation
   population; the currently used external tasks are UID-disjoint but have been
   inspected by prior project analyses.

## Go criterion

- Oracle accuracy is not below all-on.
- Mean route-sensitive visual-on layers is below 28.
- The fixed single proposer, not only a multi-proposer union, has useful
  headroom.
- Candidate generation cost and router overhead are excluded from the proxy
  until measured explicitly.
