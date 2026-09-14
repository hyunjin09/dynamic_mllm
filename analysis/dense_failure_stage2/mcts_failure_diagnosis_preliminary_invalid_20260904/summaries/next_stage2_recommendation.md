# Next Stage-2 recommendation

## One smallest next experiment

**observed-valid-set loss**

- Why it matters: the present diagnosis most directly supports this bounded discriminator without changing Stage 1 or opening test.
- Hypothesis tested: whether exact-prefix label ambiguity is suppressing corrective action learning.
- Frozen implementation: Replace single-route CE targets with -log(sum p(observed-valid actions)) while keeping the router and data fixed.
- Positive result: the targeted oracle/action or free-rollout deficit improves while A's single corrective behavior is retained.
- Negative result: this explanation is insufficient, and the next decision should revisit the remaining supported hypotheses rather than repeat the same recipe.
- Cannot establish: deployment benefit, unseen-source generalization, or final-threshold validity without a separately authorized validation experiment.

This recommendation is not executed in Phase 67.
