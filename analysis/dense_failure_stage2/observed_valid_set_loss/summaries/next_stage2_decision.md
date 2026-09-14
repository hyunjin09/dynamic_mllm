# Next Stage-2 decision

## One next step

**small partial-prefix on-policy collection**

- Why it matters: Experiment C resolves the isolated objective hypothesis under fixed data and architecture; the remaining largest uncertainty should now be tested without combining changes.
- Remaining hypothesis: whether exposure shift explains the oracle-to-rollout gap.
- Fixed implementation: collect only first-deviation states under the fixed C policy.
- Positive result: the targeted oracle or free-rollout deficit improves while preservation remains acceptable.
- Negative result: this mechanism is insufficient under the current router and the next decision should revisit the remaining diagnosed limitation rather than repeat it.
- Cannot conclude: deployment benefit, held-out test performance, unseen-source generalization, or impossibility of sequential correction.

This recommendation is not executed in Phase 68.
