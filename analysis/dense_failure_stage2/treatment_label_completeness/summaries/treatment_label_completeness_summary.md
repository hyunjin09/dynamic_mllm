# Large-scale treatment-label completeness audit

## Status and provenance

- Frozen contract: `3c371c2f4f8372b36eb5965a97344adc6d4f1a76ae906009a93c72e63bff08cc`
- Git commit at freeze: `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2` on `main`; the pre-existing dirty worktree is recorded verbatim in the contract.
- Model: Qwen2.5-VL-7B-Instruct snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Runtime: Python 3.12.7, PyTorch 2.6.0+cu124, Transformers 5.3.0, lmms-eval 0.7.3, four RTX 6000 Ada GPUs.
- All 1,200 selected states completed; zero states were quarantined.

## Population and execution

- Audited 1,200 exact states from 422 UIDs and 415 image groups, with at most four states per UID.
- Frozen old-label composition: 500 KEEP_REQUIRED, 500 INTERVENE_REQUIRED, and 200 MIXED.
- Searched 3,194 previously unobserved first-action branches.
- Ran 3,194 direct-suffix evaluations, 26,704 one-later-intervention evaluations, and 1,124 MCTS roots totaling 130,460 iterations.
- Total terminal evaluations: 103,940. Parallel wall time was 1.525 hours; summed GPU time was 5.697 hours.
- Exact replay validated all 1,200 pre-existing successful continuations and all 2,504 newly discovered actions. Every replay was correct and token-identical. No duplicate, missing-rank, or quarantine condition occurred.

## Main label result

| Old label | Audited KEEP | Audited INTERVENE | Audited MIXED | Total |
|---|---:|---:|---:|---:|
| KEEP_REQUIRED | 4 | 0 | 496 | 500 |
| INTERVENE_REQUIRED | 0 | 229 | 271 | 500 |
| MIXED | 0 | 0 | 200 | 200 |
| Overall | 4 | 229 | 967 | 1,200 |

- Old KEEP invalidation: 496/500 = 99.2%; UID-bootstrap 95% CI [98.24%, 99.81%].
- Old INTERVENE invalidation: 271/500 = 54.2%; UID-bootstrap 95% CI [49.30%, 58.89%].
- At least one new successful action was found for 1,087/1,200 states = 90.58%; UID-bootstrap 95% CI [88.81%, 92.18%].
- Mean successful-action-set cardinality increased from 1.338 to 3.425. Final cardinalities were 37/238/103/822 states with 1/2/3/4 successful first actions.

These values exceed the prospectively frozen large-invalidation criterion by a wide margin. The previous clean binary KEEP/INTERVENE labels were not sufficiently identified by the original route observations.

## Discovery details

| Newly searched action | Eligible states | Newly successful | Discovery rate |
|---|---:|---:|---:|
| FULL | 500 | 271 | 54.20% |
| READ_ONLY | 999 | 814 | 81.48% |
| WRITE_ONLY | 881 | 728 | 82.63% |
| IGNORE | 814 | 691 | 84.89% |

READ_ONLY produced the largest number of discoveries; IGNORE had the highest rate among eligible states. Discoveries were attributed to 1,018 direct all-FULL suffixes, 930 one-additional-intervention suffixes, and 556 MCTS searches.

MCTS discovery was prospectively classified as saturated at 200 iterations: 546/556 discoveries (98.20%) occurred by iteration 150 and 10/556 (1.80%) occurred in iterations 151-200, below the frozen 10% late-discovery limit. This is bounded-search saturation, not exhaustive proof that no other continuation exists.

## Stratum findings

- Dataset KEEP invalidation was 100.0% GQA, 99.48% TextVQA, and 97.44% ChartQA. INTERVENE invalidation was 57.89%, 51.66%, and 52.20%, respectively.
- Canonical/historical KEEP invalidation was 100.0%/98.97%; INTERVENE invalidation was 51.46%/54.91%.
- Early/middle/late INTERVENE invalidation was 81.82%/75.90%/49.01%. Early support was only 11 INTERVENE states, so this is descriptive.
- For MCTS-route states, any-new-action discovery was 91.34%, versus 89.27% for single-route states. More importantly, old INTERVENE invalidation was 58.82% for MCTS-route membership versus 28.00% for single-route membership. MCTS-derived labels were therefore more incomplete on this sample, although route-source strata overlap.

## Secondary probe recheck

The unchanged old-label sampled-cohort probe was reproducible: RW OOF AUROC 0.5334 and precision@5% 0.46 on the balanced 500/500 sample, consistent with the weak Phase-72 full-corpus RW AUROC of 0.5620.

The audited-label probe is **not estimable under the frozen five-fold UID/image-group-disjoint contract**. Excluding AUDITED_MIXED leaves only 233 states: 4 AUDITED_KEEP from four distinct UIDs/groups and 229 AUDITED_INTERVENE. At least one of five test folds must contain no KEEP state; the actual deterministic assignment has fold supports 1/45, 1/45, 1/46, 0/47, and 1/46 for KEEP/INTERVENE. Binary AUROC/AUPRC is undefined in the zero-KEEP fold.

No fold count, threshold, metric, or target was changed after observing this outcome. Audited RW AUROC, its gain over Phase 72, and audited high-precision metrics are recorded as not estimable. In particular, the 98.3% INTERVENE base rate makes an apparent high precision value non-diagnostic even outside cross-validation.

## Answers to the plan's final questions

1. **Scale:** 1,200 exact states, 422 UIDs, and 415 image groups were audited.
2. **Branches:** 3,194 previously unobserved first-action branches were searched.
3. **KEEP invalidation:** 496/500 = 99.2% became MIXED.
4. **INTERVENE invalidation:** 271/500 = 54.2% became MIXED because FULL was newly found.
5. **Action types:** READ_ONLY had the most discoveries; IGNORE had the highest eligible-branch rate.
6. **Search stages:** 1,018 direct, 930 one-later-intervention, and 556 MCTS-dependent discoveries.
7. **Saturation:** yes under the frozen rule; only 1.80% of MCTS discoveries occurred after iteration 150.
8. **Highest incompleteness:** KEEP invalidation was essentially universal. INTERVENE invalidation was highest in GQA and in early/middle layers; source-regime differences were small.
9. **Route source:** MCTS-route INTERVENE labels were more incomplete than single-route labels, 58.82% versus 28.00% invalidation.
10. **Cardinality:** mean action-set size rose from 1.338 to 3.425; 822/1,200 states ended with all four actions successful.
11. **Final labels:** 4 AUDITED_KEEP, 229 AUDITED_INTERVENE, and 967 AUDITED_MIXED.
12. **RW improvement:** cannot be estimated under the frozen five-fold binary protocol because only four clean KEEP groups remain.
13. **High-precision region:** no decision-grade audited estimate exists; raw precision would be dominated by the 98.3% INTERVENE base rate.
14. **Interpretation:** label incompleteness is directly and strongly supported. The planned audit cannot determine whether RW representations separate trustworthy clean binary labels, so representation limitation is not established by this phase.
15. **Limits:** the audit proves only that these actions have replay-valid successful continuations under the frozen bounded search. It does not prove structural necessity, impossibility of unobserved continuations, causal action effects, or deployment benefit.

The pre-specified A-E decision grid did not include the observed extreme depletion of one clean class. Assigning Case A or B would require an audited probe result that cannot validly be computed, so no post-hoc case is claimed.
