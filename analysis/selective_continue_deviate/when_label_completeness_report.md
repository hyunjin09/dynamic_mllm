# WHEN-Label Completeness Report

## Outcome

The prospective Phase-1 label-trust gate **fails**. Forced `FULL` at the
supposed mandatory boundary has a correct bounded continuation for
**39/128 states
(30.47%, 95% UID-bootstrap CI
[22.66%,
38.28%])**. There are zero unresolved
states. Only 89 states remain
trusted under this bounded audit, below the prospectively required 128 clean
validation DEVIATE positives.

Per the frozen protocol, linear/MLP gate training, threshold selection, and
learned-WHEN + oracle-WHAT execution were not started.

## Execution validity

- Cohort: all 128 frozen held-out W2C UIDs; no audit subsampling.
- Live executions: 252/252.
- Suffix coverage: all 256 compatible
  frozen source suffixes; 252 complete routes
  after deduplicating 4 identical routes.
- Statuses: 39
  `FULL-cache-incomplete`,
  89
  `FULL-confirmed-invalid`, 0 unresolved.
- Bootstrap: 10,000 UID-group draws with fixed overall
  seed 20260831 (fixed derived seeds for subgroups).
- Base-model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Executor: `live_unified_four_action_full_insertion_known_suffix_replay`.
- Audit-subset SHA-256: `4e8677ad39bccd30ca805a868e607be310481130936a9112ec913daf945ba3e4`.
- Audit-config SHA-256: `98c557d4db7c090f94a16299256d7c54f94e4c77cbb62383d279dedd2ed58acb`.

Every `FULL-confirmed-invalid` classification remains bounded: failure of all
known compatible suffixes is not global proof that no unobserved continuation
could work.

## Table 1 — WHEN-label completeness

| Group | States | FULL bounded rescue | Rescue rate | 95% CI |
|---|---:|---:|---:|---:|
| Overall | 128 | 39 | 30.47% | [22.66%, 38.28%] |
| ChartQA | 43 | 15 | 34.88% | [20.93%, 48.84%] |
| GQA | 43 | 12 | 27.91% | [16.28%, 41.86%] |
| TextVQA | 42 | 12 | 28.57% | [14.29%, 42.86%] |
| Early | 48 | 25 | 52.08% | [37.50%, 66.67%] |
| Middle | 43 | 12 | 27.91% | [13.95%, 41.86%] |
| Late | 37 | 2 | 5.41% | [0.00%, 13.51%] |
| IGNORE | 33 | 5 | 15.15% | [3.03%, 27.27%] |
| READ_ONLY | 33 | 13 | 39.39% | [24.24%, 54.55%] |
| WRITE_ONLY | 32 | 5 | 15.62% | [3.12%, 28.12%] |
| MULTI | 30 | 16 | 53.33% | [36.67%, 70.00%] |

## Suffixes tested per state

1 route(s): 73 states, 2 route(s): 29 states, 3 route(s): 8 states, 4 route(s): 9 states, 5 route(s): 5 states, 7 route(s): 1 states, 8 route(s): 1 states, 9 route(s): 1 states, 12 route(s): 1 states.

## Interpretation

The incompleteness is not confined to one dataset: bounded rescue is 15/43
ChartQA, 12/43 GQA, and 12/42 TextVQA. It is depth-dependent in this frozen
cohort—25/48 early, 12/43 middle, and 2/37 late—and is largest for multi-valid
boundaries (16/30) and READ_ONLY-valid boundaries (13/33). These subgroup
patterns are descriptive, not causal explanations.

The direct observation is that the existing mandatory-boundary target treats
`FULL` as invalid for 39 states where the unchanged executor finds a correct
route using a frozen compatible suffix. The cause of this cache incompleteness
remains unknown. That observation alone changes the action: a clean binary
CONTINUE/DEVIATE gate must not be trained from these labels under the frozen
minimum-validation contract.

## Conditional stop

The plan's Case A applies. The smallest defensible future action is to
repair/expand route-cache continuation coverage, rebuild WHEN labels, and
re-audit prospectively. That repair is a new research action and was not
executed here. Stage 2 and external evaluation remain out of scope.
