# Phase 20: Full10 POLAR-Style Duplicated-BCE Comparator

## Current Objective

Train the Question-only and Image+Question direct binary predictors for ten
epochs with POLAR-style duplicated-route BCE, then evaluate both frozen
best-Hit@1 checkpoints on the established 22,307-record external suite.

## Active Constraints

- Change only supervision grouping/loss relative to the exact-set-NLL full10.
- Preserve 6,043/874 positives, max-50 routes, POLAR 0.3 ALL-ON weighting,
  architecture, seed, optimizer, schedule, decoding, and evaluator.
- Run one GPU on node02 and one GPU on node07; never use node04.
- Current live ALL-ON is the external scientific baseline; DocVQA is excluded.

## Current State

- Done: completed exact-set-NLL full10 and external evaluation are preserved.
- Done: Slurm `101023` Question-only on node02 and `101022` Image+Question on
  node07 completed ten epochs, joint preflight, both external evaluations, and
  the merged integrity analysis.
- In progress: none.
- Blocked: none for execution; scientific interpretation is a separate action.
- Most recent useful observation: the earlier matched smoke used the validated
  duplicated-BCE objective, but no corresponding full10 trajectory existed.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Duplicated BCE and exact set-NLL deterministic tests pass | `tests/test_binary_policy_objective_comparison.py` | Establishes the intended loss and grouping contract | confirmed |
| Exact-set full10 used 6,043/874 inputs and saved 20 checkpoints | `outputs/binary_polar/full10/` | Defines the matched comparator settings | confirmed |
| External evaluator completed 22,307 records per modality | `outputs/binary_polar/external_eval/full10_best_v1/` | Defines the unchanged downstream protocol | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Historical cached FULL as external baseline | Cache differed from current live FULL | supported cache non-equivalence | `reports/binary_polar_full10_external_eval.md` | Recompute live FULL for every scientific comparison | Do not restore the cache as baseline |
| Initial plain-shell pipeline launch | Both jobs exited before data loading because the project root was absent from Python's import path | supported launch-wrapper import failure | `runs/binary_polar_full10_bce_{question,image_question}_v1/slurm.log` | Relaunch unchanged commands with `PYTHONPATH=.` | Do not retry the same plain command without the project module path |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Full10 duplicated BCE for both modalities | Explicitly requested missing comparator | Whether full-scale duplicated supervision differs from exact set-NLL | high | testing |

## Next-Step Decision

- Deliberation mode: fast
- Active objective and bottleneck: execute the fully specified matched BCE
  comparator without changing any other scientific setting.
- Relevant memory item used: the completed exact-set run and evaluator fix all
  non-loss settings and expose the missing baseline.
- Confirmed observation: only the bounded smoke, not full10, previously trained
  duplicated BCE.
- Unverified interpretation: longer duplicated-BCE training may or may not
  avoid the exact-set predictor's near-ALL-ON external collapse.
- Diagnosis: unknown
- Chosen action: validate the full10 BCE path, then launch concurrent node02
  Question and node07 Image+Question train-to-eval pipelines.
- How this differs from failed attempts: the objective is duplicated-route BCE;
  the exact-set artifacts remain untouched.
- Automatic execution authorized: yes
- Authorization basis: explicit user request in the current conversation.
- Stop condition: both evaluations and merged checks pass, or a technical gate
  fails.

## Latest Research-Action Result

- Action taken: implemented and completed the loss-only full10 comparator and
  both external evaluations.
- Result: ten epochs completed per modality; best-Hit@1 is epoch 2 for both;
  each external shard contains exactly 22,307 records; merged integrity PASS.
- Evidence saved: `outputs/binary_polar/full10_bce/`,
  `outputs/binary_polar/external_eval/full10_bce_v1/`, and
  `reports/binary_polar_full10_bce_external_eval.md`.
- Failure or issue: the first wrapper omitted `PYTHONPATH`; both processes
  exited before data loading and left no outputs. The unchanged commands were
  relaunched with the project module path.
- Lesson learned: plain-shell scheduler jobs must set `PYTHONPATH=.` for scripts
  that import the project package.
- Next implication: execution is closed; interpret the matched BCE versus
  exact-set-NLL results only as a separately requested research action.
