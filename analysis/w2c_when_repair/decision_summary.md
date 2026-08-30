# Phase 42 W2C WHEN-Label Repair Decision Summary

## Decision

**Stop before the full 640-sample repair.** The prospectively frozen smoke gate
failed exact original-route replay: 37/312 cached-correct routes now execute
incorrectly, affecting 10/12 smoke samples across every dataset. The complete
repair, post-repair audit, gate/router training, Stage 2, and external
evaluation did not start.

## What completed

- Verified all 640 authoritative W2C source records by SHA-256: 512 train and
  128 validation, with 16,848 cached correct routes.
- Froze the same iterative repair algorithm and budget for both splits:
  compatible known suffixes first, then at most 96 deterministic one-edit
  suffix variants per current boundary.
- Froze 12 smoke samples covering all datasets, old boundary depths,
  single/multi suffixes, and prior cache-incomplete/bounded-invalid states.
- Executed 1,401 smoke routes on four GPUs with zero quarantine, then verified
  deterministic trace reconstruction, deduplication, cache updates, iterative
  boundary movement, search ordering, and byte-stable resume.

## Failed prerequisite

Only 275/312 old routes replayed correct. The exact route identities/order match
the authoritative manifest; the 37 failures affect 26 GQA, 8 TextVQA, and 3
ChartQA routes. One failed route per GPU was repeated twice: all 4/4 current
pairs were exact, while 0/4 matched the original cached tokens. This supports
reproducible current-runtime cache drift. It does not identify the root cause.

The transferred label records and current repair config bind different hashes
for the four-action executor and input construction. That is direct contract
drift, but causality remains unknown. Proceeding would violate the frozen
`require_all_old_route_replays_correct` rule and could incorrectly retain routes
that are not correct under the executor used for repair.

## Q1-Q6 status

- Q1: population-level old-cache incompleteness cannot be estimated under a
  mismatched execution contract. The failed smoke separately establishes that
  37/312 cached positives do not reproduce here.
- Q2-Q3: the smoke exercised boundary shifts and iterative reevaluation, but
  those shifts are not promoted as scientific results because the starting
  cache failed validity.
- Q4-Q5: final train/validation DEVIATE-candidate counts are unknown because the
  full repair correctly did not start.
- Q6: repaired population-level mechanism change is unknown for the same reason.

## Smallest defensible next action

Recover and freeze the exact executor source/environment that produced the
transferred sequential labels, then rerun only the 12-sample old-route replay
gate. The source records require, among other hashes,
`binary_policy/executor/four_action.py` =
`e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868`.
That exact file is not present in the current repository state. If the original
contract cannot be recovered, the alternative is a new plan to rebuild the
authoritative W2C correct-route population under the current executor before
attempting WHEN repair. Neither action is authorized automatically.

Evidence: `smoke/smoke_report.md`, `smoke/smoke_gate.json`,
`smoke/smoke_executions.jsonl`, and `smoke/replay_failure_diagnostic.json`.
Raw immutable records are under
`/mnt/hyemin/qwen_train_eval/outputs/w2c_when_repair_v1/`.
