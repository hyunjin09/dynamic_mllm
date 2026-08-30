# Stage-1 Decision Summary

## Decision

**Stop before selective-gate training.** Phase 1 found
39/128 `FULL-cache-incomplete`
mandatory boundaries and 0 unresolved states.
Only 89 trusted validation
DEVIATE positives remain, versus the frozen requirement of
128.

## Q1 — Label validity

No. Mandatory-boundary DEVIATE labels are not sufficiently trustworthy under
the expanded audit: bounded FULL rescue is 30.47%
with 95% UID-bootstrap CI [22.66%,
38.28%].

## Q2 — Signal

Not tested in this phase. The linear and MLP gates were conditionally forbidden
after the Phase-1 failure. The earlier Phase-40 probe is motivation, not a
substitute for the clean gate experiment specified here.

## Q3 — Selectivity

Not tested. No threshold sweep or 99%/98%/95% C2C-preservation operating point
was produced because that would require training on a failed label contract.

## Q4 — Behavioral value

Not tested. Learned-WHEN + oracle-WHAT execution depends on a trained Stage-1
gate and therefore was not run.

## Q5 — Next step

Stage 1 is not good enough to justify READ_OFF/WRITE_OFF/BOTH_OFF Stage-2
training. The smallest defensible next action is a separately authorized
continuation-cache repair, WHEN-label rebuild, and repeat audit. No such repair,
training, Stage 2, or external evaluation was executed.

## Intentionally absent conditional artifacts

`gate_train_manifest.jsonl`, `gate_val_manifest.jsonl`, both gate configs and
histories/results, `threshold_sweep.csv`, `selective_operating_points.json`,
oracle-WHAT outputs, and gate figures do not exist. Their absence is the frozen
Case-A stop behavior, not missing execution.
