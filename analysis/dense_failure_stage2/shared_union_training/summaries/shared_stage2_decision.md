# Shared Stage-2 decision

## Decision

**Decision C — Stage-2 still under-generalizes.**

- Final planned training condition: Experiment B.
- Selected operating point on the frozen Historical validation set: **P98**.
- Dense/routed accuracy: 0.500000 / 0.500000 (+0.000000).
- W-to-C / C-to-W / net: 0 / 0 / +0.
- C-to-C preservation: 1.000000.
- Experiment B completed: True; Experiment-A B gate: True.

## Scope limitation

This is a Historical-800 validation result, not an all-source deployment claim. The available Canonical 4,000 are Stage-1 OOF only and contribute Stage-2 union supervision, so they are not a held-out Stage-2 evaluation set. Their requested source cells are recorded as unavailable rather than leaked.

No test set, new threshold, Stage-1 change, new search, or architecture/loss change was run.
