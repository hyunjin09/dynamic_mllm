# Invalid Phase-54 Aggregation

This run is invalid and must not be used.

- Failure: `KeyError: 'records'` in the Phase-52 prior-aggregate consistency check.
- Cause: the checker read the validation ranking CSV, which intentionally omits raw population counts; the frozen validation operating-point object in `selected_winner.json` is the authoritative source for those fields.
- Scope: all four lightweight train-score workers completed, but aggregation stopped before final trigger manifests or metrics were emitted. No Qwen forward pass or four-action search ran.
- Resolution: added a regression test for the two prior-source schemas, switched validation to the frozen selection JSON, refroze, and reran all workers under the corrected contract.
