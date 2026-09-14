# Failed full attempt 01

- Contract: `f223f56ae683ba4d6bfccbbccd9f032123f13b821defef1d48fbc080473089a3`
- Sanity gate: passed.
- Partial execution preserved: 3,455 successful records, one failed record,
  and 108 completed batch markers before the run was intentionally stopped.
- Failed UID: `gqa:gqa_ge_10447544` on rank 2.
- Direct observation: the detached layer-27 hook reconstruction and the
  model's own raw final logits differed enough to swap top-1 on this near-tie
  sample. The sanity population had shown reconstruction differences up to
  0.125 without a top-1 change.
- Correction: layers 0-26 continue to use the frozen final norm and LM head on
  their hooked states. Layer 27 is replaced with `output.logits` from the same
  forward, which is the model's numerically authoritative raw realization of
  that final norm/head. The pre-replacement discrepancy remains audit metadata.

No partial aggregate or scientific conclusion was emitted.
