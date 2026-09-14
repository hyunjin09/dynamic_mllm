# Aggregation provenance addendum

The frozen extraction/training contract is
`3a0d2251e23b43082227b9b05befd9a37d9b9ecc6bb40a570861ebd7affebb6a`.
All 2,115 checkpoints were produced and validated under that contract.

The first aggregation attempt stopped while constructing the prediction
manifest because the mapping-only `canonical_hash` helper was passed an
ordered list of state IDs. This occurred after checkpoint validation and did
not alter extraction, training, predictions, metrics, targets, folds, models,
or decision thresholds.

Aggregation was rerun without changing the worktree or frozen contract. A
process-local compatibility function SHA-256-hashed non-mapping values using
canonical JSON (`sort_keys=True`, separators `(',', ':')`, UTF-8) and delegated
mapping values to the original helper. The completed artifact manifest is
`1f9005bf11cf48e00b795e205aaec80370c218b4145d47f7548f68a67ad35961`.

After all contract-bound work completed, the identical permanent helper was
added to `experiments/analyze_read_short_horizon_propagation.py`, together with
a regression test that checks ordered-sequence stability and order
sensitivity. Post-fix file hashes are:

- analyzer: `6b18f80fd7b6957b25d8ed3e9da43ce4d464324d1b1df7b84032597d00371a43`
- test: `97508574574dff0d2448c715be763e95ac942b7dd166a9d3a3e7ef6fa47187cb`

The focused Phase 81-84 test set passes: 64 tests.

This addendum is intentionally post-run and is not included in the frozen
artifact manifest; it documents the non-scientific aggregation repair without
rewriting the historical contract or checkpoint provenance.
