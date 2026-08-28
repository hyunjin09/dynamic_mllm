# Exact Sequential Four-Action Smoke Test

- Result: **PASS**
- Samples: 8/8
- Source routes: 61 (56 replay-valid, 5 replay failures)
- Final branch occurrences: 60
- Unique final routes: 59
- Maximum active branch count: 2
- Real-data path counts: FULL restoration 96, READ_ONLY-only 10, WRITE_ONLY-only 34, both-partial branching 4, IGNORE fallback 7.
- Synthetic truth-table coverage: `tests/test_sequential_four_action_label_conversion.py`.
- Exact resume verification: passed.
- Old binary FULL/IGNORE semantic parity: passed.
- Worker topology: 8 workers, one replica on each of 8 GPUs.

All check details and Slurm provenance are in `smoke_audit_v1.json`.
