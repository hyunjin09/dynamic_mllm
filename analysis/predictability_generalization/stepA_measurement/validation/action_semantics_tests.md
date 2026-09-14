# Action semantics

- FULL = READ1 WRITE1
- READ_ONLY = READ1 WRITE0
- WRITE_ONLY = READ0 WRITE1
- IGNORE = READ0 WRITE0
- Every later layer is FULL.

Covered by `tests/test_four_action_binary_executor.py` and frozen executor hashes.
