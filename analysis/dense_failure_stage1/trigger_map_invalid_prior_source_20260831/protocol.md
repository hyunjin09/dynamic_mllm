# Frozen Stage-1 Trigger Map Audit Protocol

- Audit contract: `dde66a14fffde4deebe776216e790a7b9e4f66523e6eb614bb45a0899b791e03`
- Git commit: `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2` on `main`; the full dirty-worktree snapshot is recorded in `frozen_protocol.json`.
- Population: the frozen current-runtime dense FULL split (6,399 train / 800 validation / 800 test), with current LMMS-Eval correctness labels and zero image-group overlap.
- Gate: Shared Random-4 (`state_layer_random4`), unchanged checkpoint and global raw threshold.
- Checkpoint SHA-256: `1aeeaa278a9ac2aa99343b4ea26fad43a8b097c05b497c27554289eaef311211`.
- Normalization SHA-256: `ce4b63ab7f503d879db20f94aae01116804095eb87fd03b16a818d5904f1aed3`.
- Trigger rule: first layer in 0-27 with `score > 0.92538392806010306` (strict crossing).
- Validation/test: reuse frozen saved 28-layer trajectories and verify them against the Phase-53 trigger manifest and Phase-52 aggregate metrics.
- Train: recompute only the lightweight gate trajectory from hash-verified stored dense feature shards; four direct GPU workers own source ranks 000-003.
- Reproducibility check: 24 deterministic val/test records, stratified by split, dataset, and dense class.
- No Qwen forward pass, four-action search, Stage-2 labeling/training, threshold change, persistence, or EMA is permitted.
- Future usage: only train triggered-W may enter corrective label search; validation is model-selection/treatment-only and test remains held out.
