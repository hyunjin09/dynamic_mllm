# WeMath2.0-Pro Greedy Recovery Launch

## Status

G0--G2 passed and Phase 1 is running. Phase 2 remains gated on the complete
global Phase-1 budget center.

## Frozen population and contract

- 2,278 current-ALL-ON-wrong records with zero valid cap-400 MCTS routes.
- 1,104 image groups.
- Manifest SHA-256:
  `dd08623e412e4e55cd56cb0985ce2aabae255f6565ef6b321477070ad77d6ed5`.
- Contract-file SHA-256:
  `ec7afabdfd25c8314ddac820b8d6dea0f8d8d79b1195918e6b9839c1fea640b4`.
- Every linked MCTS record checksum passed during manifest materialization.
- The raw cache retains all positive and negative evaluated masks. Only the
  derived training view is capped at 50 diverse valid masks per sample.

## Technical gate

The initial launch stopped before scientific output because deterministic
PyTorch required `CUBLAS_WORKSPACE_CONFIG`. The launch contract was repaired
by freezing `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no fixture or tolerance changed.

The unchanged five-record rerun passed 5/5 native-versus-binary ALL-ON token
parity, cached ALL-ON and mixed-mask token/score parity, repeated new-mask
determinism, native processing, and the no-custom-image-cap check. Evidence is
`outputs/label_regeneration/wemath2pro_greedy_recovery_v1/preflight/preflight_report_v1.json`.

## Phase-1 placement

| Job | Node | GPUs | Global shards | Slurm ID |
|---|---|---:|---|---:|
| Phase 1 node06 | node06 | 2 | 0, 1 of 4 | 101708 |
| Phase 1 node07 | node07 | 2 | 2, 3 of 4 | 101709 |

Each GPU runs one independent model/search process. Per-sample outputs are
atomic and contract-validated before resume skips them. Phase 2 will use the
same four-way layout only after all 2,278 Phase-1 records and 22,780
permutation finals reconcile and one global request manifest is frozen.
