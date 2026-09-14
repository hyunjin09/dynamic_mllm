# Robust Stage-1 operating-point and label-compatibility protocol

- Frozen Stage-1 system: Phase-63 five-checkpoint ALL-source system, per-layer probability mean; no singleton checkpoint exists or is trained here.
- Strict trigger rule: first layer 0-27 with `score > tau`.
- P98 is anchored exactly at `0.97113474103143993`; P99/P97/P95/P90 are least-permissive discrete calibration breakpoints satisfying both-source preservation.
- Default retained set P98/P95/P90 is accepted only if all three are fold-stable, thresholds are distinct, and each relaxation adds at least 5% pooled calibration W recall.
- Fold-stable means threshold IQR <= 0.05 and range <= 0.1 across five existing cross-fit calibrations.
- Train maps are descriptive in-sample maps: 6,399 Historical train plus 4,000 Canonical train rows scored by the exact five-head mean.
- Existing single route structural compatibility: new trigger <= intervention layer. Existing MCTS route structural compatibility: new trigger <= first non-FULL layer.
- Every structurally compatible route/handoff pair is rerun with exact Qwen suffix execution and current LMMS scoring; current correct output plus exact stored-token parity is required for reuse.
- Stage-2 outcomes are not used to choose thresholds. No new single/MCTS search or Stage-2 training is permitted.
