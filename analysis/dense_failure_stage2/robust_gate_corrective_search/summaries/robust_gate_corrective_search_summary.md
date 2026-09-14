# Robust-gate corrective-search summary

Contract: `767284f1e8ed76d5d1c797562a3aaea4ea59183703d62b1521545cacc7c57ab4`. Thresholds are frozen at P98=0.9711347410314399, P95=0.9448015466742209, P90=0.9061332901863008 with `strict_greater_than` comparison semantics.

| Point | Triggered W | Existing covered | New single | New MCTS | Known corrective | Unresolved | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| P98 | 344 | 67 | 1 | 40 | 108 | 236 | 0.3140 |
| P95 | 727 | 146 | 12 | 76 | 234 | 493 | 0.3219 |
| P90 | 1307 | 259 | 42 | 162 | 463 | 844 | 0.3542 |

1. **Unique W UIDs searched:** 1,104.
2. **Single searches launched:** 1,104, one exhaustive search per missing-union UID.
3. **MCTS roots launched:** 1,799 at cap 200.
4. **Cross-threshold reuse:** single search avoided 802 duplicate pair-level launches and 18,777 terminal evaluations. MCTS sharing avoided 52 pair-root launches after single resolution.
5. **P98 known corrective:** 108/344.
6. **P95 known corrective:** 234/727.
7. **P90 known corrective:** 463/1,307.
8. **Unresolved:** P98=236, P95=493, P90=844. These are unresolved at the frozen budget, not proven unfixable.
9. **New single contribution:** P98=1, P95=12, P90=42 pair-level outcomes; 246 successful new single routes were retained globally.
10. **New MCTS contribution:** P98=40, P95=76, P90=162 additional pair-level outcomes; 650 successful new MCTS routes were retained globally.
11. **Dataset/source variation:** observed known-corrective coverage spans 0.0000 (canonical textvqa P98) to 0.6667 (canonical chartqa P98); cells and support are in `metrics/dataset_source_breakdown.csv`.
12. **Trigger-depth variation:** bounded correctability spans 0.2729 (P95 L19-L27) to 0.4882 (P90 L9-L18). This is descriptive and not a causal depth claim.
13. **Replay/provenance:** PASS. All 2,519 retained routes have threshold-specific exact replay evidence, current LMMS correctness, and contract/model/code/schema/source-bound routed states.

No Stage-2 training, Stage-1 change, threshold change, test deployment, or external evaluation was performed.
