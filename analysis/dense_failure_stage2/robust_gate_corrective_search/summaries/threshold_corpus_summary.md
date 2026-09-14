# Threshold-specific Stage-2 corpus summary

Each threshold view starts at that threshold's own frozen trigger layer, even when the underlying route state shard was captured once from an earlier valid trigger.

| Point | Preservation C bases | Single-corpus W bases | MCTS-corpus W bases | Single/MCTS overlap | Unresolved W | Retained routes | Routed state rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| P98 | 7 | 65 | 43 | 0 | 236 | 464 | 4141 |
| P95 | 30 | 143 | 100 | 9 | 493 | 1093 | 12258 |
| P90 | 106 | 270 | 216 | 23 | 844 | 2519 | 34253 |

Cross-threshold overlap:

- P98/P95: 108 corrective W bases; 457 shared corrective route IDs.
- P98/P90: 108 corrective W bases; 457 shared corrective route IDs.
- P95/P90: 234 corrective W bases; 1,063 shared corrective route IDs.
- P98/P95/P90 intersection: 108 corrective W bases and 457 shared corrective route IDs.

The single and MCTS corpus columns count route-family support and may overlap on a base UID because all replay-valid routes are retained. Pair-level outcome classes remain mutually exclusive in `work/final_pair_outcomes.jsonl`.
