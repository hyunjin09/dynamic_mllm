# Chartqa exhaustive audit

| Dense-W | Triggered-W | W non-FULL | W→C | Dense-C | Triggered-C | C non-FULL | C→W |
|---|---|---|---|---|---|---|---|
| 353 | 49 | 11 | 1 | 2147 | 58 | 9 | 3 |

Stage-1 W admission is 0.1388; C false admission is 0.0270. Stage-2 uses non-FULL on 0.2245 of triggered W and 0.1552 of triggered C.

Conditional treatment success is 0.0909; conditional preservation risk is 0.3333. First non-FULL counts are W: WRITE_ONLY=8, IGNORE=3, READ_ONLY=0; C: WRITE_ONLY=7, IGNORE=1, READ_ONLY=1.

All 4 correctness-changing trajectories are preserved in `../answer_changes/all_22_answer_changes.jsonl`; the descriptions are associations, not per-action causal attributions.

ChartQA has one rescue and three regressions. Although only 20 triggered rows receive non-FULL, preservation loss still exceeds rescue, and C regression risk conditional on non-FULL is 3/9 versus W rescue success 1/11.
