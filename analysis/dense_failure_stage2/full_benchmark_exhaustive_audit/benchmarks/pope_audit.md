# Pope exhaustive audit

| Dense-W | Triggered-W | W non-FULL | W→C | Dense-C | Triggered-C | C non-FULL | C→W |
|---|---|---|---|---|---|---|---|
| 1080 | 0 | 0 | 0 | 7920 | 0 | 0 | 0 |

Stage-1 W admission is 0.0000; C false admission is 0.0000. Stage-2 uses non-FULL on 0.0000 of triggered W and 0.0000 of triggered C.

Conditional treatment success is 0.0000; conditional preservation risk is 0.0000. First non-FULL counts are W: WRITE_ONLY=0, IGNORE=0, READ_ONLY=0; C: WRITE_ONLY=0, IGNORE=0, READ_ONLY=0.

All 0 correctness-changing trajectories are preserved in `../answer_changes/all_22_answer_changes.jsonl`; the descriptions are associations, not per-action causal attributions.

POPE is inactive, not successfully preserved: its maximum Stage-1 score is 0.828423, still 0.077710 below the strict P90 threshold 0.906133. Stage-2 was never called, so this audit provides no evidence about Stage-2 treatment quality on POPE.
