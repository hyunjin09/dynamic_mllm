# Textvqa exhaustive audit

| Dense-W | Triggered-W | W non-FULL | W→C | Dense-C | Triggered-C | C non-FULL | C→W |
|---|---|---|---|---|---|---|---|
| 712 | 186 | 26 | 2 | 4288 | 255 | 57 | 12 |

Stage-1 W admission is 0.2612; C false admission is 0.0595. Stage-2 uses non-FULL on 0.1398 of triggered W and 0.2235 of triggered C.

Conditional treatment success is 0.0769; conditional preservation risk is 0.2105. First non-FULL counts are W: WRITE_ONLY=18, IGNORE=8, READ_ONLY=0; C: WRITE_ONLY=45, IGNORE=10, READ_ONLY=2.

All 14 correctness-changing trajectories are preserved in `../answer_changes/all_22_answer_changes.jsonl`; the descriptions are associations, not per-action causal attributions.

TextVQA dominates the negative net because 255 Dense-C rows are admitted versus 186 Dense-W rows, 57 triggered-C rows receive non-FULL versus 26 triggered-W rows, and those interventions produce 12 regressions versus 2 rescues. Both exposure and conditional outcome are unfavorable: C regression risk is 12/57 while W rescue success is 2/26.
