# Mmmu Pro exhaustive audit

| Dense-W | Triggered-W | W non-FULL | W→C | Dense-C | Triggered-C | C non-FULL | C→W |
|---|---|---|---|---|---|---|---|
| 2235 | 261 | 82 | 0 | 1225 | 92 | 26 | 4 |

Stage-1 W admission is 0.1168; C false admission is 0.0751. Stage-2 uses non-FULL on 0.3142 of triggered W and 0.2826 of triggered C.

Conditional treatment success is 0.0000; conditional preservation risk is 0.1538. First non-FULL counts are W: WRITE_ONLY=54, IGNORE=23, READ_ONLY=5; C: WRITE_ONLY=19, IGNORE=6, READ_ONLY=1.

All 4 correctness-changing trajectories are preserved in `../answer_changes/all_22_answer_changes.jsonl`; the descriptions are associations, not per-action causal attributions.

MMMU-Pro is treatment-quality limited under the fixed rule: Stage-1 admits 261 W rows and Stage-2 intervenes on 82 of them, yet none is rescued. The four regressions occur on 26 non-FULL-treated C rows. This does not imply MMMU-Pro can never benefit.
