# Current dense 8K run summary

- Candidate samples: 4,000
- Attempted: 4,000
- Successfully completed: 4,000
- Skipped/failed: 0
- Complete 28-layer feature records: 4,000
- Candidate/completed image groups: 4,000 / 4,000
- Current-runtime wrong ratio: 0.217750

## Labels

| Dataset | Correct | Wrong | Total |
|---|---:|---:|---:|
| GQA | 1,264 | 736 | 2,000 |
| CHARTQA | 884 | 116 | 1,000 |
| TEXTVQA | 981 | 19 | 1,000 |
| **Overall** | **3,129** | **871** | **4,000** |

## LMMS-Eval metrics

- GQA: `gqa` / `exact_match`, case and punctuation ignored; correct at 1.0.
- ChartQA: `chartqa` / `relaxed_overall`; correct at 1.0.
- TextVQA: `textvqa_val` / EvalAI leave-one-out `exact_match`; raw score preserved, correct at >= 0.5.

## Historical bucket comparison (analysis only)

- `new_canonical_unlabeled_to_correct`: 3,129
- `new_canonical_unlabeled_to_wrong`: 871

## Skip reasons

- None

The current LMMS-Eval dense outcome is authoritative. Historical buckets are metadata only.
