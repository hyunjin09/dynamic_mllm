# Current dense 8K run summary

- Candidate samples: 8,000
- Attempted: 8,000
- Successfully completed: 7,999
- Skipped/failed: 1
- Complete 28-layer feature records: 7,999
- Candidate/completed image groups: 7,477 / 7,476
- Current-runtime wrong ratio: 0.500063

## Labels

| Dataset | Correct | Wrong | Total |
|---|---:|---:|---:|
| GQA | 2,000 | 2,000 | 4,000 |
| CHARTQA | 999 | 1,000 | 1,999 |
| TEXTVQA | 1,000 | 1,000 | 2,000 |
| **Overall** | **3,999** | **4,000** | **7,999** |

## LMMS-Eval metrics

- GQA: `gqa` / `exact_match`, case and punctuation ignored; correct at 1.0.
- ChartQA: `chartqa` / `relaxed_overall`; correct at 1.0.
- TextVQA: `textvqa_val` / EvalAI leave-one-out `exact_match`; raw score preserved, correct at >= 0.5.

## Historical bucket comparison (analysis only)

- `correct_to_correct`: 3,999
- `wrong_to_wrong`: 4,000

## Skip reasons

- `missing_image`: 1

The current LMMS-Eval dense outcome is authoritative. Historical buckets are metadata only.
