# Stage-1 Dense-Failure Population Audit

## Result

The physical image population is exactly 8,000 and its **historical filename
buckets** are exactly 4,000 correct / 4,000 wrong. Those historical buckets are
not the authoritative dense labels produced by the later frozen A6000 runtime.

The later current-dense counts can be reconstructed as 4,045 correct / 3,955
wrong, consistent with the preserved Phase-12 P5 report, but cannot be
independently re-audited record-by-record because the canonical 8K bundle is
missing.

| Dataset | Current correct | Current wrong | Total |
|---|---:|---:|---:|
| GQA | 2,000 | 2,000 | 4,000 |
| ChartQA | 1,011 | 989 | 2,000 |
| TextVQA | 1,034 | 966 | 2,000 |
| **Total** | **4,045** | **3,955** | **8,000** |

The reconstruction uses the 6,917 positive-route inventory statuses plus the
1,083 images absent from that positive-only manifest. The absent samples are
necessarily current-dense wrong under the recorded MCTS contract because
ALL-ON was an evaluated route and they had zero positive routes. Their dataset
counts are GQA 614, ChartQA 215, and TextVQA 254.

This is a supported count reconstruction, not a substitute for the missing
record payloads. Notably, one zero-positive TextVQA sample came from a
historical `correct__` bucket, directly confirming that filename buckets are
not authoritative current labels.

## Integrity findings

| Check | Result |
|---|---:|
| Physical image files | 8,000 |
| Unique physical basenames | 8,000 |
| Positive-inventory VQA rows | 6,917 |
| Unique UIDs among available rows | 6,917 |
| Duplicate UID rows among available rows | 0 |
| Unknown current correctness among available rows | 0 |
| Available rows missing image/question/prompt/answer | 0 |
| Available rows with null stored source dense prediction | 6,917 |
| Unrepresented zero-positive records | 1,083 |
| Missing question/prompt/answer/generated-answer payloads | 1,083 |
| Positive-inventory images missing physically | 0 |

## Gate status

The plan's Phase-1 source-population gate fails. The 4K/4K expectation is true
only for historical source buckets, not for the authoritative current dense
outcomes, and the complete authoritative records needed to verify the current
labels are absent. No records were dropped to force balance.

