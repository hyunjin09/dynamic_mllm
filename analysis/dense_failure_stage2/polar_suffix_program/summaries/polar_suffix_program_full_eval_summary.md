# POLAR suffix-program full evaluation summary

## Result

- Outcome: **B: fewer regressions but insufficient rescues**
- Full training population: 569 UIDs / 4948 programs.
- Provenance: preservation 106; single 1442; original MCTS 75; robust search 896; completeness audit 2429.
- Full-corpus refit completed: **yes**.
- Full external evaluation completed: **19,960 / 19,960** with exact Phase-69 Dense/Stage1 reuse parity.

| Scope | Dense | Sequential-A | Program | W→C | C→W | Net |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.780561 | 0.779760 | 0.780311 | 3 | 8 | -5 |
| chartqa | 0.858800 | 0.858000 | 0.856800 | 1 | 6 | -5 |
| textvqa | 0.857600 | 0.855600 | 0.857600 | 0 | 0 | 0 |
| mmmu_pro | 0.354046 | 0.352890 | 0.354046 | 2 | 2 | 0 |
| pope | 0.880000 | 0.880000 | 0.880000 | 0 | 0 | 0 |

## Required answers

1. P90-triggered training UIDs: **569**.
2. Unique complete suffix programs: **4948**.
3. Source counts are listed above; overlaps are preserved in `corpus/program_provenance.jsonl`.
4. Per-W-UID cardinality is frozen in `corpus/program_cardinality_per_uid.csv`.
5. Full-corpus training completed: **yes**, after group-disjoint epoch selection and exact reinitialization.
6. Dense / Sequential-A / Program accuracies are reported in the table.
7. W→C, C→W, and pooled Net are reported in the table.
8. Pooled Net positive: **False**.
9. Program accuracy >= Dense: **False**.
10. Dense-C preservation improved over Sequential-A: **True** (Program C→W 8 vs Sequential-A 19).
11. Treated-W success versus Sequential-A: Program W→C 3 vs Sequential-A 3.
12. MMMU-Pro rescues: **2**.
13. POPE remained inactive when Stage-1 did not hand off: Stage-1/Stage-2 funnel is in `metrics/stage1_stage2_funnel.csv`; POPE program Net 0.
14. All-FULL rate after trigger is in `metrics/program_statistics.csv`.
15. Non-FULL action counts are in `metrics/program_statistics.csv` and `metrics/action_usage.csv`.
16. Internal-dev exact match to any known program: **0.20869565217391303**; nearest-program distances are diagnostic, while LMMS end-to-end correctness is primary.

Paired bootstrap intervals are in `metrics/paired_bootstrap.csv`. A positive point estimate is not interpreted as conclusive when its interval crosses zero.
