# Label Regeneration P5 Outcome Summary

Status: **PASS**

P5 summarizes the strict P4-frozen GQA, TextVQA, and ChartQA cache. WeMath2.0-Pro is excluded. Route-transition, segment, and pairwise-Hamming analysis remains unopened for P6.

## Population and current ALL-ON

| Dataset | Samples | Current correct | Current wrong | ≥1 valid | ≥20 valid | Corrected current-wrong |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 4,000 | 2,000 | 2,000 | 3,386 | 2,668 | 1,386/2,000 (69.30%) |
| TEXTVQA | 2,000 | 1,034 | 966 | 1,746 | 1,239 | 712/966 (73.71%) |
| CHARTQA | 2,000 | 1,011 | 989 | 1,785 | 970 | 774/989 (78.26%) |
| **Total** | **8,000** | **4,045** | **3,955** | **6,917** | **4,877** | **2,872/3,955 (72.62%)** |

## Valid-route coverage

| Dataset | Zero | ≥1 | ≥5 | ≥10 | ≥20 | Mean | Median | P10 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA | 614 | 3,386 | 3,035 | 2,859 | 2,668 | 88.38 | 80.0 | 0.0 | 184.0 | 191.0 |
| TEXTVQA | 254 | 1,746 | 1,519 | 1,407 | 1,239 | 49.17 | 36.0 | 0.0 | 122.0 | 142.0 |
| CHARTQA | 215 | 1,785 | 1,458 | 1,250 | 970 | 38.09 | 18.0 | 0.0 | 114.0 | 154.0 |
| **Total** | **1,083** | **6,917** | **6,012** | **5,516** | **4,877** | **66.01** | **39.0** | **0.0** | **174.0** | **185.0** |

## Correction and preservation

A correction is a valid evaluated mask for a sample whose authoritative current ALL-ON route is wrong. For current-correct samples, preservation reports the least visual computation among valid evaluated masks.

| Dataset | Correction recovery | Correcting routes/recovered mean | Min ON median | Min ON mean | Mean OFF-layer saving |
|---|---:|---:|---:|---:|---:|
| GQA | 1,386/2,000 (69.30%) | 61.12 | 7.0 | 4.96 | 82.29% |
| TEXTVQA | 712/966 (73.71%) | 38.56 | 10.0 | 10.02 | 64.22% |
| CHARTQA | 774/989 (78.26%) | 30.18 | 11.0 | 10.26 | 63.36% |
| **Total** | **2,872/3,955 (72.62%)** | **47.19** | **9.0** | **7.58** | **72.94%** |

## Execution-contract drift

Historical easy/hard membership is metadata only; the current executor is authoritative.

| Dataset | Correct→correct | Correct→wrong | Wrong→correct | Wrong→wrong |
|---|---:|---:|---:|---:|
| GQA | 1,988 | 12 | 12 | 1,988 |
| TEXTVQA | 996 | 4 | 38 | 962 |
| CHARTQA | 994 | 6 | 17 | 983 |
| **Total** | **3,978** | **22** | **67** | **3,933** |

## Integrity and scope

- Per-sample rows: `8,000`.
- P4 audit SHA-256: `0afc2e62a0b20b5821bc847d8be2080d0f9cc9cef3cd94b829ff0e924a353cf2`.
- P4 record-index SHA-256: `f61eb0ac6c40e0498cdfaa53c328b3de34cbb67733a8fbd44c5bd590db051ebe`.
- Raw records reverified against P4 checksums: `8,000`.
- Per-sample summary: `outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl` (`4ef2d6e1f5cbdd4506068fc2b8e0896256678bc162a8c7a3e1ebb55e37ff744d`).
- Aggregate JSON: `outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json`.
- No record was excluded or replaced; no likelihood or predictor training was run.
- P6 diversity/transition analysis, P7 splits, P8 derived views, P9 final freeze, and P10 training were not executed.

## P5 decision

P5 passes because all 8,000 P4-frozen records reconcile into complete per-sample summaries and the required current ALL-ON, correction, preservation, budget, and contract-drift statistics are defined and saved. The next bounded action is P6 route-diversity and transition-structure analysis.
