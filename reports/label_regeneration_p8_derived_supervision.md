# Label Regeneration P8 Derived Supervision

Status: **PASS**

P8 derives training views from the checksum-verified raw cache and frozen P7 split. No route was re-executed and the raw cache was not modified.

## Counts

| Dataset | Split | Samples | Positive samples | Raw valid routes | Selected valid routes | Evaluated ranking routes |
|---|---|---:|---:|---:|---:|---:|
| GQA | train | 3,500 | 2,957 | 310,566 | 115,508 | 1,173,799 |
| GQA | validation | 500 | 429 | 42,952 | 16,619 | 165,600 |
| TEXTVQA | train | 1,750 | 1,525 | 85,856 | 51,311 | 572,499 |
| TEXTVQA | validation | 250 | 221 | 12,488 | 7,478 | 80,500 |
| CHARTQA | train | 1,750 | 1,561 | 66,598 | 40,875 | 569,300 |
| CHARTQA | validation | 250 | 224 | 9,587 | 6,011 | 81,300 |

- Samples: `8,000`; positive: `6,917`; zero-positive: `1,083`.
- Raw valid routes: `528,047`; selected: `237,802`; evaluated positive+negative ranking routes: `2,642,998`.
- Samples capped from more than 50 routes: `3,616`.

## Frozen selection

The max-50 view first includes the minimum-ON route and valid ALL-OFF/ALL-ON anchors. It then balances exact ON-count strata and greedily maximizes minimum Hamming distance, transition-count coverage, and transition distance, with a seeded digest used only for ties.
Duplicated BCE and exact set-NLL consume the same `binary_predictor_manifest_v1.jsonl` and therefore the identical selected masks and equal within-sample weights.

## Integrity

- Raw records checksum verified: `8,000`.
- P7 cross-split image groups: `0`.
- Selected masks missing from raw valid sets: `0`.
- Minimum-route/anchor violations: `0`.
- Shared objective route-set digest: `eafd8bb9dd66b2a800850e6f8e778eb68ad493c24c2861b921e91bb387c2bc0b`.
- P9 and predictor training were not executed.

## Artifacts

- `single_best`: `outputs/label_regeneration/v1/post_generation/derived_single_best_manifest_v1.jsonl` (`86f35c55ddf0f20eeb5f7339f7cc70045994b5aff327061977e299761e5e4c6d`)
- `valid_set`: `outputs/label_regeneration/v1/post_generation/derived_valid_set_manifest_v1.jsonl` (`dd6ac520b990a4d05316f20950b0a5ba7a3ee07cfb2c89f7d4bfba82e25f4c1b`)
- `binary_predictor`: `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl` (`3620a347a3498d16853463a6f9f8b842fecbab7b442cb869f1fb11bc9ab8aa52`)
- `route_ranking`: `outputs/label_regeneration/v1/post_generation/derived_route_ranking_manifest_v1.jsonl` (`60aa9075e775646ad8e928abf0c070bc3bbd0cb1fc7328be6b3f8d4a101daefe`)
- `polar_segment`: `outputs/label_regeneration/v1/post_generation/derived_polar_segment_manifest_v1.jsonl` (`cfe9d4cdd2e82c206ef30a348fd2f4d1554075037fce6cafbcab34b5899b47ad`)
- `audit`: `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json` (`29ed03efdb548fa19fc6eddccd03612e6f348b8f382ce11237cbeb03f9b54856`)
