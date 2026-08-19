# Label Regeneration P7 Predictor Split

The exact outcome-blind image-group-disjoint predictor split is frozen.
Selection used only dataset, image group, historical source-cell status,
and seed `20260809`. Current executor outcomes were joined only after
assignment for descriptive auditing.

| Dataset | Split | Records | Image groups | Historical correct | Historical wrong | Current correct | Current wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| GQA | train | 3500 | 3352 | 1750 | 1750 | 1748 | 1752 |
| GQA | validation | 500 | 482 | 250 | 250 | 252 | 248 |
| TEXTVQA | train | 1750 | 1684 | 875 | 875 | 903 | 847 |
| TEXTVQA | validation | 250 | 241 | 125 | 125 | 131 | 119 |
| CHARTQA | train | 1750 | 1522 | 875 | 875 | 886 | 864 |
| CHARTQA | validation | 250 | 216 | 125 | 125 | 125 | 125 |

- Train: `7000` records.
- Validation: `1000` records.
- Cross-split image groups: `0`.
- Duplicate UIDs: `0`.
- Route success, valid-route count, correction discovery, diversity, and predictor/evaluation outcomes were not selection inputs.
- P8, P9, and predictor training were not executed.

Artifacts:

- `outputs/label_regeneration/v1/post_generation/predictor_split_manifest_v1.jsonl`
- `4d12bf427f08b0cc55d21c82bf7eaac7d19d283dc514ffd4f59894d6faf1bd1a`
- `2f60e4688e7727d5f6d715c255daba4cba4a2f0e10476b390371522c0b1ad84e` (audit file checksum)
