# Mandatory-Boundary Audit

## Frozen provenance

- Source manifest: `outputs/four_action_polar/preparation_v1/manifest_v1.jsonl`
- Source manifest SHA-256: `6a50ca1a2d5c512d7bd8cededfc4732c93258264ae471e1fd4653c6c16637c28`
- Boundary manifest SHA-256: `0e11651beee39a0723fd9a973e5ae70e34314b89ac9684ba83507e74e5becd47`
- Pilot subset SHA-256: `85235eab3f61405fbc2d213cb5ae9e4390f9b231ad4ce122d830dd9a5c70b734`
- Plan SHA-256: `f61f7476ff9a5872f823c7df837e1a2ba21774c83e4efc88f152d2b77d5aceb9`

## Integrity gates

- W2C records: 2397
- Unique UIDs: 2397
- FULL invalid at boundary: 2397/2397
- At least one non-FULL valid action: 2397/2397
- Singleton boundaries: 1898/2397

## Boundary distribution

- Mean layer: 14.586
- Range: 0–27
- Valid action sets: `{"IGNORE": 427, "IGNORE+READ_ONLY": 57, "IGNORE+READ_ONLY+WRITE_ONLY": 62, "IGNORE+WRITE_ONLY": 108, "READ_ONLY": 649, "READ_ONLY+WRITE_ONLY": 272, "WRITE_ONLY": 822}`

## Dataset counts

| Dataset | Records | Singleton |
|---|---:|---:|
| gqa | 1147 | 878 |
| chartqa | 652 | 549 |
| textvqa | 598 | 471 |

All required A0 invariants pass. The canonical artifacts are under
`analysis/4action_collapse/`; the legacy A0 paths under
`analysis/4action_router/` may link to these frozen files.
