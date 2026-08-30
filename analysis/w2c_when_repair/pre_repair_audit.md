# W2C WHEN Repair — Pre-Repair Audit

Frozen before smoke or repair execution.

## Authoritative population

| Split | W2C samples | Valid routes | Mean routes/sample | Median | Single suffix | Multiple suffixes |
|---|---:|---:|---:|---:|---:|---:|
| Train | 512 | 13534 | 26.434 | 16.0 | 296 | 216 |
| Validation | 128 | 3314 | 25.891 | 15.5 | 73 | 55 |

- Total W2C samples: 640.
- Total existing valid routes: 16848.
- Overall mean/median valid routes: 26.325 /
  16.0; P95 78.0;
  maximum 103.
- Compatible known suffixes at old boundaries: 1269
  total; mean 1.983; median
  1.0; maximum
  16.
- Samples with one/multiple compatible suffixes:
  369 / 271.
- Old boundary depth distribution: {'early': 223, 'late': 194, 'middle': 223}.
- Dataset counts: {'chartqa': 214, 'gqa': 214, 'textvqa': 212}.
- Old mechanism counts: {'IGNORE': 162, 'MULTI': 156, 'READ_ONLY': 162, 'WRITE_ONLY': 160}.
- Initial known-FULL candidate routes after deduplication:
  1239.
- Initial one-edit suffix population: 55001;
  frozen-budget selected maximum 34861.
- All 640 physical source label records exist and
  match the SHA-256 stored in the authoritative manifest.

These are search-derived caches. Existing valid routes are authoritative input
to repair, not proof that their boundary action sets are complete.
