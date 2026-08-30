# Frozen Matched-Subset Boundary Audit

- Records: 1,280 total; 1,024 train and 256 validation.
- Train: 512 W2C and 512 C2C.
- Validation: 128 W2C and 128 C2C.
- Dataset cells: train 171/171/170 and validation 43/43/42 per route type for
  GQA/ChartQA/TextVQA (fixed dataset ordering in `subset_manifest.json`).
- Mandatory boundaries: 640, covering all selected W2C records.
- Boundary layers: every decoder layer 0 through 27 is represented.
- Action-set memberships: IGNORE 230, READ_ONLY 287, WRITE_ONLY 295.
- Singleton boundaries: 484; multi-valid boundaries: 156.
- Boundaries allowing FULL: 0.
- Missing images: 0; missing cached visual tensors: 0.
- Cross-split image-group leakage: 0.
- C2C rows or labels modified: 0.

Machine-readable evidence: `subset_audit.json` (SHA-256
`9245d86c1df550c580a7cf56df88c1c2620cf643d84ddd8ff664a5cfe9c904df`).
