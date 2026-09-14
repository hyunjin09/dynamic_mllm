# ALL-Source LODO TEXTVQA Protocol

- Frozen parent contract: `08dbf1464f5b1d2bdd4c9398ff11ce961a7d59aa1c52967e3505c7fac890a4fa`
- `textvqa` is absent from training, internal validation, checkpoint selection, and layer selection.
- Training uses Historical train plus all Canonical rows from `gqa, chartqa`.
- Every epoch uses exact 25% Historical-C/Historical-W/Canonical-C/Canonical-W quotas with natural within-cell dataset mixture.
- Evaluation uses every Historical and Canonical `textvqa` row; all are target-blind because `textvqa` is completely absent from fitting. No threshold is selected.
