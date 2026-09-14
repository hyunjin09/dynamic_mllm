# Stage-1 gate treatment-correctability protocol

- Contract SHA-256: `dd532bbc1a57a3f175e12807603ea1cd20b6379dc6c1ddee5433dc2a09f53d11`
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Gates: shared Random-4 sequential, independent sequential, and fixed-L27 detect-and-replay at Phase-52 selected controls only.
- Dynamic treatment reuses the exact materialized all-FULL prefix and may change only layers from the first trigger through 27.
- Fixed-L27 treatment performs a second pass from layer 0.
- Search is outcome-independent: all permitted one-layer READ_ONLY/WRITE_ONLY/IGNORE routes plus exactly 12 seeded distinct-layer pairs, or every pair if fewer than 12 exist.
- Search stops at the first current LMMS-Eval-correct route. Rates are bounded-search lower estimates, not exhaustive four-action oracle rates.
- Enrichment uses the same layer-0 replay panel on every dense-wrong sample, then compares triggered subsets with all wrong. Dynamic deployment correctability is reported separately.
- Native dense token parity and cached-suffix versus complete-route token parity must pass the 12-record smoke.
- Validation is completed before test treatment execution. Test cannot change the validation-selected dynamic substrate.
- No historical cached route correctness is authoritative.
