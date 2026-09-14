# Robust Stage-1 missing corrective-search protocol

- Frozen parent: Phase-64 contract `ab254e983ad3ed02f587e93f6a82250721916c1cc08a2e32b8400c5870546674`.
- Operating points: P98/P95/P90 with exact frozen trigger maps and strict score comparison.
- Search population: 1,104 unique Dense-W UIDs, 1,906 missing threshold pairs.
- Route discovery: exhaustive single interventions once from each UID's earliest needed trigger.
- MCTS: deterministic existing tree/action/reward semantics, at most 200 iterations per actually launched unresolved trigger root, up to eight successful routes retained per root.
- Validation: structural eligibility followed by exact threshold-specific current Qwen/LMMS replay; token parity and correctness are required for corpus admission.
- State provenance: each unique route is captured once from its earliest exact-valid threshold trigger; every threshold view binds an explicit contiguous layer slice beginning at that threshold's trigger.
- Existing route handling: Phase-64 replay-compatible routes are retained and their states are recaptured under the robust trigger contract without new search.
- Preservation: all 106 triggered-C UIDs receive the FULL suffix, with threshold-specific state slices.
- Stop boundary: build P98/P95/P90 corpora and audits only; no Stage-2 training, Stage-1 change, test deployment, or external evaluation.
