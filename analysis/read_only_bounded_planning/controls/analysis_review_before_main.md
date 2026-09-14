# Supplemental analysis freeze before main outcomes

The supervisor was paused after infrastructure smoke and before main search. The original execution/smoke contract is preserved unchanged.

- Structured-versus-random comparisons use the per-UID shared actual route cap across both algorithms and all paired seeds. Report those caps and bootstrap seed-averaged UID differences.
- Binary MCTS participates in structured comparisons. A/B coverage uses the same matched-cap population; the first-rescue<=32 gate must hold separately for every seed. Rank/CDF summaries use one observed-seed mean per UID, with censored seed counts and adaptive coverage retained.
- Correctness-stopped Hamming<=2 enumeration remains a first-class union/complexity/stopping-policy reference. It is excluded from positive A/B superiority evidence. If enumeration already recovers>=70% by128, do not call the landscape needle-like solely because the other algorithms fail; report an inconclusive planning category when appropriate.
- Planning-C's structured advantage is evaluated independently of70% ceiling recovery; low recovery alone cannot imply no structured advantage.
- Use the existing tie-aware binary_auroc implementation; no new package installation or environment change.

Search code/config, q/generation, model/runtime, population, cache acceptance, RNG seeds, and budgets remain exactly those of the original40-UID smoke. This supplemental contract binds the corrected finalizer, its statistical helper/tests and supervisor before any main-search outcome. It supersedes only the initial analyzer implementation and its conflicting operational reporting text; execution and smoke provenance are unchanged.
