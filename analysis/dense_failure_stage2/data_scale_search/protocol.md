# Stage-2 data-scale search protocol

- Contract SHA-256: `74ebd82c4d071a06a1d5efa8590d9a5614ab2308bbe026453795cb444000dcaf`
- Candidate manifest SHA-256: `a8e1daa29f3003d0de5cf7bc542feaca162eb2f4de9d54a591641bda8eb0740c`
- Population: 4,000 unique train candidates (2,000 GQA / 1,000 ChartQA / 1,000 TextVQA), each a unique SHA-256 image group, with zero overlap against all 8,000 legacy candidates.
- Selection was frozen before current dense outcomes, Stage-1 scores, triggers, or fixability. It matches recoverable pre-outcome metadata strata and uses no historical or current model correctness.
- Runtime: frozen native dense Qwen2.5-VL followed by the frozen Shared Random-4 strict global trigger. Triggered Dense-C receives FULL preservation; triggered Dense-W receives exhaustive singles, then cap-200 Phase-56-equivalent MCTS only if single-unresolved.
- Every retained route must reproduce exact tokens and current LMMS correctness. Single, MCTS, preservation, and unresolved provenance remain separate.
- Claim boundary: this is a canonical-source data-scale expansion, not a perfectly pure scale-only population replication.
- Stop after expanded corpora/audits. No router training, Stage-1 retuning, validation/test search, or held-out evaluation.
