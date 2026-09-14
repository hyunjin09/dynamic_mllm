# Full corrective-label generation protocol

- Contract SHA-256: `0931226b2c6823f2e4cd6721844163393a08df8c9f5d6c8904bf4e288da491dd`
- Frozen populations: 1,881 Stage-1-triggered Dense-W and 39 Stage-1-triggered Dense-C train rows.
- Import: 120 Phase-55 pilot W rows are transcript-equivalently reinterpreted at cap 200; the one rescue first found at iteration 294 is excluded, yielding 35 SINGLE_FIXABLE, 21 MCTS_ONLY_FIXABLE, and 64 UNRESOLVED imported rows.
- Fresh workload: 1,761 W searches and 39 C preservation replays, assigned deterministically across four direct GPUs.
- Dense prefix: exact native all-FULL execution through the state entering the frozen trigger layer. No layer before the trigger may change.
- W search: exhaustive singles from trigger through layer 27; MCTS only when no single succeeds; deterministic UCB1 2/3/4-cardinality rollout, binary current LMMS correctness, cap 200, 25 post-success iterations within the cap, retain at most 8 correct MCTS routes.
- C preservation: no search; retain and replay the known-safe all-FULL suffix.
- Every retained route is exact-token/correctness replay valid. Every saved feature is the actual pre-layer routed state for its route and is hash-bound to this contract/model/code/schema/source.
- Corpus A (`preservation_full`), B (`single`), and C (`mcts`) remain separate. Route multiplicity does not decide future sampling weight.
- This phase ends after search, replay, corpus construction, audit, figures, and artifact verification. It does not train Stage 2 or modify Stage 1.
