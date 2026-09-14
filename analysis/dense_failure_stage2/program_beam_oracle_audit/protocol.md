# Frozen program beam-oracle audit protocol

- Contract: `e274d24cf145c191929c5080bdbb83ad2342b31eb5d90f0d5e3ffc9eb06c7fe7`.
- Parent Phase-74 contract: `0c1696ed895353f31ba40201633215f235bacc7175624dccd2ac8d7fdb597a49`.
- Frozen population: 19,960 external rows; 901 triggered (496 Dense-W, 405 Dense-C).
- Candidate set: 6,944 ranked entries / 6,944 unique UID-program executions.
- Beam: width 8, sum of action log probabilities; layer-27 triggers use their exhaustive four-program support.
- Gate: all 901 top-1 paths must reproduce exact trigger layer, program, generated tokens, answer, LMMS score, and correctness; beam score tolerance is 1e-06.
- Only after the global top-1 gate passes may frozen ranks 2-8 execute.
- External labels score already-frozen programs only. No search, reranking, tuning, retraining, or model modification is allowed.
