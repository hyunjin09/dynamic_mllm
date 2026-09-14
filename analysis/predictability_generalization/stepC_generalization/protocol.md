# Predictability Step-C protocol

- Targets and model/optimization contracts: unchanged from Step B `792cc760bb41ce8c9b11776f21f43a88e54409ef1543471709000f83ec21bba4`.
- Question encoder: `Qwen/Qwen3-Embedding-0.6B` snapshot `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`.
- Shared instruction: `Given a visual-question-answering question, retrieve semantically similar questions.`
- Semantic clusters: deterministic spherical k-means K=100; balancing is label-blind.
- OOD regimes: 5 cluster folds, bidirectional global/within-dataset source transfer, 3 LODO, 6 pairwise dataset transfers.
- Controls: frozen M0 nuisance, M1 linear, M3 current/full-state, exact k=5 question kNN.
- Unsupported cells are frozen before outcomes and are not relaxed.
