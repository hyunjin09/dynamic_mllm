# Closed-loop trajectory-set Stage-2 frozen protocol

- Contract: `00639e7e5638d766ca9d83d912e827456185bfa778fda4ab702407dc6c835f6a`
- Population: 569 P90-triggered UIDs / 4,948 replay-valid complete programs.
- Dense-C: one all-FULL preservation route. Dense-W: every unique successful route.
- State: exact pre-action routed query + visual inputs, keyed by action-prefix hash.
- Objective: one length-normalized log-mean-exp successful-trajectory loss per UID.
- Router: frozen architecture initialized exactly from Sequential-A; READ/WRITE/shared head trainable, Qwen and Stage-1 frozen.
- Selection: image-group-disjoint internal dev selects only epoch count; full refit uses all 569 UIDs.
- Evaluation: exact Phase-69 19,960-row ChartQA/TextVQA/MMMU-Pro/POPE protocol with deterministic greedy closed-loop routing.
- No layer/source IDs, Stage-1 score/history input, route preference, sampling, beam, MCTS, or external-set tuning.
