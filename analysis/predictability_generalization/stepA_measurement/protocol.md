# Predictability Step-A frozen measurement protocol

- Contract: `6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d`
- Internal Stage-1 population: 6,399 Historical-train + 4,000 Canonical = 10,399 samples / 291,172 `(sample, layer)` states.
- Stage-1 state: exact existing current-runtime `text_final`, `text_mean`, and `visual_mean` BF16 features, admitted only after source hashes and fresh stratified native replay pass.
- Trigger: robust ALL-source five-checkpoint probability mean, strict P90 `p > 0.9061332901863008`.
- Primary Stage-2: all 1,413 P90-triggered UIDs / 15,185 dense pre-action states.
- Secondary Stage-2: all 35,565 unique Phase-76 exact routed-prefix states, independently reconstructed and state-hash checked.
- Branches: `FULL`, `READ_ONLY`, `WRITE_ONLY`, `IGNORE` at the current layer, followed only by `FULL`.
- Continuous q: weighted log-sum-exp of per-reference token-mean log probability, without EOS. GQA uses one evaluator-normalized annotation; ChartQA uses the literal annotated gold; TextVQA uses EvalAI-normalized references with empirical frequency weights.
- Discrete outcomes: exact current LMMS-Eval task scoring and frozen binary thresholds.
- Limitation: ChartQA relaxed numeric correctness represents a tolerance interval; finite-string q measures the annotated gold string and is not the probability mass of that full interval.
- No outcome filtering, MCTS, suffix search, utility threshold, probe/router training, OOD evaluation, or predictability claim.
