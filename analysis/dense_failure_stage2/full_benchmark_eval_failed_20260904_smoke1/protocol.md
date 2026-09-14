# Full-Benchmark End-to-End Evaluation Protocol

- Contract SHA-256: `9d7b8d32b3272813f81218c3c77fa379f67ef7743d88937b9440e35ae1a67618`
- Git commit: `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2` on `main`; the complete dirty-worktree snapshot is recorded in `frozen_protocol.json`.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, SDPA, slow processor, local files only.
- Dense control: all 28 layers `FULL`.
- Routed candidate: five-checkpoint ALL-source Shared Random-4 Stage-1 probability mean, first strict `score > 0.9061332901863008` crossing, then frozen Phase-66 Experiment A Stage-2 actions `FULL/READ_ONLY/WRITE_ONLY/IGNORE` on the actual routed state.
- Generation: greedy custom argmax, repetition penalty 1.05 in FP32, EOS `[151645]`, row-specific 16 tokens except POPE 128.
- Correctness: frozen LMMS-compatible project scorers—ChartQA relaxed accuracy; TextVQA EvalAI consensus (correct at score >=0.5); MMMU-Pro first standalone A-J accuracy; POPE yes/no accuracy.
- Full paired population: 19,960 unique UIDs: ChartQA 2,500; TextVQA 5,000; MMMU-Pro Standard/Vision 1,730 each; POPE adversarial/popular/random 3,000 each.
- Excluded: GQA, DocVQA, MMStar, and base MMMU.
- Primary criterion: `W→C > C→W`; no threshold, checkpoint, or task selection may use these external results.
- ChartQA, TextVQA, MMMU-Pro, and POPE remain separate primary family metrics because their task scorers differ. Any pooled row is a descriptive 19,960-sample micro-average, not a benchmark macro-average or a common-score claim.
