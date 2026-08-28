# Three-Action Answer-Aligned Label Conversion Completion Audit

This ledger maps `plans/4way_labeling_fix.md` to required evidence. `PROVEN` is already established, `LIVE` is being executed, and `PENDING` cannot be claimed before its named artifact exists.

| Sections | Requirement | Status |
|---|---|---|
| 0--2 | Five authoritative datasets; no MCTS/`mcts_v2`; reuse validated executor; exact three suppression aliases | **PROVEN** by source freeze, implementation audit, and tests |
| 3 | W2C fixed-target margin and C2C correct-answer support with evaluator-compatible references | **PROVEN** in implementation/tests; real-data target gate **LIVE** in pilot |
| 4 | Prospectively frozen within-unified repeatability epsilon; raw effects; no native-drift threshold | **PROVEN** by 56/56 passing records and `calibration/noise_calibration_v1.json` (`epsilon=1e-6`, 224 differences) |
| 5 | Current unified FULL and source replay define W2C/C2C; replay failures explicit/no replacement | **PROVEN** in runner/tests; population evidence **PENDING** |
| 6--9 | W2C hard/soft/redundant screening and contextual READ_OFF/WRITE_OFF/BOTH_OFF decomposition | **PROVEN** in pure tests; real-data paths **LIVE** in pilot |
| 10--11 | Bounded joint refinement, wrong partial guidance only, positive correctness, Pareto/canonical/max-margin routes | **PROVEN** in implementation/tests; beam-8/16 stability **LIVE** |
| 12--18 | C2C compensated-alignment objective, support-gain screening, three-action causal test, joint validation, semantic separation | **PROVEN** in implementation/tests; real-data C2C gain path **LIVE** |
| 19--20 | ALL-OFF W2C and every source route; sample-local preprocessing/cache; dedup provenance | Source inclusion **PROVEN**; full output accounting **PENDING** |
| 21 | Avoid blind fourth-state execution; report avoided forwards/cache hit rate | **PROVEN** in implementation/tests; population metrics **PENDING** |
| 22 | All 8 GPUs, 16 workers, two replicas/GPU, one sample/worker, balanced queue | New-method worker layout and saturated eight-GPU telemetry **PROVEN**; completed-pilot throughput remains **LIVE** |
| 23 | Five-dataset calibration/pilot, semantic/reference/cache/beam/resume/throughput gates | Calibration **PROVEN**; real-data label/beam pilot **LIVE** in job 1609 |
| 24 | Isolated source/unique/W2C/C2C/combined/canonical output views with hashes | **PENDING** |
| 25--27 | Per-dataset/combined W2C, C2C, and direct comparison analyses | **PENDING** |
| 28 | Only evaluator-correct positive routes; partial W2C candidates separate | **PROVEN** in code/tests; full integrity evidence **PENDING** |
| 29 | Prohibitions: no MCTS/exhaustive search/native drift/local composition/source overwrite/subsampling | **PROVEN** by contract; re-audit after full run |
| 30 | Final report answers all 12 required questions | **PENDING** |

## Required completion order

1. ~~Complete calibration records and freeze a passing checksum-bound epsilon artifact.~~ Done in job 1609.
2. Complete the new-semantics pilot on all 8 GPUs/16 workers; require hard, soft, C2C gain, target-policy, cache, beam, parity, checksum, and worker gates.
3. Automatically launch the clean full conversion only after pilot audit passes; dependent job 1610 is already queued fail-closed.
4. Complete/resume all 12,278 samples and reconcile all 545,531 source routes.
5. Finalize all required views, analyses, plots, report, and checksum ledger.
6. Re-run the complete active tests and change every remaining LIVE/PENDING item only when direct evidence exists.
