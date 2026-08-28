# Four-Action Label Conversion Completion Audit

This ledger maps every numbered section of `plans/4way_labeling.md` to the
authoritative evidence needed for completion. `PROVEN` means current evidence
already establishes the requirement. `LIVE` means the immutable full run is
producing the evidence. `PENDING` must not be claimed until the named final
artifact or check exists.

| Plan section | Required evidence | Current status |
|---|---|---|
| 0. High-level goal | Binary `ON/OFF` routes are mapped and refined in the unified `FULL/READ_ONLY/WRITE_ONLY/IGNORE` space. | **PROVEN** by the implementation audit, execution contract, pilot audit, and active tests. |
| 1. Datasets | Exact authoritative artifacts for GQA, TextVQA, ChartQA, WeMath Standard, and WeMath Pro. | **PROVEN** in `implementation_audit.md` and `source_inventory_summary_v1.json`. |
| 2. Convert all labels | Frozen inventory contains every positive sample/route in the selected authoritative views; no negative-mask expansion or silent subsampling. | **PROVEN** for the frozen input: 12,278 samples and 545,531 positive routes. Full output accounting is **LIVE**. |
| 3. Executor audit | Unified four-action semantics, arbitrary complete routes, shared executor/scorer/preprocessing, and binary mapping parity. | **PROVEN** by pilot audit, code hashes, and active tests. |
| 4. Source replay | Every source positive route is replayed; failures are explicit and never replaced; current unified FULL defines semantics. | **LIVE**. Final proof: `full_integrity_audit_v1.json` exact raw-route accounting and final raw view. |
| 5. W2C/C2C separation | W2C is corrective/refined; C2C remains mechanical and is never called answer-unaligned. | **PROVEN** in implementation/pilot; full-population proof is **LIVE**. |
| 6. W2C purification | Both deterministic restoration orders reach fixed point and selection uses cost, margin, stable tie-break. | **PROVEN** by converter tests and pilot. |
| 7. Contextual refinement | Relaxations are evaluated inside the current correcting route rather than an all-FULL context. | **PROVEN** by converter implementation/tests and pilot. |
| 8. Bounded monotone search | Beam width 8, permitted monotone transitions only, lexicographic objective, no MCTS. | **PROVEN** by execution contract and tests. |
| 9. Joint validity | Every retained final route is executed as a complete trajectory and evaluator-correct; independent-composition failures are counted. | **PROVEN** in pilot; full-population proof is **LIVE**. |
| 10. ALL-OFF W2C | Seeds retained and current unified ALL-OFF-correct/wrong strata recorded separately. | **PROVEN** in full record schema/tests; final strata totals are **LIVE**. |
| 11. Multiple routes | Per-sample preprocessing/FULL/route cache reuse; all source routes processed; identical final routes deduplicated with a provenance partition. | **PROVEN** in implementation/pilot; exact full provenance audit is **PENDING**. |
| 12. Output views | Raw mapping, full per-sample unique set, canonical route, deterministic max-50 training view, source metadata, scores, actions, and hashes. | **PENDING** on `finalize_four_action_label_conversion.py` after a passing full audit. |
| 13. GPU configuration | All 8 GPUs, 16 workers, two replicas/GPU, one sample/worker, load model once. | **PROVEN** by pilot audit and live job 1604 progress/telemetry; final job provenance remains **LIVE**. |
| 14. Load balancing | Cost-aware launch-scoped atomic shared queue keeps all workers supplied. | **PROVEN** by tests and live job 1604; final utilization summary is **LIVE**. |
| 15. Resumability/safety | Atomic checksum-protected records, append-only progress/failures, launch-scoped claims, deterministic contract, resume without overwrite. | **PROVEN** by pilot resume and active full records; final checksum/accounting audit is **PENDING**. |
| 16. Pilot | 40--80 stratified samples across five datasets and all specified paths; 8 GPUs/16 workers; clean gate before full. | **PROVEN** by `pilot_audit_v1.json` for 56/56 samples and 4,026 routes. |
| 17. Output location | Separate `datasets/mcts_labels_4action/` root; source labels untouched; splits preserved; image leakage check. | Separate root is **PROVEN**; finalized split/leakage manifest is **PENDING**. |
| 18. Final report | Combined and five dataset-specific summaries with scientific, throughput, utilization, memory, GPU-hour, and exclusion fields. | **PENDING** on final analyzer and manual evidence review. |
| 19. Scientific analyses | W2C redundancy/refinement/depth/dataset/ALL-OFF/diversity/joint-validity answers; separate C2C efficiency analysis. | **PENDING** on full results and `aggregate_statistics_v1.json`. |
| 20. Prohibitions | No fresh MCTS, exhaustive search, `mcts_v2`, source overwrite, semantic mixing, local-action composition, silent fallback, or silent subsampling. | **PROVEN** through the frozen source/contract and implementation audit; verify no unexpected artifacts at final audit. |
| 21. Final decision | Exact sources/counts; READ/WRITE structure; W2C/C2C separation; training readiness; remaining case for fresh search. | **PENDING**. The analyzer now emits all seven explicit decisions; research interpretation follows only after the full audit passes. |

## Finalization order and hard gates

1. Job 1604, and any checksum-compatible resume allocation if required, must
   end with all 12,278 atomic records present and no unresolved worker failure.
2. Run `audit_four_action_label_full.py` with every full-run Slurm job ID. Do
   not finalize unless every audit check passes.
3. Run `finalize_four_action_label_conversion.py`. Verify all six views and
   their sidecars, exact 545,531-row raw accounting, deterministic max-50 cap,
   canonical coverage/exclusions, and zero image-split leakage.
4. Run `analyze_four_action_label_conversion.py`. Inspect combined and all five
   dataset summaries, plots, ALL-OFF strata, W2C/C2C wording, operational
   metrics, and all seven final decisions.
5. Run `write_four_action_label_checksum_ledger.py`, then verify the ledger and
   sidecar against the final artifact set.
6. Run the active project gate with `.venv/bin/python -m pytest -q tests`.
   Repository-root pytest collection is not the project gate because it also
   imports vendored reference suites with conflicting package roots.
7. Re-read this ledger against current artifacts, change every remaining
   `LIVE`/`PENDING` item only when its direct evidence exists, and update the
   phase/global research state before declaring the goal complete.
