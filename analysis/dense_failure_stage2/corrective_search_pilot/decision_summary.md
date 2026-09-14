# Trigger-conditioned corrective-search pilot decision summary

Contract: `6b4eb6de4573a0f06397304721483348523b78a1ce905c5076e49763f15afa2c`. All rates below are bounded-search lower estimates under the frozen 120-sample diagnostic allocation; the balanced-pilot rate is primary for diagnosis, and the Phase-54 cell-weighted estimate is reported in `metrics/overall_correctability.csv` for population projection.

1. **Single-intervention rescue:** 35/120 = **0.2917**.
2. **Additional MCTS-only rescue:** 22/120 = **0.1833**.
3. **Total bounded correctability:** 57/120 = **0.4750**.
4. **Budget saturation:** Fixable@100 = **0.4333**, @200 = **0.4667**, and @300 = **0.4750**.
5. **200 versus 300:** gain from 200→300 is **0.0083**. Using the prospectively frozen **0.0100 absolute-rate** materiality yardstick, the evidence selects a cap of **200** iterations for any separately authorized full label phase.
6. **Successful-route complexity:** among one post-hoc preferred route per fixable sample, median non-FULL count is **1.00** (IQR 1.00–3.00).
7. **Multiple successful first actions:** observed for **14/57** fixable samples (0.2456). This is discovery support, not proof that unobserved actions fail.
8. **Trigger-depth variation:** the max–min total-correctability span across the four bins is **0.3491**; exact rates are in `metrics/trigger_depth_breakdown.csv`.
9. **Dataset variation:** the max–min total-correctability span across GQA/ChartQA/TextVQA is **0.1500**; exact rates are in `metrics/dataset_breakdown.csv`.
10. **Projected full-cohort cost:** approximately **362007 terminal routes**, **31.15 GPU-hours**, or **7.79 wall-hours** at four equally utilized GPUs, based on dataset×depth-cell means.
11. **Sequential Stage-2 justification:** **yes, within this bounded pilot**. MCTS-only rescue was observed beyond exhaustive singles; this says whether multi-layer search added discovered labels, not whether a learned policy will generalize.
12. **Recommended MCTS cap:** **200 iterations/sample**, conditional on separate authorization for full corrective-label generation.

## Execution and label integrity

- All 120 frozen UIDs completed exactly once across four ranks; there were no failed or partial sample records.
- Every sample reproduced its current dense generated tokens, answer, prompt hash, and LMMS correctness under native all-FULL execution before search.
- Layers before each Phase-54 trigger remained FULL. All suffix rewards were binary current LMMS-Eval correctness.
- All 360 retained successful routes were exactly replayed; 7776 compact pre-layer action-state rows were stored with contract/model/code/schema/source/run provenance.
- Single-fixable samples were excluded from MCTS. The 39 triggered Dense-C rows, validation/test cohorts, remaining train candidates, Stage-2 training, and external evaluation were not executed.

## Stop decision

Phase 55 stops here. The cap recommendation and whether to scale label generation are conclusions for user review, not authorization to search the remaining 1,761 rows or train Stage 2.
