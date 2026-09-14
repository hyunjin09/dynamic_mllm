# Full corrective-label generation decision summary

Contract: `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`. The final corpus covers exactly the frozen 1,881 triggered Dense-W and 39 triggered Dense-C train samples. `UNRESOLVED` means no rescue was found under this bounded contract, not proven unfixability.

1. **SINGLE_FIXABLE:** 698/1,881 = **0.3711**.
2. **Additional MCTS_ONLY_FIXABLE:** 209/1,881 = **0.1111**.
3. **Final bounded correctability:** 907/1,881 = **0.4822**; 974/1,881 remain unresolved.
4. **Pilot comparison:** cap-200 pilot projection was 56/120 = 0.4667; the full-cohort rate differs by **+0.0155**.
5. **Successful single routes:** **7628** exact-replay routes were retained across 698 samples.
6. **Successful MCTS routes:** **442** exact-replay routes were retained across 209 samples.
7. **Route complexity:** preferred successful routes have median **1.00** non-FULL actions (IQR 1.00–1.00, 95th percentile 4.00); exact all-fixable and MCTS-only bins are in `metrics/route_complexity.csv`.
8. **Single intervention location/action:** exact layer×action counts are in `metrics/action_by_layer.csv`; the dedicated distribution is `figures/single_intervention_layer_distribution.png`.
9. **Dataset variation:** total correctability spans **0.4462** across GQA/ChartQA/TextVQA; exact simple/MCTS/unresolved counts and actions are in `metrics/dataset_breakdown.csv`.
10. **Trigger-depth variation:** total correctability spans **0.3415** across L0/L1-8/L9-18/L19-27; exact results are in `metrics/trigger_depth_breakdown.csv`.
11. **Training states:** Corpus A has **400**, Corpus B **199193**, and Corpus C **8687** pre-layer routed-state records.
12. **Stage-2 V1 support:** **yes**—Corpus A contains 39 preservation routes and Corpus B contains 7628 simple corrective routes, kept explicitly separate.
13. **Stage-2 V2 support:** **yes**—Corpus C adds 209 MCTS-only samples and 442 retained trajectories, but whether they improve a learned policy remains untested.
14. **Replay/provenance integrity:** **PASS**. All 8109 routes are current-LMMS-correct and exact-token replay valid; all 208280 states are in 946 contract/model/code/schema/source-bound shards. The 120-row pilot import uses only canonical cap-200 routes; the iteration-294 rescue and its tensor rows were excluded.

## Corpus boundary and stop

- Corpus A is only `preservation_full`; Corpus B only `single`; Corpus C only `mcts`.
- Multiple successful actions sharing the same routed prefix are retained as `observed_successful_actions` in `states/state_index.jsonl`; this is not a claim that all valid actions were found.
- Route multiplicity has not been converted into future sample weight.
- No Stage-2 model was trained, Stage 1 was not changed, validation/test were not searched, and no external/downstream evaluation ran.
