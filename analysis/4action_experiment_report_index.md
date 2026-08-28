# Four-Action Answer-Alignment Experiment Reports

This document is the entry point for the two completed four-action studies:

- [Plan 1: full-context layerwise decomposition](../plans/4way.md)
- [Plan 2: route-conditioned decomposition](../plans/4way_2.md)

Both studies use the unified four-action executor, but they answer different questions. Plan 1 changes one layer while every other layer remains `FULL`. Plan 2 changes an OFF position inside an existing successful binary correcting route while preserving all other route decisions. Their effects therefore have different conditioning contexts and should not be treated as interchangeable estimates.

## Completion summary

| Study | Conditioning context | Analysis population | Evaluated cells/actions | Status |
|---|---|---:|---:|---|
| `4way.md` | One target layer; all other layers `FULL` | 1,880 current-valid A+ samples: 1,222 GQA, 658 TextVQA | 28 layers × 4 actions per primary sample, plus controls and trajectory analyses | Complete |
| `4way_2.md` | One OFF position inside a fixed successful binary route | 1,804 samples with a current-correct cached anchor route: 1,170 GQA, 634 TextVQA | 17,262 route cells; 69,048 action rows; 51,786 new evaluations | Complete |

No four-action project job is currently pending or running. The bounded joint-refinement study described at the end of Plan 2 is a proposal only; it has not been launched.

## Plan 1: full-context local causal landscape

### Scientific question

For each sample and layer, what are the READ, WRITE, and interaction effects when that layer alone is changed and all remaining layers execute unified `FULL`?

The four margins were defined entirely inside the unified executor:

- `M00`: `IGNORE`
- `M10`: `READ_ONLY`
- `M01`: `WRITE_ONLY`
- `M11`: unified `FULL`

Native Qwen `FULL` was retained only for cohort provenance, semantic checking, and implementation-drift measurement. It was not used as `M11`.

### Main findings

- The matched cache contained 1,912 candidate A+ samples. Re-evaluation with the current unified executor retained 1,880 primary samples; 32 were excluded because current unified `FULL` was no longer wrong.
- A local single-layer behavioral rescue occurred for 836/1,880 samples: 380/1,222 GQA and 456/658 TextVQA. Samples without a discrete rescue were retained in the continuous analysis.
- The most negative population-mean READ effect with WRITE enabled was at layer 19 (`-0.0992`). The most negative population-mean WRITE effect with READ enabled was at layer 0 (`-0.0945`). The layer profiles were not reduced to a pre-imposed early/middle/late narrative.
- Layers turned OFF by cached correcting routes were enriched for locally harmful visual operations. Relative to layers kept ON, OFF layers had `+0.1081` larger `M00-M11`, `-0.0674` lower `Delta_READ_W1`, and `-0.0382` lower `Delta_WRITE_R1` on average.
- The number of locally rescuing layers was strongly negatively associated with nearest correcting-route distance (Spearman `-0.5606`). Other count and magnitude relationships were weaker.
- 93.7% of primary samples had a positive intermediate correct-vs-FULL-wrong margin, while mean peak-to-final erosion was `4.3271`. The strongest local harmful operation did not localize the largest subsequent trajectory drop better than the matched random comparison (within two layers: 16.1% versus 16.2%).
- Population trajectory interventions nevertheless improved final margins in most selected cases: 96.45% for harmful-READ candidates, 96.60% for harmful-WRITE candidates, and 93.03% for joint candidates.
- The primary cohort showed somewhat more negative local effects than the matched “no correcting route found” control. Vision-required, FULL-correct controls instead showed positive average visual effects. “No correcting route found” means only under the matched search budget, not that no such route exists.

### Validation and numerical consistency

- Unified `FULL` matched native deterministic token sequences, answers, evaluator decisions, and correctness on all 72 semantic-validation cases.
- Unified `IGNORE` matched the old binary single-layer OFF executor semantically on all 1,816 checked cases.
- Native-versus-unified continuous margin drift was kept separate from the causal calculation. Signed drift had mean approximately `0`, median `0`, standard deviation `0.0517`, p95 `0.0899`, p99 `0.125`, and maximum absolute outlier `1.125`. Absolute drift had mean `0.0257`, median `0.0033`, p95 `0.125`, p99 `0.1388`, and maximum `1.125`.
- All READ/WRITE factorial effects in the report are within-unified differences and do not use this implementation drift as an effect threshold.

### Plan 1 reports and evidence

Start with:

- [Final scientific report](4action_answer_alignment/4action_answer_unaligned_report.md)
- [Numerical consistency report](4action_answer_alignment/numerical_consistency_report.md)
- [Implementation audit](4action_answer_alignment/implementation_audit.md)
- [Experiment log](4action_answer_alignment/experiment_log.md)

Primary data and aggregate tables:

- [Raw per-sample/layer/action results](4action_answer_alignment/per_sample_layer_actions.parquet)
- [Analysis summary](4action_answer_alignment/aggregate/analysis_summary.json)
- [Per-layer effects](4action_answer_alignment/aggregate/per_layer_effects.parquet)
- [Per-layer rescue taxonomy](4action_answer_alignment/aggregate/rescue_taxonomy_per_layer.parquet)
- [Per-sample effect summary](4action_answer_alignment/aggregate/per_sample_effect_summary.parquet)
- [Route overlap per sample](4action_answer_alignment/aggregate/route_overlap_per_sample.parquet)
- [Route overlap summary](4action_answer_alignment/aggregate/route_overlap_summary.json)
- [Hamming-distance associations](4action_answer_alignment/aggregate/hamming_distance_associations.parquet)
- [Hamming strata](4action_answer_alignment/aggregate/hamming_strata.parquet)
- [Answer-erosion table](4action_answer_alignment/aggregate/answer_erosion.parquet)
- [Answer-erosion summary](4action_answer_alignment/aggregate/answer_erosion_summary.json)
- [Native/unified drift table](4action_answer_alignment/aggregate/native_unified_full_drift.parquet)
- [Native/unified drift summary](4action_answer_alignment/aggregate/native_unified_full_drift_summary.json)
- [Control-cohort comparisons](4action_answer_alignment/aggregate/cohort_comparisons.parquet)
- [Semantic-validation summary](4action_answer_alignment/aggregate/validation_semantic_summary.json)
- [Trajectory-rescue summary](4action_answer_alignment/trajectory_rescue/summary.json)
- [Trajectory selection summary](4action_answer_alignment/trajectory_rescue/selection_summary.json)

Population figures:

- [READ effects by layer](4action_answer_alignment/figures/read_effect_vs_layer.png)
- [WRITE effects by layer](4action_answer_alignment/figures/write_effect_vs_layer.png)
- [Interactions by layer](4action_answer_alignment/figures/interaction_vs_layer.png)
- [Effect distributions](4action_answer_alignment/figures/effect_distributions.png)
- [Local rescue prevalence](4action_answer_alignment/figures/local_rescue_prevalence_vs_layer.png)
- [Route OFF frequency versus harmfulness](4action_answer_alignment/figures/binary_off_frequency_vs_local_harmfulness.png)
- [Hamming-distance stratification](4action_answer_alignment/figures/hamming_distance_stratification.png)
- [Answer-erosion curves](4action_answer_alignment/figures/answer_erosion_curves.png)
- [Culprit layer versus collapse layer](4action_answer_alignment/figures/culprit_layer_vs_collapse_layer.png)
- [Native/unified FULL margin drift](4action_answer_alignment/figures/native_unified_full_margin_drift.png)

The [historical mid-run status snapshot](4action_answer_alignment/run_status_summary.md) is preserved for provenance, but it is superseded by the final report and experiment log. In particular, the final CPU analysis job that was pending in that snapshot later completed successfully.

## Plan 2: route-conditioned mechanism decomposition

### Scientific question

Given an already successful cached binary correcting route, which of its OFF positions are actually necessary in that fixed route context, and when a position is necessary, is the rescue attributable to suppressing READ, WRITE, either component, or both?

Plan 2 did not invent replacement routes. Of the 1,880 Plan 1 primary samples, 1,804 had at least one cached correcting route that remained correct under current re-evaluation. The other 76 were excluded from route-conditioned analysis because no cached route remained current-correct.

### Main findings

- Of 17,262 anchor-route OFF positions, 7,880 (45.65%, 95% bootstrap CI 44.31–46.87%) were necessary for the fixed route and 9,382 (54.35%) were redundant in that context.
- Among necessary positions, the mechanism taxonomy was:
  - READ suppression sufficient: 1,619 (20.55%, CI 19.4–21.7%)
  - WRITE suppression sufficient: 3,379 (42.88%, CI 41.5–44.4%)
  - Either suppression sufficient: 783 (9.94%, CI 9.2–10.7%)
  - Both suppressions required: 2,099 (26.64%, CI 25.3–28.1%)
- Necessary READ positions occurred later than necessary WRITE positions: mean layer 17.79 versus 9.97, a mean difference of 7.82 layers (CI 7.31–8.31).
- Redundancy was greatest for route sizes 5–8 (70.8%) and declined for larger routes: 47.5% necessary for sizes 9–12, 62.8% for 13–16, and 71.0% for routes over 16. The 2–4 bin contained only 12 samples and should be interpreted cautiously.
- Full-context discrete local rescue agreed with route-conditioned necessity only 51.8% of the time. It recalled just 7.3% of route-necessary positions at 36.3% precision; 7,305/7,880 necessary route positions were missed by the prior discrete full-context rescue test.
- Continuous cross-context agreement was real but incomplete: pooled Spearman correlations were 0.3573 for READ and 0.1771 for WRITE; median within-sample correlations were 0.4424 and 0.2473, respectively.
- Route-conditioned continuous effects were predominantly negative: mean `Delta_READ_W0=-0.1620`, `Delta_WRITE_R0=-0.3174`, `Delta_READ_W1=-0.1428`, and `Delta_WRITE_R1=-0.2982`.
- 5,781/7,880 necessary positions (73.4%) allowed at least one partial component restoration while preserving route correctness; 2,099 required suppressing both. This is conditional evidence about successful route contexts, not a claim of globally harmful layers.
- Most samples exhibited mixed mechanisms (1,203). The remaining structural categories were no essential positions (195), one dominant position (161), multiple WRITE positions (111), either/ambiguous (69), multiple READ positions (46), and joint-both (19).

### Validation and execution

- The final integrity audit covered 1,804 samples, 17,262 cells, and 69,048 action rows. It found zero factorial-formula errors and passed action-semantics, fixed-target, `M00`, and taxonomy checks.
- Job 1576 failed before scientific evaluation because the submitted environment could not load CUBLAS. The corrected launch, job 1578, completed anchor construction successfully.
- Pilot jobs 1579 and 1580 completed successfully. Two replicas per GPU achieved 12.1839 cells/s, 1.414× the one-replica configuration, and were selected for the full sweep.
- Full sweep job 1581 completed successfully in 29 minutes 13 seconds.
- The completed implementation/test state passed 90 tests, and all 54 generated shard sidecars passed validation.

### Plan 2 reports and evidence

Start with:

- [Final route-conditioned report](4action_route_conditioned/route_conditioned_decomposition_report.md)
- [Implementation audit](4action_route_conditioned/implementation_audit.md)
- [Pilot report](4action_route_conditioned/pilot_report.md)
- [Experiment log](4action_route_conditioned/experiment_log.md)

Primary data and aggregate tables:

- [Raw route-conditioned cells](4action_route_conditioned/route_conditioned_cells.parquet)
- [Anchor-route manifest](4action_route_conditioned/anchor_route_manifest.parquet)
- [Aggregate summary](4action_route_conditioned/aggregate_summary.json)
- [Anchor-route summary](4action_route_conditioned/anchor_route_summary.json)
- [Pilot benchmark summary](4action_route_conditioned/pilot_benchmark_summary.json)
- [Compute estimate](4action_route_conditioned/compute_estimate.json)
- [Final integrity audit](4action_route_conditioned/final_integrity_audit.json)
- [Interpretation decision](4action_route_conditioned/interpretation_decision.json)
- [Necessity taxonomy](4action_route_conditioned/aggregate/necessity_taxonomy.parquet)
- [Continuous effects](4action_route_conditioned/aggregate/continuous_effects.parquet)
- [Depth effects](4action_route_conditioned/aggregate/depth_effects.parquet)
- [Depth taxonomy](4action_route_conditioned/aggregate/depth_taxonomy.parquet)
- [Category/depth comparison](4action_route_conditioned/aggregate/category_depth_comparison.parquet)
- [Route-size stratification](4action_route_conditioned/aggregate/route_size_stratification.parquet)
- [Per-sample mechanism structure](4action_route_conditioned/aggregate/sample_structure.parquet)
- [Full-context comparison](4action_route_conditioned/aggregate/full_context_comparison.parquet)
- [Within-sample context correlations](4action_route_conditioned/aggregate/within_sample_context_correlations.parquet)

Population figures:

- [Necessity taxonomy](4action_route_conditioned/figures/necessity_taxonomy.png)
- [Taxonomy by layer](4action_route_conditioned/figures/taxonomy_by_layer.png)
- [Continuous effects by layer](4action_route_conditioned/figures/continuous_effects_by_layer.png)
- [Redundancy by route size](4action_route_conditioned/figures/redundancy_by_route_size.png)
- [Sample mechanism structure](4action_route_conditioned/figures/sample_mechanism_structure.png)
- [Full-context versus route-conditioned effects](4action_route_conditioned/figures/full_vs_route_effects.png)

The [bounded joint-refinement proposal](4action_route_conditioned/proposed_joint_refinement_plan.md) is a documented possible next study, not a completed result and not an authorized or running experiment.

## Combined interpretation

The two completed studies support four conclusions.

1. Visual READ and WRITE operations can causally erode the correct-answer signal in this selected A+ cohort. Plan 1 establishes this in a controlled single-layer, otherwise-FULL context and links local harmfulness to the layers suppressed by successful binary routes.
2. Correcting routes are strongly context-dependent and often redundant. Plan 2 shows that fewer than half of anchor-route OFF decisions are necessary once the rest of the route is fixed, while necessity rises for the largest routes.
3. Full-context local rescue is not an adequate proxy for route-conditioned necessity. The 7.3% recall demonstrates that many operations become causally decisive only after other route changes have altered the computation. This explains why the full-context landscape can enrich for route choices without reconstructing the route itself.
4. Mechanisms are component- and depth-dependent. WRITE suppression is the most common sufficient route-conditioned mechanism and tends to occur earlier; READ suppression tends to occur later; roughly one quarter of necessary positions require suppressing both components.

The evidence does not establish that a globally fixed set of layers should always be disabled, that every OFF position chosen by search is necessary, or that the strongest full-context local effect directly marks the later answer-collapse layer. It instead supports a conditional picture: multi-layer route context changes which READ/WRITE operations are necessary, and those route-conditioned mechanisms can differ substantially from isolated interventions around native-like FULL computation.

## Reproducibility boundary

- Plan 1 causal effects use only unified-executor margins. Native/unified numerical drift is a separate implementation diagnostic.
- Plan 2 necessity and mechanism labels are conditional on the selected current-correct cached anchor route and fixed target answers.
- Bootstrap intervals are over samples, using image grouping where the underlying analysis made it practical.
- Raw tables, aggregate summaries, figures, validation reports, and execution logs are linked above. These artifacts are the authoritative evidence; scheduler job IDs are historical provenance rather than live state.
