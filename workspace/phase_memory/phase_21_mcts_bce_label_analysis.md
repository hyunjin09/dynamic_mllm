# Phase 21: MCTS Label Geometry and Duplicated-BCE Oracle

## Current Objective

Execute the label-only analysis authorized by `plans/mcts_bce_analysis.md` to
separate raw MCTS geometry, max-50 selection effects, duplicated-BCE target
geometry, and predictor fitting/generalization effects.

## Active Constraints

- Use only the frozen 8,000-record GQA/TextVQA/ChartQA cache, frozen 7,000/1,000
  image-group split, and frozen max-50 selected supervision.
- Do not train, regenerate labels, run new Qwen inference, change the route
  selector, or design a new predictor/loss/search method.
- Match the actual full10 BCE weighting: per-sample normalized route weights,
  with ALL-ON raw weight 0.3 only when a cheaper selected valid route exists.
- Match deployed decoding: `sigmoid(logit) >= 0.5`, so exact ties resolve ON.
- Run the CPU-heavy aggregation through Slurm on node05; never use node04.

## Current State

- Done: source plan, cache schemas, selected-route schema, BCE weighting, and
  decode tie rule inspected.
- Done: all 8,000 frozen records checksum/semantics verified; raw/selected
  geometry, exact weighted/unweighted BCE oracles, clustering, Pareto,
  diversity, balance, and cross-sample summaries completed.
- Done: required report, machine-readable tables, per-sample records, figures,
  and checksum manifest completed.
- In progress: none.
- Blocked: none.
- Most recent useful observation: the exact weighted BCE label oracle is
  selected-valid for only 5.93% of positive inputs; Pareto filtering raises
  diagnostic Hit@1 to 73.41% while reducing mean ON from 17.21 to 9.78.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Raw cache contains all evaluated masks and authoritative score/validity fields | `outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl` and indexed raw records | Supports raw geometry, taxonomy, deduplication, and Pareto audits | confirmed |
| Selected manifest stores the exact deterministic max-50 route set | `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl` | Defines the supervision actually seen in full10 training | confirmed |
| Full10 BCE weights ALL-ON at 0.3 only when cheaper valid masks coexist | `binary_policy/dataset.py`, `configs/binary_polar_full10_polar_bce_v1.yaml` | Defines the exact weighted BCE label oracle | confirmed |
| Deployed threshold decoder resolves exact 0.5 ties ON | `binary_policy/decode.py` | Defines deterministic oracle masks | confirmed |
| Raw/selected routes remain highly diverse | `outputs/binary_mcts_label_geometry_v1/raw_selected_summary.csv`, `cluster_summary.csv` | Rejects poor-search and selector-collapse explanations | confirmed |
| Exact weighted BCE oracle is invalid for 6,507/6,917 positives | `weighted_bce_oracle_summary.csv`, `invalid_hybrid_summary.csv` | Supports complete-mask objective/label mismatch | confirmed |
| 95.83% of selected route occurrences are Pareto-dominated | `pareto_summary.csv` | Supports dominated-supervision pressure as a major contributor | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Initial entropy pass | A marginal equaled `1.0000000000000002` | supported floating-point summation roundoff | `runs/mcts_bce_label_analysis_v2/slurm.log`; UID `gqa:gqa_gh_12749098` | Clamp only within 1e-12 and retain hard failure outside tolerance | Do not treat harmless roundoff as semantic invalidity |
| First final aggregation | Singleton valid sets have undefined pairwise Hamming and a mean consumed `None` | supported reporting edge case | `runs/mcts_bce_label_analysis_v3/slurm.log` | Preserve singleton samples and omit only undefined pairwise values | Do not impute a zero pairwise distance or exclude singleton inputs |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Raw MCTS geometry is poor | Search may return redundant/dense modes | Whether label generation is the bottleneck | medium | rejected |
| Max-50 selection degrades useful raw diversity | Selection may overrepresent dense/redundant routes | Whether supervision construction is the bottleneck | medium | rejected |
| Duplicated-BCE oracle forms invalid hybrids | Per-bit averaging can combine incompatible valid modes | Whether the loss geometry is the bottleneck | medium | supported |
| Dominated supervision amplifies hybridization | Most correct routes may be compute-dominated | Whether Pareto filtering is a necessary matched factor | low | supported |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: deterministically locate the first dominant
  source of the observed near-ALL-ON BCE predictor behavior.
- Relevant memory item used: P6 already established broad raw Hamming and
  transition diversity, but did not test the exact weighted BCE oracle.
- Confirmed observation: raw valid-mask diversity is high in aggregate and the
  trained BCE predictors largely decode ALL-ON.
- Unverified interpretation: max-50 selection, BCE bitwise averaging, dominated
  FULL labels, or predictor fitting may be the principal cause.
- Diagnosis: unknown
- Viable alternatives considered: analyze only selected labels; analyze only
  raw labels; execute the approved matched raw/selected/oracle counterfactuals.
- Chosen action: implement and run the complete approved label-only analysis,
  including raw/selected geometry, exact weighted and unweighted BCE oracles,
  clustering, Pareto and diversity counterfactuals, and fixed balance ratios.
- Strongest objection: some raw-cache comparisons are computationally larger
  than selected-label summaries; bounded streaming aggregation avoids changing
  the scientific scope.
- How this differs from failed attempts: it inspects the labels and induced
  optimum directly, without retraining or inferring from predictor outcomes.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform `plans/mcts_bce_analysis.md`.
- Stop condition: required artifacts/report are complete, or a frozen-data
  integrity/semantic mismatch prevents valid analysis.

## Latest Research-Action Result

- Action taken: completed the approved label-only MCTS/BCE geometry analysis.
- Result: **Outcome C + Outcome E**. Raw/search and max-50 diversity are sound;
  the exact duplicated-BCE oracle is an invalid hybrid for 94.07% of positive
  inputs, and 95.83% of selected route occurrences are Pareto-dominated.
- Evidence saved: `reports/binary_mcts_label_geometry_and_bce_oracle_report.md` and
  `outputs/binary_mcts_label_geometry_v1/`.
- Failure or issue: two deterministic engineering edge cases were repaired
  without changing data or scientific definitions; frozen histories lack
  UID-level BCE predictions, so actual-vs-oracle comparison is aggregate only.
- Lesson learned: route count/diversity is not the current bottleneck. Complete
  mask coherence and dominated-label pressure must be separated before any
  further predictor claim.
- Next implication: stop. A matched Pareto-efficient BCE versus coherent
  complete-route objective comparison requires separate user approval.
