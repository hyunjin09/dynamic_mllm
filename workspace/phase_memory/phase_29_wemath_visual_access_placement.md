# Phase 29: WeMath Visual-Access Placement Memory

## Current Objective

Use the frozen hard-cap-400 WeMath2.0-Pro route cache to test whether the
placement and re-entry schedule of direct visual access across 28 decoder
layers changes with difficulty among V+ samples, while holding route budget
approximately fixed.

## Active Constraints

- Read-only analysis of the same 4,544 eligible raw records and source metadata
  used by phases 27 and 28; no new MCTS, inference, training, or routes.
- Primary cohort is V+ only: FULL correct and exact ALL-OFF wrong.
- Use all discovered minimum-budget routes per sample, sample-balanced, with
  prospectively frozen +2 and +4 near-minimum sensitivities.
- Preserve all eight strata and prioritize same-family/same-image paired
  evidence over unpaired degree trends.
- Interpret route locations as discovered direct-visual-access schedules, not
  causal layer necessity or semantic re-grounding.

## Current State

- Done: phases 27 and 28 verified the frozen population, cache hashes, anchors,
  and V0/V+/A0/A+ definitions.
- Done: the full minimum/near-min placement, re-entry, amount-adjusted,
  family-paired, same-image, axis, metadata, and A+ analyses passed.
- In progress: none.
- Blocked: none.
- Most recent useful observation: aggregate V+ family-paired normalized
  centroid delta is 0.0053 with 95% CI [-0.0091, 0.0190]; same-image delta is
  0.0041 with CI [-0.0138, 0.0214]. Every amount-adjusted degree and axis
  aggregate crosses zero across exact-min, min+2, and min+4.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| V+ contains 428 FULL-correct, ALL-OFF-wrong samples | `outputs/wemath2pro_visual_dependence_reanalysis_v1/analysis_summary.json` | Defines the primary placement cohort | confirmed |
| Positive minimum-ON amount has no stable global degree trend within V+ | `reports/wemath2pro_visual_dependence_reanalysis_v1.md` | Motivates the orthogonal placement question and amount adjustment | confirmed |
| `x/y/z` mean contextual/visual/step complexity | authors' `dynamic_scheduling/verl/utils/dataset.py`, lines 303--311 | Authorizes semantic axis labels rather than inferred names | confirmed |
| Frozen records carry complete knowledge-point annotations | frozen 4,544-record manifest | Supports an authoritative knowledge-count covariate, but not manufactured reasoning categories | confirmed |
| Visual-access schedules vary materially across V+ samples | `vplus_placement_metrics_by_sample.csv` | Centroid spans 0.210--0.794 and ON segments span 1--11 | confirmed |
| Difficulty does not explain that schedule variation | `vplus_family_paired_transitions.csv`, `vplus_same_image_analysis.csv`, `axis_placement_summary.csv` | Global, axis, amount-adjusted, and same-image intervals cross zero under all route-set definitions | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Infer axis meaning from local examples or label order | Local dataset card names three dimensions but does not map them to letters | supported documentation gap | local HF dataset card | Use the released scheduler's explicit mapping | Do not infer semantics from examples |
| Invoke `pytest` through the stale console script | It launched system Python and could not import `experiments` | supported environment-entrypoint issue | local test log/output | Use `.venv/bin/python -m pytest` | Do not use the stale `.venv/bin/pytest` launcher |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Sample-balanced minimum/near-min placement analysis | Directly implements the approved orthogonal question without new model execution | Whether access schedules shift with aggregate difficulty or specific axes | medium | completed |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: determine whether the finite cache contains a
  stable difficulty-linked depth schedule after controlling for minimum ON.
- Relevant memory item used: phase 28 showed amount and direct-vision
  dependence must be separated before interpretation.
- Confirmed observation: the cache, cohorts, family IDs, image IDs, and axis
  semantics are available and checksum-bound.
- Unverified interpretation: other input properties, rather than the official
  difficulty construction, may explain schedule heterogeneity.
- Diagnosis: supported that official difficulty/axes do not explain schedule
  placement in this frozen finite cache.
- Evidence path if diagnosis is not unknown:
  `reports/wemath2pro_visual_access_placement_v1.md`.
- Viable alternatives considered: none; the user specified one bounded,
  complete read-only analysis.
- Chosen action: implement and run all sample-balanced route-set, placement,
  re-entry, amount-adjusted, paired-family, same-image, axis, A+, and sensitivity
  analyses in `plans/motivation_check3.md`.
- Strongest objection: minimum/near-min route sets are finite-search artifacts,
  so apparent layer preferences may be unstable under +2/+4 expansion.
- How this differs from failed attempts: it uses authoritative axis semantics,
  all raw valid routes, and does not collapse multiple minimum routes to one.
- Automatic execution authorized: yes.
- Authorization basis: explicit request to read and perform
  `plans/motivation_check3.md`.
- Stop condition: any cache/hash/anchor/cohort mismatch, invalid raw route,
  missing required metadata identity, or need for new model/search execution.

## Latest Research-Action Result

- Action taken: validated all 4,544 raw records and analyzed every V+ minimum,
  min+2, and min+4 valid-route set sample-balanced, plus paired, same-image,
  amount-adjusted, axis, metadata, and A+ secondary analyses.
- Result: Outcome D — route placement varies, but not with difficulty. Exact-min
  normalized centroid spans 0.210--0.794 and segment count 1--11, but aggregate
  family-paired centroid, late fraction, latest ON, segments, and late re-entry
  all cross zero. Same-image and every axis aggregate also cross zero across
  all three route-set definitions.
- Evidence saved: `outputs/wemath2pro_visual_access_placement_v1/`,
  `reports/wemath2pro_visual_access_placement_v1.md`, and
  `runs/wemath2pro_visual_access_placement_v1.log`.
- Failure or issue: exact layer profiles broaden under min+2/min+4 (median
  exact-to-min+4 L1 7.65), but aggregate placement summaries and the null
  difficulty conclusion remain stable. One stale pytest launcher issue was
  repaired by using the project interpreter without changing tests.
- Lesson learned: WeMath aggregate difficulty, contextual complexity, visual
  complexity, and step complexity do not provide a stable conditioning signal
  for discovered direct-visual-access schedules.
- Next implication: stop. The completed result does not motivate a
  difficulty-conditioned schedule router or another unapproved experiment.
