# Phase 30: Cross-Dataset Visual-Access Memory

## Current Objective

Use the authoritative unrestricted 28-bit binary MCTS caches to test whether
GQA, TextVQA, ChartQA, and WeMath2.0-Pro differ in direct visual dependence,
minimum positive visual-access amount, and placement schedule under matched
search budgets.

## Active Constraints

- Read-only analysis: no new MCTS, model inference, training, or route labels.
- Use raw authoritative caches, never max-50 supervision views.
- Primary search prefixes include anchors plus the first 200 simulations for
  FULL-correct records and anchors plus the first 400 simulations for
  FULL-wrong records; all-available search is sensitivity only.
- Preserve V0/V+/A0/A+/D taxonomy and analyze V+ and A+ separately.
- Sample-balance route summaries and interpret discovered schedules as
  search-conditioned evidence, not causal layer necessity.

## Current State

- Done: phases 27--29 validated all WeMath records, anchors, route semantics,
  and placement metrics.
- Done: all 12,544 authoritative records and 4,301,483 evaluated routes passed
  checksum, semantics, anchor, threshold, and MCTS linkage validation.
- Done: matched-prefix and all-available sensitivity analyses, controls,
  figures, manifest, and report are complete.
- In progress: none.
- Blocked: none.
- Most recent useful observation: task family strongly separates V+ minimum ON
  (8.66 to 13.86), while placement profiles remain highly shape-similar.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Legacy task cache contains 8,000 contract-valid records | `reports/label_regeneration_p4_cache_audit.md` | Defines GQA/TextVQA/ChartQA source population | confirmed |
| WeMath cache contains 4,544 valid hard-cap-400 records | `outputs/wemath2pro_mcts_label_analysis_v1/analysis_manifest.json` | Defines the fourth source population | confirmed |
| Both contracts use the same executor, 28 layers, model revision, BF16/SDPA, and unrestricted full masks | frozen execution contracts | Supports semantic comparability despite dataset-specific scoring and generation lengths | confirmed |
| Legacy wrong searches may extend to 600, WeMath wrong searches stop at 400 | frozen execution contracts | Requires matched-prefix primary analysis | confirmed |
| V+ minimum ON differs substantially across datasets after matching | `vplus_min_on_by_dataset.csv`, `visual_token_control.csv` | Supports a task-family amount association | confirmed |
| Placement profiles have cosine 0.982--0.996 at min and 0.994--0.999 at min+4 | `profile_distance_matrix.csv` | Rejects a strong task-specific depth-schedule interpretation | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Invoke `pytest` through the stale console script | It launched system Python | supported environment-entrypoint issue | phase 29 test log | Use `.venv/bin/python -m pytest` | Do not use `.venv/bin/pytest` |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Matched-budget raw-cache analysis | Directly answers the approved plan without new model work | Dataset differences in dependence, amount, and placement | medium | completed |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: compare task families without confounding
  their differing wrong-sample search ceilings.
- Relevant memory item used: phase 29 established sample-balanced placement
  summaries and showed finite-search route-set sensitivity must be explicit.
- Confirmed observation: 200/400 simulations are guaranteed across all four
  datasets for current FULL-correct/FULL-wrong records.
- Unverified interpretation: residual dataset differences may reflect task
  format, image-token geometry, or search discovery rather than task semantics.
- Diagnosis: supported that task family is associated with dependence and
  amount, but not strongly with placement shape in this frozen population.
- Evidence path if diagnosis is not unknown:
  `reports/cross_dataset_visual_access_v1.md`.
- Viable alternatives considered: unmatched all-available caches were rejected
  as primary because only legacy wrong samples can receive 600 simulations.
- Chosen action: execute the complete matched-prefix and all-available
  sensitivity analysis specified in `plans/motivation_check4.md`.
- Strongest objection: observational task families differ in scoring, prompt
  format, answer format, and visual-token geometry, so adjusted associations
  remain descriptive rather than causal task effects.
- How this differs from failed attempts: it aligns search opportunities first,
  keeps V+ and A+ separate, and controls amount and token geometry explicitly.
- Automatic execution authorized: yes
- Authorization basis: explicit request to read and perform
  `plans/motivation_check4.md`.
- Stop condition: incompatible route semantics, missing anchors/trace linkage,
  invalid mask/threshold data, or any need for new model/search execution.

## Latest Research-Action Result

- Action taken: validated and reanalyzed all authoritative GQA, TextVQA,
  ChartQA, and WeMath raw routes under matched 200/400 simulation prefixes,
  with all-search sensitivity, V+/A+ separation, amount and token controls,
  route-set sensitivity, profiles, distances, and clustered uncertainty.
- Result: Outcome C. V+ minimum ON rises 8.66 -> 10.74 -> 12.47 -> 13.86
  across GQA/TextVQA/ChartQA/WeMath. V+ dependence is 57.3%, 93.3%, 82.3%,
  and 50.9% in the selected populations. Placement shifts are small: centroid
  range 0.473--0.492 and profile cosine at least 0.982.
- Evidence saved: `outputs/cross_dataset_visual_access_v1/`,
  `reports/cross_dataset_visual_access_v1.md`, and
  `runs/cross_dataset_visual_access_v1_r2.log`.
- Failure or issue: the first Slurm invocation stopped before data loading
  because file-path execution lacked the project package root. Module-mode
  invocation repaired it without changing code or analysis rules. Cross-
  dataset sampling/scoring differences remain an observational limitation.
- Lesson learned: broad task regime is a stronger descriptor of whether/how
  much direct vision is discovered than of where access is placed.
- Next implication: stop. No additional experiment is authorized by this
  action, and the result does not justify a task-conditioned depth schedule.
