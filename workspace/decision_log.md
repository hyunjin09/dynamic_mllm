# Decision Log

Record only decisions, pivots, and lessons that should affect later phases.
Do not copy raw logs or provisional explanations here.

## 2026-08-29 — Require internal routed benefit before external router evaluation

- Decision or promoted lesson: Do not spend the restricted external-evaluation
  budget on a router checkpoint family that repeatedly provides no internal
  W2C rescue and deploys an effectively all-FULL route. Preserve the validated
  negative result and stop before external outcomes are opened.
- Triggering evidence: The online four-action router completed nine atomic
  866-record routed validations. Every epoch had zero W2C rescues; epochs 2--8
  selected FULL for all 24,248 layer decisions, and epoch 9 selected 24,247
  FULL plus one IGNORE. C2C preservation was 1.0 from epoch 2 onward. This
  occurred after a real semantic smoke passed and training loss improved, so it
  is not explained by the earlier smoke-scheduler defect.
- Evidence paths: `outputs/four_action_online_router/training_v3/history.json`,
  `reports/four_action_online_router_early_stop_20260829.md`, and
  `workspace/phase_memory/phase_37_online_four_action_router.md`.
- Confidence: high for stopping this checkpoint family; unknown for why the
  learned policy collapsed.
- Applies when: Deciding whether to open ChartQA/MMMU-Pro/POPE external
  outcomes for this online-router run or a later run with the same repeated
  internal all-FULL/zero-rescue behavior.
- Does not apply when: Claiming every online router or every alternative
  supervision/architecture must collapse, or diagnosing the cause without a
  separate controlled action.
- Consequence for future actions: Internal routed execution must show a
  decision-relevant non-FULL policy and W2C benefit before external evaluation.
  Lower training loss or higher node Valid-Action@1 alone is insufficient.
- Revisit condition: A prospectively specified router produces reproducible
  internal rescue/preservation evidence under the same executor and frozen
  validation contract.

## 2026-08-28 — Preserve complete cross-server handoff evidence

- Decision or promoted lesson: This project is operated concurrently from
  multiple servers. Every bounded research or implementation action must leave
  enough tracked evidence for an agent on another server to continue without
  relying on conversation history, local scheduler state, or inferred assets.
- Triggering evidence: Explicit user operating instruction on 2026-08-28.
- Evidence paths: `workspace/workflow_state.md` for the global dashboard,
  `workspace/phase_memory/` for active phase decisions, this decision log for
  promoted lessons, and phase-specific experiment logs/reports for execution
  evidence.
- Confidence: high; this is a user-defined operating constraint.
- Applies when: Implementing, launching, monitoring, pausing, resuming,
  interpreting, or handing off any Dynamic MLLM work.
- Does not apply when: Treating server-local topology, live job IDs, symlink
  targets, environments, or ignored payloads as portable facts. Those must be
  verified independently on each server.
- Consequence for future actions:
  1. Whenever the user asks to follow up work from another server, first read
     `workspace/workflow_state.md`, `workspace/decision_log.md`, the relevant
     active file under `workspace/phase_memory/`, and the newest matching
     phase-specific handoff or report. This four-file handoff read is required,
     not optional context discovery.
  2. Before work, fetch the shared branch and reconcile remote commits without
     force-pushing or discarding another server's changes.
  3. Commit and push portable code, configs, plans, tests, compact reports,
     checksums, and updated workflow/phase state at bounded handoff points.
  4. Record exact Git commit, config and source hashes, commands, cohort/count
     contracts, output locations, completion boundaries, failures, and the
     scientific implication of the latest result.
  5. Treat datasets, labels, checkpoints, raw outputs, and generated analysis
     as separately transferred assets. Record their real paths, sizes/counts,
     and checksums; never infer their presence from Git reports.
  6. Keep `ACCESS_POLICY.md`, `infra/`, scheduler state, and
     `workspace/env_state.md` machine-local. Record server differences and
     revalidate live GPU/scheduler state instead of reusing historical jobs.
- Revisit condition: The user replaces the concurrent multi-server workflow
  or defines a different artifact synchronization mechanism.

## 2026-08-27 — Restrict prospective 4-action POLAR external evaluation to three benchmark families

- Decision or promoted lesson: For future 4-action POLAR evaluation, run only
  ChartQA, MMMU-Pro Standard and Vision, and all three POPE splits. Do not run
  TextVQA, DocVQA, MMStar, or base MMMU unless the user later expands the
  scope explicitly.
- Triggering evidence: Explicit user scope decision on 2026-08-27.
- Evidence paths: `eval/reference/shared_prefix_eval_20260812/` retains the
  historical seven-benchmark-family protocol; this entry defines the narrower
  prospective 4-action POLAR evaluation scope.
- Confidence: high.
- Applies when: Building, launching, estimating, or reporting external
  evaluation for the new 4-action POLAR predictor.
- Does not apply when: Describing or reproducing the historical shared-prefix
  evaluation bundle, whose original benchmark coverage remains unchanged.
- Consequence for future actions: Materialize and evaluate ChartQA (2,500),
  MMMU-Pro Standard (1,730), MMMU-Pro Vision (1,730), and POPE adversarial,
  popular, and random (3,000 each), for 14,960 total rows. Report Core VQA,
  multiple-choice, and POPE metrics separately rather than pooling them.
- Revisit condition: The user explicitly approves adding or replacing an
  evaluation benchmark.

## 2026-08-22 — Task family predicts visual-access amount more than placement

- Decision or promoted lesson: Preserve task-family differences in direct
  visual dependence and positive visual-access amount, but do not infer a
  strongly task-specific depth schedule from the current MCTS caches.
- Triggering evidence: Under matched 200-simulation FULL-correct prefixes, V+
  minimum ON means are 8.66 (GQA), 10.74 (TextVQA), 12.47 (ChartQA), and 13.86
  (WeMath2.0-Pro), and visual-token-adjusted dataset coefficients remain large.
  Exact-min normalized centroids differ by at most 0.019, however; pairwise
  profile cosine similarities are 0.982--0.996 and rise to 0.994--0.999 at
  min+4. The result is Outcome C.
- Evidence paths: `reports/cross_dataset_visual_access_v1.md` and
  `outputs/cross_dataset_visual_access_v1/`.
- Confidence: high for the matched-prefix frozen-cache description; low for a
  causal task effect because source sampling, prompts, scorers, answer formats,
  and input geometry differ.
- Applies when: Motivating visual-access amount controls or interpreting these
  four raw route caches.
- Does not apply when: Treating dataset identity as scalar difficulty, claiming
  a layer is necessary, or using these selected populations for natural
  prevalence claims.
- Consequence for future actions: If task family is used as context, separate
  V0/V+ and amount from placement. Do not justify a task-conditioned depth
  schedule from the small aggregate placement shifts.
- Revisit condition: prospective image/query-matched evidence shows a material
  and reproducible task-specific profile shape after equal search and amount.

## 2026-08-22 — Do not condition visual-access schedules on WeMath difficulty

- Decision or promoted lesson: The official WeMath2.0-Pro difficulty degree
  and contextual (`x`), visual (`y`), and step (`z`) axes are not supported as
  stable predictors of where direct visual access appears across decoder depth.
- Triggering evidence: Among 428 V+ samples, exact-minimum schedules are
  heterogeneous, but the aggregate family-paired normalized-centroid delta is
  0.0053 (95% CI [-0.0091, 0.0190]) and the same-image delta is 0.0041 (CI
  [-0.0138, 0.0214]). Latest access, late fraction, segment count, late
  re-entry, amount-adjusted degree, and every axis aggregate cross zero across
  exact-min, min+2, and min+4.
- Evidence paths: `reports/wemath2pro_visual_access_placement_v1.md` and
  `outputs/wemath2pro_visual_access_placement_v1/`.
- Confidence: high for the frozen-cache descriptive conclusion; low for causal
  layer necessity because MCTS does not exhaust the valid route space.
- Applies when: Proposing difficulty-conditioned layer-placement or direct-
  visual-access schedule predictors from this WeMath cache.
- Does not apply when: Claiming schedules are identical across inputs or that
  other question/image properties cannot predict them.
- Consequence for future actions: Do not motivate a schedule router from
  WeMath difficulty labels alone. Any future conditioning variable requires a
  separately approved hypothesis and actual executed-route validation.
- Revisit condition: independent prospective data show a robust paired
  schedule shift under an exhaustively or independently validated route set.

## 2026-08-22 — Separate direct visual dependence from positive visual-access budget

- Decision or promoted lesson: Treat ALL-OFF correctness as a distinct
  no-direct-visual-K/V regime. Estimate positive VISUAL_ON budgets only among
  samples where FULL is correct and ALL-OFF is wrong; do not mix the zero mass
  with positive visual routes.
- Triggering evidence: Of 841 FULL-correct WeMath2.0-Pro records, 413 are V0.
  V0 prevalence rises from 32.5% at degree 0 to 73.4% at degree 3 and explains
  83.7–94.9% of the degree-level mean decline. The V+-only rho is -0.057 with
  family-clustered 95% CI [-0.154, 0.037], and the paired V+ aggregate is null.
- Evidence paths: `reports/wemath2pro_visual_dependence_reanalysis_v1.md` and
  `outputs/wemath2pro_visual_dependence_reanalysis_v1/`.
- Confidence: high for the frozen-cache decomposition; low for causal
  necessity outside the finite binary search space.
- Applies when: Interpreting minimum-ON routes, route sparsity, or difficulty
  relationships in binary visual-routing caches.
- Does not apply when: Claiming ALL-OFF removes every structural image side
  channel, or that a V+ minimum is an identified physical requirement.
- Consequence for future actions: Always report V0 prevalence separately and
  condition positive visual-budget analyses on ALL-OFF failure. The previous
  x pattern should be described primarily as visual-dependence composition.
- Revisit condition: A prospectively independent cache or expanded action
  space shows a robust conditional budget relationship among V+ samples.

## 2026-08-22 — Do not use WeMath difficulty degree as a monotonic visual-depth proxy

- Decision or promoted lesson: Preserve the eight WeMath2.0-Pro difficulty
  strata in visual-compute analyses; coarse degree conflates distinct axes and
  must not be interpreted as a monotonic requirement for visual decoder depth.
- Triggering evidence: Among 841 FULL-correct records, minimum discovered ON
  falls from 9.74 at degree 0 to 3.66 at degree 3 (Spearman -0.225, family-
  clustered 95% CI [-0.291, -0.159]). The change is concentrated in
  x-containing strata, while FULL-wrong correction discovery separately falls
  from 50.0% to 26.7% across degree 0 to 3.
- Evidence paths: `reports/wemath2pro_visual_compute_difficulty_v1.md` and
  `outputs/wemath2pro_visual_compute_difficulty_v1/`.
- Confidence: high for the frozen-cache descriptive result; low for any causal
  statement about intrinsic computation need because FULL-correct survivors
  and discovered routes are search-selected.
- Applies when: Interpreting or stratifying the completed WeMath2.0-Pro binary
  route cache.
- Does not apply when: Claiming difficult samples need less computation,
  treating zero-positive as requiring more than FULL, or assessing REPEAT.
- Consequence for future actions: Do not motivate a larger visual-depth or
  REPEAT experiment from aggregate difficulty degree alone. Any follow-up must
  state an axis-specific hypothesis and handle route-coverage selection.
- Revisit condition: Independent, prospectively powered execution evidence
  shows a different within-family relationship under an expanded valid action
  space.

## 2026-08-17 — Do not use multimodal valid-mask sets as unfiltered duplicated-BCE targets

- Decision or promoted lesson: When complete valid masks form separated modes,
  preserve them as grouped exact-set supervision or use a prospectively frozen
  structured alternative; do not assume duplicated per-route BCE defines a
  coherent complete-mask target.
- Triggering evidence: The result repeated across two independent label
  populations. The GQA/TextVQA/ChartQA weighted per-sample BCE oracle has 5.93%
  valid-set Hit@1 and the WeMath2.0-Pro oracle has 13.72%, despite raw mean
  pairwise Hamming distances of 13.36 and 13.26. Selected route occurrences
  are 95.83% and 94.93% Pareto-dominated, respectively.
- Evidence paths: `outputs/binary_mcts_label_geometry_v1/`,
  `reports/binary_mcts_label_geometry_and_bce_oracle_report.md`,
  `outputs/wemath2pro_mcts_label_analysis_v1/`, and
  `reports/wemath2pro_mcts_training_suitability.md`.
- Confidence: high for label-objective incoherence; unknown for held-out
  execution of a factorized exact-set predictor.
- Applies when: Selecting supervision for complete 28-bit binary routes from
  these MCTS caches.
- Does not apply when: Declaring every uncached predicted mask behaviorally
  invalid, or claiming exact-set NLL solves cross-layer factorization.
- Consequence for future actions: Do not launch another unfiltered duplicated-
  BCE training run as the default. Freeze grouped sets and evaluate actual
  executed masks if exact-set training is later approved.
- Revisit condition: A matched held-out executed-mask experiment shows that
  duplicated BCE reliably outperforms complete-route-coherent supervision.

## 2026-08-12 — Preserve non-contiguous full-mask diversity in regenerated supervision

- Decision or promoted lesson: Construct later capped valid-set supervision
  from the unrestricted full masks using explicit Hamming, ON-count, and
  transition diversity. Keep POLAR segmentation as a derived controlled
  baseline rather than assuming the regenerated labels are naturally
  contiguous.
- Triggering evidence: Across 528,047 valid masks, the sample-balanced mean
  transition count is 13.20 and within-sample pairwise Hamming distance is
  13.36/28. Only 1.02% have at most three transitions, while 50.79% have at
  least 14 transitions.
- Evidence paths:
  `outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json`,
  `reports/label_regeneration_p6_route_diversity.md`.
- Confidence: high for label geometry; unknown for predictor generalization.
- Applies when: Building P8 derived views and comparing the direct factorized
  binary head with a POLAR-style representation on this regenerated cache.
- Does not apply when: Claiming that high-transition masks are semantically
  necessary, that a structured predictor must fail, or that routes accelerate
  wall-clock inference.
- Consequence for future actions: Do not cap routes by count or compute alone;
  preserve distinct complete-mask modes under the frozen max-32 policy.
- Revisit condition: A held-out executed-mask comparison shows that the
  apparent full-mask diversity is not predictive or useful for generalization.

## Entry Template

### <date> — <short title>

- Decision or promoted lesson:
- Triggering evidence:
- Evidence paths:
- Confidence: high / medium / low
- Applies when:
- Does not apply when:
- Consequence for future actions:
- Revisit condition:

## 2026-08-04 — Preserve stock eager as the primary causal runtime

- Decision or promoted lesson: Use the unchanged Transformers stock-eager
  decoder for the causal FULL path; do not substitute SDPA or query-chunked
  eager without a new prospective equivalence validation.
- Triggering evidence: Both substitutes passed some local checks but exceeded
  the frozen suffix-logit RMS equivalence threshold.
- Evidence paths: `outputs/stage_a_sdpa_reference_probe_attempt_03_valid_sdpa_rejected/stage_a_summary.json`, `outputs/stage_a_chunked_stock_equivalence_boundary/chunked_eager_equivalence.json`.
- Confidence: high
- Applies when: Running this Qwen2.5-VL-7B revision and the approved READ/WRITE
  counterfactuals.
- Does not apply when: A different runtime is prospectively shown equivalent at
  hook, suffix logits, scores, and ordering under a declared numerical budget.
- Consequence for future actions: Later stages inherit the stock-eager runtime
  and the currently validated prompt-length domain (≤4,861 tokens).
- Revisit condition: Faithful higher-memory execution or a new substitute passes
  the frozen equivalence protocol.

## 2026-08-04 — Inherited pool buckets require pinned-revision revalidation

- Decision or promoted lesson: Treat `easy_hard_5k` correct/wrong buckets as
  Stage A sampling metadata only until regenerated or revalidated under the
  pinned checkpoint.
- Triggering evidence: Stored evaluator scores reproduce 23/23, but fresh pinned
  checkpoint bucket scores reproduce only 22/23 (`boy` → `boys` on one GQA item).
- Evidence paths: `outputs/stage_a/benchmark_scoring_reproduction.csv`, `workspace/dataset_inventory.md`.
- Confidence: high
- Applies when: Stratifying or interpreting effects by full-correct/full-wrong
  status in later stages.
- Does not apply when: Labels are generated and frozen under the exact pinned
  runtime and evaluator.
- Consequence for future actions: No discovery or prevalence analysis may use
  inherited bucket labels as confirmatory model outcomes.
- Revisit condition: A pinned-revision label manifest is produced and audited.

## 2026-08-04 — Expand Stage B discovery to 400 samples

- Decision or promoted lesson: Use 400 discovery samples: 100 each from
  GQA/TextVQA × inherited complete-correct/complete-wrong cells.
- Triggering evidence: Explicit user approval after reviewing the source plan's
  suggested 100–200 discovery size.
- Evidence paths: `workspace/stage_b_protocol.md`,
  `data_manifests/stage_b_discovery_candidates_400_audit.json`.
- Confidence: high
- Applies when: Stage B exploratory discovery only.
- Does not apply when: Planning held-out Stage C sample size or claiming
  confirmatory prevalence.
- Consequence for future actions: Compute and artifact planning may cover 400
  intervention samples, but discovery remains exploratory and cannot support the
  final prevalence claim.
- Revisit condition: User changes the discovery budget or the protocol becomes
  infeasible under the faithful stock-eager runtime.

## 2026-08-04 — Use reference-answer likelihood for exploratory Stage B

- Decision or promoted lesson: Do not construct distractors. For Stage B only,
  use answer-token sequence log-likelihood as the primary within-sample effect
  and per-token mean log-likelihood for aggregation and robustness.
- Triggering evidence: Explicit user amendment after confirming the selected
  GQA/TextVQA records are open-ended.
- Evidence paths: User task “Revise and Execute Stage B Using Reference-Answer
  Likelihood”; `workspace/stage_b_protocol.md`; `configs/stage_b.yaml`.
- Confidence: high
- Applies when: Exploratory Stage B layer/state discovery on the frozen 400 samples.
- Does not apply when: Claiming correct-over-alternative preference, harmful
  participation, prevalence, or choosing the Stage C primary endpoint.
- Consequence for future actions: Report only signed reference-answer evidence
  shifts; retain generation/correctness as secondary behavior; no synthetic
  distractors or router training.
- Revisit condition: Stage B evidence is complete and a held-out Stage C endpoint
  is proposed for user approval.

## 2026-08-04 — Treat accepted-answer normalization as part of the estimand

- Decision or promoted lesson: Use the official EvalAI/VQA normalization for
  TextVQA both when constructing weighted accepted-reference targets and when
  scoring generated answers; do not substitute a generic punctuation normalizer.
- Triggering evidence: A pre-sweep conformance review found that the initial
  simplified routine omitted number mapping, contraction handling, and official
  punctuation behavior. The corrected two-dataset validity run then passed.
- Evidence paths: `scoring/benchmark_metrics.py`,
  `outputs/stage_b_validity_v4/stage_b_validity_summary.json`.
- Confidence: high
- Applies when: TextVQA accepted answers or generated predictions are scored in
  Stage B or a later protocol inherited from Stage B.
- Does not apply when: A different dataset specifies a different official
  evaluator.
- Consequence for future actions: Evaluator normalization must be frozen before
  intervention outcomes because it changes the positive-reference mixture and
  therefore the measured likelihood effect.
- Revisit condition: The project adopts an independently versioned official
  evaluator whose behavior differs and the user approves the estimand change.

## 2026-08-05 — Freeze a narrow reference-likelihood Stage C endpoint

- Decision or promoted lesson: Confirm only TextVQA layer-0 conditional READ
  with WRITE enabled, using accepted-reference per-token `FULL - WRITE_ONLY`.
  Primary success is an image-clustered 95% bootstrap CI for the held-out mean
  entirely below zero. The `-0.05` threshold is secondary only.
- Triggering evidence: Explicit user approval with modifications after the
  open-ended TextVQA design made option/label controls inapplicable without
  prohibited distractors.
- Evidence paths: User's 2026-08-05 “Stage C Amendment Decision”;
  `workspace/stage_c_reference_likelihood_proposal.md`.
- Confidence: high
- Applies when: Preparing and executing Stage C on a new, outcome-blind,
  non-overlapping TextVQA manifest.
- Does not apply when: Searching other layers/datasets/operations, executing
  Stage D, or claiming a harmful mechanism or accuracy improvement.
- Consequence for future actions: Use covariance/subspace and norm-matched real
  residual controls, wrong-answer contrast, aggregation/prefix robustness, and
  image clustering; actual READ removal must beat both structured null families
  before “confirmed answer-misaligned READ effect” is permitted.
- Revisit condition: A required control is technically invalid or the user
  explicitly amends the endpoint.

## 2026-08-05 — Retain 800 unique-image records for Stage C

- Decision or promoted lesson: Target 800 eligible unique-image TextVQA records
  and prohibit outcome-dependent resizing.
- Triggering evidence: Stage B layer-0 `read_w1` mean -0.052953 nats/token and
  SD 0.407473; normal-approximation power at n=800 is 95.7% at the observed
  mean and 93.5% at magnitude 0.05.
- Evidence paths: `workspace/stage_c_power_analysis.md`;
  `outputs/stage_c_prerun/power_analysis_v1.json`.
- Confidence: medium
- Applies when: Building the Stage C manifest under the same primary endpoint.
- Does not apply when: Eligibility yields fewer records, image clustering has a
  material design effect, or the endpoint changes.
- Consequence for future actions: Do not reduce the target based on the
  discovery estimate and do not adapt it after held-out outcomes.
- Revisit condition: An outcome-blind eligibility audit shows 800 unique images
  are unavailable or a predeclared design-effect calculation changes the
  effective sample size.

## 2026-08-05 — Use official validation rather than correctness-selected remainder

- Decision or promoted lesson: Draw the frozen Stage C population from the
  pinned official TextVQA validation split, not the unused `easy_hard_5k`
  remainder selected by inherited model correctness.
- Triggering evidence: The official pool supplied 4,991 technically eligible
  records across 3,162 unique images and yielded 800 records with zero Stage B
  record/effective-image overlap. The local remainder remains
  correctness-selected discovery/calibration data.
- Evidence paths: `workspace/stage_c_eligibility_overlap_audit.md`;
  `outputs/stage_c/manifest/stage_c_eligibility_overlap_audit_v1.json`.
- Confidence: high
- Applies when: Interpreting the frozen Stage C population mean.
- Does not apply when: Using Stage B samples solely to fit outcome-blind
  residual geometry.
- Consequence for future actions: Never replace held-out records with the
  correctness-selected remainder after intervention outcomes are visible.
- Revisit condition: Only an explicit protocol amendment changes the primary
  Stage C population.

## 2026-08-05 — Validate covariance before row interpolation and freeze donor coverage

- Decision or promoted lesson: Test random-null affine-subspace membership on
  the native fixed grid, test row mapping/norm separately, and freeze real
  donors with a Stage-B-only composite geometry cap that guarantees eight
  eligible calibration donors before confirmation.
- Triggering evidence: The first smoke conflated interpolation deviation with
  subspace leakage and provisional independent calipers failed for two
  calibration targets. A single geometry diagnostic ruled out norm width alone;
  the corrected cap froze at 1.5 and the final smoke passed.
- Evidence paths:
  `outputs/stage_c/nulls/null_calibration_and_smoke_v1_first_failed.json`,
  `outputs/stage_c/nulls/null_calibration_and_smoke_v1.json`, and
  `workspace/stage_c_structured_null_spec.md`.
- Confidence: high
- Applies when: Generating or validating either frozen Stage C residual-null
  family.
- Does not apply when: Interpreting the held-out likelihood result itself.
- Consequence for future actions: A target outside the frozen 1.5 donor cap
  fails closed; do not adapt matching or mistake expected interpolation error
  for covariance-fit failure after outcomes are visible.
- Revisit condition: The user prospectively approves a different null family
  before held-out outcome inspection.

## 2026-08-05 — Freeze Outcome B and close the harmful layer-0 READ hypothesis

- Decision or promoted lesson: Record a held-out TextVQA layer-0
  reference-support replication, but do not call it a confirmed
  answer-misaligned READ effect. Close the harmful layer-0 READ hypothesis,
  cancel Stage D for this path, and do not search for a READ-specific harmful
  mechanism. The actual intervention was not distinguishable from either
  frozen structured null.
- Triggering evidence: The all-800 primary mean was `-0.07294332` nats/token
  with image-clustered 95% CI `[-0.14127645, -0.01710262]`, but covariance and
  real-residual paired CIs both crossed zero. The 798-target original-caliper
  sensitivity also failed, and contextual-prefix robustness crossed zero.
- Evidence paths: `outputs/stage_c/analysis_v1/primary_endpoint_summary.json`,
  `outputs/stage_c/analysis_v1/structured_null_comparison.json`,
  `outputs/stage_c/analysis_v1/real_residual_null_original_caliper_798_sensitivity.json`,
  `reports/stage_c_conclusion.md`.
- Confidence: high
- Applies when: Interpreting this frozen model/dataset/layer/endpoint and
  deciding whether the approved harmfulness-confirmation path advances.
- Does not apply when: Making claims about other layers, tasks, models, or a
  prospectively approved redesigned estimand.
- Consequence for future actions: Preserve Outcome B and stop; correction
  counts and the wrong-answer contrast remain secondary and cannot rescue the
  failed structured-null gate. The 22 corrections, 12 regressions, and net
  `+10/800` correct are descriptive rather than an accuracy claim. Any new
  direction requires explicit approval and cannot be framed as Stage D rescue.
- Revisit condition: Only a new prospectively approved protocol, not post-hoc
  analysis of this held-out set.

## 2026-08-06 — Supersede v2 with the policy-conditional v3 plan

- Decision or promoted lesson: Plan v3 is active. Represent local visual
  participation by the complete dense-suffix four-action vector
  `[Q(0,0), Q(1,0), Q(0,1), Q(1,1)]`; retain READ/WRITE effects only as derived
  conditional contrasts. Reuse the valid v2 Stage A implementation and Stage B
  four-cell results as discovery evidence, but do not transfer the v2 Stage C
  confirmatory label or inspected population to v3.
- Triggering evidence: The v3 migration audit verified 400 complete Stage B
  records, 3,200 sample-layer pairs, and 12,800 finite action cells under the
  identical dense prefix/suffix protocol. v2 Stage C contains only the frozen
  layer-0 `FULL`/`WRITE_ONLY` endpoint and failed both structured-null
  superiority gates.
- Evidence paths: `reports/v3_migration_audit.md`,
  `outputs/v3_migration/v2_artifact_audit_v1.json`,
  `reports/stage_c_frozen_outcome_b_closure.md`.
- Confidence: high.
- Applies when: Preparing any new analysis or confirmation under plan v3.
- Does not apply when: Interpreting v2 within its frozen protocol; those reports
  and checksums remain unchanged.
- Consequence for future actions: Do not rerun the existing 400-cell discovery
  sweep merely to rename its states. First perform deterministic v3 reanalysis,
  then close the missing query-invariance and search-adjusted-null preflight
  gates before freezing a new, nonoverlapping held-out manifest. Do not resume
  old Stage D or train a probe/router.
- Revisit condition: Static audit evidence is shown incorrect, dense-suffix
  semantics fail a prospective preflight, or the user approves a different
  scientific plan.

## 2026-08-06 — Advance only to a search-matched v3 preflight

- Decision or promoted lesson: The inspected Stage B four-action landscape is
  heterogeneous enough to justify a bounded preflight, but not confirmation.
  The next causal gate is a structured null with exactly the same action/layer
  search budget as the real statistic.
- Triggering evidence: The per-token sample-layer oracle gain over FULL is
  `0.0976`, while fixed per-layer and dataset/layer schedules gain only `0.0045`
  and `0.0075`; independent main effects miss the exact best cell in about 23%
  of pairs. Medians are small, heavy tails are material, and v2 Outcome B shows
  that a replicating narrow contrast can remain nonspecific.
- Evidence paths: `outputs/v3_discovery/analysis_manifest.json`,
  `outputs/v3_discovery/fixed_policy_regret_v1.csv`,
  `outputs/v3_discovery/interaction_summary_v1.csv`,
  `reports/v3_stage_b_reanalysis.md`, `reports/v3_stage_b_decision.md`.
- Confidence: medium.
- Applies when: Deciding whether and how to prepare a new held-out v3
  confirmation under the active plan.
- Does not apply when: Claiming harmful participation, policy benefit,
  acceleration, prevalence, or independent replication from inspected Stage B.
- Consequence for future actions: Before any held-out manifest is frozen or
  opened, validate the search-matched null pipeline, numerical same-image
  invariance, exact inspected-image exclusion, and missing residual/state
  diagnostics on discovery/calibration data only.
- Revisit condition: The bounded preflight shows the real statistic is
  indistinguishable from matched nulls, mostly under the practical tie band, or
  cannot support a prospectively fixed endpoint.

## 2026-08-07 — Stop v3 confirmation after independent null redesign fails

- Decision or promoted lesson: Stop the v3 causal-confirmation direction. Do
  not freeze or run the 1,600-record held-out manifest because neither a joint
  covariance null nor a completely well-matched paired real-residual null
  survives outcome-blind calibration under unchanged validity thresholds.
- Triggering evidence: On 2,000 then 4,000 independent train records, the
  empirical eight-donor global caliper was `2.625` then `3.09375`; rare shape
  tails persisted. Fixed 32-row and exact-native-shape representations failed,
  and the native-row model still exceeded 0.50 cross-validated error in
  multiple strata after every stratum reached the 85% target at rank cap 1,024.
- Evidence paths: `reports/v3_null_redesign_v2.md`,
  `outputs/v3_null_redesign/donor_coverage_v2.json`, and
  `outputs/v3_null_redesign/covariance_representation_c_rank_extension.json`.
- Confidence: high.
- Applies when: Deciding whether the current maximum-over-21 v3 causal endpoint
  has a valid specificity test under the approved null families.
- Does not apply when: Describing the inspected Stage B action landscape or
  proposing a genuinely new, explicitly approved scientific objective.
- Consequence for future actions: Do not retune calipers, distance, ranks,
  representations, or calibration size again and do not substitute an
  isotropic null for specificity. Held-out scoring, Stage C2, and Stage D remain
  closed.
- Revisit condition: Only an explicit strategic-plan amendment that changes
  the scientific objective or supplies a prospectively justified new
  specificity-null family.

## 2026-08-07 — Pivot to query-conditional value without reopening harmfulness

- Decision or promoted lesson: Close v3 harmfulness confirmation permanently
  under its frozen protocol and make same-image, different-question four-action
  value the v4 objective. Start with a GQA-only discovery and mandatory
  common-padding identity gate; reserve TextVQA for later replication.
- Triggering evidence: v3 structured-null gates fail despite 4,000 independent
  calibration images, while exact interventions remain valid, common padding
  restores visual-state/WRITE equality, and GQA provides 9,800 metadata-eligible
  multi-question images plus semantic/object annotations.
- Evidence paths: `reports/v3_null_redesign_v2.md`,
  `workspace/v3_query_invariance_validation.md`,
  `outputs/v3_preflight/stage_c2_reserved_pool_audit.json`, and
  `reports/v4_strategy_transition.md`.
- Confidence: medium-high.
- Applies when: Testing whether the same visual computation has different
  downstream value for different questions under the pinned architecture.
- Does not apply when: Claiming harmfulness, a semantic mechanism, a deployable
  policy, acceleration, cross-model generality, or rescuing v3 confirmation.
- Consequence for future actions: The next authorized experiment, if approved,
  is 120 GQA images with two questions each, seven frozen layers, all four
  actions, and image-level tie-aware statistics after a 12-image exact identity
  preflight. No outcome-dependent task substitution or layer search is allowed.
- Revisit condition: The identity gate fails, semantic pair construction is
  infeasible, or discovery dependence collapses under tie, robustness, and
  semantic controls.

## 2026-08-07 — Stop before v4 confirmation pending a paraphrase control

- Decision or promoted lesson: The GQA discovery supports query-associated
  four-action disagreement and transfer regret under identical same-image
  visual state/WRITE, but it does not open held-out confirmation. Request a
  separately approved paraphrase-only discovery amendment; otherwise close v4.
- Triggering evidence: Equal-layer transfer regret is `0.1121` mean, `0.0634`
  median, and `0.0711` 20%-trimmed nats/token, while the direct image+query
  oracle gap is only `0.0144` mean and `0.00344` median. Different-evidence
  semantic ordering is mixed, and zero official paraphrase pairs meet the
  prospective metadata rule.
- Evidence paths: `reports/v4_discovery_results.md`,
  `reports/v4_discovery_decision.md`,
  `outputs/v4_discovery/analysis_v1/analysis_manifest.json`, and
  `outputs/v4_discovery/analysis_v1/semantic_control_sensitivity_v1.json`.
- Confidence: high for stopping before confirmation; medium for the value of a
  paraphrase amendment.
- Applies when: Interpreting this pinned-model, GQA-only inspected discovery and
  deciding whether v4 confirmation may begin.
- Does not apply when: Claiming a semantic mechanism, policy benefit,
  acceleration, harmfulness, or generality beyond the studied data/model.
- Consequence for future actions: Do not freeze a held-out manifest or run a
  confirmation/probe. A paraphrase arm must be prospectively frozen and user
  approved; it remains discovery evidence and may not change existing Q values.
- Revisit condition: A valid approved paraphrase control shows stable
  within-question action patterns relative to different-evidence pairs, without
  changing the frozen actions, layers, scoring, or claim boundary.

## 2026-08-07 — Close the dynamic-policy direction after the v4 cost frontier

- Decision or promoted lesson: Frequent same-image query-conditioned action
  disagreement does not, by itself, imply a meaningful compute-allocation
  advantage. Close the dynamic-policy direction under the frozen v4 protocol
  and do not run the proposed paraphrase control.
- Triggering evidence: Across all 840 image-layer pairs, the maximum pooled
  matched-compute mean utility gain is `0.02370` nats/token, mean
  matched-utility saving is `1.40%` of FULL cost, and only `3.40%` of utility
  targets save at least 10% FULL. The unconstrained query-conditioned oracle
  uses `2.62%` more FULL-equivalent compute, rejecting the conservative-FULL
  explanation. Robust non-ties and trimmed summaries do not materially change
  the result.
- Evidence paths: `reports/v4_cost_utility_reanalysis.md`,
  `outputs/v4_discovery/cost_utility_frontier_v1.csv`, and
  `outputs/v4_discovery/cost_utility_frontier_summary_v1.json`.
- Confidence: medium-high; the median frontier and layer-0 raw mean retain
  localized counterevidence but not a stable pooled practical gain.
- Applies when: Deciding whether this pinned-model GQA discovery justifies
  further semantic controls, confirmation, or policy development.
- Does not apply when: Claiming that query-conditioned values do not exist, or
  generalizing to other models, datasets, actual sparse kernels, or latency.
- Consequence for future actions: Preserve v4 as discovery evidence and stop
  paraphrase, held-out confirmation, probe/router, and acceleration work under
  this protocol.
- Revisit condition: Only an explicitly approved new scientific direction with
  independently motivated costs, real sparse execution, and a new claim—not a
  continuation or retuning of the closed v4 oracle analysis.

## 2026-08-07 — Stop frozen query-conditioned visual refinement

- Decision or promoted lesson: A clean frozen question-to-visual replay edge is
  technically feasible, but this output-boundary operator does not provide
  robust target-question-specific answer value. Close the direction under its
  frozen kill rule.
- Triggering evidence: Zero of three preselected layers passed. Layer 4's
  conditioning mean was `0.01918` nats/token with a positive CI, but its median
  was `0.000021`, 20%-trimmed mean `0.00212`, and paired-other-question CI
  crossed zero. Layers 12/20 were negative or near zero. Target replay had one
  net greedy regression versus baseline at every anchor.
- Evidence paths: `reports/query_refinement_preflight.md`,
  `reports/query_refinement_gqa_discovery.md`,
  `outputs/query_refinement/analysis_v1/layer_summaries.json`, and
  `outputs/query_refinement/analysis_manifest.json`.
- Confidence: high for stopping this frozen operator; low for claims about
  different trained visual-memory architectures.
- Applies when: Deciding whether to replicate, train, deepen, or retune this
  frozen existing-token replay approach.
- Does not apply when: Claiming all query-conditioned visual architectures are
  impossible or interpreting the result as harmfulness/efficiency evidence.
- Consequence for future actions: Do not run TextVQA replication, search replay
  depth/layers/subgroups, or train a selector/adapter to rescue this result.
- Revisit condition: Only an explicit strategic amendment defining a genuinely
  different architecture and claim, not a replay-depth or layer-search variant.

## 2026-08-09 — Open a distinct supervised binary visual-token policy detour

- Decision or promoted lesson: The user explicitly opened a new supervised
  policy direction using layer-local visual-token ON/OFF actions and existing
  MCTS v2 labels. This does not reopen the closed v2–v4 harmfulness, local
  four-action, or frozen-refinement claims.
- Triggering evidence: the binary Qwen reference supplies a concrete token-
  presence executor, POLAR supplies a lightweight pre-action predictor recipe,
  and the audited source contains 4,000 labeled samples with 3,408 positive-
  route records.
- Evidence paths: `reference/binary_action_qwen/`, `reference/polar/`,
  `/home/hyemin/data/dataset/dynamic_mllm/mcts_v2/final/audit_summary_full_v2.json`,
  and `reports/binary_visual_polar_adaptation_audit.md`.
- Confidence: high that the method is implementable; unknown whether the
  policy will generalize or preserve accuracy because training and model-scale
  validity checks have not run.
- Applies when: planning and validating the 28-bit visual-token policy under
  `plans/dynamic_mllm_binary_visual_polar_plan_v1.md`.
- Does not apply when: claiming harmful visual participation, reinterpreting
  prior negative results, fine-tuning the base MLLM, or treating MCTS oracle
  savings as realized acceleration.
- Consequence for future actions: require full label/factorization audit,
  all-ON/OFF executor validity, exact cached-label reproduction, and an
  image-grouped split before training.
- Revisit condition: any validity gate fails, direct mask factorization is
  inadequate, or fresh top-1 evaluation lacks the frozen correctness/compute
  tradeoff.

## 2026-08-10 — Regenerate binary route labels under one native-Qwen contract

- Decision or promoted lesson: Stop treating the old MCTS v2 cache as
  ground-truth supervision. Regenerate unrestricted complete 28-bit visual
  ON/OFF masks for the fixed 8K GQA/TextVQA/ChartQA pool under one frozen
  native-Qwen execution contract. Old masks may propose candidates, but every
  answer, score, and valid/invalid label must be recomputed.
- Triggering evidence: after input geometry and native ALL-ON parity were
  repaired, the frozen BP-1 suite still contained two cached-positive routes
  that became invalid under the target executor.
- Evidence paths: `plans/dynamic_mllm_label_regeneration_plan.md`,
  `reports/binary_polar_bp1_input_contract_repair.md`, and
  `outputs/binary_polar/preflight/executor_preflight_v3.json`.
- Confidence: high that regeneration is necessary; unknown whether the native
  processor and planned MCTS budget are feasible at 8K scale.
- Applies when: constructing supervision for later binary, top-K, reranking,
  single-route, or POLAR-derived predictor comparisons.
- Does not apply when: changing prior v2–v4 scientific conclusions or
  authorizing predictor/base-model training.
- Consequence for future actions: execute P0–P9 in order, use native processing
  without the old token cap, freeze image-disjoint splits before search, retain
  every evaluated positive/negative mask, and block training until P9 passes.
- Revisit condition: a hard gate fails or native processing produces a resource
  blocker requiring an explicitly approved fixed alternative contract.

## 2026-08-10 — Use a minimal smoke and split after route extraction

- Decision or promoted lesson: Before regenerating labels, run only a frozen
  15-record executor smoke—five records per dataset, 15/15 exact ALL-ON/native
  generated-token parity, and repeated mixed-route token/score equality. If it
  passes, immediately extract all 8K labels. Construct image-disjoint predictor
  splits after extraction and before predictor training.
- Triggering evidence: explicit user amendment; MCTS search is independent per
  sample and therefore does not learn across future predictor splits.
- Evidence paths: `plans/dynamic_mllm_label_regeneration_plan.md` and
  `workspace/phase_memory/phase_12_label_regeneration.md`.
- Confidence: high that this is the approved order; native-processing resource
  feasibility remains unknown until execution.
- Applies when: executing the current 8K unrestricted routing-label plan.
- Does not apply when: relaxing exact executor parity/determinism, choosing
  predictor splits using route outcomes, or authorizing predictor training.
- Consequence for future actions: do not run the former 20–50-per-dataset pilot
  or freeze predictor splits before extraction. Stop immediately if the minimal
  smoke fails; otherwise start full MCTS without new pre-generation gates.
- Revisit condition: the smoke exposes a concrete executor failure or the full
  run encounters the plan's native-processing hard stop.

## 2026-08-12 — Preserve an in-domain predictor test alongside external transfer

- Decision or promoted lesson: use an exact image-group-disjoint 6,500 train /
  1,000 validation / 500 internal-test split for the regenerated MCTS
  population, and reserve the disjoint 5,807-record MMStar/MMMU/MMMU-Pro bundle
  as a separate external transfer test after checkpoint freezing.
- Triggering evidence: the external bundle has zero overlap with the MCTS pool
  across IDs, exact image hashes, normalized text/prompts, and image-question
  pairs, but its SW31/admission outcomes are already inspected and its
  multiple-choice task distribution differs from GQA/TextVQA/ChartQA. Exact
  6,500/1,000/500 grouped internal targets are feasible.
- Evidence paths:
  `reports/binary_router_p7_split_and_external_eval_audit.md`,
  `outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json`,
  and
  `outputs/label_regeneration/v1/post_generation/predictor_split_design_audit_v1.json`.
- Confidence: high for the split-role decision; predictor performance remains
  unknown because no real training or evaluation has run.
- Applies when: freezing P7 identities and evaluating the matched duplicated-BCE
  versus exact-set-NLL direct-mask predictors.
- Does not apply when: treating the external bundle as untouched publication
  confirmation, reusing its SW31/forced-K8/admission policy, or selecting
  checkpoints from external outcomes.
- Consequence for future actions: select checkpoints only on internal
  validation, use internal test for the primary in-domain comparison, and run
  the full external bundle once as transfer evidence through a validated static
  mask adapter.
- Revisit condition: a split-freeze implementation cannot meet the exact group
  and source-cell constraints without using route outcomes, or the external
  adapter fails baseline/executor parity.

## 2026-08-12 — Replace the internal test with expanded task-native evaluation

- Decision or promoted lesson: revise the internal split to 7,000 train / 1,000
  validation and use the expanded 27,656-record bundle as final evaluation,
  reported separately as core VQA, multiple-choice transfer, and POPE.
- Triggering evidence: the updated bundle added 12,849 core-VQA and 9,000 POPE
  records. Full image verification passed. ChartQA test, TextVQA validation,
  DocVQA validation, and every MMStar/MMMU benchmark have zero MCTS image
  overlap. POPE has only one overlapping image in 18 rows, which can be exposed
  by a pre-specified 8,982-row image-disjoint sensitivity result.
- Evidence paths: `reports/binary_router_expanded_eval_suite_audit.md` and
  `outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json`.
- Confidence: high for split roles and evaluation populations; router
  performance remains unknown because the direct predictor is not trained.
- Applies when: freezing P7 and evaluating the matched direct-mask predictors.
- Does not apply when: pooling incompatible suite metrics, using SW31/K8/gating
  as the new predictor, or tuning on final evaluation outcomes.
- Consequence for future actions: validation selects checkpoints; the expanded
  bundle is opened only after freezing. Preserve official full POPE results and
  the strict image-disjoint sensitivity.
- Revisit condition: the direct-mask adapter cannot reproduce cached all-ON
  outputs under the bundle contract or a manifest/checksum changes.

## 2026-08-12 — Exclude DocVQA from active binary-router evaluation

- Decision or promoted lesson: remove all 5,349 DocVQA validation records from
  the planned direct-predictor evaluation. Preserve the bundle and historical
  DocVQA reference artifacts unchanged.
- Triggering evidence: explicit user direction after reviewing the expanded
  suite.
- Evidence path: `reports/binary_router_expanded_eval_suite_audit.md`.
- Confidence: definitive scope decision.
- Applies when: building the static-mask evaluation adapter and final benchmark
  selection for duplicated-BCE and exact-set-NLL predictors.
- Does not apply when: deleting or rewriting the reference bundle, its
  integrity audit, or historical SW31 results.
- Consequence for future actions: active final evaluation is 22,307 records—
  2,500 ChartQA, 5,000 TextVQA, 9,000 POPE, and 5,807 MMStar/MMMU/MMMU-Pro.
- Revisit condition: explicit user approval to restore DocVQA.

## 2026-08-12 — Match POLAR's 50-route supervision cap

- Decision or promoted lesson: use at most 50 valid masks per image-query in
  the primary derived training view. Duplicated BCE and exact set-NLL must use
  the identical deterministic diverse subset.
- Triggering evidence: released POLAR defaults to 50 valid paths per sample,
  and the user explicitly approved matching that cap. The regenerated raw
  cache contains 3,616 samples with more than 50 valid masks, so the choice is
  consequential rather than cosmetic.
- Evidence paths: `reference/polar/PoLar/polar/data.py`,
  `plans/dynamic_mllm_label_regeneration_plan.md`, and
  `outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl`.
- Confidence: definitive protocol choice.
- Applies when: constructing P8 valid-set supervision and running the matched
  loss comparison.
- Does not apply when: truncating raw MCTS outputs, changing route weights, or
  giving one objective a different subset.
- Consequence for future actions: P8 retains all valid masks when count is at
  most 50 and applies the frozen diversity selection only above 50. The former
  32-route view is secondary ablation material only.
- Revisit condition: explicit user approval of a matched route-cap ablation.

## 2026-08-12 — Freeze regenerated routing labels as the sole P10 supervision source

- Decision or promoted lesson: P0-P9 of label regeneration are complete. Any
  downstream binary predictor comparison must use the checksum-bound 8K cache,
  frozen 7K/1K image-group split, and identical deterministic max-50 route sets
  produced by P8.
- Triggering evidence: P9 binds 8,000/8,000 raw records through the P4 record
  index, freezes 50 primary/code/provenance files, and independently verifies
  all 53 entries in `P9_SHA256SUMS`.
- Evidence paths: `reports/label_generation_report.md`,
  `outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json`,
  `p9_artifact_inventory_v1.json`, and `p9_run_provenance_v1.json`.
- Confidence: high; all integrity gates pass and the final cache has zero
  incomplete or invalid terminal records.
- Applies when: launching the matched duplicated-BCE versus exact-set-NLL P10
  smoke or tracing its exact data/code provenance.
- Does not apply when: authorizing full training, changing route cap/weights,
  changing the split, or treating cached Hit@1 as behavioral correctness.
- Consequence for future actions: P10 may start only as a separately approved
  bounded smoke. Full training and external evaluation remain later gates.
- Revisit condition: an independent checksum failure or a concrete defect in a
  frozen P9 artifact.

## 2026-08-13 — Probability-level input signal does not admit full binary-head training

- Decision or promoted lesson: do not launch full matched training merely
  because exact valid-set NLL beats duplicated BCE or aligned questions beat a
  shuffle in set-NLL. Admission also requires materially nonconstant decoded
  masks and useful real execution.
- Triggering evidence: P11 exact set-NLL improves aligned versus shuffled
  set-NLL (`14.8699` versus `15.5089`) but predicts ALL-ON on 147/150
  route-validation records and 57/60 actual executions. Its three non-FULL
  masks are uncached and remain wrong; accuracy equals FULL at 50% on the
  balanced subset and recovers none of the 100% MCTS-oracle result.
- Evidence paths: `reports/binary_polar_p11_results.md`,
  `outputs/binary_polar/p11/question/exact_set_nll_conditioning_v1.json`, and
  `outputs/binary_polar/p11/execution/exact_set_nll_v1.json`.
- Confidence: high that the P11 full-training admission gate fails; medium that
  factorized decoding is the next bottleneck rather than limited smoke
  optimization.
- Applies when: deciding whether to scale the current direct factorized 28-bit
  question-conditioned head.
- Does not apply when: claiming no question signal exists, proving a structured
  decoder will work, or authorizing a head redesign.
- Consequence for future actions: preserve P11 as Outcome C, stop full matched
  training, and require explicit approval for any controlled structured-head
  comparison.
- Revisit condition: a prospectively approved experiment supplies materially
  stronger nonconstant execution evidence under the unchanged data/evaluator.

## 2026-08-13 — Canonical run segmentation does not repair top-1 collapse

- Decision or promoted lesson: close the P12 maximal-run structured-head pivot
  as Outcome B and do not launch its full training.
- Triggering evidence: all 237,802 selected supervision masks reconstruct
  exactly, but their mean/median segment count is 14.11/14. The selected P12
  checkpoint predicts ALL-ON on 150/150 validation and 60/60 execution inputs.
  Aligned set-NLL is better than shuffled (`17.4918` versus `18.1939`), while
  actual accuracy remains 50%, W→C=0, C→W=0, and compute reduction is zero.
- Evidence paths: `reports/binary_polar_p12_results.md`,
  `outputs/binary_polar/p12/segment_geometry_v1.json`,
  `outputs/binary_polar/p12/structured_conditioning_v1.json`, and
  `outputs/binary_polar/p12/structured_execution_v1.json`.
- Confidence: high for the frozen two-epoch P12 gate; low for claims about all
  possible structured decoders or much longer optimization.
- Applies when: deciding whether this canonical boundary/operation head earned
  full matched training or whether independent bit thresholding alone explains
  P11's collapse.
- Does not apply when: claiming question signal is absent, comparing raw P11
  and P12 NLL values across different output spaces, or ruling out every future
  structured model.
- Consequence for future actions: do not full-train P12, substitute the
  non-selected epoch, tune its decoder, or stack another route-structure module.
- Revisit condition: a separately approved research direction changes the
  feature/supervision hypothesis and defines a new bounded gate.

## 2026-08-13 — Native visual context improves probability mass, not route selection

- Decision or promoted lesson: close P13 as Outcome B and do not full-train the
  current direct valid-set route generator merely because native visual input
  lowers set-NLL.
- Triggering evidence: the selected Image+Question checkpoint lowers aligned
  set-NLL from Question-only `14.8699` to `14.4944`, and image shuffling worsens
  it to `14.8748`. It nevertheless predicts ALL-ON for 150/150 validation
  records, exactly matching the constant baseline's 57.33% Hit@1, 3.693
  nearest-valid Hamming, and 28 mean ON layers. The prospective execution gate
  therefore fails.
- Evidence paths: `reports/binary_polar_p13_results.md`,
  `outputs/binary_polar/p13/conditioning_diagnostic_v1.json`, and
  `outputs/binary_polar/p13/analysis_manifest_v1.json`.
- Confidence: high for the frozen two-epoch admission decision; low for claims
  about every possible multimodal predictor or longer optimization.
- Applies when: deciding whether missing image information alone justifies
  scaling the direct factorized exact-set route-generation pipeline.
- Does not apply when: claiming visual input has no probability-level signal,
  claiming all multimodal predictors must fail, or evaluating a separately
  approved candidate-route utility objective.
- Consequence for future actions: do not run the P13 60-record execution, do
  not substitute the more diverse but worse epoch-2 checkpoints, and require a
  new approved plan for any objective-level pivot.
- Revisit condition: a prospectively approved, separately gated objective
  tests route validity/utility without using P13 outcomes for tuning.

## 2026-08-13 — Longer direct-head optimization yields diversity without route-quality gain

- Decision or promoted lesson: Do not admit the current direct factorized
  exact-valid-set predictor to external evaluation. A longer POLAR-style
  schedule can emit diverse masks, but diversity alone is not evidence of
  useful routing and does not overcome the constant ALL-ON solution.
- Triggering evidence: On 874 validation positives, Question-only and
  Image+Question best-Hit@1 checkpoints both equal constant ALL-ON at `58.12%`
  and are `99.66%`/`100%` ALL-ON. Epoch-10 diversity rises to 122/64 masks but
  Hit@1 falls to `55.03%`/`55.84%`. Both frozen-60 best-checkpoint executions
  remain `50%` with W→C=0 and C→W=0.
- Evidence paths: `reports/binary_polar_full10_polar_matched_results.md`;
  `outputs/binary_polar/full10/question_v1/history.json`;
  `outputs/binary_polar/full10/image_question_v1/history.json`;
  `outputs/binary_polar/full10/execution_*best_hit_at_1_v1.json`.
- Confidence: high for this frozen direct predictor and training setup.
- Applies when: deciding whether to scale or externally evaluate the current
  direct 28-Bernoulli head with exact valid-set NLL.
- Does not apply when: judging every possible structured predictor or a
  separately approved candidate-ranking objective.
- Consequence for future actions: require complete-mask route quality and
  execution evidence, not reduced ALL-ON rate or lower set-NLL alone.
- Revisit condition: a prospectively approved formulation beats the ALL-ON
  internal baseline under matched held-out validation.

## 2026-08-16 — Duplicated BCE hybridization and dominated supervision are distinct label bottlenecks

- Decision or promoted lesson: do not attribute the current binary-router
  failure to poor raw MCTS diversity or max-50 collapse. Treat exact bitwise
  duplicated-BCE hybridization as the primary label/objective mismatch and
  Pareto-dominated routes as a major separable contributor.
- Triggering evidence: raw/selected mean pairwise Hamming is 13.36/13.44 and
  entropy is 0.5989/0.5986, while the exact training-weighted per-sample BCE
  oracle is selected-valid for only 5.93% of positive inputs. Invalid-oracle
  samples have 36.46 effective modes and 0.6353 bit entropy. Of 237,802
  selected route occurrences, 95.83% are Pareto-dominated; diagnostic Pareto
  filtering raises oracle Hit@1 to 73.41% but does not eliminate all invalidity.
- Evidence paths: `reports/binary_mcts_label_geometry_and_bce_oracle_report.md`,
  `outputs/binary_mcts_label_geometry_v1/weighted_bce_oracle_summary.csv`,
  `invalid_hybrid_summary.csv`, `raw_selected_summary.csv`, and
  `pareto_summary.csv`.
- Confidence: high for the frozen 8K label geometry and exact duplicated-BCE
  oracle; no claim is made about a new trained formulation.
- Applies when: deciding whether to regenerate labels, alter max-50 selection,
  or train another predictor on the current supervision.
- Does not apply when: claiming Pareto-filtered training will generalize,
  declaring every uncached learned mask invalid, or choosing a replacement
  objective without a matched experiment.
- Consequence for future actions: any approved next study should hold the raw
  cache/split/model/training budget fixed, remove dominated-label pressure as a
  matched factor, and compare against a complete-route-coherent objective.
- Revisit condition: a checksum defect in the frozen analysis or a matched
  held-out execution study contradicts the label-oracle diagnosis.

## 2026-08-17 — Exclude node03 and node04 from project compute

- Decision: do not submit or run project jobs on node03 or node04.
- Triggering evidence: generic NLL Slurm job `101490` was placed on node03 and
  cancelled before training at `2026-08-17 18:41:09 KST`.
- Consequence: all future Slurm submissions must use an explicit allowed-node
  constraint; generic A6000 placement is not permitted.
- Evidence: `runs/binary_pareto_v1/nll/slurm.log`.

## 2026-08-18 — Re-enable node04; keep node03 excluded

- Decision: node04 is allowed for future scheduled work by explicit user
  amendment. Node03 remains prohibited.
- Consequence: Slurm jobs may target node04 again when its partition and GPU
  type fit the task; CPU jobs still prefer node05.
- Does not alter: historical node03/node04 failures or the placement of jobs
  that were already running when the amendment was given.

## 2026-08-18 — Pareto filtering does not repair predictor training fit

- Decision evidence: across all ten saved checkpoints, best train Pareto
  Hit@1 is 18.27% for duplicated BCE and 17.95% for exact set NLL, versus a
  73.92% frozen train BCE-label oracle.
- Supported lesson: the current bottleneck is primarily complete-mask
  training fit under the shared predictor/optimization/input pipeline, not
  held-out generalization. Residual multimodal/factorized failure coexists:
  multi-route Hit is approximately zero, but singleton Hit is also only ~24%.
- Collapse lesson: Pareto filtering removes ALL-ON collapse but replaces it
  with substantial ALL-OFF concentration and later diverse mostly non-Pareto
  masks; increased diversity is not evidence of route learning.
- Consequence: adding more data or performing another loss-only comparison is
  not justified by this evidence. A future action, if explicitly authorized,
  must first distinguish architecture/input capacity from optimization fit.
- Evidence: `reports/binary_pareto_training_fit_analysis.md` and
  `outputs/binary_pareto_v1/training_fit_analysis_v1/`.

## 2026-08-23 — Re-freeze runtime-defined cohort predicates after server transfer

- Decision or promoted lesson: a transferred matched cache may supply candidate
  IDs, route evidence, and historical provenance, but any cohort predicate
  defined by current model generation/correctness must be re-evaluated and
  frozen under the exact current executor before an intervention sweep.
- Triggering evidence: four-action primary job `1497` encountered
  `gqa:gqa_ge_16564303`, whose transferred FULL answer was wrong while current
  native and unified FULL agreed on a correct answer. An all-candidate unified-
  FULL freeze subsequently excluded 32/1,912 primary candidates and 26/2,110
  FULL-correct-control candidates, with zero native/unified semantic issue.
- Evidence paths:
  `analysis/4action_answer_alignment/cohort_eligibility__unified_v1/summary.json`,
  `logs/slurm/four-action-unified-primary-r2-20260823-1497.log`, and
  `workspace/phase_memory/phase_31_four_action_answer_alignment.md`.
- Applies when: a cohort definition contains a generated-answer or evaluator-
  correctness condition and the runtime, server, model stack, or execution path
  differs from the cache-producing environment.
- Does not apply when: discarding historical routes or redefining continuous
  factorial effects; those remain within the unified executor and use no drift
  threshold.

## 2026-08-24 — Treat binary route choices and visual-operation effects as context-dependent

- Decision or promoted lesson: do not treat every OFF layer in a discovered
  correcting route as individually causal, and do not expect a FULL-context
  single-layer rescue map to recover the operations required inside a
  successful multi-layer route. Use route-conditioned intervention when the
  claim concerns why a cached correcting route works.
- Triggering evidence: among 17,262 OFF positions in 1,804 current-correct
  anchors, 9,382 (54.35%) were individually redundant, while 7,880 were
  necessary. The earlier FULL-context discrete local-rescue map recalled only
  575/7,880 (7.30%) route-necessary positions; 92.70% were revealed only with
  the other anchor suppressions held fixed. Continuous FULL-versus-route effect
  Spearman correlations were 0.357 for READ and 0.177 for WRITE.
- Evidence paths:
  `analysis/4action_route_conditioned/aggregate_summary.json`,
  `analysis/4action_route_conditioned/aggregate/full_context_comparison.parquet`,
  `analysis/4action_route_conditioned/final_integrity_audit.json`, and
  `analysis/4action_route_conditioned/route_conditioned_decomposition_report.md`.
- Confidence: high for the frozen A+ population, current unified executor, and
  selected cached anchors.
- Applies when: interpreting cached search routes, designing route-mechanism
  studies, or deciding whether dense-context local effects explain a
  multi-layer correction.
- Does not apply when: claiming an operation is globally harmful, claiming the
  selected route is globally minimal, or claiming individually valid partial
  restorations compose into a better four-action route/router.
- Consequence for future actions: require current-runtime anchor validation and
  route-conditioned necessity tests; test composability separately before any
  true four-action search/router pivot.

## 2026-08-25 — Do not use an unstable bounded beam as a four-action label oracle

- Decision or promoted lesson: a bounded beam may be useful for exploration,
  but do not promote its canonical route or positive set into supervision when
  the prospectively chosen width is materially unstable against the wider
  validation width. Preserve executor-valid evidence separately from
  search-policy validity.
- Triggering evidence: among 1,417 replay-valid route conversions from 24
  completed five-dataset pilot samples, beam 8 and beam 16 disagreed on 322
  canonical routes; 167 positive-set Jaccards were below 0.50 and the minimum
  was 0.0. Binary parity, evaluator correctness, C2C gain, cache, checksum, and
  worker checks had zero failures.
- Evidence path:
  `analysis/three_action_answer_aligned_label_conversion/early_stop_audit.md`.
- Confidence: high for rejecting this beam-8 label contract; this does not
  imply a unified-executor or model failure.
- Consequence for future actions: use an approved exact verified policy when
  every valid branch is required, or prospectively validate a bounded policy
  before treating it as a label generator.

## 2026-08-29 — Sample balance does not prevent four-action FULL collapse

- Decision or promoted lesson: do not treat 50:50 W2C:C2C sampling as balanced
  four-action supervision, and do not retry the online router without directly
  covering the mandatory departure from its deployed all-FULL prefix.
- Triggering evidence: under the exact planned ten-epoch sampler, combined
  teacher actions are 66.735% FULL, 25.335% IGNORE, 4.795% READ_ONLY, and
  3.136% WRITE_ONLY. FULL is uniquely valid at 55.360% of sampled prefix nodes,
  versus 3.107%/2.216% singleton READ/WRITE. The sampler never visits the
  latest valid all-FULL-prefix deviation boundary for 1,045/2,397 W2C samples,
  although READ/WRITE are valid at 43.388%/52.733% of those boundaries.
- C2C qualification: the exact all-FULL route is present for 3,501/3,548 C2C
  train samples and is a plausible shared-route shortcut, but removing it does
  not change the measured W2C boundary-coverage defect. C2C labels also contain
  no READ_ONLY or WRITE_ONLY positives by construction.
- Evidence path:
  `reports/four_action_router_collapse_label_audit_20260829.md`.
- Confidence: high for the label/sampler geometry and the next diagnostic;
  unresolved for the sole cause of learned collapse or the final remedy.
- Consequence for future actions: first test isolated W2C mandatory-boundary
  coverage with the unchanged router, loss, and C2C population. Keep C2C
  all-FULL removal, action weighting, and on-policy exposure as separate
  prospectively defined ablations rather than bundling them into one retry.
- Revisit condition: an explicitly covered boundary-capacity pilot fails to
  fit or free-run its fixed W2C subset, which would shift attention to router
  state features, action-head capacity, or gradient allocation.

## 2026-08-29 — Exact boundary exposure establishes online-router pilot capacity

- Decision or promoted lesson: do not attribute the prior online all-FULL
  collapse to an immediate inability of the existing state-conditioned
  architecture or head to represent corrective actions. Test population-level
  generalization with the same architecture and guaranteed boundary exposure
  before redesigning it.
- Triggering evidence: the prospectively frozen 96-W2C/24-C2C pilot passed all
  gates at epoch 30: boundary Valid-Action@1/non-FULL recall 0.9583/0.9583,
  singleton IGNORE/READ/WRITE recall 0.9583/1.0000/0.9167, all-FULL departure
  1.0000, W2C rescue 0.8958, and C2C preservation 0.9167.
- Evidence paths: `analysis/4action_collapse/mandatory_boundary_overfit_report.md`,
  `analysis/4action_collapse/mandatory_boundary_overfit_history.jsonl`, and
  `outputs/four_action_collapse/mandatory_boundary_overfit_v1/training_summary.json`.
- Confidence: high for local capacity on the fixed pilot; unresolved for
  held-out generalization and for whether one scheduled exposure per full-data
  W2C sample is sufficient.
- Consequence for future actions: run the matched A2 schedule before action
  weighting, on-policy data, DAgger, or architecture redesign.
