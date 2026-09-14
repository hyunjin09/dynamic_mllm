# Decision Log

Record only decisions, pivots, and lessons that should affect later phases.
Do not copy raw logs or provisional explanations here.

## 2026-09-09 — Treat the current Stage-1 signal as source/dataset-specific, not semantic-neighbor memorization

- Decision or promoted lesson: retain the current Stage-1 hidden-state signal
  as robust to semantic-neighbor distance and unseen label-blind question
  clusters, but do not treat it as source- or dataset-transferable. Any future
  use must calibrate and validate on the intended source/task family. Continue
  to treat the current Stage-2 local READ/WRITE utility predictor as weak.
- Triggering evidence: Stage-1 M3 Q1/Q5 AUROC is 0.7828/0.8012 and its
  Q5-minus-Q1 delta is +0.0183 with 5,000-draw image-group-bootstrap 95% CI
  [-0.0050, 0.0423]. Concatenated K=100 cluster-OOD AUROC is 0.7725 versus ID
  0.7869. In contrast, Historical→Canonical/Canonical→Historical AUROC is
  0.4342/0.5563, LODO AUROC is 0.5508-0.6043, and pairwise transfer is
  0.4985-0.5807. Cluster-OOD Stage-2 READ/WRITE Spearman remains only
  0.0453/0.0164, with LODO near zero.
- Qualification: GQA within-dataset source transfer is stronger (~0.74), so
  source specificity is heterogeneous rather than universal collapse. K=100
  cluster OOD establishes robustness to this frozen question encoder's
  partition, not arbitrary semantic shifts. LODO threshold drift is especially
  severe for ChartQA/GQA and should not be generalized to every dataset.
- Evidence paths:
  `analysis/predictability_generalization/stepC_generalization/statistics/similarity_q1_q5_bootstrap.csv`,
  `question_cluster_ood/stage1_metrics.csv`,
  `source_transfer/*_stage1.csv`, `dataset_lodo/lodo_stage1.csv`, and
  `summaries/stepC_generalization_summary.md`.
- Confidence: high for split isolation, all 300 fit completions, frozen encoder
  and model contracts, exact kNN support accounting, and the S1-C/S2-A
  descriptive categories; unknown for the causal feature or preprocessing
  responsible for source/dataset failure.
- Consequence for future actions: do not deploy a global Stage-1 threshold or
  claim transferable failure prediction from Phase-79 ID AUROC. A separately
  authorized Step D may measure external transfer, but redesigning Stage 1 or
  Stage 2 is a strategic pivot and must be planned prospectively.

## 2026-09-09 — Separate learnable dense failure risk from weak local READ/WRITE utility

- Decision or promoted lesson: retain the Phase-79 Stage-1 state signal as
  genuine in-domain learnability, but do not treat the tested current-state
  representation/head family as a useful predictor of immediate READ or WRITE
  utility. Keep these as distinct scientific conclusions.
- Triggering evidence: leakage-free five-fold OOF Stage-1 AUROC is 0.6814 for
  nuisance, 0.7665 linear, 0.7874 MLP, and 0.7869 current head; current-head
  AUROC peaks at layer 20 (0.8257), and its group-bootstrap 95% interval is
  [0.7789, 0.7946]. For 15,185 primary dense post-trigger states, joint-router
  READ/WRITE Spearman is 0.0416/0.0347, harmful AUROC is 0.5193/0.5115, and
  top-10% harmful precision is 0.4937/0.4775 versus prevalence 0.480/0.472.
  z_R does not beat z_W for READ, z_W does not beat z_R for WRITE, nuisance is
  the strongest READ correlate, and routed/transfer correlations remain weak.
- Qualification: Step B is same-population in-domain OOF. It does not test
  question/template independence, dataset LODO, external transfer, causal
  branch necessity, or deployable routing. Routed-state evidence covers a
  selected 569-UID successful-route corpus. Weak Stage-2 predictability does
  not prove controlled effects are absent; Phase 78 directly measured them.
- Evidence paths:
  `analysis/predictability_generalization/stepB_id_learnability/`, especially
  `stage1/overall_metrics.csv`, `stage2_dense/continuous_metrics.csv`,
  `stage2_dense/representation_specificity.csv`,
  `statistics/uid_bootstrap_ci.csv`, and
  `summaries/stepB_id_learnability_summary.md`.
- Confidence: high for fold isolation, full OOF completeness, metric
  reproducibility, and the Case-D in-domain classification; unknown for why
  the immediate utilities are weakly predictable and for all OOD behavior.
- Consequence for future actions: do not deploy or add another local
  READ/WRITE utility gate from this frozen ladder. A separately authorized
  Step C may test semantic/source generalization without redesigning targets;
  any Stage-2 target/representation pivot requires its own prospective plan.

## 2026-09-08 — Step-A controlled utility measurements are valid and non-degenerate

- Decision or promoted lesson: retain the frozen Phase-78 census as the
  authoritative measurement basis for a separately authorized predictability
  phase. Keep dense-origin and routed-state utilities as distinct domains, and
  do not infer predictability from their label distributions alone.
- Triggering evidence: all 10,399 internal UIDs / 291,172 Stage-1 layer states
  are represented; strict P90 yields 1,413 UIDs and 15,185 primary dense-origin
  states. All 60,740 primary and 142,260 secondary four-action branches are
  complete. Primary `U_READ` has 7,962 positive and 7,223 negative states;
  `U_WRITE` has 7,353 positive, 6,419 negative, and 1,413 zero states. The
  routed census contains 35,565 unique states deduplicated from 69,178 route
  occurrences and likewise has broad signed utility support. Primary controlled
  correctness labels contain 1,048 local-rescue and 83 local-regression states.
- Qualification: these are teacher-forced annotated-answer likelihood effects
  plus separately preserved LMMS correctness flips, not learned-predictor
  performance. ChartQA's relaxed numeric acceptance interval has no finite
  answer-string representation, so continuous q uses the literal annotation
  while discrete correctness retains LMMS semantics. Routed states cover the
  existing 569-UID exact corpus rather than all reachable routed states.
- Evidence paths:
  `analysis/predictability_generalization/stepA_measurement/`, especially
  `summaries/stepA_measurement_summary.md`, `summaries/stepB_readiness.md`, and
  `artifact_manifest.json`.
- Confidence: high for census completeness, action semantics, replay parity,
  utility algebra, and label provenance; unknown for actual Stage-1 or Stage-2
  predictability.
- Consequence for future actions: Step B is measurement-ready, but it must be a
  separately authorized research action. Do not train a probe/router or choose
  a supervision depth from Step-A distributions alone.

## 2026-09-07 — Diagnose the trajectory-set router as corrective-action-underlearned before exposure-limited

- Decision or promoted lesson: do not treat strong route-level geometric
  probability or free-run support departure as evidence that exposure bias is
  the primary Stage-2 bottleneck. First require high top-1 recall for sparse
  corrective actions on exact expert states. For the frozen Phase-76 router,
  change the objective before collecting on-policy labels.
- Triggering evidence: the Phase-77 full-refit seen audit scores all 69,178
  route-state occurrences and obtains 99.47% FULL recall but only 1.83%
  non-FULL and 2.54% first-nonFULL recall. Even the highest-responsibility route
  per W UID reaches only 11.02% first-correction recall. On 463 W UIDs, R0 free
  rollout rescues 42 and R1 exact-state release rescues 43, while R2 forcing
  the first corrective action rescues 264. Thus removing pre-intervention drift
  adds 0.22 points, but executing the missed corrective action adds 47.73
  points. Frozen held-out first-correction recall is likewise 1.79%.
- Qualification: R2 still leaves 199/463 W UIDs wrong, so later exposure,
  incomplete route support, representation, or action-sequence problems may
  remain. Off-support means outside observed successful routes, not invalid.
  This audit does not prove that an auxiliary loss will work or that internal
  gains will transfer externally.
- Evidence paths:
  `analysis/dense_failure_stage2/teacher_forced_free_run_audit/teacher_forced/action_class_metrics.csv`,
  `teacher_forced/highest_responsibility_route_metrics.csv`,
  `release/release_success_summary.csv`,
  `generalization/seen_vs_heldout_summary.csv`, and
  `summaries/teacher_forced_free_run_audit_summary.md`.
- Confidence: high for the exact-state recall, hybrid-release effects, frozen
  held-out comparison, and execution provenance; moderate for calling later
  exposure secondary rather than irrelevant.
- Consequence for future actions: if separately authorized, test one bounded
  objective change that retains the trajectory marginal and adds a fixed-weight
  first-nonFULL auxiliary cross-entropy term. Do not start DAgger/on-policy
  relabeling first, and do not add more route labels as a substitute for action
  recall.
- Revisit condition: a corrective-sensitive objective produces high seen and
  held-out first-nonFULL recall but free rollout remains weak, at which point a
  bounded on-policy exposure experiment becomes justified.

## 2026-09-06 — Retire the old binary Stage-2 treatment labels; preserve expanded action sets

- Decision or promoted lesson: do not train or evaluate another Stage-2
  KEEP-versus-INTERVENE head from the old observed single-route labels. Preserve
  the replay-validated expanded action sets and their MIXED status. Any future
  target change, including set-valued supervision, requires a separately
  approved prospective plan.
- Triggering evidence: a frozen four-GPU audit of 1,200 exact states searched
  all 3,194 unobserved first-action branches. It invalidated 496/500 old KEEP
  labels and 271/500 old INTERVENE labels to MIXED. Final bounded labels are 4
  AUDITED_KEEP, 229 AUDITED_INTERVENE, and 967 AUDITED_MIXED; 1,087/1,200
  states gained at least one action and mean action-set cardinality increased
  from 1.338 to 3.425. All 2,504 discoveries replayed exactly, zero states were
  quarantined, and MCTS discovery met the frozen saturation rule.
- Qualification: these are bounded-search successful-action sets, not exhaustive
  causal treatment labels. The unchanged audited-label probe is not estimable:
  four clean KEEP UID/groups cannot populate both classes in five disjoint test
  folds. Therefore this phase directly supports label incompleteness but does
  not establish whether the frozen RW representation can separate trustworthy
  binary treatment need.
- Evidence paths:
  `analysis/dense_failure_stage2/treatment_label_completeness/metrics/overall_completeness.csv`,
  `labels/old_vs_audited_action_sets.jsonl`, `search/replay_validation.jsonl`,
  `probe_recheck/probe_estimability_note.md`, and
  `summaries/treatment_label_completeness_summary.md`.
- Confidence: high for the sampled-state invalidation rates, exact replay, and
  bounded-search saturation; none for an audited binary AUROC because it is not
  statistically estimable under the frozen split contract.
- Consequence for future actions: do not interpret Phase-72 RW AUROC 0.562 as a
  clean representation test and do not add another binary head. If explicitly
  authorized, prospectively define and validate one set-valued treatment target
  before fitting Stage 2.
- Revisit condition: a new identification protocol yields adequate clean support
  in both binary classes, or a prospectively approved set-valued objective
  changes the treatment question.

## 2026-09-04 — Do not equate MCTS oracle-label yield with learned Stage-2 value

- Decision or promoted lesson: treat the current shared single-label Stage-2
  router as under-generalized and do not run its held-out test. Richer MCTS
  supervision is not automatically beneficial: under a matched architecture,
  optimizer, update budget, C:W mixture, and validation population, it removed
  the small P90 gain observed with single-only supervision. Any next repair
  requires a separately approved prospective plan; do not add more search or
  MCTS labels by default.
- Triggering evidence: Experiment A (preservation+single) achieved P90
  W-to-C/C-to-W/net `4/1/+3`, raising Historical-800 accuracy from 0.5000 to
  0.50375. Experiment B (same system plus 1:1 single:MCTS W supervision)
  produced `0/0/0` at P98, P95, and P90. Final teacher-forced non-FULL recall
  was 0.1920 for A and 0.1687 for B; B used at least one non-FULL action on only
  4.13% of P90-triggered samples. Both overfit gates and full execution audits
  passed, so this is a learned rollout result rather than a failed launch.
- Evidence paths:
  `analysis/dense_failure_stage2/shared_union_training/comparison/A_vs_B.csv`,
  `experiment_A_single/metrics/threshold_comparison.csv`,
  `experiment_B_single_plus_mcts/metrics/threshold_comparison.csv`,
  `summaries/shared_stage2_decision.md`, and
  `workspace/phase_memory/phase_66_stage2_shared_union_training.md`.
- Confidence: high for the matched Historical-800 comparison and exact artifact
  provenance; low for the cause of B's under-generalization. Canonical OOF rows
  contributed Stage-2 supervision and are not a leakage-free Stage-2 holdout.
- Applies when: deciding whether the current shared Stage-2 formulation or more
  MCTS supervision is ready for test/deployment on this three-dataset regime.
- Does not apply when: claiming MCTS search itself lacks oracle value, claiming
  the A/P90 +3 generalizes beyond Historical validation, or diagnosing the
  representation/objective/exposure cause without a targeted experiment.
- Consequence for future actions: preserve A/P90 as a small validation-only
  signal and B/P98 as the plan's conservative tied final-condition point, but
  promote neither to test. A new plan must address learned-policy
  under-generalization rather than merely increasing route count.
- Revisit condition: a prospectively specified objective/representation or
  exposure-correction experiment yields reproducible positive net rollout on a
  genuinely held-out Stage-2 population.

## 2026-09-04 — Keep robust-gate corrective corpora threshold-specific and treat MCTS as essential supervision

- Decision or promoted lesson: preserve separate P98/P95/P90 Stage-2 corpora
  and do not select a deployment threshold from corrective-label yield. New
  corrective supervision is predominantly MCTS-derived, while GQA and late
  trigger handoffs remain substantially less correctable under the bounded
  executor/search contract. A future comparison must use the same Stage-2
  architecture and evaluation semantics across operating points and account
  explicitly for their unequal preservation/corrective support.
- Triggering evidence: shared search resolves 333/1,906 missing pairs, with 55
  new-single and 278 new-MCTS outcomes. Final known-corrective coverage is
  108/344 P98, 234/727 P95, and 463/1,307 P90. P90 GQA coverage is
  0.1946 historical and 0.2544 canonical versus 0.48397 historical TextVQA and
  about 0.533 ChartQA. L19-L27 bounded coverage is only 0.277/0.273/0.276 for
  P98/P95/P90, below the supported middle-depth cells. All 2,519 routes and
  34,253 states passed exact replay and provenance verification.
- Evidence paths:
  `analysis/dense_failure_stage2/robust_gate_corrective_search/metrics/threshold_coverage.csv`,
  `metrics/dataset_source_breakdown.csv`, `metrics/trigger_depth_breakdown.csv`,
  `summaries/threshold_corpus_summary.md`, and `artifact_manifest.json`.
- Confidence: high for population counts, bounded-search outcomes, route replay,
  and state provenance; low for predicting learned end-to-end benefit before a
  matched Stage-2 validation experiment.
- Applies when: designing Stage-2 training for these frozen robust operating
  points and the Historical+Canonical GQA/ChartQA/TextVQA mixture.
- Does not apply when: calling unresolved pairs unfixable, choosing P90 because
  it has the most labels, claiming trigger depth is causal, or transferring
  these rates to unseen benchmarks.
- Consequence for future actions: if separately authorized, first freeze a
  matched threshold-specific Stage-2 training/evaluation design; preservation
  plus single supervision is the clean first comparison, with MCTS as an
  explicit second ablation rather than silently mixing label families.
- Revisit condition: matched Stage-2 rollout results contradict the label-yield
  ordering or a new search/treatment contract materially changes coverage.

## 2026-09-03 — Carry P98/P95/P90 as treatment-dependent operating points and reuse only replay-verified routes

- Decision or promoted lesson: keep the Phase-63 five-head ALL-source scoring
  system fixed and carry P98/P95/P90 as conservative, middle, and permissive
  candidates. None is the final threshold. For future Stage-2 reconstruction,
  reuse only routes whose first intervention is at or after the new trigger and
  whose exact current-runtime replay is LMMS-correct; search only the remaining
  triggered-W population.
- Triggering evidence: held-out worst-source C preservation is
  0.9808/0.9501/0.9003 and pooled W recall is 0.1054/0.2179/0.3572. On the
  10,399 train rows, P98/P95/P90 trigger 344/727/1,307 W; 67/146/259 already
  have replay-compatible routes. All 2,618 route×operating-point replays pass
  exact token parity and current LMMS correctness. Residual workloads are
  277/581/1,048 W, with 1,104 unique UIDs across the three points.
- Evidence paths:
  `analysis/dense_failure_stage1/robust_operating_points_and_compatibility/thresholds/named_operating_points.csv`,
  `metrics/operating_point_summary.csv`, `metrics/route_reuse_summary.csv`,
  `compatibility/replay_results.jsonl`, and
  `summaries/next_search_recommendation.md`.
- Confidence: high for the bound artifacts, trigger maps, structural checks,
  and exact replay compatibility. Train admission rates are in-sample
  five-head-ensemble descriptions and must not be substituted for held-out
  calibration estimates.
- Applies when: rebuilding Stage-2 supervision for the frozen ALL-source gate
  on the Historical+Canonical GQA/ChartQA/TextVQA populations.
- Does not apply when: selecting a final deployment threshold from these label
  counts, treating unsearched rows as unfixable, or transferring the gate to a
  new benchmark/source regime.
- Consequence for future actions: a separately authorized corrective-search
  plan should deduplicate the 1,104-UID missing union while preserving each
  operating point's trigger-layer handoff. Final threshold choice requires
  measured W-to-C benefit and C-to-W harm under actual Stage-2 treatment.
- Revisit condition: new search finds materially different correctability by
  operating point, replay semantics change, or held-out source evidence changes
  the preservation frontier.

## 2026-09-03 — Freeze the conservative ALL-source gate only for the observed three-benchmark mixture

- Decision or promoted lesson: use the Phase-62 ALL-source Shared Random-4
  system with one strict global threshold `0.9711347410314399` for subsequent
  in-scope trigger-map compatibility work. This is a conservative gate for the
  observed Historical+Canonical GQA/ChartQA/TextVQA mixture, not evidence of
  benchmark-universal calibration.
- Triggering evidence: the prospectively selected threshold preserves
  Historical/Canonical Dense-C at 98.75%/98.08% while recalling 6.50%/12.40%
  of Dense-W (10.54% pooled) with 67.34% pooled trigger precision. The untouched
  Historical test preserves 99.75% C with 6.50% W recall and 96.30% precision.
  Five canonical cross-fit thresholds span only 0.96859-0.97220 and no supported
  dataset/source cell has C preservation below 90%.
- Evidence paths:
  `analysis/dense_failure_stage1/all_source_threshold_calibration/frozen/robust_stage1_gate.json`,
  `metrics/reference_operating_points.csv`, `crossfit/threshold_stability.csv`,
  `metrics/dataset_source_breakdown.csv`, and
  `summaries/threshold_calibration_summary.md`.
- Confidence: high in the exact artifact-level threshold calculation,
  Historical validation/test behavior, and current Canonical OOF estimate;
  low for Canonical TextVQA benefit because the gate detects 0 of only 19 W.
  Canonical OOF rows are single held-out-fold predictions whereas deployment
  uses the named five-head probability ensemble, so the next trigger-map phase
  must preserve and audit that exact scoring provenance.
- Applies when: regenerating the Stage-1 trigger map or auditing compatibility
  of prior Stage-2 corrective labels within the declared three datasets.
- Does not apply when: claiming unseen-dataset calibration, creating
  dataset-specific thresholds, relaxing to the 95% point, or reusing old
  trigger layers/labels without a compatibility audit.
- Consequence for future actions: the next separately authorized action is a
  new robust trigger map followed by old/new sample-and-layer compatibility
  accounting. Do not automatically launch new corrective search or Stage-2
  training.
- Revisit condition: exact ensemble trigger-map scoring violates the calibrated
  preservation behavior, a supported dataset/source cell becomes catastrophic,
  or new held-out source evidence materially changes the operating point.

## 2026-09-03 — Use the ALL-source head only for the declared mixture; do not claim benchmark-general transfer

- Decision or promoted lesson: Exact source×correctness-balanced Historical +
  Canonical training produces a useful shared Stage-1 boundary across the two
  observed source regimes, but leave-one-dataset-out transfer remains mixed.
  Treat this head as a candidate for the current three-benchmark mixture, not a
  universal failure detector.
- Triggering evidence: The ALL-source head reaches Historical/Canonical AUROC
  0.8394/0.7689 and worst-source 0.7689, versus specialist worst-source AUROCs
  0.4056 and 0.5511. Per-source dataset AUROC is Historical
  0.7370/0.9487/0.9488 and Canonical 0.7538/0.5826/0.6111 for
  GQA/ChartQA/TextVQA. Target-blind LODO Historical/Canonical AUROC is
  0.7016/0.5691 ChartQA, 0.6243/0.6259 TextVQA, and 0.6034/0.5931 GQA, failing
  the frozen all-target 0.60 worst-source rule.
- Evidence paths:
  `analysis/dense_failure_stage1/all_source_robustness/summaries/all_source_training_summary.md`,
  `summaries/dataset_ood_summary.md`, `main_all/metrics/source_summary.csv`,
  `main_all/metrics/dataset_source_breakdown.csv`, `lodo/lodo_summary.csv`, and
  `workspace/phase_memory/phase_62_stage1_all_source_robustness.md`.
- Confidence: high for aggregate source robustness, exact sampling/provenance,
  and the observed mixed OOD results; low for Canonical TextVQA precision
  because it has only 19 wrong samples. L26's minimum 0.6270 across six OOD
  cells is post-hoc diagnostic evidence, not an authorized layer selection.
- Applies when: choosing a Stage-1 substrate for the current Historical+
  Canonical GQA/ChartQA/TextVQA mixture or describing its scope.
- Does not apply when: claiming unseen-benchmark generalization, calibrating a
  threshold, selecting L26 from these target results, or regenerating Stage-2
  labels without a separate compatibility decision.
- Consequence for future actions: if separately authorized, calibrate a robust
  threshold only on the declared ALL mixture and retain dataset×source safety
  reporting. Do not automatically generate a trigger map or modify Stage 2.
- Revisit condition: a prospectively frozen common-layer experiment or new
  unseen benchmark contradicts the current OOD limitation, or threshold
  calibration cannot maintain required safety across dataset×source cells.

## 2026-09-03 — Keep the Stage-1 architecture, but treat fitted boundaries as source-regime-specific

- Decision or promoted lesson: The historical Shared Random-4 feature/head
  architecture can learn canonical current-runtime dense-failure ranking for
  GQA and ChartQA without changing the old normalization. The main old-gate
  deployment failure was therefore its historical fitted boundary/population
  regime, not a demonstrated inability of the architecture to encode canonical
  failure. Do not, however, deploy the canonical-only fit as a universal gate:
  reverse transfer to the historical population is weak, and TextVQA remains
  unresolved.
- Triggering evidence: Five-fold group-disjoint canonical refitting raises
  overall max-score AUROC from 0.4056 to 0.8178, a paired +0.4123 with 95% CI
  [+0.3862,+0.4376]. GQA/ChartQA OOF AUROC is 0.7214/0.7021, removing the old
  ChartQA inversion. TextVQA OOF AUROC is 0.4640 with only 19 W. The
  full-canonical fit reaches only 0.5511 on frozen historical val+test versus
  0.8885 for the old head. Frozen-old score reproduction differs by at most
  6.22e-08.
- Evidence paths:
  `analysis/dense_failure_stage1/canonical_refit_diagnostic/summaries/canonical_refit_summary.md`,
  `metrics/canonical_oof_summary.csv`, `metrics/canonical_dataset_breakdown.csv`,
  `metrics/paired_bootstrap_difference.csv`, `metrics/historical_cross_eval.csv`,
  and `workspace/phase_memory/phase_61_stage1_canonical_refit_diagnostic.md`.
- Confidence: high that the unchanged architecture can rank canonical GQA and
  ChartQA failures and that source-specific fitted boundaries transfer poorly;
  low for TextVQA because only 19 canonical wrong examples exist. Overall AUROC
  also reflects cross-dataset prevalence differences, so within-dataset results
  are required alongside it.
- Applies when: selecting the next Stage-1 repair direction for the historical
  and canonical GQA/ChartQA/TextVQA populations.
- Does not apply when: claiming TextVQA recovery, claiming old normalization is
  irrelevant, treating OOF probabilities as calibrated deployment scores, or
  reusing either source-specific checkpoint as a final mixed-source gate.
- Consequence for future actions: if separately authorized, test one unchanged
  head with explicitly source-balanced historical + canonical training and
  group-disjoint evaluation in both regimes. Do not calibrate a new threshold
  or regenerate Stage-2 labels until cross-regime ranking is established.
- Revisit condition: source-balanced mixed training fails within GQA/ChartQA,
  or a canonical-normalization Arm B materially changes the unresolved
  TextVQA/calibration behavior.

## 2026-09-03 — Treat old Stage-1 success as same-selection-regime evidence, not canonical robustness

- Decision or promoted lesson: The historical 7,999 Stage-1 train/validation/
  test population was constructed from fixed previous-Qwen Dense-C/Dense-W
  quotas. Its group-disjoint validation/test performance demonstrates identity
  generalization within that selected regime, not robustness to an outcome-
  blind canonical source. Before changing architecture, test whether the same
  head can relearn canonical-label ranking with the frozen old normalization.
- Triggering evidence: Within current Dense-C, group-held-out historical-vs-
  canonical L21 source AUROC is 0.586 GQA, 0.938 ChartQA, and 0.937 TextVQA;
  canonical-minus-old-C frozen max-score means shift by +0.089/+0.689/+0.712.
  Old test max-score AUROC falls only from 0.886 to 0.840 with exact visual-
  token matching and 0.820 with fixed multi-variable matching, so observable
  nuisances contribute but do not explain all same-regime ranking. Canonical
  correct ChartQA/TextVQA states also move toward the historical wrong feature
  direction.
- Evidence paths:
  `analysis/dense_failure_stage1/historical_population_shortcut_audit/summaries/old_head_shortcut_summary.md`,
  `metrics/source_probe.csv`, `metrics/matched_head_performance.csv`,
  `metrics/feature_geometry.csv`, and
  `workspace/phase_memory/phase_60_stage1_historical_population_shortcut_audit.md`.
- Confidence: high for population construction, feature accessibility, score
  association, and matched observational results; medium for the mixed
  shortcut/boundary interpretation; no causal claim that token count or source
  identity is directly used by the head.
- Applies when: interpreting the old gate's in-regime validation/test success or
  selecting the smallest next Stage-1 repair diagnostic for these three tasks.
- Does not apply when: claiming the Stage-1 representation is unusable,
  identifying a sole causal nuisance, or asserting that normalization cannot
  matter. The original pre-quota pool is absent, so causal selection rates are
  unavailable.
- Consequence for future actions: if separately authorized, first fit the same
  head on canonical training labels while retaining old normalization. Only if
  that fails should a canonical train-fold normalization arm be added; defer
  source-mixed training until the simpler boundary test is resolved.
- Revisit condition: a canonical-label same-head fit fails to recover held-out
  ranking or a controlled nuisance intervention contradicts the observational
  shortcut evidence.

## 2026-08-31 — Prefer fixed L27 over sequential gates for the first treatment-feasibility test

- Decision or promoted lesson: Under the prospectively frozen aggregate
  admission utility and 90% failure-precision floor, carry the shared
  Random-4 predictor's fixed layer-27 score at threshold
  `0.8497647428417646` into the next separately authorized treatment-feasibility
  experiment. Do not pay for sequential gating when it provides no measured
  utility advantage under these controls.
- Triggering evidence: Fixed L27 led validation by only three net utility
  samples, with preservation/recall/precision/utility rate
  `0.9400/0.5400/0.9000/0.2400`, and transferred to
  `0.9500/0.5275/0.9134/0.23875` on test. Independent sequential tied its test
  utility rate exactly; shared Random-4 and All-28 reached `0.2250` and
  `0.2175`. Against the validation runner-up, 95% paired-bootstrap intervals
  span zero for both test utility-rate difference `[-0.00375, 0.03125]` and
  wrong-recall difference `[-0.0125, 0.0475]`.
- Evidence paths:
  `analysis/dense_failure_stage1/gate_winner_selection/decision_summary.md`,
  `validation_summary.csv`, `test_comparison.csv`,
  `bootstrap_comparison.json`, and
  `workspace/phase_memory/phase_52_stage1_gate_winner_selection.md`.
- Confidence: medium for the fixed candidate ranking and high for exact
  reproduction of the frozen criterion. Fixed L27 had already been selected
  among L14/L21/L27 on Phase-51 validation, so this is not an unbiased claim
  over all possible fixed depths.
- Applies when: choosing the one Stage-1 admission substrate and operating
  point for the next matched four-action treatment-feasibility experiment on
  the same current-runtime three-dataset mixture.
- Does not apply when: claiming fixed L27 is universally superior, claiming
  early failure signal is useless, transferring the threshold to a new task,
  or treating this admission threshold as the final routed-system threshold.
- Consequence for future actions: Retain the frozen winner plus its 99/98/95
  reference points, and judge the eventual system by actual W-to-C rescue,
  C-to-W regression, and routed accuracy. Keep independent and shared
  sequential gates as baselines, not co-winners.
- Revisit condition: a prospective treatment experiment shows that earlier
  admission materially improves rescue/safety, or task-calibration evidence
  changes the aggregate deployment objective. Winner test utility is highly
  heterogeneous (`0.0775/0.3850/0.4150` on GQA/ChartQA/TextVQA), with only
  `0.7460` GQA precision.

## 2026-08-31 — Shared scores repair aggregate preservation drift but not task calibration

- Decision or promoted lesson: One shared state predictor and a single
  trajectory-level raw-score threshold can stabilize aggregate conservative
  preservation, but this is not sufficient for a robust Stage-1 admission
  gate. Do not connect the Phase-51 gate to visual treatment because its
  cross-dataset failure coverage remains strongly heterogeneous.
- Triggering evidence: The validation-designated state+layer Random-4 gate at
  the 99% point transfers from 0.9900/0.3950 preservation/recall on validation
  to 0.9875/0.4050 on test, reducing preservation drift from the independent
  gate's 0.0425 to 0.0025. Yet GQA/ChartQA/TextVQA test wrong recall is
  0.040/0.740/0.800, a 0.760 spread versus Phase 50's already-large 0.725.
  Preservation spread is 0.030 versus 0.025. Mean test layer AUROC remains
  strong at 0.8769 versus 0.8737 for independent probes, so the negative gate
  decision is not caused by loss of ranking signal.
- Evidence paths:
  `analysis/dense_failure_stage1/shared_global_gate/decision_summary.md`,
  `evaluation/test_results.csv`, `evaluation/dataset_breakdown.csv`, and
  `workspace/phase_memory/phase_51_shared_stage1_global_risk_gate.md`.
- Confidence: high for this exact current three-dataset mixture, compact
  representation, pooled train-only normalization, shared MLP, and frozen
  global-threshold protocol; none for another calibration objective or new
  benchmark family.
- Applies when: deciding whether the current shared predictor/global threshold
  is ready to admit samples into four-action treatment.
- Does not apply when: claiming shared predictors cannot match independent
  ranking, because mean AUROC is matched; claiming layer identity is necessary,
  because state-only validation AUROC is essentially equal; or ruling out a
  separately specified calibration/risk-control method.
- Consequence for future actions: preserve shared state prediction as an
  informative substrate but reject this raw global-threshold gate for
  treatment. Any new risk-control phase requires explicit authorization and
  must target sample-level task heterogeneity prospectively without post-hoc
  dataset thresholds.
- Revisit condition: a prospectively frozen calibration or risk-control method
  reduces dataset preservation/recall mismatch while retaining aggregate
  conservative preservation on untouched data.

## 2026-08-31 — Do not treat independent per-layer tail calibration as a robust sequential gate

- Decision or promoted lesson: Reusing the 28 independent Phase-48 probes with
  one shared validation-correct tail parameter produces early, nontrivial
  failure triggers, but the union of layer crossings is not a robust
  conservative gate. Do not connect this V1 gate to treatment or treat it as a
  final admission mechanism.
- Triggering evidence: Validation selected `alpha=0` for the 99% and 98%
  targets, yielding 1.000 correct preservation and 0.4100 wrong recall, but the
  frozen thresholds transferred to only 0.9575 preservation on untouched test
  with 0.4175 wrong recall. The 95% point reached 0.9550/0.4775 on test.
  Sequential recall exceeded the best fixed L14/L21/L27 comparator by only
  +0.0200 at the 99% target and was lower by 0.0225/0.0500 at 98%/95%.
  Dataset test wrong-recall spread was 0.7250 at the 99%/98% points and 0.6650
  at 95%, with GQA far below ChartQA and TextVQA.
- Evidence paths:
  `analysis/dense_failure_stage1/independent_sequential_gate/decision_summary.md`,
  `test_results.csv`, `dataset_breakdown.csv`, `single_layer_comparison.csv`,
  and `workspace/phase_memory/phase_50_independent_sequential_gate.md`.
- Confidence: high for this exact known three-dataset mixture, Phase-48 split,
  frozen independent probes, empirical higher-quantile rule, and first-trigger
  policy; none for another calibration family or jointly trained gate.
- Applies when: deciding whether to deploy or connect the current independent
  layer-wise sequential gate to an intervention.
- Does not apply when: claiming that early failure signal is absent (median
  detected trigger is layer 3 at the conservative point), that all sequential
  models must fail, or that a shared/global risk-budget method has been tested.
- Consequence for future actions: preserve this as a negative admission-policy
  result. A shared predictor or global risk-budget formulation would be a new,
  separately authorized action and must address sample-level union risk and
  dataset calibration explicitly.
- Revisit condition: a prospectively specified joint/sample-level calibration
  method meets conservative preservation on untouched data without
  dataset-specific post-hoc tuning.

## 2026-08-31 — Do not promote in-domain failure accessibility to a benchmark-general predictor claim

- Decision or promoted lesson: The strong Phase-48 in-domain linear signal is
  not sufficient evidence for a benchmark-general, calibration-stable dense-
  failure predictor. Preserve the leave-one-dataset-out result as mixed: useful
  mid-layer ranking transfers to TextVQA and ChartQA and weakly to GQA, but a
  source-validation-selected layer and conservative threshold do not transfer
  reliably across all three targets.
- Triggering evidence: Source-only selection chose layers 22/26/20 for held-out
  TextVQA/ChartQA/GQA. Their full-target AUROCs were 0.7236/0.4502/0.6160,
  compared with native pre-language-decoder AUROCs 0.5219/0.5472/0.5432.
  Descriptively, fixed layer 21 reached 0.8046/0.8018/0.5975, so the signal is
  not absent, but transfer is nonuniform and the source-optimal depth can be
  anti-predictive on a target. Source-calibrated 99%-preservation thresholds
  achieved actual target preservation 0.9950/0.3053/0.1380; only TextVQA
  retained the intended conservative operating regime.
- Evidence paths:
  `analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md`,
  `input_only_vs_hidden.csv`, `in_domain_vs_ood.csv`, and
  `workspace/phase_memory/phase_49_ood_failure_signal_diagnostic.md`.
- Confidence: high for these three leave-one-dataset-out transfers under the
  frozen current model, compact representation, current LMMS labels, and
  Phase-48 split memberships; low for broader benchmark families or another
  backbone/runtime.
- Applies when: motivating a shared Stage-1 predictor as benchmark-general,
  selecting layers across tasks, or transferring conservative failure-gating
  thresholds to an unseen task.
- Does not apply when: claiming no hidden-state failure signal exists, because
  mid-layer OOD ranking remains strong on two targets; evaluating a predictor
  explicitly trained for the known deployment mixture; or extrapolating to
  external benchmark families not tested here.
- Consequence for future actions: do not proceed directly to the shared
  predictor with learnable layer embeddings on a benchmark-general rationale.
  A later experiment requires explicit authorization and must state whether it
  targets the known three-dataset deployment mixture or introduces a separate
  prospective mechanism for cross-task layer/calibration robustness.
- Revisit condition: a prospectively fixed shared-depth/calibration protocol,
  additional task families, or a deployment-mixture objective supplies
  decision-changing evidence without target tuning.

## 2026-08-30 — Treat dense failure as linearly accessible across the full stack

- Decision or promoted lesson: For the frozen 7,999-sample current-dense
  population and concatenated `text_final`/`text_mean`/`visual_mean` compact
  representation, final LMMS failure is linearly predictable from layer 0.
  Use layers 0-27 as the informative range; layers 16-27 are a stronger
  descriptive plateau, not evidence that early layers should be excluded.
- Triggering evidence: on the untouched image-group-disjoint 800-record test
  split, layer-0 AUROC/AUPRC are 0.8421/0.8519 and peak AUROC is 0.8992 at
  layer 21. All datasets are above chance at layer 0, although GQA is weaker
  (0.6981; peak 0.7713) than ChartQA/TextVQA (about 0.94 at layer 0; peaks above
  0.97). Validation-threshold transfers that still satisfy test correct
  preservation detect 33.50%/42.75%/52.25% of wrong samples at 99%/98%/95%
  preservation.
- Evidence paths:
  `analysis/dense_failure_stage1/layerwise_failure_probe/analysis_summary.md`,
  `test_metrics.csv`, `dataset_layerwise_metrics.csv`, and
  `workspace/phase_memory/phase_48_layerwise_dense_failure_predictability.md`.
- Confidence: high for held-out linear accessibility under this exact split,
  feature contract, label population, and regularized probe protocol; no claim
  about causal failure awareness or another representation/model.
- Applies when: choosing supervision depths for the next shared Stage-1
  predictor on the current compact dense-state dataset.
- Does not apply when: claiming explicit answers form early, attributing signal
  to one feature component, transferring precision/AUPRC to natural benchmark
  prevalence, or choosing an intervention policy without execution evidence.
- Consequence for future actions: do not compare all-layer supervision with
  “informative-only layers 0-27,” because they are the same arm. If separately
  authorized, prioritize all-layer versus random-k over 0-27; a fixed 16-27
  arm may be included only as a stronger-plateau efficiency sensitivity.
- Revisit condition: component ablation, another model/runtime/population, or a
  shared-predictor experiment supplies decision-changing generalization or
  efficiency evidence.

## 2026-08-30 — Treat answer emergence as late and keep raw logits separate from generation processors

- Decision or promoted lesson: For the frozen Qwen2.5-VL Stage-1 population,
  answer-token emergence under the final norm/head lens is a late-stack event.
  Layers 25–27, especially 26–27, are the only evidence-backed candidate region
  from this diagnostic. Early zero-threshold sign runs must not be interpreted
  as formed answers, and this result does not itself select a supervision
  strategy or prove that early hidden states lack correctness information.
- Triggering evidence: At the true assistant answer-start state, correct GT
  first reaches raw top-1 at median layer 26 (IQR 26–27), with the population
  top-1 fraction rising 22.3% → 47.1% → 94.9% at layers 25–27. For wrong
  samples, the eventual generated token reaches rank ≤10 for 64.1%, 86.8%, and
  100% at layers 25–27, while its raw top-1 fraction reaches 20.3%, 42.9%, and
  91.2%. Earlier wrong GT-minus-generated margins are close to zero with both
  tokens ranked in the tens of thousands, even though the fixed sign-crossing
  rule reports median layer 2.
- Evidence paths:
  `analysis/dense_failure_stage1/answer_logit_emergence_v2/analysis_summary.md`,
  `layerwise_correct_summary.csv`, `layerwise_wrong_summary.csv`, and
  `workspace/phase_memory/phase_47_answer_position_logit_emergence_v2.md`.
- Confidence: high for the descriptive raw-logit trajectories under this exact
  model/runtime and answer-position contract; moderate for using 25–27 as a
  candidate supervision region; none for a strategy choice not yet tested.
- Applies when: deciding the next Stage-1 supervision-depth comparison for the
  current 7,999-sample dense population.
- Does not apply when: claiming the full early hidden state is uninformative,
  transferring the depth boundary to another model, or treating raw top-1 as
  identical to a token selected after generation processors.
- Consequence for future actions: if separately authorized, design the next
  comparison around a prospectively fixed late candidate range rather than
  using the literal early sign-crossing median. At layer 27 use the model's
  native raw `output.logits`; for cached continuations validate raw logits and
  processed generated tokens in their own score spaces.
- Revisit condition: a learned failure-prediction experiment, alternate fixed
  margin rule, or different model/runtime provides decision-changing evidence.

## 2026-08-30 — Do not use final-user-token states as answer-start logit lenses

- Decision or promoted lesson: The Phase-45 `text_final` feature is the final
  literal user-prompt token and must not be treated as the assistant answer-start
  position for an unfiltered final-head vocabulary competition. It may remain a
  predictor input, but answer-logit emergence requires an actually aligned
  answer-start state or a separately justified readout contract.
- Triggering evidence: Applying the exact frozen Qwen2.5-VL final RMSNorm and
  LM head to all 7,999 records produced zero persistent correct GT-vs-strongest-
  token emergence events among 3,999 correct samples. The layer-27 correct
  margin averaged -29.683. In the one allowed cheap validity diagnostic,
  `<|im_end|>` was top-1 at layer 27 for all 54 correct records in a frozen
  source shard, as expected at the token immediately preceding the chat
  delimiter.
- Evidence paths:
  `analysis/dense_failure_stage1/logit_emergence/analysis_summary.md`,
  `analysis/dense_failure_stage1/logit_emergence/validity_diagnostic.json`, and
  `workspace/phase_memory/phase_46_dense_answer_logit_emergence.md`.
- Confidence: high that the correct-sample strongest-token comparator is
  positionally invalid; moderate for the descriptive wrong GT-vs-generated
  token trajectories, which remain a query-position relative readout.
- Applies when: Interpreting or designing answer-logit analyses from the
  Phase-45 saved feature tensors.
- Does not apply when: Claiming the hidden state itself lacks correctness
  information, rejecting `text_final` as a learned-predictor input, or
  interpreting a genuinely assistant-start-aligned state.
- Consequence for future actions: Do not choose a Stage-1 supervision start
  layer from Phase-46 curves. If answer-emergence evidence is still required,
  prospectively extract the actual assistant-start state before comparing GT
  tokens with vocabulary competitors.
- Revisit condition: A frozen answer-position extraction shows exact token
  alignment and a valid strongest-competitor interpretation.

## 2026-08-30 — Use current native-dense LMMS labels for Stage 1

- Decision or promoted lesson: The authoritative Stage-1 target population is
  the current native Qwen2.5-VL dense all-on output scored by the official
  task-specific LMMS-Eval contracts. Historical correct/wrong buckets remain
  metadata even when they happen to agree with the regenerated labels.
- Triggering evidence: The four-GPU run attempted all 8,000 recovered
  GQA/ChartQA/TextVQA candidates. It completed 7,999 with 3,999 current-correct
  and 4,000 current-wrong outcomes; one ChartQA image was missing. An independent
  LMMS rescore found zero score or binary-label mismatches. Fresh generated text
  exactly matched the historical stored dense prediction for all 7,999
  executable samples, so there were zero bucket flips under this exact dense
  model/runtime.
- Evidence paths: `analysis/dense_failure_stage1/current_dense_8k/dense_outputs.jsonl`,
  `analysis/dense_failure_stage1/current_dense_8k/generation_summary.json`,
  `analysis/dense_failure_stage1/current_dense_8k/lmms_eval_contract.md`, and
  `workspace/phase_memory/phase_45_current_dense_8k_lmms.md`.
- Confidence: high for this exact snapshot, prompt, dense executor, and
  executable recovered population.
- Applies when: Defining labels and data membership for the next Stage-1 dense
  failure-predictor experiment.
- Does not apply when: Interpreting four-action/sparse route replay parity,
  claiming parity under another model/runtime, or treating the one missing
  sample as executed.
- Consequence for future actions: Join predictor inputs through the frozen
  `features/feature_index.jsonl` and labels through `dense_outputs.jsonl`; use
  image-group-disjoint splits over the 7,999 completed records. Do not recover,
  rebalance, or overwrite labels from historical buckets.
- Revisit condition: The model snapshot, prompt, generation policy, LMMS task
  semantics, or completed sample population changes.

## 2026-08-30 — Audit mandatory-boundary FULL validity before gate training

- Decision or promoted lesson: Do not treat a mandatory W2C boundary inferred
  from discovered correct routes as proof that `FULL` is invalid. Before
  training another CONTINUE/DEVIATE gate from these labels, repair or expand
  compatible-continuation coverage, rebuild WHEN labels, and repeat a frozen
  FULL-insertion audit.
- Triggering evidence: The complete 128-state held-out W2C census executed all
  252 deduplicated routes induced by every compatible frozen suffix. Forced
  FULL yielded a correct bounded continuation for 39/128 states (30.47%;
  10,000-draw 95% UID-bootstrap CI [22.66%, 38.28%]), across ChartQA, GQA, and
  TextVQA. Zero states were unresolved.
- Evidence paths:
  `analysis/selective_continue_deviate/when_label_completeness_report.md`,
  `analysis/selective_continue_deviate/when_full_insertion_executions.jsonl`,
  and `workspace/phase_memory/phase_41_selective_continue_deviate.md`.
- Confidence: high that the current frozen WHEN cache is materially incomplete
  under the bounded known-suffix executor; unknown why cache discovery omitted
  these continuations or how many unobserved continuations remain.
- Applies when: Constructing or evaluating binary KEEP/DEVIATE supervision from
  the current four-action W2C boundary cache.
- Does not apply when: Claiming FULL is globally valid at the rescued states,
  estimating natural prevalence outside the frozen held-out cohort, or
  diagnosing the discovery failure's cause.
- Consequence for future actions: Do not train the planned linear/MLP gate on
  the current labels or relax the clean 128-positive validation contract after
  seeing this result. A label-repair phase requires explicit authorization.
- Revisit condition: A prospective repaired cache supplies at least 128 trusted
  held-out DEVIATE positives and passes a new exhaustive bounded audit.

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

## 2026-09-04 — Expand the standing external-evaluation scope to four benchmark families

- Decision or promoted lesson: interpret future user requests for external
  evaluation as ChartQA, TextVQA, MMMU-Pro, and POPE unless the user explicitly
  narrows or replaces the scope for that action.
- Required sub-suites: report MMMU-Pro Standard and Vision separately, and
  POPE adversarial, popular, and random separately.
- Triggering evidence: explicit user instruction on 2026-09-04 to add TextVQA
  to the prior ChartQA/MMMU-Pro/POPE evaluation set.
- Supersedes: the 2026-08-27 three-family restriction above only with respect
  to TextVQA. DocVQA, MMStar, and base MMMU remain excluded.
- Operational qualification: this standing scope does not itself authorize an
  evaluation run. Before execution, freeze and verify the TextVQA evaluation
  split, sample manifest, images, LMMS task evaluator, and provenance. Do not
  substitute TextVQA training-label samples for the evaluation benchmark.
- Confidence: high; this is a user-defined scope decision.

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

## 2026-08-29 — Isolated coverage and shortcut fixes do not select a four-action architecture

- Decision or promoted lesson: do not select either the online state-conditioned
  router or the upfront POLAR router from the current recipes, and do not treat
  a small-subset overfit result as population-level rescue. One scheduled
  boundary visit per W2C sample and removal of the exact C2C all-FULL route are
  each insufficient to break held-out all-FULL collapse.
- Triggering evidence: A2 activated exactly 2,397/2,397 mandatory-boundary
  visits but had zero validation boundary Valid@1 and zero W2C rescue at every
  epoch. B1 removed 3,501 exact all-FULL train-C2C routes yet its selected
  checkpoint executed all-FULL on 866/866 with zero W2C rescue. The matched
  probe found upfront/online AUROC 0.5764/0.5751 and online-minus-upfront 95%
  CI [-0.0548, 0.0534], so it supplies no evidence for an online-state
  advantage.
- Evidence paths: `analysis/4action_collapse/online_boundary_coverage_v2_report.md`,
  `analysis/4action_collapse/polar_c2c_no_allfull_report.md`,
  `analysis/4action_collapse/upfront_vs_online_boundary_probe_report.md`, and
  `analysis/4action_collapse/decision_summary.md`.
- Confidence: high that both isolated interventions are insufficient and
  medium that neither substrate should currently be prioritized; richer state
  summaries and persistent targeted supervision remain untested.
- Consequence for future actions: do not rerun A2 or B1 unchanged and do not
  start external evaluation from their selected checkpoints. A future
  comparison, if explicitly authorized, should match persistent targeted
  W2C/non-FULL supervision mass across both substrates and select on held-out
  W2C rescue plus C2C preservation rather than C2C-dominated overall route
  membership.

## 2026-08-30 — Persistent correction yields small rescue but no architecture advantage

- Decision or promoted lesson: Persistent mandatory-boundary supervision can
  break exact zero-rescue behavior for both fixed four-action recipes, but the
  observed held-out gains are small and do not support choosing the online
  state-conditioned architecture over upfront POLAR. Use the prospective
  operational tie-break to prefer POLAR for this comparison only.
- Triggering evidence: On the same frozen 128-W2C/128-C2C validation set,
  POLAR epoch 15 rescued 7 W2C and preserved 124 C2C, while online epoch 14
  rescued 6 and preserved 122. The paired online-minus-POLAR rescue difference
  was -0.0078125 with 10,000-draw 95% bootstrap interval
  [-0.0625, 0.0390625]. Both were trained on the same 512-W2C/512-C2C subset
  for 20 epochs with every W2C boundary supervised every epoch.
- Validity qualification: one frozen C2C record failed a current-runtime direct
  all-FULL check. Excluding it gives POLAR 124/127 and online 122/127 C2C
  preservation and leaves both selected epochs and the architecture decision
  unchanged; the drift cause remains unknown.
- Evidence paths:
  `analysis/persistent_corrective_supervision/decision_summary.md`,
  `analysis/persistent_corrective_supervision/matched_comparison.md`, and
  `analysis/persistent_corrective_supervision/runtime_cohort_sensitivity.md`.
- Confidence: high for the fixed matched recipes and frozen decision rule;
  low for any claim of robust population-level correction or architectural
  impossibility.
- Consequence for future actions: Do not interpret train-boundary fit or
  nonzero rescue alone as robust generalization, do not claim an online feature
  advantage, and do not open external evaluation for these checkpoint
  families without a new prospective authorization.

## 2026-08-30 — Four-action held-out failure is dominated by WHEN, with incomplete cached WHAT labels

- Decision or promoted lesson: for the frozen persistent-supervision routers,
  treat failure to recognize held-out mandatory departures from `FULL` as the
  dominant deployed decision failure. Do not infer that the online state lacks
  all transferable signal, and do not treat every action absent from the
  discovered valid-route cache as execution-invalid.
- Triggering evidence: exact split/dataset/layer matching gives POLAR/online
  validation KEEP-vs-DEVIATE AUROC 0.542877/0.507751 and argmax deviation
  recall 0.054688/0.148438, versus train AUROC 0.915585/0.994514. A fresh
  prespecified linear probe on the frozen online state reaches 0.737976 WHEN
  AUROC, while the trained router remains at 0.507751. In the bounded selected
  cached-invalid audit, 6/14 states have an execution-correct supposedly
  invalid non-`FULL` action under a compatible known suffix.
- Mechanism qualification: READ_OFF and WRITE_OFF validation discrimination is
  weak, IGNORE-only recall is zero for both routers, and k=10 exact mechanism
  purity is only 0.371--0.431. The label audit is conditional and small; it
  proves incompleteness for six states but does not estimate global prevalence
  or show that WHAT incompleteness causes WHEN failure.
- Evidence paths:
  `analysis/4action_generalization_diagnostics/decision_summary.md`,
  `when_keep_vs_deviate.csv`, `representation_probe_results.json`, and
  `label_incompleteness_results.json` in the same directory.
- Confidence: high for the frozen matched state population and selected
  checkpoints; medium for future-action ranking; unknown for the exact causal
  optimizer/head/objective subcomponent.
- Consequence for future actions: before a new two-stage head or broader route
  repair, prospectively test WHEN-label completeness by inserting `FULL` at a
  bounded stratified set of mandatory boundaries with compatible known
  suffixes. This is a recommendation, not authorization.

## 2026-08-30 — Separate source-contract recovery from numerical replay parity

- Decision or promoted lesson: when transferring discrete four-action labels,
  preserve per-file source hashes and distinguish a semantic executor change
  from a hardware/runtime numerical-path change. A changed aggregate code hash
  is not sufficient evidence that READ/WRITE semantics changed, and exact source
  recovery is not sufficient evidence of cached-token parity on another GPU
  architecture.
- Triggering evidence: the label contract reconstructs all 16/16 historical
  source files and the frozen YAML exactly from a dirty worktree at recorded
  `HEAD` `a3c6a411...`. The historical fixed-route action implementation is
  scientifically valid and source-equivalent to the current fixed path, yet the
  RTX 6000 Ada replay differs from 41/312 cached token sequences. The
  discriminating recovered-source H100 replay was canceled before allocation,
  so H100-versus-Ada numerical drift remains suspected rather than supported.
- Evidence paths:
  `analysis/executor_provenance_audit/executor_provenance_audit.md`,
  `historical_vs_current_executor_diff.md`, and `replay_parity_report.md`.
- Confidence: high for exact source recovery and semantic equivalence; unknown
  for the end-to-end replay cause.
- Consequence for future actions: before repairing or relabeling, rerun the
  fixed 12-sample/312-route smoke under the recovered H100 contract. Do not
  silently drop mismatching cached routes or attribute the mismatch to source
  semantics without new evidence.
## 2026-08-31 — Failure detection modestly enriches visual-treatment opportunity, but late replay remains competitive

- Decision or promoted lesson: treat the frozen Stage-1 risk gate as a useful
  admission signal for a future learned action head, not as proof that dynamic
  intervention is necessary. Keep fixed-L27 detect-and-replay as a serious
  fallback, and do not generalize treatment feasibility from ChartQA/TextVQA
  to GQA.
- Triggering evidence: under one current-runtime all-single plus seeded
  12-pair bounded search, validation-selected shared Random-4 rescued
  81/400 dense-wrong samples (0.2025 population rescue), versus 73/400 for
  independent sequential and 79/400 for fixed L27. Held-out rescue was
  84/400, 79/400, and 88/400, respectively. Shared-gate full-replay
  correctability enrichment transferred from 1.1211 validation to 1.1132
  test, and every triggered-correct sample was preservable by at least one
  bounded route. Test population rescue for shared Random-4 was only
  2/200 on GQA, versus 34/100 ChartQA and 48/100 TextVQA.
- Evidence paths:
  `analysis/dense_failure_stage1/treatment_correctability/decision_summary.md`,
  `metrics/gate_correctability.csv`, `metrics/enrichment.csv`, and
  `metrics/dataset_breakdown.csv` under the same root.
- Confidence: high for the frozen split, current native full-row executor, and
  bounded search contract; low for exhaustive oracle coverage or a causal
  claim that earlier triggering itself creates the observed enrichment.
- Consequence for future actions: a separately authorized Stage-2 action-head
  experiment may use the frozen 463-record selected-gate handoff, but it must
  compare against fixed-L27 replay and report GQA separately. Do not interpret
  the bounded lower estimates as exhaustive correctability or select a final
  deployment architecture from this phase alone.

## 2026-08-31 — Freeze the exact dynamic-gate population before Stage-2 labeling

- Decision or promoted lesson: any future Shared Random-4 Stage-2 label
  generation must use the frozen Phase-54 train trigger manifests rather than
  reconstructing admission ad hoc. Validation and test trigger cohorts remain
  non-training populations, and the large GQA coverage deficit must be
  reported separately.
- Triggering evidence: the complete current-dense map contains 1,881 triggered
  train Dense-W samples and 39 triggered train Dense-C samples. Validation/test
  preservation, wrong recall, and precision are `0.9425/0.5300/0.9021` and
  `0.9400/0.5100/0.8947`, but wrong recall is only `0.180/0.185` for GQA versus
  `0.880/0.800` ChartQA and `0.880/0.870` TextVQA. All 1,600 prior trigger rows
  and 18 prior aggregate fields reproduce exactly.
- Evidence paths: `analysis/dense_failure_stage1/trigger_map/decision_summary.md`,
  `manifests/train_triggered_wrong.jsonl`,
  `manifests/train_triggered_correct.jsonl`, and
  `metrics/dataset_breakdown.csv` under the same root.
- Confidence: high for the frozen checkpoint, threshold, current split, labels,
  and trigger identities; no claim is made that triggered errors are fixable or
  that earlier triggers are causally better.
- Consequence for future actions: a separately authorized corrective suffix
  search may label only the 1,881 train triggered-W rows and use the 39 train
  triggered-C rows for FULL-suffix preservation. Do not train on validation or
  test trajectories, and do not treat the 1,319 train Dense-W misses as Stage-2
  opportunities under the current gate.

## 2026-08-31 — Sequential corrective search adds support, but saturates by 200 iterations

- Decision or promoted lesson: for the frozen Shared Random-4 train triggered
  Dense-W cohort, retain exhaustive single-intervention discovery and a bounded
  sequential suffix search; multi-action search adds meaningful oracle support,
  but use a 200-iteration cap rather than 300 if full label generation is later
  authorized. This is search-label evidence, not learned-policy generalization.
- Triggering evidence: in a prospectively balanced 120-row pilot, exhaustive
  singles fixed 35 samples and sequential trajectory-conditioned MCTS fixed 22
  additional samples. Fixable@100/200/300 was 52/56/57, so 200→300 added only
  1/120 = 0.0083, below the frozen 0.01 material-gain rule. Phase-54
  dataset×depth-cell weighting estimates total bounded support at 0.5309.
  Preferred successful routes use median one non-FULL action (IQR 1–3), while
  14/57 fixable samples have multiple observed successful trigger actions.
- Qualification: the pilot deliberately balances datasets and partially
  balances trigger depth; weighted results are model-based cell estimates, not
  a new random full-population sample. Deterministic 2/3/4-cardinality rollouts
  make depth comparisons interpretable but can miss corrections requiring more
  interventions. GQA correctability remains lower than TextVQA, and late
  triggers are least correctable in this sample.
- Evidence paths:
  `analysis/dense_failure_stage2/corrective_search_pilot/decision_summary.md`,
  `metrics/overall_correctability.csv`, `metrics/budget_saturation.csv`, and
  `future_stage2_labels/successful_route_manifest.jsonl` under the same root.
- Confidence: high for the frozen pilot contract, current-runtime executor,
  binary LMMS reward, and bounded-search counts; medium for the cell-weighted
  full-cohort projection; unknown for a learned Stage-2 policy's generalization.
- Consequence for future actions: do not rerun this pilot or default to 300.
  Any separately authorized full label phase should use exactly the Phase-54
  1,881-row train triggered-W manifest, preserve single/MCTS provenance, use a
  200-iteration MCTS cap, keep the 39 triggered-C FULL-suffix supervision
  separate, and stop again before Stage-2 training unless that training is
  explicitly authorized.

## 2026-09-01 — Full corrective support is real but strongly dataset- and depth-dependent

- Decision or promoted lesson: preserve both simple single-intervention and
  trajectory-conditioned MCTS supervision, but do not interpret the full
  bounded oracle corpus as evidence that a learned Stage-2 router will
  generalize. A future V1 should use preservation plus single routes first;
  MCTS-only corpus C is a separately measurable V2 addition.
- Triggering evidence: over all 1,881 frozen train triggered Dense-W samples,
  exhaustive singles fix 698 (0.3711) and cap-200 MCTS adds 209 (0.1111), for
  total bounded support 907/1,881 = 0.4822. This is close to the cap-200 pilot
  projection 0.4667, but total support is only 0.1986 for GQA versus 0.4854
  ChartQA and 0.6448 TextVQA, and falls from 0.5923 at trigger layer 0 to
  0.2507 for layers 19-27. Exact replay produced 7,628 single, 442 MCTS, and 39
  preservation routes with 208,280 routed state rows.
- Qualification: `UNRESOLVED` is bounded-search failure, not proof of
  unfixability. Route multiplicity must not silently determine training
  weight, and train-only oracle support does not select a deployment policy.
- Evidence paths:
  `analysis/dense_failure_stage2/full_corrective_labels/decision_summary.md`,
  `metrics/overall_fixability.csv`, `metrics/dataset_breakdown.csv`,
  `metrics/trigger_depth_breakdown.csv`, and
  `stage2_corpora/corpus_manifest.json` under the same root.
- Confidence: high for population coverage, current-runtime LMMS reward,
  exact route replay, cap-200 semantics, and corpus provenance; unknown for
  learned-policy generalization or final architecture choice.
- Consequence for future actions: do not rerun label generation or merge
  route sources. A separately authorized Stage-2 V1 should begin with Corpus
  A+B, explicitly choose sample- versus route-balanced sampling, and report
  GQA/depth breakdowns. Add Corpus C only as a controlled V2 comparison.

## 2026-09-01 — Stage-2 V1 must balance samples and retain post-trigger FULL timing

- Decision or promoted lesson: do not train Corpus A+B by expanding every
  retained route and every suffix state equally. For a separately authorized
  Stage-2 V1, use each W sample as the sampling unit, choose one successful
  single route uniformly, retain its corrective state plus a small bracketing
  set of pre/post FULL states, and explicitly oversample the 39 preservation
  samples. Keep alternative successful actions as provenance/metadata rather
  than collapsing them to one supposedly unique oracle action.
- Triggering evidence: the 698 SINGLE_FIXABLE samples have 1-60 successful
  routes (median 7), while naive Corpus B contains 191,565 FULL versus 7,628
  non-FULL states (96.17% FULL, 25.11:1). Exact routed-state identity finds
  90,609 route-derived duplicate rows. Moreover, 491/698 samples are
  delayed-only, median intervention delay is 13 layers, and 2,685/4,421
  successful sample/layer positions have multiple observed non-FULL actions.
- Evidence paths:
  `analysis/dense_failure_stage2/single_label_audit/summaries/single_label_audit_summary.md`,
  `metrics/sample_route_multiplicity.csv`,
  `metrics/naive_state_class_balance.csv`,
  `metrics/immediate_vs_delayed.csv`, and
  `metrics/state_redundancy.csv` under the same root.
- Confidence: high for corpus geometry and exact state redundancy; medium for
  the proposed S3+S1 and C:W=1:2 loader because no Stage-2 training or free
  rollout has tested it.
- Consequence for future actions: the simplest defensible V1 loader is one
  uniformly sampled route per W sample, one corrective plus up to two pre- and
  two post-FULL states, and C:W=1:2 sample mixing. Treat S2 K=2 as the simple
  more-balanced runner-up, do not add Corpus C automatically, and do not infer
  learned-policy generalization from this audit.

## 2026-09-01 — Stage-2 V1 proves narrow correction, not broad action generalization

- Decision or promoted lesson: retain the shared exact-state Stage-2 router as
  proof that conservative learned correction is possible, but do not treat V1
  as broad action/timing generalization and do not automatically add Corpus C.
- Triggering evidence: the prospectively frozen 39-C/698-W run passes local
  overfit (0.7292 non-FULL recall) and improves validation by 5 net samples
  with zero C→W, but its final train diagnostic has only 0.0315 non-FULL
  recall. Free rollout is 98.397% FULL; 96.30% of intervened samples act at the
  trigger, WRITE_ONLY is never selected, GQA receives no non-FULL action, and
  all five rescues are ChartQA/TextVQA.
- Qualification: the strict prospective FULL-collapse flag is false because
  54/235 triggered samples intervene at least once, but action-level behavior
  is still near-FULL and immediate-intervention collapsed. Teacher-forced
  metrics are train diagnostics, so data diversity, label ambiguity,
  sampler/loss imbalance, and exposure shift are not separated.
- Evidence paths:
  `analysis/dense_failure_stage2/v1_training_revised/free_rollout/overall_metrics.json`,
  `free_rollout/action_behavior.csv`, `free_rollout/dataset_breakdown.csv`,
  `diagnostics/collapse_check.json`, and `summaries/stage2_v1_decision.md`.
- Confidence: high for exact validation transitions, preservation, action
  behavior, and accepted contract/checkpoint hashes; unknown for which one of
  the remaining bottlenecks is causal.
- Consequence for future actions: stop V1. A separately authorized next plan
  must choose a bounded discriminator for coverage/timing failure. Do not open
  test, add the 209 MCTS-only samples, change loss/layer inputs, or rerun V1
  automatically.

## 2026-09-01 — Canonical-source scale-up adds corrective bases but exposes frozen-gate source shift

- Decision or promoted lesson: preserve the expanded A/B/C corpora as useful
  disjoint supervision, but describe Phase 59 as canonical-source expansion
  rather than a pure sample-count replication. Any later matched training must
  keep the frozen Stage-1 gate and report dataset/source composition and trigger
  behavior explicitly.
- Triggering evidence: 4,000 prospectively frozen candidates completed with
  zero skips and yielded 75 single-fixable plus 33 MCTS-only bases, including
  44 new bounded-fixable GQA bases. Bounded fixability among triggered W is
  0.4202 versus 0.4822 in the old pool, within the prospective ±0.10 tolerance.
  However, `P(trigger|C)` shifted from 0.0122 old to 0.5404 new and
  `P(trigger|W)` from 0.5878 to 0.2951. Expanded A/B/C contain
  1,730/8,578/508 routes over 1,730/773/242 bases.
- Qualification: candidates were selected outcome-blind from pinned canonical
  training sources with metadata stratification and zero UID/SHA-group overlap;
  the resulting dense correctness and gate mix were not rebalanced post hoc.
  `UNRESOLVED` remains bounded-search failure, not proof of unfixability.
- Evidence paths:
  `analysis/dense_failure_stage2/data_scale_search/summaries/data_scale_search_summary.md`,
  `metrics/old_vs_new_yield.csv`, `metrics/dataset_breakdown.csv`, and
  `combined_corpora/corpus_manifest.json` under the same root.
- Confidence: high for candidate disjointness, current-runtime labels, trigger
  identities, exact route replay, and artifact provenance; unknown for learned
  policy generalization or the cause of the gate's source shift.
- Consequence for future actions: if separately authorized, run Scaled-Single
  with Expanded A+B first under the frozen V1 training/evaluation contract,
  then add Corpus C as a controlled comparison. Do not retune Stage 1 or alter
  architecture/loss simultaneously.
## 2026-09-04 — MCTS supervision failure combines weak action learning, interference, and exact-prefix ambiguity

- Decision or promoted lesson: do not attribute the Phase-66 zero-gain MCTS arm
  primarily to free-rollout exposure shift, and do not add more of the same
  single-label MCTS supervision. The smallest next discriminator is an
  unchanged-router observed-valid-set loss, subject to separate authorization.
- Triggering evidence: on exact successful-route states, B recalls only 8.45%
  of MCTS non-FULL actions (first/later 9.93%/7.82%) and first disagrees at the
  first intervention on 88.83% of MCTS routes. B reduces single corrective
  recall from A's 10.78% to 1.36%; its UID-weighted delta is -21.74 points with
  95% CI [-25.56,-18.01]. Multi-valid B rows have 30.96% nominal error versus
  5.65% for single-valid rows, while accepting any exact-prefix successful
  action recovers 21.63% of multi-valid rows. After C1-C3 oracle-prefix forcing,
  only 8.35%/6.75%/8.33% of remaining corrective actions are reproduced.
- Qualification: state drift is measurable after a wrong action and forced
  corrections improve final accuracy, so exposure remains a secondary factor.
  This training-route diagnosis does not establish deployment benefit,
  unseen-source generalization, or that sequential correction is impossible.
- Evidence paths: `analysis/dense_failure_stage2/mcts_failure_diagnosis/`,
  especially `summaries/mcts_failure_diagnosis_summary.md`,
  `negative_transfer/paired_bootstrap.csv`, and
  `ambiguity/error_by_valid_action_count.csv`.
- Confidence: high for the frozen exact-replay diagnosis; unknown for the
  proposed set-valued objective until separately tested.
- Consequence for future actions: keep A/B checkpoints and Stage 1 frozen. If
  authorized, test only `-log(sum p(observed-valid actions))` first, and require
  both improved MCTS oracle corrective recall and retention of A-like single
  corrective behavior before any free-rollout promotion.

## 2026-09-04 — Observed-valid supervision improves oracle compatibility but not deployment

- Decision or promoted lesson: retain exact-prefix observed-valid sets as a
  better supervision contract for ambiguous successful-route states, but do
  not promote Experiment C as a deployment winner. Label ambiguity was a
  causal oracle-learning bottleneck for MCTS states, not the sole cause of the
  Stage-2 failure.
- Triggering evidence: with every Phase-66 B variable except the loss fixed, C
  improves MCTS nominal/observed-valid non-FULL recall from 0.0845/0.1036 to
  0.1616/0.2126. First/later observed-valid recall improves from
  0.1352/0.0900 to 0.2593/0.1925. Overall valid probability mass rises from
  0.8022 to 0.9056. Yet single corrective nominal recall remains 0.0136
  (versus A's 0.1078), and Historical-800 P98/P95/P90 net corrections are
  -1/-1/0; P90 contains one rescue and one regression.
- Qualification: observed-valid action sets contain only actions seen in
  bounded successful searches, not every valid action. Oracle improvement does
  not establish unseen-source/test generalization or deployment benefit.
- Evidence paths:
  `analysis/dense_failure_stage2/observed_valid_set_loss/`, especially
  `oracle_eval/overall_metrics.csv`,
  `oracle_eval/single_negative_transfer.csv`,
  `free_rollout/threshold_comparison.csv`, and
  `summaries/observed_valid_set_loss_summary.md`.
- Confidence: high for the matched Historical-800 and exact-replay conclusion;
  unknown for on-policy correction and unseen-source/test behavior.
- Consequence for future actions: do not repeat one-hot multi-route
  supervision and do not open the held-out test. If separately authorized,
  isolate exposure shift with a small partial-prefix on-policy collection under
  fixed C before changing the router, Stage 1, thresholds, or architecture.

## 2026-09-04 — The frozen P90 plus Stage-2 A candidate is regression-dominated at full external scale

- Decision or promoted lesson: do not promote the exact Robust ALL-source
  Shared Random-4 P90 gate plus Phase-66 Experiment A as the deployed method.
  It produced more regressions than rescues on the complete established
  external population.
- Triggering evidence: all 19,960 reference UIDs completed under contract
  `63379eef...27e83`. ChartQA/TextVQA/MMMU-Pro/POPE W→C/C→W/net are
  `1/3/-2`, `2/12/-10`, `0/4/-4`, and `0/0/0`; pooled is `3/19/-16`.
  Delta accuracy is -0.000802 with paired-bootstrap 95% CI
  [-0.001253,-0.000351]. Stage 1 triggers 901 samples, but only 211 use any
  non-FULL action; POPE has zero triggers.
- Qualification: this rejects only the exact frozen threshold/checkpoint and
  execution candidate. It does not show that dynamic routing, four-action
  decomposition, or a better preservation policy cannot work. Current dense
  accuracies differ slightly from the rounded original-server reference even
  with exact manifests/prompt/scorers, so cross-server output identity is not
  claimed.
- Evidence paths: `analysis/dense_failure_stage2/full_benchmark_eval/`,
  especially `metrics/benchmark_summary.csv`, `metrics/paired_bootstrap.csv`,
  `metrics/stage1_admission.csv`, `metrics/stage2_action_behavior.csv`, and
  `summaries/method_level_decision.md`.
- Confidence: high for population coverage, paired outcomes, current-runtime
  scoring, and the negative method decision; unknown for the causal source of
  individual regressions.
- Consequence for future actions: stop this candidate. The single retained,
  unexecuted direction is conservative Stage-2 action-selection calibration
  focused on preservation, subject to a new prospective plan and explicit
  authorization.

## 2026-09-04 — Full-eval loss is preservation-dominated in pooled accounting, with family-specific failure modes

- Decision or promoted lesson: do not expand Stage-1 admission for the frozen
  candidate before Stage-2 preservation improves. Conditional Stage-2
  activation is nearly the same on triggered W and C, but its outcome is much
  worse on C; the family-level diagnosis is not uniform.
- Triggering evidence: Stage 1 admits 496/4,380 W and 405/15,580 C, then
  Stage 2 uses non-FULL on 119/496 W and 92/405 C. Only 3/119 treated W rescue,
  whereas 19/92 treated C regress. TextVQA contributes 12 regressions and two
  rescues; MMMU-Pro treats 82 W without any rescue; POPE never triggers and its
  maximum score remains 0.077710 below P90.
- Qualification: this is deterministic funnel accounting, not proof that one
  action caused any individual transition. The audit does not establish that
  current Stage-2 logits can separate rescues from regressions, nor does POPE
  inactivity measure Stage-2 quality.
- Evidence paths:
  `analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/`, especially
  `funnel/pooled_funnel.csv`, `funnel/bottleneck_classification.csv`,
  `answer_changes/all_22_answer_changes.jsonl`, and
  `summaries/exhaustive_audit_summary.md`.
- Confidence: high for the exact accounting and benchmark heterogeneity;
  unknown for action-level causality and confidence-margin separability.
- Consequence for future actions: if separately authorized, test exactly one
  preservation-calibrated Stage-2 abstention margin selected on development
  data under a C-preservation constraint. Do not tune it on these 22 external
  answer changes and do not lower Stage 1 first.

## 2026-09-05 — The current Stage-2 global confidence margin does not improve treatment selectivity

- Decision or promoted lesson: do not add a positive global
  best-non-FULL-minus-FULL margin to the frozen Experiment-A method. Keep δ=0
  only as the development winner; because this does not change the failed
  external candidate, do not rerun external evaluation.
- Triggering evidence: on Historical-800, δ=0 gives W→C/C→W/net `4/1/+3`.
  q10, q25, and q40 give `3/1/+2`, `2/1/+1`, and `2/0/+2`; q50/q60 give
  `1/0/+1`, and q70+ gives `0/0/0`. All five image-group-disjoint cross-fit
  training folds select δ=0, with pooled held-out `4/1/+3`. Median maximum
  executed margins are 0.6747 for rescues and 0.7210 for the sole regression.
- Qualification: only four rescues and one regression occur on this small
  development set, and the grid comes from sampler-weighted final-epoch states.
  This rejects the prespecified one-dimensional global margin for the current
  router; it does not prove Stage-2 representations, actions, or retraining
  cannot improve treatment quality.
- Evidence paths: `analysis/dense_failure_stage2/abstention_margin/`, especially
  `development/per_margin_rollout_summary.csv`,
  `crossfit/stability_summary.json`, and
  `diagnostics/margin_by_transition.csv`.
- Confidence: high for the frozen-grid sequential-rollout result and selection;
  low for fine-grained margin-distribution comparisons because transition
  counts are sparse.
- Consequence for future actions: move away from post-hoc global confidence
  gating. If separately authorized, revisit the Stage-2 representation or
  training signal; do not retune this margin on development or external data.

## 2026-09-05 — Frozen Stage-2 READ/WRITE summaries do not support another treatment gate

- Decision or promoted lesson: do not add a KEEP-vs-INTERVENE head or a learned
  four-logit abstention readout to the current Experiment-A representation. The
  exact routed-state probe evidence is weak and lacks a useful high-precision
  intervention region; the next authorized work should diagnose representation
  or training-state diversity, not another confidence/readout layer.
- Triggering evidence: all 21,071 exact prefix-states were extracted with exact
  router-logit parity and split across five UID/image-group-disjoint folds.
  Clean labels contain 18,438 KEEP_REQUIRED and 1,657 INTERVENE_REQUIRED states;
  976 MIXED states are excluded from fitting. OOF AUROC/AUPRC is 0.4632/0.1050
  for the scalar margin, 0.4640/0.0939 for four logits, 0.5370/0.0931 for z_R,
  0.5219/0.1088 for z_W, and 0.5620/0.1086 for [z_R;z_W]. The prespecified
  optional MLP reaches only 0.5583/0.1128. RW precision at top 5/10/20% is
  0.123/0.135/0.123; recall at 90% precision is 0.0012. Nuisance-only AUROC is
  0.8150, while matched RW remains weak at 0.5763.
- Qualification: KEEP_REQUIRED means FULL is the only action observed in a
  successful replay-valid continuation, not proof that every unsearched
  intervention fails. The corpus is highly imbalanced and route-source support
  differs; matched AUPRC uses deliberately balanced nuisance-cell weights and
  is not prevalence-comparable to natural AUPRC. This rejects readily usable
  selectivity in this frozen representation/probe family, not corrective
  routing, READ/WRITE control, or richer/diversified representations.
- Evidence paths:
  `analysis/dense_failure_stage2/treatment_selectivity_separability/`, especially
  `metrics/probe_summary.csv`, `metrics/matched_sensitivity.csv`,
  `metrics/nuisance_controls.csv`, and
  `summaries/treatment_selectivity_summary.md`.
- Confidence: high for exact-state provenance, group isolation, extracted
  representation identity, and the negative prospective decision; moderate
  for route-specific comparisons because positive support varies materially.
- Consequence for future actions: stop Phase 72. If separately authorized,
  choose one bounded representation/training-state-diversity discriminator.
  Do not train or deploy another Stage-2 treatment gate from these summaries.

## 2026-09-06 — Complete suffix-program supervision improves preservation but not corrective transfer

- Decision or promoted lesson: retain complete-program supervision only as a
  conservative Stage-2 formulation; do not promote the Phase-74 predictor as a
  deployment winner. Modeling the joint post-trigger trajectory sharply
  reduces regressions relative to independent local actions, but does not
  increase rescues and still underperforms Dense.
- Triggering evidence: all 4,948 eligible programs over 569 UIDs replayed
  LMMS-correct with exact token and cached/live trigger-state parity. On all
  19,960 external rows, Program W-to-C/C-to-W/net is `3/8/-5`, compared with
  Sequential-A `3/19/-16`. Program accuracy is 0.780311, Dense accuracy is
  0.780561, and the paired-bootstrap 95% interval for Program-minus-Dense is
  [-0.000601, 0.000050].
- Qualification: the program decoder exactly matches only 20.87% of known
  development programs, and successful-route labels are bounded by prior
  search. The result establishes improved preservation for this frozen
  formulation, not an upper bound on program prediction or corrective
  routing.
- Evidence paths:
  `analysis/dense_failure_stage2/polar_suffix_program/`, especially
  `metrics/full_benchmark_summary.csv`, `metrics/paired_bootstrap.csv`,
  `metrics/dense_c_preservation.csv`, and
  `summaries/polar_suffix_program_full_eval_summary.md`.
- Confidence: high for corpus validity, paired external outcomes, and the
  preservation improvement; unresolved for how to improve treatment transfer.
- Consequence for future actions: do not deploy this candidate or interpret
  reduced C-to-W alone as success. Any follow-on must directly address why
  known corrective programs fail to transfer, under a new prospective plan
  and explicit authorization.

## 2026-09-06 — Frozen beam-8 exposes secondary ranking capacity but dominant candidate failure

- Decision or promoted lesson: do not treat top-1 ranking as the primary
  remaining Stage-2 bottleneck. The frozen program beam contains meaningful
  additional rescue capacity, especially for TextVQA, but candidate
  generation/representation dominates population-level failures.
- Triggering evidence: under contract `e274d24c...06c7fe7`, all 901 top-1 paths
  reproduced exactly before 6,944/6,944 frozen unique candidates executed.
  W-to-C@1/@2/@4/@8 is `3/10/24/33`. Of 493 top-1 W failures, 30 (6.09%) have
  a correct lower-ranked candidate and 463 (93.91%) have none. TextVQA rises
  from 0 to 19 oracle-available rescues and MMMU-Pro from 2 to 12. Median
  top1-minus-correct sequence-score gap for ranking failures is 1.886, and
  best-correct programs use a median one non-FULL action versus zero for
  top-1 wrong programs.
- Qualification: W-to-C@8 is an external-label oracle ceiling, not a deployable
  method or evidence that a reranker can identify the correct candidate.
  Candidate support is limited to the frozen width-8 beam and does not measure
  all possible suffix programs.
- Evidence paths:
  `analysis/dense_failure_stage2/program_beam_oracle_audit/`, especially
  `metrics/benchmark_breakdown.csv`, `metrics/w_failure_decomposition.csv`,
  `metrics/score_gap_analysis.csv`, and
  `summaries/program_beam_oracle_audit_summary.md`.
- Confidence: high for frozen-beam completeness, exact top-1 parity, and the
  ranking-versus-generation decomposition; unknown for whether a different
  trigger representation will improve candidate support.
- Consequence for future actions: if separately authorized, test one minimal
  trigger-state representation enrichment while keeping Stage 1, action
  semantics, beam width, evaluator, and evaluation firewall fixed. Do not fit
  a reranker to these external oracle outcomes.

## 2026-09-07 — Offline closed-loop trajectory-set supervision does not repair corrective transfer

- Decision or promoted lesson: do not promote the Phase-76 closed-loop
  trajectory-set router as a deployment winner. Recomputing the unchanged
  READ/WRITE representation on actual routed states and marginalizing over all
  known successful trajectories improves preservation relative to independent
  local labels, but still produces no external corrective rescues.
- Triggering evidence: all 4,948 programs over 569 UIDs exact-replayed into
  35,565 unique prefix states with zero quarantines. All 19,960 external rows
  completed under contract `00639e7e...5f6a`. Closed-loop W-to-C/C-to-W/net is
  `0/8/-8` at accuracy 0.780160, versus Dense 0.780561, Open-loop Program
  `3/8/-5`, and Sequential-A `3/19/-16`. Only 39/496 triggered Dense-W samples
  receive any non-FULL action, and none rescue, even though the median
  best-route geometric action probability on the training corpus is 0.7269.
- Qualification: feedback and the set-valued objective changed together, and
  successful trajectories remain bounded by offline search. The result rejects
  this exact offline formulation; it does not establish that dynamic routing,
  richer representations, on-policy labels, Stage 1, or MCTS are generally
  ineffective.
- Evidence paths:
  `analysis/dense_failure_stage2/closed_loop_trajectory_set/`, especially
  `summaries/closed_loop_trajectory_set_full_eval_summary.md`,
  `metrics/full_benchmark_summary.csv`, `metrics/stage1_stage2_funnel.csv`,
  `training/responsibility_statistics.csv`, and `artifact_manifest.json`.
- Confidence: high for replay completeness, objective correctness, paired
  external outcomes, and the negative deployment decision; moderate for the
  on-policy-shift diagnosis because the combined formulation does not causally
  isolate representation from state-support mismatch.
- Consequence for future actions: stop this candidate. If separately
  authorized, run one bounded on-policy state-distribution diagnostic/relabeling
  experiment with architecture, Stage-1 gate, action semantics, evaluator, and
  external firewall fixed.

## 2026-09-09 — Current dense-failure and local-utility heads do not robustly transfer externally

- Decision or promoted lesson: do not treat pooled external Stage-1
  discrimination as robust transfer, and do not proceed to deployment with the
  current-state local READ/WRITE utility heads. Per-benchmark evidence is the
  controlling result: Stage 1 is category `D1-C`, and Stage 2 is `D2-A`.
- Triggering evidence: all 19,960 external UIDs reproduced the frozen Dense and
  strict-P90 trigger traces exactly. Stage-1 M3 AUROC is 0.4998 ChartQA, 0.6794
  TextVQA, 0.5399 MMMU-Pro, and 0.5582 POPE (macro 0.5693; pooled 0.6980).
  All 8,442 triggered dense states received complete four-branch measurement.
  READ Spearman is 0.0848/0.1095/0.0246 and WRITE Spearman is
  0.0335/0.0158/0.0204 on ChartQA/TextVQA/MMMU-Pro; POPE has no P90-triggered
  states. Predictions were frozen before utility labels, and all 33,768 branch
  parity checks passed.
- Qualification: this rejects broad transfer for the exact frozen
  representations, objectives, and linear/joint-head families. It does not
  establish that richer representations, benchmark-calibrated Stage 1,
  history/nonlocal Stage-2 information, or counterfactual probes cannot work.
  The pooled Stage-1 AUROC is confounded by cross-benchmark prevalence/score
  structure and is not a substitute for within-benchmark transfer.
- Evidence paths:
  `analysis/predictability_generalization/stepD_external_transfer/`, especially
  `summaries/stepD_external_transfer_summary.md`,
  `stage1/benchmark_metrics.csv`,
  `stage2_predictability/read_benchmark_metrics.csv`,
  `stage2_predictability/write_benchmark_metrics.csv`, and
  `statistics/group_bootstrap_ci.csv`.
- Confidence: high for completeness, parity, per-benchmark metrics, and the
  fixed D1-C/D2-A decisions; unknown for untested richer/nonlocal formulations.
- Consequence for future actions: if separately authorized, first use one
  bounded diagnostic to distinguish Stage-1 source calibration from
  representation failure. Do not add routing complexity or train another
  current-state local utility head merely from the pooled score.

## 2026-09-10 — Exact one-step action effects do not reliably identify full-suffix READ/WRITE utility

- Decision or promoted lesson: do not build a one-layer speculative
  probe-and-route controller from the current pooled or token-aware
  representations. Exact one-step alternatives remain weak predictors of the
  frozen full-suffix utility targets.
- Triggering evidence: all 15,185 dense states / 45,555 one-step branches and
  35,565 routed states / 106,695 branches passed same-prestate, action-bit,
  deterministic-repeat, and exact FULL-to-canonical post-state checks. Dense
  OOF READ PRE/DELTA/token Spearman is `0.0626/0.0773/0.0694`; the DELTA-minus-
  PRE 95% image-group-bootstrap interval is `[-0.0089, 0.0381]`. Dense WRITE
  PRE/PAIR/token is `0.0354/0.0421/0.0383`. Harmful-flip AUROC is 0.4561 for
  READ and 0.4106 for WRITE. Routed token OOF reaches only `0.1290/0.0726`,
  and Dense-to-routed token transfer is `0.1141/0.0699` for READ/WRITE.
- Qualification: this rejects one-step identifiability for the exact targets,
  populations, representations, and fixed capacity ladder. It does not show
  that a bounded multi-layer probe, a short rollout, or other nonlocal/history
  information cannot identify utility; it establishes no benchmark gain,
  compute saving, deployment safety, or causal optimality.
- Evidence paths:
  `analysis/dense_failure_stage2/counterfactual_effect_identifiability/`,
  especially `summaries/counterfactual_effect_identifiability_summary.md`,
  `statistics/uid_bootstrap_ci.csv`,
  `statistics/pairwise_model_differences.csv`, and
  `routed_secondary/dense_to_routed_transfer.csv`.
- Confidence: high for execution validity, population completeness, and the
  Case-D conclusion; unknown for horizons beyond one layer.
- Consequence for future actions: stop one-layer controller investment. If
  separately authorized, the single smallest remaining discriminator is the
  prospectively specified two-layer / short-horizon counterfactual-
  identifiability audit; do not run it automatically.

## 2026-09-11 — Explicit local READ-operation structure does not robustly identify full-suffix harm

- Decision or promoted lesson: do not train another local READ-harm classifier
  or promote the selection-qualified routed result. Explicit attention,
  compatibility, update, value, and spatial-concentration features do not make
  the frozen full-suffix READ-harm target robustly predictable.
- Triggering evidence: the complete primary census contains 15,185 states over
  1,413 UIDs, with 7,285 harmful and 7,900 beneficial READ effects. Harmful
  adjacent persistence fails its shuffled-null gate, while span and immediate-
  neighborhood gates pass only marginally, yielding `R-STRUCT-B`. Only 1/34
  matched feature effects has a pooled group-bootstrap interval excluding zero;
  the harmful-vs-beneficial matched probe AUROC is 0.4186, yielding `R-MECH-B`.
  The prospectively selected F_ALL/MLP dense OOF Spearman/AUROC is
  `0.0697/0.5338`, and historical↔canonical and LODO transfer remain weak,
  yielding `R-LEARN-C`.
- Qualification: the result applies to the frozen current-operation F1-F7
  features, full-suffix target, populations, folds, and linear/small-MLP
  capacity ladder. It does not establish that READ is harmless, that attention
  lacks causal structure, or that short-horizon propagation/history/planning
  information cannot identify harm. Routed OOF Spearman 0.1623 is secondary and
  selection-qualified, not evidence of dense-primary robustness.
- Evidence paths: `analysis/read_harm_structure_learnability/`, especially
  `summaries/read_harm_structure_summary.md`,
  `summaries/read_harm_mechanism_summary.md`,
  `summaries/read_harm_learnability_summary.md`, and `artifact_manifest.json`.
- Confidence: high for census completeness, operation fidelity, hash integrity,
  and the fixed categories; moderate for the qualitative "mostly isolated"
  label because two of three structure components pass narrowly.
- Consequence for future actions: stop this local-classifier branch. The sole
  unexecuted recommendation is one bounded short-horizon READ
  effect-propagation/planning audit, requiring separate user authorization.

## 2026-09-12 — READ harm remains non-identifiable through eight FULL continuation layers

- Decision or promoted lesson: do not train or deploy another one-state or
  H<=8 READ-harm controller from the tested pooled or token-comparator family.
  Propagated counterfactual effect magnitude grows, but predictive information
  does not satisfy the prospective materiality or high-precision gates.
- Triggering evidence: all 15,185 dense states yielded 47,133 valid horizon
  pairs / 94,266 ON/OFF branches after exact H1 parent parity, canonical ON,
  fresh-cache repeat, swapped-order H8, action-trace, census, and cache-readback
  checks. On the 6,916-state H8-common support, pooled DELTA Spearman is
  `0.0756/0.0688/0.0646/0.1097` and harmful AUROC is
  `0.5316/0.5287/0.5314/0.5483` for H1/H2/H4/H8. H8-minus-H1 gains are only
  `+0.0340/+0.0167`, with 95% image-group-bootstrap lower bounds
  `-0.0009/-0.0020`; token H8 Spearman is 0.0854, metrics are non-monotone,
  and precision@10% 0.5462 is only +0.0536 over prevalence. Category:
  **H-READ-D**.
- Qualification: this result is limited to the fixed current-runtime dense
  population, target `H_R`, FULL-vs-WRITE_ONLY intervention, common FULL
  continuations through H<=8, inherited group folds, fixed models, and tested
  pooled/token representations. It does not establish that READ is harmless or
  rule out longer-horizon planning/search, history, or a different causal
  information family. Routed confirmation was optional for H-READ-D and was
  not run because it is selection-biased and cannot override the dense-primary
  non-material result.
- Evidence paths: `analysis/read_harm_short_horizon_propagation/`, especially
  `summaries/read_short_horizon_propagation_summary.md`,
  `statistics/group_bootstrap_ci.csv`, `controls/`, and
  `artifact_manifest.json` (`1f9005bf...35961`).
- Confidence: high for execution validity, census completeness, artifact
  integrity, and the H-READ-D decision; unknown for horizons beyond eight.
- Consequence for future actions: stop this short-horizon branch. The sole
  unexecuted recommendation is a separately authorized bounded longer-horizon
  READ planning/search study; do not start it automatically.


## 2026-09-13 — Preserve original Stage1 output indexing and separate score shift from target shift

- Direct evidence: Phase86 original native features are decoder outputs indexed l, including valid layer27; compact Stage2 caches retain the last control token and cannot substitute for Stage1 final-user and mean-user-text summaries. Exact reconstruction passed all 15,185 states, with original ON scores reproduced within 1.37e-7.
- Timing consequence: historical trigger l already used dense post-layer-l features, while StepA acts pre-layer l. First-trigger branch comparison is retrospective; any future controller must explicitly resolve rollback or action timing rather than silently shift indices/labels.
- Interpretation consequence: ON/OFF failure AUROC 0.6728/0.5949 alone does not diagnose counterfactual representation collapse. Holding targets fixed yields ON/OFF-score AUROC 0.6728/0.6710 on ON outcomes and 0.5948/0.5949 on OFF outcomes. Compare scores on fixed labels before attributing a cross-branch metric difference to the representation.
- Evidence: `analysis/read_counterfactual_stage1_branch_critic/stage1_post_action_indexing_contract.md`, `scores/fixed_target_cross_score_diagnostic.csv`, `parity/independent_result_verification.json`.
- Scope: frozen Stage1 and this exact internal branch corpus. Qualified BC-C is a measured relative-choice failure, not a permanent impossibility claim or proof calibration will recover it. No follow-on execution authorized.


## 2026-09-14 — WRITE propagation influence does not imply useful harm identification

- Direct evidence: Phase87 FULL versus READ_ONLY preserves same-layer text, while every one of 6,044 H8-common states has text divergence after one subsequent FULL layer. HW labels remain frozen. Complete 15,185-state local and 57,538-record propagation evidence passed parity and raw hash readback.
- Learnability: local F_ALL rho .04752 / AUC .48188; H8 common delta rho .09732 / AUC .53872. H8−H0 gains +.05613 [.02670,.08523] rho and +.02255 [.00514,.03957] AUC are real but below the fixed gain thresholds and yield no useful precision. Dense-W confirms. Category is qualified W-PROP-C / WRITE-S4, not zero signal or intrinsic nonlocality.
- Reusable interpretation rules: keep absolute and relative representation norms separate; compare horizons on exact common support; align READ/WRITE horizon conventions and baseline target signs explicitly. WRITE H8 includes eight subsequent layers; prior READ H8 includes seven. Utility signs and correctness flips are distinct targets.
- Scope and consequence: close the tested local/H≤8 family; longer horizons/representations remain unknown. A metadata-only H16 audit reveals strongly reduced and shifted support, not a new result. No transfer failure may be inferred from skipped positive-gated tests. No search/router or strategic pivot is authorized.
- Evidence: `analysis/write_harm_structure_learnability/summaries/final_write_characterization.md`, `write_propagation_summary.md`, `read_vs_write_characterization.md`, `final_review_reconciliation.md`, and `analysis/write_harm_structure_learnability/artifact_manifest.json`.

### Phase88 — Native generation continuation requires its exact position convention
Direct evidence: `analysis/benchmark_calibrated_fixed_rw_schedule/parity/dense_scorer_mismatch_diagnostic.json`. All28 prefill states and initial logits can match while generated correctness differs. On MMMU-Pro Standard Psychology128, the inherited cached decoder used full length plus maximum-based RoPE delta (position323); installed native HF5 generation increments the final prompt position (position194). Aligning only continuation restores exact native output. Apply the same convention to generation and multi-token q, and regenerate affected evidence after a correction. Phase88 uses a scoped adapter; historical shared files and prior-phase reported results were not modified or re-audited. This finding does not itself quantify any effect on earlier conclusions.
