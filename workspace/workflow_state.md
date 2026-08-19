# Workflow State

- Repository organization update (2026-08-19): root-level `analysis*` packages
  were migrated to versioned packages under `tools/research_analysis/`, and
  `03_experiments/` was moved to `runs/experiments/`. The scheduler default and
  all live imports/tests were updated; loose generated reports were moved into
  `reports/`. Focused verification preserved the baseline result (60 passing,
  one pre-existing Transformers 5.3 `DynamicCache` failure). Active jobs were
  unaffected. Evidence: `reports/repository_root_cleanup_20260819.md`.

- Label-storage update (2026-08-19): canonical MCTS labels were atomically
  relocated to the dataset tree. GQA/TextVQA/ChartQA now live at
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1/` (8,000 records), and
  WeMath2.0-Pro lives at
  `datasets/math_labels/wemath20_pro_mcts_max400_v2/` (4,544 records). Stored
  manifest/audit SHA-256 checks pass. The former `outputs/label_regeneration/v1`
  and `outputs/label_regeneration/wemath2pro_cap400_v2` paths are compatibility
  symlinks; active recovery jobs 101708/101709 remained running. Evidence:
  `reports/mcts_label_relocation_20260819.md`.

- Active detour (2026-08-18): WeMath2.0-Pro conditional greedy route recovery
  for exactly 2,278 current-FULL-wrong records with zero valid hard-cap-400
  MCTS routes (1,104 image groups).
- Plan: `plans/dynamic_mllm_wemath2pro_greedy_recovery_plan.md`. Preserve the
  supplied Phase-1/Phase-2 search semantics while adapting only data/executor/
  scorer interfaces to the current Transformers 5.3.0 verified WeMath runtime.
- Authorization: full bounded G0--G5 recovery search, with at most 50 valid
  masks in the derived training view and no truncation of raw route evidence.
- G0 PASS: exactly 2,278 records / 1,104 image groups; every linked MCTS cache
  checksum passed. G1 PASS: deterministic search/dedup/max-50 tests pass. G2
  PASS: unchanged 5/5 native/binary/cached/new-mask gate.
- Current stage: G3 Phase 1 runs one process per GPU in four global shards:
  Slurm 101708 on node06 (0--1) and 101709 on node07 (2--3).
- First G2 launch stopped before scientific output on a missing deterministic
  CuBLAS setting. Supported repair: `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the
  unchanged rerun passed.
- Next gate: 2,278 atomic Phase-1 records and 22,780 finals with zero unresolved
  errors, then freeze one global Phase-2 budget/request manifest.
- Evidence: `reports/wemath2pro_greedy_recovery_package_audit.md` and
  `workspace/phase_memory/phase_23_wemath_greedy_recovery.md`.

- Active phase completed (2026-08-18): training-set fitting diagnosis for the
  completed Pareto-filtered Image+Question duplicated-BCE and exact-set-NLL
  runs. No new training was performed; all 20 saved checkpoints were evaluated
  read-only on the frozen 6,043 positive training inputs.
- Decision: both objectives exhibit primary training-fit failure. Best train
  Pareto Hit@1 is 18.27% for BCE and 17.95% for NLL, versus the frozen 73.92%
  train BCE-label oracle. Singleton train Hit is only ~24%; doubleton and
  three-plus Hit is approximately zero. Residual multimodality and a late
  generalization gap coexist but are not the primary bottleneck.
- Collapse diagnosis: Pareto filtering removes ALL-ON collapse (essentially
  0% throughout), but creates 34–60% ALL-OFF concentration. Later mask
  diversity increases without coherent Pareto-route learning.
- Validation source: original frozen A6000 histories. A4000 read-only train
  metrics reproduced early decoded validation Hit exactly, but small
  cross-hardware continuous/threshold differences prevented exact validation
  re-evaluation; no tolerance was tuned further.
- Compute placement: node03 remains prohibited. Node04 is allowed again by
  explicit user amendment; CPU work continues to prefer node05.
- Evidence: `reports/binary_pareto_training_fit_analysis.md` and
  `outputs/binary_pareto_v1/training_fit_analysis_v1/`.

- Proposed detour audited (2026-08-17), not executed: use the frozen greedy
  Phase-1/Phase-2 algorithm to search the 2,278 WeMath2.0-Pro Group-D samples
  (1,104 image groups) with no valid cap-400 MCTS route.
- The reproduction package is checksum-clean but not directly runnable for
  WeMath: it targets an old 10K four-benchmark manifest, Transformers 4.57.1,
  old project imports, per-row image caps, and non-WeMath preflight/auditing.
- Provisional protocol: preserve the package, port only its ten greedy orders
  and Phase-2 candidate rules onto the current Transformers 5.3.0 verified
  executor/MathRuler/native-processing contract, then pass a five-sample gate.
  No manifest was frozen, no code adapter implemented, and no GPU search run.
- Evidence: `reports/wemath2pro_greedy_recovery_package_audit.md` and
  `workspace/phase_memory/phase_23_wemath_greedy_recovery.md`.

- Active phase completed (2026-08-17): hard-cap-400 WeMath2.0-Pro MCTS cache
  audit and training-suitability analysis. Exactly 4,544/4,544 eligible records
  pass; eight source rows remain prospectively technical-invalid, and no
  timeout/temp/error record exists.
- WeMath label coverage: current FULL 841 correct / 3,703 wrong; 1,425 current-
  wrong samples have a correcting route. Overall 2,266 samples have at least
  one valid route and 2,278 have none. The raw cache holds 107,671 positive and
  1,550,814 negative evaluated routes.
- Suitability decision: conditionally usable for image-group-disjoint exact
  valid-set NLL and positive/negative ranking; not suitable for unfiltered
  duplicated-route BCE. The ideal weighted BCE label oracle has 13.72% Hit@1
  and mean nearest-valid Hamming 5.10; 94.93% of diagnostic selected routes are
  Pareto-dominated. No training was authorized or run.
- Evidence: `reports/wemath2pro_mcts_training_suitability.md` and
  `outputs/wemath2pro_mcts_label_analysis_v1/analysis_manifest.json`.

- Active phase completed (2026-08-16): label-only MCTS geometry and exact
  duplicated-BCE oracle analysis authorized by `plans/mcts_bce_analysis.md`.
- Decision: **Outcome C + Outcome E**. Raw MCTS and max-50 labels retain high
  route diversity, but the exact weighted per-sample BCE oracle has only 5.93%
  selected-valid Hit@1 (6,507/6,917 invalid hybrids). Separately, 95.83% of
  selected route occurrences are Pareto-dominated; diagnostic Pareto filtering
  raises oracle Hit@1 to 73.41% and lowers mean ON 17.21 -> 9.78.
- Outcome A rejected: raw mean pairwise Hamming is 13.36/28 and radius-4 mean
  effective mode count is 75.14. Outcome B rejected: selected Hamming 13.44 and
  entropy 0.5986 preserve raw geometry. Outcome D is not primary because the
  ideal per-input BCE target is already poor.
- Evidence: `reports/binary_mcts_label_geometry_and_bce_oracle_report.md` and
  `outputs/binary_mcts_label_geometry_v1/analysis_manifest.json` (SHA-256
  `0fd58601b811d5bfdd4785dc5bc804e90c1e90463fe228554ebee1d02257b36c`).
- No training, Qwen inference, label regeneration, selector change, or new
  research method was executed. The phase is closed; any matched
  Pareto-efficient BCE versus complete-route-coherent objective study requires
  explicit approval.

- Active phase (completed 2026-08-15): full10 POLAR-style duplicated-BCE
  comparator authorized by `plans/full_train_polar_bce.md`. Both ten-epoch
  trainings, the joint external preflight, both 22,307-record evaluations, and
  the merged analysis completed with integrity PASS. Exact-set-NLL full10
  artifacts remain unchanged.
- Runtime gate PASS: `outputs/binary_polar/full10_bce/preflight_v1.json`
  (Slurm `101019`) validated physical batch 128 on longest cached images,
  finite BCE/gradients for both modalities, zero frozen-encoder gradients,
  exact repeated logits, and matched shared initialization.
- Completed jobs: Question-only pipeline Slurm `101023` on node02 and
  Image+Question pipeline Slurm `101022` on node07. Best-Hit@1 selection is
  epoch 2 for both modalities. Final report:
  `reports/binary_polar_full10_bce_external_eval.md`.
- The completed BCE report now includes, for every benchmark and suite,
  FULL-normalized router accuracy ratio, fixes, regressions, mean visual-ON
  layers, distinct predicted-mask count, and ON-layer reduction.

- Active phase: full10 best-checkpoint external evaluation is complete with
  integrity PASS; no new predictor or evaluation action is authorized.
- Active source plan: `plans/dynamic_mllm_label_regeneration_plan.md`.
- Plan SHA-256:
  `634f2736d287c647cda7b21755b2ace753db29316ecc9c51523218b498380918`.
- Current step: stop after reporting. Both predictors completed exactly 22,307
  records. Image+Question selected ALL-ON universally; Question-only selected
  non-ALL-ON only 44 times, with no prediction, score, or correctness change.
- External baseline amendment: the first 192 live ALL-ON rows per job exposed
  7 ChartQA disagreements with the historical bundle cache (including one
  correctness change), despite the 9-row preflight passing. The cache is now
  audit-only. Every scientific comparison uses current live ALL-ON: reuse the
  predicted execution when its mask is ALL-ON; otherwise execute a paired live
  ALL-ON baseline. The 192 partial rows were atomically canonicalized with
  original backups and checksummed repair ledgers preserved.
- External jobs `100788`, `100787`, and aggregation job `100790` completed.
  Exact merged-result and analysis checksums pass. Final report:
  `reports/binary_polar_full10_external_eval.md`.
- Frozen execution contract SHA-256:
  `64f525f5d0a4333e1aeae27f41b9055c8da19a9a0fc566ab3c7db270ea37fc7d`.
- Frozen artifacts: `outputs/label_regeneration/v1/`; checksums pass for the
  contract, 8,000-record source manifest, and 15-record smoke manifest.
- Data scope: 8,000 historical-balanced records—4,000 GQA, 2,000 TextVQA,
  2,000 ChartQA; DocVQA excluded.
- Predictor split decision: exact image-group-disjoint 7,000 train / 1,000
  validation, seed 20260809, constructed without current route outcomes. The
  expanded 22,307-record active evaluation replaces the need for an internal
  test: ChartQA/TextVQA core VQA 7,500, MMStar/MMMU 5,807, and POPE 9,000,
  reported as three separate suites after checkpoint freezing. DocVQA is
  excluded by explicit user direction.
- Route semantics: unrestricted complete 28-bit layer-wise visual ON/OFF masks;
  POLAR segments are a derived predictor representation only.
- Label authority: fresh greedy output and benchmark score under the new frozen
  executor. Historical all-ON buckets and old cached masks are metadata or
  proposal-only; old valid/invalid labels cannot be copied.
- Processor contract: native Qwen defaults with no project-specific
  `max_image_tokens` override.
- Pre-extraction gate: exactly 15 smoke records (five per dataset); require
  15/15 binary ALL-ON/native token parity plus exact repeated mixed-route
  tokens/scores. Passed on Slurm job `99740`; report:
  `outputs/label_regeneration/v1/smoke_report_v1.json`.
- Search budget: 200 simulations for current all-ON-correct; 400 default and at
  most 600 adaptively for current all-ON-wrong.
- Raw-cache rule: retain every evaluated positive and negative mask and MCTS
  metadata. Target about 20 diverse positives when found; derive at most 50
  diverse positives for later training, matching POLAR's cap. Both objectives
  use the identical deterministic subset; the raw cache remains untruncated.
- Predictor status: audited and ready for the bounded P10 smoke only. P9 and
  all P10 static/real-encoder readiness checks pass. No optimizer step or real
  predictor training was run. Full mode is programmatically blocked by the
  readiness gate until a post-smoke decision explicitly opens it.
- Preserved evidence: BP-1 showed the old cache is not portable ground truth;
  two cached-positive fixtures became invalid under the repaired target
  executor. The direct binary head and exact valid-set NLL remain prospective
  downstream comparisons, not part of label generation.
- P0 provenance: the project root is not a Git checkout, so deterministic
  hashes of the active MCTS/executor/evaluator/runner source files are frozen
  in the contract.
- Initial P3 allocation: Slurm job `99741` on node07, 4 A6000 GPUs, 32 CPUs,
  240 GB RAM, and four workers. It was cleanly stopped at 2,291 completed
  records for the user-approved scale-up; no zero-byte, temporary, or error
  artifacts remained.
- Completed P3 runtime: Slurm job `99758` on node02, 8 NVIDIA RTX A6000 GPUs,
  64 CPUs, 400 GB RAM, and eight workers. Cross-shard-count resume discovery
  validates and skips completed records from the four-worker layout. All eight
  ranks have produced new contract-bound records with zero errors and no
  duplicates.
- Resume amendment: `outputs/label_regeneration/v1/p3_resume_amendment_v1.json`
  records the allocation change and the runner-only resume-index hash change;
  all other 13 frozen source hashes and the scientific contract are unchanged.
- P4 result: PASS on 4,000 GQA, 2,000 TextVQA, and 2,000 ChartQA records;
  WeMath2.0-Pro was explicitly excluded. Exactly 8,000 terminal records passed
  source binding, frozen contract, 28-bit route, anchor, score/validity,
  search-budget, and trace-linkage checks. Zero missing, unexpected, duplicate,
  invalid, error, temporary, or zero-byte records. Evidence:
  `outputs/label_regeneration/v1/post_generation/cache_audit_v1.json`.
- P5 result: PASS on all 8,000 checksum-reverified records. Current ALL-ON is
  correct for 4,045 and wrong for 3,955; 2,872/3,955 current-wrong records
  (72.62%) have a correcting evaluated route. Valid-route coverage is
  6,917/8,000 for at least one route and 4,877/8,000 for at least 20. Evidence:
  `outputs/label_regeneration/v1/post_generation/label_quality_summary_p5_v1.json`
  and `reports/label_regeneration_p5_summary.md`.
- P6 result: PASS on all 528,047 valid masks from 6,917 positive samples and
  36,163,535 exact unordered within-sample route pairs. Sample-balanced means
  are 13.20 ON/OFF transitions and 13.36/28 pairwise Hamming distance; only
  1.02% of valid masks have at most three transitions. Evidence:
  `outputs/label_regeneration/v1/post_generation/route_diversity_summary_p6_v1.json`
  and `reports/label_regeneration_p6_route_diversity.md`.
- Initial P7 design audit: zero overlap with the original 5,807-record external
  bundle by UID, sample ID, benchmark, exact image SHA-256, normalized
  question/instruction, normalized prompt, and exact image-question pair.
  Evidence:
  `outputs/label_regeneration/v1/post_generation/external_eval_overlap_split_audit_v1.json`,
  `outputs/label_regeneration/v1/post_generation/predictor_split_design_audit_v1.json`,
  and `reports/binary_router_p7_split_and_external_eval_audit.md`.
- Expanded evaluation audit: the updated bundle passes full image verification.
  The active selection contains 7,500 ChartQA/TextVQA core-VQA, 5,807 prior
  multiple-choice, and 9,000 POPE records. ChartQA test, TextVQA validation,
  MMStar, MMMU, and MMMU-Pro have zero exact MCTS image overlap. DocVQA is
  excluded. POPE has one shared image repeated in 18 records; report full 9,000
  plus a pre-specified 8,982-record image-disjoint sensitivity. Evidence:
  `outputs/label_regeneration/v1/post_generation/eval_suite_overlap_audit_v1.json`
  and `reports/binary_router_expanded_eval_suite_audit.md`.
- P7 split result: PASS. All 8,000 source records are frozen into exactly 7,000
  train and 1,000 validation records with zero cross-split image groups. The
  historical validation strata are exact: GQA 250 correct/250 wrong, TextVQA
  125/125, and ChartQA 125/125. Current outcomes were joined only after
  assignment and were not selection inputs. Manifest SHA-256:
  `4d12bf427f08b0cc55d21c82bf7eaac7d19d283dc514ffd4f59894d6faf1bd1a`.
  Evidence:
  `outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json`
  and `reports/label_regeneration_p7_predictor_split.md`.
- P8 result: PASS. The unchanged raw cache yielded 8,000 single-best and
  valid-set sample rows, 237,802 selected valid routes from 528,047 raw valid
  routes, 2,642,998 positive/negative ranking rows, and 237,802 exactly
  reconstructing POLAR-segment rows. There are 6,917 positive and 1,083
  zero-positive samples; 3,616 samples required the deterministic max-50 cap.
  Both predictor losses consume the identical selected route set with equal
  within-sample weights. Generation and independent streaming audits pass:
  `outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json`
  and `derived_supervision_verification_v1.json`.
- Evaluation adapter: reuse bundle manifest/input/scorer/reporting contracts,
  but replace SW31, forced K=8, and admission with each trained question-only
  predictor's static 28-bit mask and verified executor. Do not aggregate core
  VQA, multiple choice, and POPE into one overall accuracy.
- P9 result: PASS. The final audit binds all 8,000 raw records through the P4
  checksum index and freezes 50 primary/code/provenance files, the final report,
  and a 53-entry checksum ledger. Independent checksum verification passed
  53/53. Evidence: `reports/label_generation_report.md` and
  `outputs/label_regeneration/v1/post_generation/p9_final_audit_v1.json`.
- Next bounded action: with explicit approval, run only the matched P10 smoke
  for duplicated BCE versus exact set-NLL. Do not begin full training or
  external evaluation automatically.
- P10 readiness audit: PASS. The active config is
  `configs/binary_polar_loss_comparison_v2.yaml`; the smoke freezes 300 positive
  train and 150 positive validation records (balanced 100/50 per dataset), two
  epochs, and 18 actual-execution records per objective. The real Qwen3 BF16
  encoder preflight produced finite losses and finite gradients for both
  objectives from the same initialization, with zero encoder gradients and
  zero optimizer steps. Evidence:
  `outputs/binary_polar/preflight/p10_readiness_gate_v1.json` and
  `reports/binary_polar_p10_readiness_final.md`.
- Updated at: 2026-08-12
- Concurrent action: We-Math2.0-Pro all-sample binary-route extraction is
  approved under `plans/dynamic_mllm_wemath2pro_label_extraction_plan.md`.
  The benchmark adapter and focused tests pass; the 4,552-record manifest is
  frozen before a five-record smoke. On smoke pass, launch node06 with 8 GPUs,
  96 CPUs, and 240 GB RAM without cancelling job `99758`.
- Combined scheduler ceiling for the two explicitly concurrent jobs: 16 GPUs.
- We-Math validity amendment: all 4,552 records remain in the frozen inventory;
  the exact eight records with an empty question and/or answer are marked
  technical-invalid, leaving 4,544 records in the MCTS manifest. No We-Math
  GPU outcome was inspected while applying the rule. The original node02 job
  `99758` remains unchanged.
- We-Math frozen contract SHA-256:
  `96b2c632ebc6e020c607b3d9a0eddd2a29f7aff1912f5219327ae96a507c3a50`.
- We-Math smoke status: passed 5/5 ALL-ON/native generated-token parity and all
  repeated mixed-mask token/score checks. Evidence:
  `outputs/label_regeneration/wemath2pro_v1/smoke_report_v1.json`.
- Superseded We-Math sweep: Slurm job `99850` initially ran on node06 with
  eight workers and produced 1,156 retained complete records before the
  scoring-stall repair below.
- We-Math scoring-stall repair (2026-08-12): old job `99850` was cancelled only
  after a checksum-bound audit preserved 1,156 complete records. MathRuler is
  now bounded at five seconds; timeout is explicitly recorded and scored
  conservatively incorrect. Amended contract SHA-256:
  `fc4a1df38925d20816770b861989b87d119bcdbf13b3bdff26a89b7abc90d485`.
  Active resumable job `100398` uses the six currently available node06 GPUs,
  72 CPUs, and 180 GB RAM. All six ranks completed a new record with zero
  errors; the two previously stalled samples each completed 600/600 simulations
  with zero scorer-timeout flags. The old stall's exact cause therefore remains
  unknown, while the bounded scorer prevents one important recurrence class.
- We-Math cap amendment (2026-08-13): the user removed the 600-simulation
  extension after only 25/528 completed extensions found a correction after
  simulation 400. The new hard cap is 400. Only terminal 200/400 predecessor
  records may be reused; all 600-simulation records must be rerun. The planned
  replacement was audited into
  `outputs/label_regeneration/wemath2pro_cap400_v2/`: 640 terminal records were
  retained (207 at 200 simulations and 433 at 400), 529 post-cap records were
  excluded, and 3,904 remain. Slurm job `100407` is now running seven workers
  on node06 with 7 GPUs, 84 CPUs, and 197 GiB. The active cap-400 contract is
  `80c7ea4ca2ca9df091696290dc644a4092508337f89cf85ecc5b849a0f4092c7`.
  The 197G request is the largest whole-GiB allocation that fits beside the
  existing 48,000 MiB job on node06; it does not change model/search semantics.
  Live startup validation passed on the first three new terminal records: all
  completed 400/400 under the active contract, with no extension, errors,
  temporary files, or overlap with retained UIDs.
- Portable MCTS handoff: `handoff/binary_visual_mcts_reproduction_v1/`
  contains the verified executor/search/evaluator sources, portable dataset
  manifest and contract tooling, Slurm template, detailed runbook, six passing
  CPU-only tests, and a complete SHA-256 inventory. Packaging did not alter
  either active extraction job or its frozen contract.
- Updated at: 2026-08-11
- Predictor loss-comparison implementation (2026-08-12): implemented matched
  duplicated-route BCE and exact valid-set NLL paths without training. Stable
  complete-mask `logsigmoid`/`logsumexp`, variable-K padding, equal route
  weights, duplicate rejection, 50-route-cap enforcement, split-group leakage
  checks, matched unique-input batching, route microbatching, and predictor
  initialization hashing are in place.
- Objective sanity: pass. Single-route and padding errors are `0.0`; exact
  set-NLL reduced the contradictory-set loss `2.7723 -> 0.7028` and selected a
  coherent valid mode; duplicated BCE converged to bit marginals. Frozen
  encoder gradients are zero and all 21 predictor parameter tensors have
  finite gradients. Evidence:
  `outputs/binary_polar/preflight/loss_comparison_sanity_v1.json`.
- Training status: not started. The regenerated-label P9 cache/split/derived-
  view gates now pass. The next action remains a separately approved bounded
  matched smoke; do not launch full training.
- Active phase memory:
  `workspace/phase_memory/phase_14_binary_polar_loss_comparison.md`.
- Updated at: 2026-08-12
- P10 matched smoke (2026-08-13): completed two epochs on frozen 300-train and
  150-validation positives per objective, followed by 18 actual executions
  each. Exact set-NLL improved validation Hit@1 `0.1333 -> 0.5733` and
  execution accuracy `0.2222 -> 0.5000` versus duplicated BCE. However, every
  BCE execution decoded ALL-OFF and every exact-set execution decoded ALL-ON;
  exact set-NLL matched FULL accuracy with zero compute reduction. A constant
  policy audit reproduces the route metrics, so the smoke advantage is
  predominantly a constant ALL-ON prior rather than demonstrated conditional
  routing. Full training remains unexecuted pending explicit user direction.
  Evidence: `reports/binary_polar_p10_smoke_results.md`.
- P11 result (2026-08-13): complete, Outcome C. Validation label geometry has
  58.12% ALL-ON coverage and 57.89% ALL-ON-plus-cheaper-valid prevalence.
  Weighted exact set-NLL improves aligned versus shuffled set-NLL (`14.8699`
  versus `15.5089`) but decodes ALL-ON on 147/150 route-validation records and
  57/60 bounded executions. Execution accuracy is 50%, identical to FULL;
  W→C=0, C→W=0, and the three non-FULL masks remain wrong. Full training was
  not launched. Evidence: `reports/binary_polar_p11_results.md`.
- P12 result (2026-08-13): complete, Outcome B. All 237,802 selected route
  occurrences pass exact canonical maximal-run round trip, but label geometry
  is weakly segment-compressible (mean/median 14.11/14 segments). The selected
  structured checkpoint is 150/150 ALL-ON on route validation and 60/60
  ALL-ON in actual execution. Aligned structured set-NLL remains better than
  shuffled (`17.4918` versus `18.1939`), but decoded metrics are identical.
  Actual accuracy is 50%, W→C=0, C→W=0, and compute reduction is zero. Full
  structured-head training was not launched. Evidence:
  `reports/binary_polar_p12_results.md`.
- Active phase memory:
  `workspace/phase_memory/phase_17_p13_multimodal_input_isolation.md`.
- P13 result (2026-08-13): complete, Outcome B. The experiment reused the
  direct P11 head, exact valid-set NLL, P11 weighting, 300/150 identities,
  two-epoch budget, and seed, changing only visible predictor input among
  Question, Image, and Image+Question. The cache contains 502 frozen native
  projected-visual feature records (500 image groups), with no answer/outcome
  leakage and all preflight/checksum gates passing. Selected Image+Question
  aligned set-NLL is `14.4944`, versus Question-only `14.8699`; shuffling its
  image worsens NLL to `14.8748`. However, Image+Question decodes ALL-ON on
  150/150, exactly matching the constant baseline's 57.33% Hit@1, 3.693
  nearest-valid Hamming, and 28 mean ON layers. The prospective execution gate
  failed, so no P13 Qwen route execution or full training ran. Evidence:
  `reports/binary_polar_p13_results.md` and
  `outputs/binary_polar/p13/analysis_manifest_v1.json`.
- Full10 result (2026-08-13): complete. Both runs used 6,043 positive train
  and 874 positive validation inputs, exact set-NLL, max-50 route sets, batch
  128, AdamW `5e-4`, cosine schedule, and 10 epochs. All 20 checkpoints and
  hashes are preserved. Best-Hit@1 is epoch 2 for Question-only and epoch 4
  for Image+Question; both equal constant ALL-ON at `58.12%`. Epoch-10
  diversity rises to 122/64 unique masks but Hit@1 falls to
  `55.03%`/`55.84%`. Actual frozen-60 best-checkpoint executions are both
  `50%`, W→C=0, C→W=0. Question-only epoch 10 has two uncached ChartQA
  corrections but is not the selected checkpoint. Do not admit the direct
  predictor to external evaluation. Evidence:
  `reports/binary_polar_full10_polar_matched_results.md` and
  `outputs/binary_polar/full10/`.
- Active phase memory:
  `workspace/phase_memory/phase_19_full10_external_evaluation.md`.
- Updated at: 2026-08-13
