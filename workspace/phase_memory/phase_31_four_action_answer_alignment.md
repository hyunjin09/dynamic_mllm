# Phase 31: Four-Action Answer Alignment Memory

## Current Objective

Test whether binary-route-correctable GQA and TextVQA errors are causally
attributable to layer-local visual READ, visual WRITE/update, or their
interaction, following `plans/4way.md`.

## Active Constraints

- Use Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, and the frozen binary
  executor semantics.
- The matched cache supplies A+ candidates and correcting routes. Primary
  eligibility additionally requires current unified FULL to remain wrong;
  current unified-FULL-correct candidates are retained as provenance but are
  excluded from the primary factorial analysis.
- A local intervention changes exactly one of 28 decoder layers; every other
  layer remains FULL.
- Do not train, fine-tune, rerun MCTS, Pareto-filter, or select only short
  Hamming-distance cases.
- Unit checks, real-example semantics, about-5 smoke, and about-50 pilot gate
  the exhaustive run.
- GPU work uses the machine-local Slurm scheduler; completed shards are
  resumable and never overwritten. Every GPU-stage submission requests all
  eight H100s and launches eight parallel workers; do not downsize while the
  server is occupied.

## Current State

- Done: read the complete four-action plan and audit irreplaceable assets.
- Done: confirmed 1,235 GQA and 677 TextVQA A+ records; both prescribed
  control cohorts are present.
- Done: unified materialized-mask four-action extension, continuation
  scoring/generation, frozen cohort, and focused static/unit checks.
- Done: deterministic causal-result flattening and aggregate-analysis tooling
  for factorial effects, rescue taxonomy, sample/image-group bootstrap,
  Hamming strata, correcting-route overlap, control comparisons, and core
  figures. Its focused suite passes 23/23 together with the executor/cohort
  tests.
- Done: resumable population trajectory-followup selection/execution and final
  bootstrap/report automation.
- Done: unified preflight, 8-example all-layer smoke, and adjudicated 56-example
  validation passed every current semantic/structural gate.
- Done: an all-eight-GPU unified-FULL eligibility freeze evaluated all 4,890
  primary/control candidates in job `1505`. It retained 1,880 primary A+
  samples (1,222 GQA, 658 TextVQA), all 868 no-correction-found controls, and
  2,084 vision-required controls; 58 candidates were excluded.
- In progress: the two-replica ramp was rejected after preserving 1,574/1,880
  unique eligible primary results because throughput fell to 0.8004x the
  one-replica baseline. One-replica resume job `1557` is pending behind another
  user's all-GPU job. Jobs `1558`, `1561`, `1562`, `1563`, and `1564` form the
  replacement control, trajectory-rescue, aggregate-analysis, and final-report
  dependency chain.
- Blocked: no scientific decision blocker. GPU validation may queue behind
  other users, but the full requested pipeline is authorized.
- Most recent useful observation: one transferred FULL anchor changed one token
  while current native and unified FULL stayed token-identical and evaluator-
  wrong. Exact historical token identity is provenance-only; correctness still
  gates cohort validity. The unified executor passes 50 focused tests;
  its target layer always performs the same full-row and compacted-row calls
  from identical pre-layer state, while the action selects only the prescribed
  text output, visual output, and cache.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| All 8,000 indexed raw cache records are present and unique | `datasets/mcts_labels/gqa_textvqa_chartqa_v1/post_generation/cache_record_index_v1.jsonl` | Cohorts and all known binary routes can be recovered without new MCTS | confirmed |
| All 6,000 GQA/TextVQA manifest images map to the transferred image tree | `datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images/` | No primary/control image payload is missing | confirmed |
| Frozen binary executor source hashes all match | `datasets/mcts_labels/gqa_textvqa_chartqa_v1/frozen_execution_contract.json` | Preserves the label-generation ON/OFF mechanism | confirmed |
| Legacy four-state execution does not use compacted text attention for READ-off | `interventions/read_path.py`; `interventions/four_state.py` | It cannot satisfy the new plan's required IGNORE parity without adaptation | confirmed |
| Existing static unit checks pass 34/34 | focused pytest run in the 2026-08-23 readiness audit | Existing executor and scoring primitives are intact | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat legacy four-state `IGNORE` as binary `VISUAL_OFF` | The former subtracts a visual value path under full-attention normalization; the latter runs a compacted text/control layer | supported semantic mismatch | `interventions/read_path.py`; `binary_policy/executor/layers.py` | Extend the binary executor from its ON/OFF primitives | Do not validate only by action names or legacy reports |
| Launch all transferred A+ candidates without a current unified-FULL correctness freeze | Job `1497` stopped on `gqa:gqa_ge_16564303`: transferred FULL was wrong but current native and unified FULL both generated the correct answer | supported cohort-boundary drift, not an executor semantic disagreement | `logs/slurm/four-action-unified-primary-r2-20260823-1497.log`; `analysis/4action_answer_alignment/primary__unified_v1/shard_04/results.jsonl` | Preserve matched-cache route provenance but gate causal cohorts on current unified-FULL correctness | Do not force current FULL-correct rows into a FULL-wrong estimand |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Compose four actions from frozen full-row and compacted-row calls | Gives exact FULL/IGNORE endpoints and independent READ/WRITE hybrids | Required intervention semantics and cache geometry | medium | testing |
| Reuse the hook-based legacy executor unchanged | Prior four-action experiments used it | Minimal implementation effort | low | rejected |

## Next-Step Decision

- Deliberation mode: standard; the user specified the estimand and continuation
  policy.
- Active objective and bottleneck: complete the exhaustive eligible A+ sweep,
  controls, trajectory rescues, and report. The current bottleneck is another
  user's live all-eight-GPU allocation; one-replica primary resume `1557` is
  safely pending.
- Relevant memory item used: native/materialized BF16 drift cannot enter a
  factorial contrast; every causal comparison is now within the unified path.
- Confirmed observation: the implementation and analysis suite passes 62
  focused tests. Validation passed 56/56 current semantic checks. Eligibility
  job `1505` passed all candidate, shard, worker, and failure gates and froze
  1,880 primary, 868 no-correction, and 2,084 vision-required eligible rows.
- Chosen action and strongest objection: reject the slower two-replica ramp and
  run the resumable one-replica jobs `1557`--`1564` automatically. The strongest
  objection is lower instantaneous GPU utilization, but observed aggregate
  sample throughput is the relevant efficiency measure and favored one replica.
- How this differs from failed attempts: M11 and every intervention now use one
  materialized-mask executor; native FULL and old binary OFF are external
  semantic diagnostics only.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform `plans/4way.md`, with
  permission to submit pending Slurm work and the explicit requirement that
  every GPU stage allocate and parallelize across all eight H100s. The smoke
  and pilot are correspondingly fixed at 8 and 56 examples.
- Stop condition: only an unresolved semantic READ/WRITE or generation/
  correctness validation failure stops the chain. Native/unified continuous
  BF16 drift alone does not.

## 2026-08-23 Trajectory Target-Identity Repair

- Deliberation mode: deep; the supporting trajectory stage failed twice, with
  the second failure exposing a distinct multi-target comparison error.
- Active objective and bottleneck: finish the authorized population trajectory
  rescue while preserving a fixed correct-answer token sequence across FULL
  and operation-suppressed trajectories. The current gate compares that fixed
  trajectory to the intervention state's best evaluator-valid target, which
  may be a different phrase.
- Confirmed observation / unverified interpretation: for
  `textvqa:textvqa_27002`, the trajectory target `not question` ends at
  `-5.5426580686`, exactly its candidate-specific intervention margin, while
  the state margin `-5.0508334417` selects the alternate valid target `yes`.
  This is direct evidence of a target-identity mismatch, not a READ/WRITE or
  cache semantic failure. Evidence is in trajectory selection
  `trajectory_006051` under `trajectory_rescue/results/shard_03/`.
- Diagnosis: supported gate-definition mismatch; the stored candidate scores
  exactly reconstruct the observed trajectory endpoint.
- Viable alternatives considered: gate against the fixed baseline-selected
  target and report target switching separately; dynamically reselect among
  all valid targets throughout the trajectory; or stop. Fixed-target identity
  is the only option that preserves a longitudinal FULL/suppressed comparison.
- Chosen action and strongest objection: reconstruct the intervention margin
  for the exact fixed trajectory target, gate against it, retain evaluator-best
  state margins, and add phrase-switch diagnostics. A fixed phrase can
  understate semantic support transferred to another valid phrase, so the
  endpoint equivalence-set sidecar remains explicit.
- How this differs from failed attempts: this changes the compared quantity,
  rather than relaxing another numerical tolerance.
- Independent review: `trajectory_target_review` returned `stable`, ranked the
  fixed-target repair first with high confidence, and recommended the same
  endpoint target-switch sidecar.
- Authorization and stop condition: this is repair required to finish the
  already authorized trajectory-rescue research action; resume unfinished
  cells after focused and full regression tests pass, and stop only for a new
  unresolved semantic failure.

## 2026-08-23 Current-Runtime Eligibility Repair

- Confirmed observation: primary job `1497` failed after 46 seconds because one
  matched-cache A+ candidate is now correct under both native and unified FULL.
  The two current paths agreed semantically; the stale cohort anchor was the
  mismatch.
- Independent review verdict: revise. The matched cache remains authoritative
  for candidate discovery, ALL-OFF status, and binary correcting routes, while
  current unified FULL must satisfy the defining wrong/correct condition of
  each factorial cohort.
- Selected repair: freeze unified-FULL generation and evaluator correctness for
  all 4,890 candidates before production. Exclude current unified-FULL-correct
  rows from both FULL-wrong cohorts and current unified-FULL-wrong rows from the
  FULL-correct control. Never use this gate as a continuous effect threshold.
- Result: job `1505` completed on eight H100s with 4,890/4,890 rows, eight of
  eight workers, and zero failures. Eligible counts are primary 1,880
  (GQA 1,222; TextVQA 658), no-correction 868 (614; 254), and vision-required
  2,084 (1,137; 947). Exclusions are 32 primary and 26 vision-required.
- Historical execution status: primary job `1506` began automatically and
  produced the preserved one-replica prefix before the authorized utilization
  relaunch described below.

## 2026-08-23 Utilization Repair

- Confirmed observation: one-replica job `1506` used about 17.5--17.9 GiB per
  80-GiB H100 and sustained roughly 24--30% sampled SM utilization while every
  Python worker saturated one CPU core. The original layout ran one sample per
  GPU and left both GPU memory and execution capacity underused.
- Authorized action: the user explicitly allowed cancelling the current runs
  and relaunching for better utilization.
- Repair: production and trajectory runners now support multiple independent
  replicas per GPU. The first ramp uses two replicas/H100, maps 16 workers into
  two disjoint streams per original shard, preserves deterministic GPU-local
  seeds, and writes per-replica append-only artifacts. Mixed old/new artifacts
  merge only after exact uniqueness and all-eight-GPU worker coverage checks.
- Verification: 62 focused tests pass. Job `1506` was recoverably cancelled at
  1:23:07 with 1,501 unique eligible results and 379 remaining. No completed
  result was removed.
- Ramp result: monitor `1554` found complete 16-worker coverage, unique samples,
  passing semantic gates, no failure artifacts, and no OOM. Live GPUs reached
  about 99% utilization and 35 GiB/H100, but throughput was 14.4067 samples/min,
  only 0.8004x the one-replica baseline. The optimization was rejected.
- Live state: one-replica primary `1557` is pending `AssocGrpGRES` because
  another user's job `1551` currently owns all eight H100s. Replacement chain:
  controls `1558`/`1561`, CPU selection `1562`, one-replica trajectory rescue
  `1563`, and CPU final analysis/report `1564`.
- Performance gate: keep two replicas only after live samples/minute materially
  exceeds the one-replica baseline with no OOM or semantic/determinism failure.
  CPU monitor `1533` automatically checks the first 64 new samples and gates
  jobs `1527`--`1529`; the threshold is at least 1.20x the measured 18.0
  samples/minute baseline.

## Latest Research-Action Result

- Action taken: completed the primary and control sweeps and the full
  population trajectory-rescue stage, including resumable repairs of the
  downstream audit boundaries.
- Result: primary passed exact 1,880-row coverage and all semantic gates;
  trajectory rescue passed exact 10,196-cell coverage (5,630 GQA and 4,566
  TextVQA), all result/worker/failure gates, and both saved checksums. Eleven
  evaluator-unscorable TextVQA rows are excluded from Control A, leaving 857
  analyzable controls. The focused suite passes 67 tests.
- Evidence saved: `analysis/4action_answer_alignment/experiment_log.md`, all
  primary/control stage summaries, `trajectory_rescue/summary.json`, the
  checksum-verified trajectory merge, preserved failure artifacts, and jobs
  `1565`/`1567`/`1572`/`1573`.
- Failure or issue: no READ/WRITE semantic failure. Control A contained 11 rows
  for which the requested correct-answer margin is mathematically undefined;
  two supporting trajectory comparisons crossed an overly strict `1e-5`
  numerical-identity gate but stayed within `1e-4` with semantics unchanged.
- Lesson learned: scorer-target validity must be audited separately from FULL
  correctness, and logit-lens/direct-score equality is a BF16 diagnostic rather
  than a causal effect.
- Next implication: let dependent CPU final analysis/report `1573` complete,
  verify all current-experiment outputs, and only then begin the approved
  route-conditioned decomposition in `plans/4way_2.md`.

## 2026-08-24 Final Aggregate Completion

- Action taken: completed and verified the dependent aggregate analysis and
  final report after the trajectory-rescue prerequisite passed.
- Result: job `1573` completed with exit `0:0`; all generated SHA-256 sidecars
  verify; the analysis contains 4,821 samples, all 1,880 primary A+ samples,
  539,952 flat action rows, and 2,000 bootstrap replicates.
- Evidence saved: `logs/slurm/four-action-final-analysis-r9-target-identity-
  20260823-1573.log`, `analysis/4action_answer_alignment/aggregate/
  analysis_summary.json`, `numerical_consistency_report.md`, and
  `4action_answer_unaligned_report.md`.
- Failure or issue: none disqualifying. Native/unified FULL semantics match on
  72/72 comparisons and unified IGNORE/old binary OFF semantics match on
  1,816/1,816 comparisons.
- Lesson learned: the completed FULL-context causal map supports route overlap
  but does not identify which READ/WRITE component is responsible inside a
  multi-layer correcting route; the discovered routes remain search-selected.
- Next implication: Phase 31 is complete. Begin the explicitly approved,
  route-conditioned audit and deterministic anchor freeze in
  `plans/4way_2.md`.

## 2026-08-23 Parity-Failure Addendum

- Deliberation mode: deep after the same parity class failed twice.
- Confirmed observation: independent review recommended a decision-metric
  bridge. Job `1483` passed every structural/native/generation preflight gate,
  but the bridge failed: selected-correct mean-logprob drift was
  0.0771--0.1230 and correct-vs-wrong margin drift was 0.0625--0.1250 across
  layers 0, 13, and 27 on the completed sample. Margin signs remained stable.
- Diagnosis: supported BF16 dispatch drift between native maskless FULL
  prefix/suffix and the frozen mixed-route materialized-mask prefix/suffix;
  evidence is `analysis/4action_answer_alignment/preflight__bridge_v1/`.
- Viable alternatives: preserve native FULL and explicitly treat frozen-route
  score parity as approximate; switch to frozen materialized-mask FULL and no
  longer exactly reproduce the original model; or stop.
- Chosen action and strongest objection: stop and request user direction rather
  than post-hoc relax a second gate. The cost is idle GPUs, but it preserves the
  approved estimand and avoids launching an expensive ambiguous analysis.
- How this differs from failed attempts: no third tolerance-only retry.
- Authorization and stop condition: a baseline/validation convention change
  requires explicit user approval before smoke.

## 2026-08-23 Unified-Executor Decision

- Deliberation mode: standard; the user supplied the missing estimand choice.
- Active objective and bottleneck: eliminate cross-path numerical drift from
  causal contrasts by defining all M00/M10/M01/M11 inside one executor.
- Confirmed observation / unverified interpretation: native-maskless versus
  mixed materialized-mask execution causes score drift up to about 0.125; a
  materialized-mask unified FULL prefix/baseline/suffix should remove that
  cross-path comparison from the factorial without changing target actions.
- Diagnosis: supported execution-path numerical inconsistency; evidence under
  `preflight__bf16_diagnostic_v1/` and `preflight__bridge_v1/`.
- Viable alternatives considered: native M11 was explicitly rejected by the
  user; stopping is unnecessary because target semantics passed; unified FULL
  is the authorized and scientifically coherent option.
- Chosen action and strongest objection: use unified materialized-mask FULL for
  every factorial branch and retain native/binary paths only as external
  semantic diagnostics. The strongest objection is native/unified score drift,
  which will be reported separately and never used as an effect threshold.
- How this differs from failed attempts: no factorial quantity mixes native
  and materialized execution paths.
- Authorization and stop condition: the user explicitly authorized validation,
  full A+ sweep, controls, erosion/trajectory analysis, and final reporting;
  stop only for an unresolved READ/WRITE semantic bug.
