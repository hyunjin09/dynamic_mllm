# Phase 32: Route-Conditioned Four-Action Decomposition Memory

## Current Objective

Execute `plans/4way_2.md`: decompose every OFF layer of one deterministic,
currently correcting binary anchor route per frozen A+ sample into READ and
WRITE components while preserving all other route actions.

## Active Constraints

- Use the frozen current-runtime 1,880-sample A+ cohort; do not redefine it.
- Select cached correcting routes deterministically by minimum OFF count, then
  the plan's stable tie-break; validate correction in the current unified
  executor, fall back in order, and exclude only if no cached route remains
  current-correct.
- At one anchor OFF layer evaluate BOTH_OFF/M00, WRITE_OFF/READ_ONLY/M10,
  READ_OFF/WRITE_ONLY/M01, and FULL-restore/M11 while every other anchor-route
  action remains fixed.
- Run no combinatorial four-action search and no optional joint refinement.
- Pass unit, semantic, numerical, resume, and throughput gates on a 48--64
  stratified pilot before automatically launching the full A+ sweep.
- GPU work uses Slurm and all eight H100s; CPU-only work runs locally by
  default under the machine-local server policy.
- Preserve the dirty worktree and all append-only completed artifacts.

## Current State

- Done: Phase 31 trajectory rescue, aggregate analysis, numerical-consistency
  report, and final report completed with verified checksums and no
  disqualifying failure.
- Done: audit, deterministic 1,880-row anchor-candidate manifest, minimal
  arbitrary-route unified executor extension, current-runtime anchor validator,
  pilot/full runner, resumable mergers, GPU monitor, pilot comparator, and
  aggregate-analysis implementation. The complete four-action suite passes 84
  tests.
- Done: current-runtime validation and local anchor freeze. Job `1578` completed
  `0:0`; 1,804/1,880 samples retain a current-correct cached anchor (GQA 1,170;
  TextVQA 634), while 76 are excluded under the prespecified no-current-correct-
  cached-route rule. The frozen anchors contain 17,262 OFF positions and 51,786
  new production cells across 32 cost-balanced work units.
- Done: both matched 56-sample pilots passed every semantic/numerical gate and
  zero failures. Two replicas/GPU delivered 12.183885 valid new cells/s,
  1.414456x one replica, at 34,745 MiB peak VRAM/H100, and was selected.
- Done: full job `1581` completed `0:0` in 29m13s; the local exact merger,
  aggregate analysis, figures, final report, proposed-but-unlaunched joint
  refinement plan, and final integrity audit all pass. Final coverage is 1,804
  samples, 17,262 OFF positions, 51,786 new evaluations, 69,048 action rows,
  16 workers, and zero failures. The focused suite passes 90 tests.
- Blocked: none.
- Most recent useful observation: 45.65% of anchor-OFF positions are necessary,
  but FULL-context local rescue recalls only 7.30% of them; among necessary
  positions WRITE-mediated suppression is most common (42.88%), and READ-
  mediated positions occur 7.82 layers later on average than WRITE-mediated.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Prior final pipeline completed with exact primary coverage and checksum-clean outputs | `analysis/4action_answer_alignment/4action_answer_unaligned_report.md`; job `1573` log | Satisfies the explicit prerequisite in `plans/4way_2.md` | confirmed |
| Frozen current-runtime primary cohort has 1,880 samples | `analysis/4action_answer_alignment/cohort_eligibility__unified_v1/summary.json` | Defines the route-conditioned population | confirmed |
| Nearest-route OFF layers are more locally harmful in FULL context | `analysis/4action_answer_alignment/aggregate/route_overlap_summary.json` | Motivates, but does not answer, route-conditioned decomposition | confirmed |
| Discovered routes are search-selected and do not prove layer necessity | `workspace/decision_log.md` | Bounds causal interpretation of anchor-route results | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Cross-path native FULL factorial baseline | BF16 margin drift was comparable to intended effects | supported execution-path mismatch | Phase 31 numerical reports | Keep all route-conditioned M00/M10/M01/M11 cells inside the unified executor | Native FULL as M11 |
| First current-runtime anchor launch, job `1576` | Two ranks stopped before producing a scientific result because deterministic CuBLAS lacked its required workspace setting | supported launch-contract omission | `logs/slurm/four-action-route-anchor-v1-20260824-1576.log`; two preserved `anchor_validation/shard_*/failures.jsonl` rows | Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` for every route-conditioned GPU launch; resumable merger treats the old failures as recovered only after successful rows exist | Launching this runtime without the deterministic CuBLAS environment variable |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Bounded joint-relaxation pilot | 73.36% of necessary positions allow an individual partial restoration | Whether several partial restorations compose | proposed 4,096 evaluations / ~0.75 GPU-hours | not authorized |
| Stop after completed evidence | The approved route-conditioned question is answered | Avoids overclaiming router value | none | valid fallback |

## Next-Step Decision

- The approved action is complete; no next experiment is authorized.
- Preserve `proposed_joint_refinement_plan.md` as a proposal only. Its purpose
  would be to test composability before any four-action search/router pivot.
- Stop condition reached: full validated sweep, local analysis, final report,
  and integrity audit all pass.

## Latest Research-Action Result

- Action taken: completed current-runtime anchor validation, matched concurrency
  pilots, the full route-conditioned sweep, local merge/analysis, and reporting.
- Result: PASS. 1,804/1,880 samples retain current-correct cached anchors.
  7,880/17,262 OFF positions are individually necessary and 9,382 redundant.
  Among necessary positions, READ-mediated/WRITE-mediated/either/both shares
  are 20.55%/42.88%/9.94%/26.64%. READ-mediated depth is later by 7.82 layers
  on average (95% CI 7.31--8.31). FULL-context discrete rescue recalls only
  7.30% of route-necessary positions.
- Evidence saved: `analysis/4action_route_conditioned/`, especially
  `route_conditioned_decomposition_report.md` and
  `final_integrity_audit.json`.
- Failure or issue: none unresolved; the initial CuBLAS launch omission was
  repaired before scientific output and did not affect results.
- Lesson learned: cached binary routes contain substantial individually
  redundant OFF choices, and operation necessity is trajectory-context
  dependent.
- Next implication: stop. The approved experiment is complete. The optional
  joint-refinement pilot is proposed only and requires explicit approval.

## Final Interpretation Decision

- Deliberation mode: STANDARD, with one required independent review.
- Active objective and bottleneck: determine whether the evidence justifies a
  true four-action search/router; composability of individually valid partial
  restorations remains untested.
- Confirmed observation / unverified interpretation: 5,781/7,880 necessary
  positions permit at least one partial restoration individually; whether
  several such restorations compose is unknown.
- Diagnosis: unknown for joint composability; the current study did not test it.
- Viable alternatives considered: stop; propose a bounded 64-sample joint-
  relaxation beam pilot; move directly to a full search/router.
- Chosen action and strongest objection: save the bounded proposal without
  launching it. It is the smallest test of composability, but would still not
  establish learned-router value or global optimality. Independent verdict:
  `stable`, high confidence, ranking proposal > stop > direct pivot.
- How this differs from failed attempts: it does not infer a new router from
  FULL-context effects or one-position route-conditioned relaxations.
- Authorization and stop condition: reporting/proposal only is within the
  approved plan; any execution is a strategic next action requiring explicit
  user approval. Stop after the verified final report and audit.
