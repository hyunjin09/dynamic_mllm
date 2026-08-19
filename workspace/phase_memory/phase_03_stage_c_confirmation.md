# Phase 03: Stage C Held-Out Confirmation Memory

## Current Objective

Preserve the completed frozen Outcome B and its interpretation boundary. The
harmful layer-0 READ hypothesis is closed; do not enter Stage D or search for a
READ-specific harmful mechanism.

## Active Constraints

- Primary endpoint is exactly per-token accepted-reference
  `FULL - WRITE_ONLY` at TextVQA layer 0.
- Primary success requires an image-clustered 95% bootstrap CI entirely below
  zero; `-0.05` nats/token is secondary only.
- Target 800 new eligible unique-image records with no Stage B record/image
  overlap; freeze the manifest before intervention outcomes.
- Actual READ removal must outperform covariance/subspace-matched and
  norm-matched real cross-sample residual controls for the “confirmed
  answer-misaligned” description.
- Other layers, datasets, operations, strata, interactions, and greedy outcomes
  cannot replace the primary analysis.
- No router training, model fine-tuning, Stage D execution, harmful-mechanism
  claim, or accuracy-improvement claim.

## Current State

- Done: The approved endpoint, 800-record manifest, structured nulls, exact
  `19/12` donor amendment, fresh full sweep, frozen aggregation, and Outcome
  A/B/C assignment are complete.
- In progress: None.
- Closed: Progression to a harmfulness claim and Stage D. Both structured-null
  superiority gates failed, and the user explicitly closed this path.
- Most recent useful observation: The primary reference-support effect
  replicated, but it was indistinguishable from both structured null families.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Explicit modified Stage C approval | User's 2026-08-05 “Stage C Amendment Decision” | Resolves the former metric/control authorization conflict. | confirmed |
| Frozen amended endpoint and gates | `workspace/stage_c_reference_likelihood_proposal.md` | Prevents layer/dataset/operation search and overclaiming. | confirmed |
| Stage B variance-based size analysis | `outputs/stage_c_prerun/power_analysis_v1.json` | Retains the target of 800 unique-image records. | confirmed for planning |
| Stage B distribution is heavy-tailed | `outputs/stage_c_prerun/power_analysis_v1.json` | Power is approximate and outcome-dependent resizing is invalid. | confirmed |
| Structured null modules are absent | `workspace/research_plan.md` repository map | Required controls must be specified and validated before confirmation. | confirmed |
| Frozen non-overlapping manifest | `outputs/stage_c/manifest/stage_c_manifest_v1.jsonl`; `workspace/stage_c_eligibility_overlap_audit.md` | Establishes the immutable held-out population before outcomes. | confirmed |
| Frozen null fit and final smoke | `workspace/stage_c_structured_null_spec.md`; `outputs/stage_c/nulls/null_calibration_and_smoke_v1.json` | Establishes rank, matching, seeds, comparison, parity, reconstruction, and generator validity. | confirmed |
| Full-sweep outcome-blind preflight failure | `outputs/stage_c/failures/stage_c_preflight_failure.json`; `outputs/stage_c/preflight/stage_c_prefix_span_diagnostic_v1.json` | Shows the required literal prefix is incompatible with the frozen exact span identity on 1,834/2,279 answer components; no endpoint was computed. | confirmed |
| Approved prefix amendment and revised pass | `workspace/stage_c_prefix_contextual_tokenization_amendment_v1.md`; `outputs/stage_c/prefix_preflight_v1/summary.json` | Contextual suffix IDs preserve the exact literal text; all 2,279 components pass and reproduce scores exactly before outcome execution. | confirmed |
| Frozen real-null coverage failure | `outputs/stage_c/failures/stage_c_shard_00_failure.json`; `reports/stage_c_full_sweep_failure.md` | One held-out target has only 7 eligible donors under the 1.5 cap; the required eight-draw null cannot be generated without amendment. | confirmed |
| Outcome-blind donor-coverage audit | `outputs/stage_c/preflight/stage_c_donor_coverage_audit_v1.json`; `workspace/stage_c_donor_coverage_audit.md` | Exact geometry-only audit finds 798/800 supported at 1.5 and minimum common `c_star = 1.5833333333333333`; no Stage C outcome was loaded. | confirmed |
| Frozen Stage C aggregation | `outputs/stage_c/analysis_v1/analysis_manifest.json`; `reports/stage_c_results.md`; `reports/stage_c_conclusion.md` | Primary CI is wholly negative, but both structured-null paired CIs cross zero; exact decision is Outcome B. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Prior Stage C draft under unchanged controls | Open-ended TextVQA could not support option/label permutations without prohibited synthetic distractors. | supported | Superseded status in `workspace/stage_c_reference_likelihood_proposal.md`; user amendment | Use the explicitly approved replacement controls. | Reintroducing distractors or option permutations. |
| First Stage C entry smoke | FULL/reconstruction parity, execution, shapes, norm matching, donor exclusion, determinism, and serialization passed, but one mapped covariance draw exceeded the provisional native-subspace tolerance (`0.0603 > 0.05`) and two calibration targets had fewer than eight donors under independent `1.5/1.25/1.25` ratio caps. | supported | `outputs/stage_c/nulls/null_calibration_and_smoke_v1_first_failed.json`; one geometry-only diagnostic showed widening norm alone through `3.0` left one target with only one donor. | Validate affine-subspace membership before lossy row interpolation. Freeze a composite max-ratio caliper from the Stage B eighth-nearest leave-one-image-out geometry, then fail closed for future targets beyond that cap. | Reusing independent fixed caps or treating interpolation error as covariance-subspace leakage. |
| Frozen full-sweep preflight | The literal `Answer: ` robustness prompt failed prompt-prefix and standalone-suffix token identity on 1,834/2,279 answer components. | supported | `outputs/stage_c/preflight/stage_c_prefix_span_diagnostic_v1.json` | A trailing-space BPE boundary cannot satisfy the frozen standalone-token identity for most answers. Approval is required to use contextual combined-text suffix IDs or drop the robustness check. | Launching the sweep, silently filtering components, or changing prefix/scoring rules without approval. |
| First final aggregation attempt | All calculations completed, then the integrity conjunction failed because a truthful negative status field (`prior_partial_records_reused: false`) was placed in an all-true assertion map. | supported | `runs/stage_c_full_v2_20260805/analysis/slurm.log`; `tools/research_analysis/v2/run_stage_c_analysis.py` | Use the positive assertion `prior_partial_records_not_reused: true`; retry the identical frozen calculation without changing outcomes or seeds. | Interpreting this as a scientific failure or changing the protocol. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Preserve and archive frozen Outcome B | Preserves the complete positive and negative evidence with interpretation boundaries. | Closes the approved hypothesis without overclaiming secondary behavior. | low | selected and complete |

## Next-Step Decision

- Deliberation mode: STANDARD result interpretation under a frozen decision
  table.
- Active objective and bottleneck: Decide whether Stage C confirms the
  answer-misaligned READ effect; structured-null specificity is the bottleneck.
- Confirmed observation / unverified interpretation: Primary mean
  `-0.07294332` has CI `[-0.14127645, -0.01710262]`, but covariance and
  real-residual paired CIs cross zero. The reason the real and null effects are
  similar is not identified and need not be diagnosed for the frozen decision.
- Diagnosis: **supported** Outcome B gate result; exact evidence is in
  `outputs/stage_c/analysis_v1/`.
- Viable alternatives considered: Close as Outcome B; or seek approval for a
  strategic redesign. Stage D is not viable because the confirmation gate did
  not pass.
- Chosen action and strongest objection: Close Stage C as Outcome B and stop.
  The strongest case for continuing is that the primary CI replicated and the
  wrong-answer contrast was positive, but neither secondary result can override
  the prospectively required structured-null conjunction; prefix robustness is
  also weak.
- How this differs from failed attempts: This is the complete fresh 800-record
  protocol, not a partial run or a coverage diagnostic.
- Authorization and stop condition: Freeze and archive Outcome B. Do not search
  another endpoint, enter Stage D, or pursue a READ-specific harmful mechanism.

### Full-sweep failure response

- Deliberation mode: DEEP because the frozen null-coverage assumption repeated
  a prior donor-coverage risk and affects confirmation validity.
- Active objective and bottleneck: Complete Stage C without changing its
  frozen null; one target has only seven of eight required donors.
- Confirmed observation / unverified interpretation: The selector raised on
  exact coverage. Whether a minimally wider cap would remain scientifically
  acceptable is unverified; no likelihood aggregate was inspected.
- Diagnosis: **supported** immediate cause—insufficient donors under the fixed
  1.5 geometry rule; evidence in the shard failure JSON and log.
- Viable alternatives considered: outcome-blind manifest-wide geometry audit;
  expand the donor pool; globally reduce draws; or stop inconclusive.
- Chosen action and strongest objection: Stop and request approval for only the
  geometry audit. It uses held-out activation geometry after partial outcome
  files exist, although those outcomes remain unopened.
- How this differs from failed attempts: No caliper, donor pool, draw count,
  sample, or outcome was changed or searched after failure.
- Authorization and stop condition: The current full-sweep action is complete
  as a failed execution. Any diagnostic or null amendment needs new approval.

### Donor-coverage audit action

- Deliberation mode: FAST; the user specified one deterministic geometry-only
  audit and prohibited alternative distances, pools, subsets, and outcomes.
- Active objective and bottleneck: Determine the exact minimum common caliper
  supplying eight frozen donors for every one of the 800 unchanged targets.
- Confirmed observation / unverified interpretation: One target failed at 1.5;
  the manifest-wide eighth-neighbor distribution and `c_star` are unknown.
- Diagnosis: **supported** immediate coverage failure; this audit measures its
  extent without testing a new scientific hypothesis.
- Chosen action and strongest objection: Re-extract only layer-0 postvisual
  READ-residual geometry for all targets, apply the unchanged distance and
  exclusions, and merge only geometry. The strongest objection is using
  held-out activation geometry after partial score files exist; the audit code
  therefore has no path to those files and loads no answer/outcome data.
- Authorization and stop condition: Explicitly authorized by the user's donor-
  coverage audit task. Stop after reporting the single candidate caliper
  amendment; do not freeze it or resume Stage C.

### Donor-coverage result interpretation

- Deliberation mode: STANDARD because minimal-versus-substantial interpretation
  affects whether the local amendment should be recommended.
- Active objective and bottleneck: Restore eight-donor coverage without
  weakening any other frozen Stage C null rule; the 1.5 cap supports 798/800.
- Confirmed observation / unverified interpretation: The exact common cap is
  `1.5833333333333333`, 5.56% above 1.5; only two targets need widening. Whether
  this limited geometric relaxation will affect the scientific null comparison
  is unverified because outcomes remain unopened.
- Diagnosis: **supported**. The two failures arise from their frozen geometric
  eighth-neighbor distances, recorded in the audit JSON.
- Viable alternatives considered: the task permits only the exact `c_star`
  candidate; changing donor pool, distance, covariates, draw count, or subset is
  prohibited.
- Chosen action and strongest objection: Recommend the exact `c_star` as a
  minimal local repair and stop for approval. The determining target has only
  three donors at 1.5 and needs five donors at the new boundary, so its local
  image-token match is weaker than the cohort norm.
- How this differs from failed attempts: It changes no protocol item and opens
  no outcome; it only measures the previously unknown frozen geometry.
- Authorization and stop condition: Audit and report only. Do not amend the
  donor index, restart Stage C, or enter Stage D without explicit approval.

### First-failure response

- Direct observation: the first smoke gate was false for the two conditions in
  the failure table; the held-out endpoint remained uncomputed.
- Diagnosis: **supported**. The subspace failure was a validation-location bug:
  it tested after a lossy 32-row-to-target-row-to-32-row interpolation rather
  than on the sampled native grid. The donor failure was genuinely caused by
  shape/visual geometry, not norm width alone.
- Decision-changing diagnostic used: one Stage-B-only donor-geometry check;
  no held-out answer scores or intervention effects were opened.
- Smallest repair: replace the provisional independent caps with the smallest
  common composite max-ratio cap covering the eighth nearest eligible donor
  for every Stage B calibration target; validate the covariance draw on its
  native grid and mapping/norm separately. If this changed second smoke fails,
  stop rather than tune again.

## Latest Research-Action Result

- Action taken: Froze the approved `19/12` donor index, recomputed all 800
  records in fresh shards, merged them under parity/donor-integrity checks, and
  executed the frozen aggregation plus the prespecified 798-target sensitivity.
- Result: **Outcome B.** Held-out reference support replicated, but neither
  structured-null superiority comparison passed.
- Evidence saved: `outputs/stage_c/stage_c_results_v1.jsonl`,
  `outputs/stage_c/analysis_v1/`, `reports/stage_c_results.md`, and
  `reports/stage_c_conclusion.md`.
- Failure or issue: Covariance paired CI `[-0.02833798, 0.01255147]` and
  all-800 real-residual paired CI `[-0.01228950, 0.01741056]` cross zero. The
  798-target sensitivity also fails. Contextual-prefix robustness crosses zero.
- Lesson learned: A negative held-out mean alone is insufficient; the effect is
  not specific relative to the frozen structured interventions and is
  heavy-tail/scoring-context sensitive.
- Next implication: The harmful layer-0 READ hypothesis is closed under the
  frozen protocol. Preserve the descriptive wrong-answer margin shift, 22
  corrections, 12 regressions, and net `+10/800` without promoting them to an
  accuracy or mechanism claim. Stage D is cancelled for this path.
