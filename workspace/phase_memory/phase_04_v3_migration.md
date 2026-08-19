# Phase 04: v3 Policy-Conditional Migration Memory

## Current Objective

Migrate the completed v2 project state to plan v3 without changing or
reinterpreting preserved v2 outcomes and without executing a new experiment.

## Active Constraints

- Plan v3 supersedes the earlier READ/WRITE causal-analysis plans.
- Every valid v3 cell is one action from an identical dense pre-layer state,
  followed by an unchanged dense suffix and identical answer scoring.
- All previously inspected Stage B/C results are discovery evidence only.
- The v2 Stage C 800-record set and selected cases cannot serve as v3 held-out
  confirmation or mechanistic replication.
- No old Stage D, large sweep, router/controller/policy/probe training, or base
  model fine-tuning is authorized.

## Current State

- Done: Static implementation audit, Stage A/B/C artifact audit, four-action
  completeness verification, preservation check, migration report, and compact
  state update.
- In progress: None.
- Blocked: Immediate v3 confirmation is not authorized or ready; v3 discovery
  reanalysis and bounded null/query-invariance preflight remain unfrozen.
- Most recent useful observation: All 400 v2 Stage B records form complete,
  finite dense-suffix four-action vectors at eight layers with exact FULL score
  parity and exactly recomputable conditional contrasts.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Cached branch runner clones the dense prefix/prestate, applies one mapped action, and executes the remaining decoder layers unchanged. | `interventions/four_state.py`; `interventions/prompt_cache.py` | Establishes the v3 policy-conditional execution semantics. | confirmed |
| Stage A parity, reconstruction, stability, scoring, and architecture checks passed on 23 samples. | `outputs/stage_a/stage_a_summary.json`; `outputs/stage_a/` | Supports reuse within the documented stock-eager/length domain. | confirmed |
| Stage B has 400 records, 3,200 sample-layer pairs, and 12,800 complete finite action cells; FULL score parity and stored contrast recomputation errors are zero. | `outputs/v3_migration/v2_artifact_audit_v1.json` | Establishes reusable v3 discovery `Q` data. | confirmed |
| v2 Stage C contains only `FULL` and `WRITE_ONLY` for the frozen endpoint and all 800 outcomes were inspected. | `outputs/stage_c/stage_c_results_v1.jsonl`; `reports/stage_c_frozen_outcome_b_closure.md` | Excludes it from v3 confirmation and full-vector mechanistic replication. | confirmed |
| The v2 checksum archive remains unchanged. | `archives/stage_b_stage_c_frozen_outcome_b_v1/` | Preserves the prior scientific record. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| v2 layer-0 READ confirmation | Reference support replicated but the real intervention did not beat either structured null; prompt robustness was weak. | supported | `reports/stage_c_conclusion.md` | Retain as discovery under v3; do not resume v2 Stage D or relabel it as a v3 confirmation. | Post-hoc rescue on the inspected 800 records. |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| `REUSE_AND_CONFIRM` | Core Stage A validity and complete Stage B four-cell values directly match v3's formal object. | Avoids repeating valid discovery while preserving a path to a new v3 confirmation. | low before confirmation | selected |
| `RERUN_DISCOVERY` | Would collect every v3 diagnostic and same-image pairs in one new sweep. | Missing serialized diagnostics and exploratory same-image analysis. | high | rejected as unnecessary now |
| `REPAIR_STAGE_A` | Could add the numerical same-image WRITE-invariance sanity check. | One v3 C2 premise. | low | rejected as classification; no intervention-validity failure exists |
| `STOP_AND_REDESIGN` | Appropriate only if the action semantics or reconstruction fail. | Fundamental invalidity. | high | rejected by direct evidence |

## Next-Step Decision

- Deliberation mode: deep, because v3 supersedes the active formal claim and
  the reuse decision affects future confirmatory cost.
- Active objective and bottleneck: Decide whether v2 Stage A/B can support v3
  discovery without a new large sweep; missing v3 diagnostics block immediate
  confirmation but not four-cell reconstruction.
- Relevant memory item used: The stock-eager runtime is the only validated
  causal runtime, and v2 Outcome B cannot be rescued by post-hoc reuse.
- Confirmed observation: Existing Stage B action cells exactly map to
  `[Q(0,0), Q(1,0), Q(0,1), Q(1,1)]` under an identical dense prefix and suffix.
- Unverified interpretation: Which v3 FULL-relative endpoint, if any, will
  survive a full search-budget-matched structured null.
- Diagnosis: supported.
- Evidence path if diagnosis is not unknown:
  `outputs/v3_migration/v2_artifact_audit_v1.json`.
- Viable alternatives considered: Reuse and prepare a new confirmation; rerun
  discovery; repair Stage A; or stop/redesign.
- Chosen action: `REUSE_AND_CONFIRM`, with confirmation explicitly gated on a
  deterministic v3 reanalysis, bounded discovery-only preflight, and a new
  frozen held-out manifest.
- Strongest objection: v2 omitted output KL, final-state distance, residual
  norms, same-image pairs, and a complete-search structured null. These gaps
  could alter endpoint selection, so reuse cannot mean immediate Stage C entry.
- How this differs from failed attempts: The inspected v2 Stage C endpoint is
  demoted to discovery and is not used as the new confirmatory population.
- Automatic execution authorized: no.
- Authorization basis: The user authorized one bounded migration audit only.
- Stop condition: Reached after the audit and state migration; no experiment or
  training action follows automatically.

## Latest Research-Action Result

- Action taken: Audited v2 implementation and artifacts against the complete
  policy-conditional v3 formal object.
- Result: Reuse Stage A and Stage B core evidence; preserve but demote all v2
  Stage C evidence to discovery; do not rerun the 400-sample sweep.
- Evidence saved: `reports/v3_migration_audit.md` and
  `outputs/v3_migration/v2_artifact_audit_v1.json`.
- Failure or issue: Immediate v3 confirmation is not ready because several v3
  diagnostics, full-search null controls, same-image preparation, and a new
  held-out manifest are absent.
- Lesson learned: A complete dense-suffix four-cell dataset is reusable even
  when an older operation-specific confirmation failed, but the old held-out
  label and search budget do not transfer to a broader v3 claim.
- Next implication: Seek approval for the bounded v3 discovery reanalysis and
  preflight; do not collect held-out outcomes yet.

