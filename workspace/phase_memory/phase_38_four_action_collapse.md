# Phase 38: Four-Action Collapse Isolation Memory

## Current Objective
Execute `plans/four_action_collapse.md` as a causally isolated comparison of
the unchanged online router with guaranteed W2C mandatory-boundary exposure
and the upfront POLAR router with only the C2C exact all-FULL route removed.

## Active Constraints
- Preserve the existing four-action labels, unified executor semantics,
  Qwen2.5-VL backbone, Image+Question inputs, data split, optimizer, and losses
  except for the one intervention assigned to each track.
- Track A1 changes only W2C mandatory-boundary exposure. Full online A2 is
  gated on the fixed pilot demonstrating boundary fit and useful free rollout.
- Track B1 retains C2C but removes only exact `[FULL] * 28`; the 35 resulting
  route-empty C2C samples are excluded and recorded, without invented labels.
- POLAR B1 uses exact-set NLL as the scientific objective. The prior duplicated
  BCE result remains historical evidence and is not rerun as the main model.
- External evaluation is not authorized by this plan; actual routed internal
  execution is required before any later external-evaluation decision.
- GPU work uses machine-local Slurm policy. CPU preparation runs directly.
- Preserve portable evidence for concurrent-server handoff; ignored outputs
  are not implied by Git.

## Current State
- Done: collapse evidence and exact sampler/label audit are complete; repository,
  assets, environment, symlinks, live eight-H100 topology, and empty user queue
  were reverified at commit `2144c38fc2f294473f2e770b2e272e2f961b4c10`.
  A0 now freezes all 2,397 W2C boundaries and the 96-W2C/24-C2C pilot.
- In progress: launch and monitor the reviewed A1 implementation against the
  frozen artifacts. Implementation and pre-launch verification are complete.
- Blocked: none.
- Most recent useful observation: the old sampler never reaches the latest
  all-FULL-prefix boundary for 1,045/2,397 W2C train samples, while both prior
  architecture families deploy all-FULL.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Nine online epochs produced zero W2C rescues and nearly exact all-FULL deployment | `reports/four_action_online_router_early_stop_20260829.md` | Establishes the online collapse to diagnose | confirmed |
| Both completed upfront objectives predict all-FULL on all 866 validation samples | `reports/four_action_polar_action_collapse_audit_20260829.md` | Establishes cross-architecture collapse | confirmed |
| Mandatory W2C boundaries are under-covered by the exact old sampler | `reports/four_action_router_collapse_label_audit_20260829.md` | Supports boundary exposure as the isolated first online intervention | confirmed |
| User-authored ordered experiment contract | `plans/four_action_collapse.md` (SHA-256 `f61f7476ff9a5872f823c7df837e1a2ba21774c83e4efc88f152d2b77d5aceb9`) | Authorizes both architecture tracks and their gates | confirmed |
| Exact A0 boundary manifest passes all invariants | `analysis/4action_collapse/boundary_manifest.jsonl` (SHA-256 `0e11651beee39a0723fd9a973e5ae70e34314b89ac9684ba83507e74e5becd47`) | Freezes one latest boundary per W2C sample before outcomes | confirmed |
| Fixed pilot spans layers 0–27 with balanced singleton actions | `analysis/4action_collapse/pilot_subset.json` (SHA-256 `85235eab3f61405fbc2d213cb5ae9e4390f9b231ad4ce122d830dd9a5c70b734`) | Prevents outcome-dependent pilot selection | confirmed |
| A1 config is parent-bound and prospectively gated | `analysis/4action_collapse/mandatory_boundary_overfit_config.yaml` (SHA-256 `0ccf117c902283714156aa01976ef26b64521eeeb4a50423989dc0df9d98ff5b`) | Freezes 4-GPU/50-epoch maximum, five-epoch validation, and six behavioral gates | confirmed |
| Full project suite passes after three adversarial review cycles | `PYTHONPATH=/home/hyunjin/projects/dynamic_mllm .venv/bin/pytest -q tests` | Verifies A0/A1 helpers, metrics, provenance, resume, locking, and existing behavior | confirmed (491 passed) |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Upfront four-action POLAR BCE and exact-set NLL | Both selected all-FULL everywhere | C2C complete-route shortcut and objective geometry are supported contributors; sole cause unknown | `reports/four_action_polar_action_collapse_audit_20260829.md` | Remove only the C2C exact all-FULL route in a matched exact-NLL ablation | Do not drop C2C or bundle new weighting/architecture |
| Online state-conditioned router | Nine validations had zero W2C rescue | Missing mandatory-boundary exposure and action/prefix imbalance are supported contributors; capacity remains unknown | `reports/four_action_router_collapse_label_audit_20260829.md` | First run a fixed overfit-capacity pilot with exact boundary exposure | Do not launch unchanged full training or alter architecture yet |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| A1 fixed mandatory-boundary overfit pilot | Tests the measured coverage defect without changing architecture/loss | Whether current online states/head can learn when FULL must stop | medium | preparing |
| A2 guaranteed-coverage full online retrain | Directly repairs the measured schedule defect | Whether boundary exposure produces population W2C rescue | high | gated on A1 |
| B1 C2C-no-all-FULL POLAR exact-NLL | Removes the dominant upfront complete-route shortcut only | Whether that shortcut causes upfront collapse | medium | pending |
| Matched upfront-vs-online boundary probe | Compares initial and current-state separability | Which architecture family has the more informative state | medium | pending |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: determine whether the current online router
  can use exact all-FULL-prefix boundary states before committing to another
  full online training run, while retaining the separately required POLAR
  shortcut ablation.
- Relevant memory item used: phase-37 supported coverage failure and the
  promoted 2026-08-29 sample-balance lesson.
- Confirmed observation: 43.6% of W2C train samples never received the critical
  boundary state under the previous schedule.
- Unverified interpretation: guaranteed exposure alone may be sufficient; the
  current representation/head may instead be unable to separate those states.
- Diagnosis: supported contributor, not sole cause.
- Evidence path if diagnosis is not unknown:
  `reports/four_action_router_collapse_label_audit_20260829.md`.
- Viable alternatives considered: immediate full retrain, bundled sampler plus
  weighting/architecture changes, or the ordered isolated plan. The ordered
  pilot has the lowest decision-changing compute and preserves attribution.
- Chosen action: build/audit exact boundary metadata, then run a deterministic
  96-W2C pilot (32 per dataset) plus 24 fixed C2C preservation samples. Use the
  unchanged router/loss/optimizer and only guarantee exact boundary-reaching
  W2C trajectories.
- Strongest objection: a small overfit population can prove capacity but not
  generalization; therefore it gates rather than substitutes for A2.
- How this differs from failed attempts: every pilot W2C is guaranteed to visit
  its latest all-FULL-prefix mandatory deviation state.
- Automatic execution authorized: yes.
- Authorization basis: the user explicitly requested execution of
  `plans/four_action_collapse.md` and testing/training both architecture tracks.
- Stop condition: stop full online progression if A1 cannot fit mandatory
  boundaries and free-run non-FULL behavior after its frozen budget; otherwise
  run A2. Run B1 separately either way, then the matched representational probe.

## Latest Research-Action Result
- Action taken: implemented and tested exact mandatory-boundary extraction,
  generated all W2C records, froze the pilot population, and completed the A1
  trainer/metrics/resume/Slurm launch slice.
- Result: 2,397/2,397 records have unique UIDs, FULL invalid, and at least one
  valid non-FULL action. The pilot has 32 W2C plus 8 C2C per dataset, covers
  boundary layers 0–27, and includes 24 singleton cases for each minority
  action plus 24 multi-valid cases.
- Evidence saved: `analysis/4action_collapse/boundary_manifest.jsonl`,
  `boundary_audit.{md,json}`, `pilot_subset.json`,
  `mandatory_boundary_overfit_config.yaml`, their checksums, and 491 passing
  project tests.
- Failure or issue: none.
- Lesson learned: the fixed subset can satisfy all planned dataset, depth, and
  action-diversity requirements without relaxing label semantics.
- Next implication: commit/push the portable slice, submit A1 through Slurm,
  monitor early samples/checkpoints, and apply the frozen A2 gate.
