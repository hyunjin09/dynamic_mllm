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
- Done: A1 Slurm job `1700` passed every frozen behavioral gate at epoch 30.
- In progress: launch the matched ten-epoch A2 run with exactly one guaranteed
  mandatory-boundary visit per W2C sample and ordinary sampling elsewhere.
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
| A1 passes all prospective gates at epoch 30 | `analysis/4action_collapse/mandatory_boundary_overfit_report.md` | Establishes local discrimination and free-rollout capacity without an architecture change | confirmed |
| Full project suite passes after three adversarial review cycles | `PYTHONPATH=/home/hyunjin/projects/dynamic_mllm .venv/bin/pytest -q tests` | Verifies A0/A1 helpers, metrics, provenance, resume, locking, and existing behavior | confirmed (491 passed) |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Upfront four-action POLAR BCE and exact-set NLL | Both selected all-FULL everywhere | C2C complete-route shortcut and objective geometry are supported contributors; sole cause unknown | `reports/four_action_polar_action_collapse_audit_20260829.md` | Remove only the C2C exact all-FULL route in a matched exact-NLL ablation | Do not drop C2C or bundle new weighting/architecture |
| Online state-conditioned router | Nine validations had zero W2C rescue | Missing mandatory-boundary exposure and action/prefix imbalance are supported contributors; capacity remains unknown | `reports/four_action_router_collapse_label_audit_20260829.md` | First run a fixed overfit-capacity pilot with exact boundary exposure | Do not launch unchanged full training or alter architecture yet |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| A1 fixed mandatory-boundary overfit pilot | Tests the measured coverage defect without changing architecture/loss | Whether current online states/head can learn when FULL must stop | medium | passed at epoch 30 |
| A2 guaranteed-coverage full online retrain | Directly repairs the measured schedule defect | Whether boundary exposure produces population W2C rescue | high | authorized; launching |
| B1 C2C-no-all-FULL POLAR exact-NLL | Removes the dominant upfront complete-route shortcut only | Whether that shortcut causes upfront collapse | medium | pending |
| Matched upfront-vs-online boundary probe | Compares initial and current-state separability | Which architecture family has the more informative state | medium | pending |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: determine whether A1-demonstrated local
  boundary capacity transfers to the full held-out population under exactly
  one guaranteed boundary visit per W2C sample, while retaining the separately
  required POLAR shortcut ablation.
- Relevant memory item used: phase-37 supported coverage failure and the
  promoted 2026-08-29 sample-balance lesson.
- Confirmed observation: A1 achieved 0.9583 boundary Valid-Action@1, 0.8958 W2C
  free-rollout rescue, and 0.9167 C2C preservation on its frozen population.
- Unverified interpretation: the exposure correction will generalize beyond
  the overfit pilot to the full validation population.
- Diagnosis: supported contributor, not sole cause.
- Evidence path if diagnosis is not unknown:
  `reports/four_action_router_collapse_label_audit_20260829.md`.
- Viable alternatives considered: stop at the pilot, change architecture, or
  run the plan's matched full-data exposure test. A2 is the only authorized
  option that tests generalization without bundling another mechanism.
- Chosen action: run the matched ten-epoch A2 comparison. Across the unchanged
  61,440-visit schedule, mark the first visit for each of 2,397 W2C UIDs as its
  exact boundary route; retain ordinary deterministic valid-route sampling for
  every remaining visit.
- Strongest objection: most guaranteed visits occur early (2,274 in epoch 1),
  so one exposure may not be retained through the full schedule. That timing is
  frozen by the plan's minimal one-visit intervention and is not outcome-tuned.
- How this differs from failed attempts: boundary exposure is 2,397/2,397
  instead of 1,352/2,397, with architecture/loss/C2C/optimizer unchanged.
- Automatic execution authorized: yes.
- Authorization basis: the user explicitly requested execution of
  `plans/four_action_collapse.md` and testing/training both architecture tracks.
- Stop condition: complete all ten A2 epochs and internal routed validations;
  do not launch external evaluation. Run B1 separately, then the matched probe.

## Latest Research-Action Result
- Action taken: ran the frozen A1 pilot without outcome-dependent tuning.
- Result: job `1700` stopped automatically at the first passing checkpoint,
  epoch 30. Boundary Valid-Action@1 and non-FULL recall were both 0.9583;
  singleton IGNORE/READ/WRITE recalls were 0.9583/1.0000/0.9167; free rollout
  left all-FULL on 1.0000; W2C rescue was 0.8958; C2C preservation was 0.9167.
- Evidence saved: `analysis/4action_collapse/mandatory_boundary_overfit_history.jsonl`,
  `mandatory_boundary_overfit_report.md`, and checksum-valid epoch artifacts
  under `outputs/four_action_collapse/mandatory_boundary_overfit_v1/`.
- Failure or issue: none. The result establishes pilot capacity, not full-data
  generalization.
- Lesson learned: the unchanged online state/head can recognize and act on exact
  mandatory-deviation states when those states are explicitly exposed.
- Next implication: A2 is authorized. Its frozen schedule covers each W2C UID
  exactly once and preserves all original balanced sample visits.
