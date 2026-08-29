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
- Done: A2 job `1725`, B1 job `1729`, and matched boundary-probe job `1749`
  all completed `0:0` in their frozen dependency order. All required internal
  reports and the four-question decision summary are complete. No external
  evaluation ran.
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
| B1 changes only train-C2C exact all-FULL labels | `analysis/4action_collapse/polar_c2c_no_allfull_manifest_audit.json` | Freezes 3,501 route removals, 35 explicit exclusions, and unchanged validation/W2C labels | confirmed |
| B1 static/cache preflight passes | `analysis/4action_collapse/polar_c2c_no_allfull_preflight.json` | Binds 6,776 records, 245,303 routes, model/embedding assets, and reused visual cache | confirmed |
| Matched probe cohort is frozen before results | `analysis/4action_collapse/upfront_vs_online_boundary_probe_manifest_audit.json` | Binds 2,584 balanced pairs within split, dataset, and target layer; terminal unmatched boundaries are excluded explicitly | confirmed |
| A2 completes but remains collapsed | `analysis/4action_collapse/online_boundary_coverage_v2_report.md` | Exactly 2,397 boundary exposures were active, yet all ten validations had zero boundary Valid@1 and zero W2C rescue | confirmed |
| B1 completes but remains collapsed | `analysis/4action_collapse/polar_c2c_no_allfull_report.md` | Selected epoch executes all-FULL on 866/866 after removing 3,501 train-C2C all-FULL routes | confirmed |
| Matched probe finds no online advantage | `analysis/4action_collapse/upfront_vs_online_boundary_probe_report.md` | Upfront/online AUROC 0.5764/0.5751; paired difference CI includes zero in both directions | confirmed |
| Full project suite passes | `PYTHONPATH=/home/hyunjin/projects/dynamic_mllm .venv/bin/pytest -q tests` | Verifies A0/A1/A2/B1/probe helpers, metrics, provenance, resume, locking, and existing behavior | confirmed (499 passed) |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Upfront four-action POLAR BCE and exact-set NLL | Both selected all-FULL everywhere | C2C complete-route shortcut and objective geometry are supported contributors; sole cause unknown | `reports/four_action_polar_action_collapse_audit_20260829.md` | Remove only the C2C exact all-FULL route in a matched exact-NLL ablation | Do not drop C2C or bundle new weighting/architecture |
| Online state-conditioned router | Nine validations had zero W2C rescue | Missing mandatory-boundary exposure and action/prefix imbalance are supported contributors; capacity remains unknown | `reports/four_action_router_collapse_label_audit_20260829.md` | First run a fixed overfit-capacity pilot with exact boundary exposure | Do not launch unchanged full training or alter architecture yet |
| A2 one-visit guaranteed boundary coverage | Ten validations retained zero boundary Valid@1 and zero W2C rescue | One mostly early visit per W2C sample is insufficient; whether persistent exposure would work remains unknown | `analysis/4action_collapse/online_boundary_coverage_v2_history.jsonl` | Do not repeat the one-visit schedule or infer that A1 capacity generalized | Do not rerun A2 unchanged |
| B1 exact C2C-all-FULL removal | Selected checkpoint remained all-FULL on 866/866, with zero W2C rescue | The universal C2C route is not the sole sufficient cause; residual FULL pressure/capacity cause remains unknown | `analysis/4action_collapse/polar_c2c_no_allfull_report.md` | Do not repeat shortcut removal alone or select by C2C-dominated overall Hit@1 | Do not rerun B1 unchanged |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| A1 fixed mandatory-boundary overfit pilot | Tests the measured coverage defect without changing architecture/loss | Whether current online states/head can learn when FULL must stop | medium | passed at epoch 30 |
| A2 guaranteed-coverage full online retrain | Directly repairs the measured schedule defect | Whether boundary exposure produces population W2C rescue | high | complete; negative |
| B1 C2C-no-all-FULL POLAR exact-NLL | Removes the dominant upfront complete-route shortcut only | Whether that shortcut causes upfront collapse | medium | complete; negative |
| Matched upfront-vs-online boundary probe | Compares initial and current-state separability | Which architecture family has the more informative state | medium | complete; no online advantage |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: choose whether the completed evidence favors
  upfront POLAR, online state-conditioned routing, or neither as the next
  four-action training substrate.
- Relevant memory item used: phase-37 supported coverage failure and the
  promoted 2026-08-29 sample-balance lesson.
- Confirmed observation: A1 established online local capacity, but A2 and B1
  both produced zero held-out W2C rescue; the matched probe found no online
  AUROC advantage (difference -0.0013, 95% CI [-0.0548, 0.0534]).
- Unverified interpretation: persistent targeted W2C/non-FULL supervision may
  succeed where A2's single front-loaded exposure and B1's route removal did
  not.
- Diagnosis: supported that each isolated intervention is insufficient;
  underlying learned-collapse cause remains unknown. Evidence paths are the
  three phase-38 reports under `analysis/4action_collapse/`.
- Viable alternatives considered: prioritize online because A1 is positive
  capacity evidence; prioritize POLAR because it is simpler and the probe shows
  no online advantage; or select neither until a matched persistent-targeting
  discriminator is run.
- Chosen action: select neither architecture as final. If explicitly
  authorized later, prospectively specify a matched low-budget comparison with
  equal persistent targeted W2C/non-FULL supervision mass for both substrates.
- Strongest objection: A1 is unique positive evidence and might justify
  prioritizing online immediately. It does not establish population
  generalization or an online representational advantage.
- How this differs from failed attempts: the proposed discriminator would
  distribute persistent targeted signal across epochs and compare both
  substrates under the same held-out W2C-rescue/C2C-preservation gate; it would
  not repeat A2's one visit or B1's shortcut removal alone.
- Automatic execution authorized: no new action is authorized after completion
  of the user-authored plan.
- Stop condition: preserve and push all phase-38 reports and recommend the
  matched discriminator without implementing or running it.

## Latest Research-Action Result
- Action taken: completed A2 online training, B1 POLAR training/internal
  execution, and the matched representation probe in frozen dependency order.
- Result: all jobs completed `0:0`. A2 and B1 each had zero held-out W2C rescue;
  B1 selected all-FULL on 866/866. Upfront and online probe AUROCs were
  0.5764/0.5751 with no supported difference.
- Evidence saved: required histories/reports/summaries under
  `analysis/4action_collapse/`, with raw checkpoints, validation outputs,
  feature shards, and execution rows under `outputs/four_action_collapse/`.
- Failure or issue: no validity or runtime failure. The scientific result is
  negative for both isolated remedies.
- Lesson learned: neither one-visit online boundary coverage nor removal of the
  exact C2C all-FULL route is sufficient to break held-out deployment collapse;
  current-state features show no measured advantage over upfront features.
- Next implication: do not select a final architecture or run external
  evaluation. Any matched persistent-targeting follow-up requires a new
  prospectively approved plan.
