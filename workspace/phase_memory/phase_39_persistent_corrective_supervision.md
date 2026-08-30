# Phase 39: Persistent Corrective Supervision Memory

## Current Objective
Execute the user-authorized matched low-budget persistent-corrective comparison
between the unchanged upfront POLAR and online four-action router substrates,
then select either substrate or neither using held-out routed behavior.

## Active Constraints
- Follow `plans/four_action_generalization.md` (SHA-256
  `79c159af4aa451cdbb153e95b7145566f77835770c1408765f1fafe1d35837b5`).
- Use the same fixed 512-W2C/512-C2C train and 128-W2C/128-C2C validation
  subsets, 20 epochs, one mandatory-boundary term per W2C sample per epoch,
  `lambda_boundary = 1.0`, and the same behavioral checkpoint rule.
- Retain the original exact-set NLL POLAR base loss, set-valued online base
  loss, C2C records and labels, four-action semantics, and architectures.
- Use all four local GPUs through direct execution; this server has no Slurm.
- Do not run external evaluation, scale either model, tune lambda, regenerate
  labels, or start any follow-up action after the architecture decision.

## Current State
- Done: the deterministic matched subsets, protocol, audit, four-GPU smoke,
  both 20-epoch training runs, all-checkpoint internal execution, prospective
  selection, paired bootstrap, and compact reports are complete.
- Done: POLAR selected epoch 15 (7/128 W2C rescue; 124/128 C2C preservation)
  and online selected epoch 14 (6/128; 122/128).
- Done: the paired online-minus-POLAR W2C difference is -0.0078125 with 95%
  bootstrap interval [-0.0625, 0.0390625]; no supported architecture advantage
  was found, so the frozen operational tie-break prefers POLAR.
- Stopped: no external evaluation or follow-up scientific action was run.
- Blocked: none.
- Most recent useful observation: persistent boundary supervision produced
  small held-out rescue for both substrates but did not establish an
  architecture advantage; substantial teacher-forced/free-rollout gaps and
  predominantly FULL deployment remain.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| A1 learned fixed mandatory boundaries and rescued its pilot | `analysis/4action_collapse/mandatory_boundary_overfit_report.md` | Establishes local online capacity but not generalization | confirmed |
| A2 covered all 2,397 W2C boundaries once but had zero held-out rescue | `analysis/4action_collapse/online_boundary_coverage_v2_report.md` | Rules out the front-loaded one-visit schedule | confirmed |
| B1 removed 3,501 C2C all-FULL routes but deployed all-FULL on 866/866 | `analysis/4action_collapse/polar_c2c_no_allfull_report.md` | Rules out shortcut removal alone | confirmed |
| Matched upfront/online probe found no online AUROC advantage | `analysis/4action_collapse/upfront_vs_online_boundary_probe_report.md` | Prevents choosing online from state separability alone | confirmed |
| User-authored persistent-supervision contract | `plans/four_action_generalization.md` | Authorizes exactly this bounded comparison | confirmed |
| Matched all-checkpoint result | `analysis/persistent_corrective_supervision/decision_summary.md` | Resolves the authorized architecture decision | confirmed |
| Current-runtime cohort sensitivity | `analysis/persistent_corrective_supervision/runtime_cohort_sensitivity.md` | One C2C mismatch does not change selection or decision | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| A2 one scheduled W2C boundary visit | Zero validation boundary Valid@1 and W2C rescue | One front-loaded visit is insufficient; underlying cause unknown | `analysis/4action_collapse/online_boundary_coverage_v2_report.md` | Give every selected W2C one boundary term in every epoch | Do not repeat the A2 schedule |
| B1 C2C all-FULL removal | All-FULL deployment on all validation samples | Shortcut removal alone is insufficient; underlying cause unknown | `analysis/4action_collapse/polar_c2c_no_allfull_report.md` | Keep C2C unchanged and match targeted supervision instead | Do not repeat B1 or remove C2C |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Upfront POLAR with persistent boundary loss | Simpler substrate and no measured online representation advantage | Whether initial features generalize corrective boundaries | medium | viable; operational preference |
| Online router with persistent boundary loss | A1 supplies positive local-capacity evidence | Whether current-state conditioning generalizes under matched supervision | high | viable; no supported advantage |
| Select neither | Both prior families collapsed | Whether neither clears the frozen rescue/preservation decision rule | low | not selected by frozen gate |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: distinguish memorization from held-out
  corrective generalization under a matched persistent schedule; the current
  bottleneck is zero held-out W2C rescue for both prior recipes.
- Relevant memory item used: the 2026-08-29 promoted lesson that isolated
  coverage and shortcut fixes do not select a four-action architecture.
- Confirmed observation: A1 establishes local capacity; A2 and B1 establish
  that their isolated interventions are insufficient; the matched probe does
  not favor online features.
- Unverified interpretation: persistent per-epoch boundary signal may overcome
  insufficient corrective supervision mass for one or both substrates.
- Diagnosis: supported that the prior isolated remedies are insufficient;
  underlying learned-collapse cause remains unknown.
- Evidence path if diagnosis is not unknown: the three completed reports under
  `analysis/4action_collapse/` cited above.
- Viable alternatives considered: no additional research action is viable
  within this authorization; the plan already fixes the matched comparison and
  forbids the follow-up pivots.
- Chosen action: execute the prospectively frozen 20-epoch matched discriminator
  through every-checkpoint internal execution and its architecture decision.
- Strongest objection: equal semantic boundary targets and lambda do not imply
  equal gradient scale across different base objectives and output spaces; the
  result is a matched recipe comparison, not an architecture impossibility
  proof.
- How this differs from failed attempts: every selected W2C receives a separate
  mandatory-boundary loss in every epoch in both substrates, rather than one
  early visit for online or C2C shortcut removal for POLAR.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request on 2026-08-30 to read and perform
  `plans/four_action_generalization.md`, using all four GPUs by default.
- Stop condition: produce the required matched reports and prospective
  architecture decision, update compact research state, and do not execute a
  scale-up, external evaluation, or follow-up diagnostic.

## Latest Research-Action Result
- Action taken: completed the prospectively frozen matched 20-epoch persistent
  corrective-supervision comparison and every-checkpoint internal execution.
- Result: POLAR epoch 15 rescued 7/128 W2C and preserved 124/128 C2C; online
  epoch 14 rescued 6/128 and preserved 122/128. Online minus POLAR was
  -0.0078125 with paired 95% CI [-0.0625, 0.0390625]. Decision: no supported
  architecture advantage; operationally prefer POLAR.
- Evidence saved: required protocol, manifests, histories, execution rows,
  reports, comparison table, decision summary, and cohort sensitivity under
  `analysis/persistent_corrective_supervision/`. The raw machine-local root is
  `/mnt/hyemin/qwen_train_eval/outputs/persistent_corrective_supervision_v1`
  (172 files; 2,607,339,142 bytes), exposed through the ignored repository
  symlink `outputs/persistent_corrective_supervision_v1`. Key SHA-256 values:
  protocol `97d00243481855e9283e5de7ed9b0228894b6a8f9cf91d306a51c67db8de8b70`,
  subset manifest `92dc068566b799f852c30d0e459aa4f231d6082f2a59c78c64c56c0b9c66936b`,
  POLAR config `654192728a1284a1eba22d49593c781f35460f477de64eed47450c77cc119ec1`,
  and online config `e535873878ae3bdfcc8bdaf4c7580be674ede73e871c7f379a505870b16b69f0`.
- Failure or issue: one frozen validation C2C UID currently executes all-FULL
  incorrectly. The frozen analysis remains primary; exclusion sensitivity
  changes the C2C denominators to 127 but leaves selections and decision
  invariant. Cause is unknown.
- Lesson learned: persistent targeted supervision is sufficient for small
  held-out non-FULL rescue in both fixed recipes, unlike A2/B1, but does not
  support an online-state architecture advantage and does not establish robust
  population-level corrective generalization.
- Next implication: stop. Any scale-up, objective change, architecture change,
  external evaluation, or new diagnostic requires a new prospective plan and
  explicit user approval.
