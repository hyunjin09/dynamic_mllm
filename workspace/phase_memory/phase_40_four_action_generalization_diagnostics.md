# Phase 40: Four-Action Generalization Diagnostics Memory

## Current Objective
Execute the user-authorized mechanism diagnostic on the frozen Phase-39
POLAR epoch-15 and online epoch-14 checkpoints, separating WHEN, WHAT,
READ-OFF, WRITE-OFF, BOTH-OFF, state usage, probe predictability, label
smoothness, and bounded label incompleteness before any new router training.

## Active Constraints
- Follow `plans/four_action_generalization_diagnostic_plan.md` (SHA-256
  `cdc39940a1e19c22f17771bc535d22be792b97cd7f45d7c6128ce816e933e446`).
- Reuse exactly the frozen Phase-39 512-W2C/512-C2C train and
  128-W2C/128-C2C validation split, selected checkpoints, labels, executor,
  and model revision. Diagnostic state construction uses only the 640 W2C
  records and does not change their split.
- Match each mandatory-deviation positive to one FULL-unique correcting-route
  node at the same split, dataset, and layer. Do not use C2C states as clean
  negatives and do not force arbitrary labels on multi-valid mechanisms.
- Freeze state matching, deterministic state shuffles, probe capacity/seeds,
  kNN fallbacks, and the label-audit cap before observing diagnostic outcomes.
- Use all four local GPUs through direct `torchrun` for frozen-state extraction
  and bounded executor work. CPU-only deterministic analysis may run directly;
  this server has no Slurm.
- Do not fine-tune the base model/router, change architecture/objective, run
  external evaluation, or start a follow-up training action.

## Current State
- Done: the complete frozen Priority-1/2/3 diagnostic ran on both selected
  checkpoints. Four-GPU extraction covered all 1,280 states, the bounded
  four-GPU label audit executed all 19 frozen routes for 14 states, and every
  required table, result, figure, summary, and Q1--Q9 decision exists under
  `analysis/4action_generalization_diagnostics/`.
- Done: the supported dominant deployed failure is WHEN generalization; online
  representations contain signal unused by the trained router, exact
  mechanisms are weakly smooth, and 6/14 selected cached-invalid states prove
  bounded WHAT-label incompleteness.
- Stopped: no implied WHEN-label audit, label repair, router/objective change,
  training run, or external evaluation was executed.
- Blocked: none.
- Most recent useful observation: validation KEEP-vs-DEVIATE AUROC is
  0.542877 POLAR and 0.507751 online, while a fresh online-state linear probe
  reaches 0.737976 and 6/14 cached-invalid selected states execute correctly
  under a bounded compatible known suffix.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Matched persistent comparison has no supported architecture advantage | `analysis/persistent_corrective_supervision/decision_summary.md` | Defines the failure to decompose | confirmed |
| Frozen W2C boundary and route labels cover 512 train + 128 validation | `analysis/persistent_corrective_supervision/boundary_manifest.jsonl` | Supplies positive states and exact route tries | confirmed |
| Every split/dataset/layer positive cell has enough FULL-unique W2C nodes for exact matching | read-only pre-protocol audit on 2026-08-30 | Avoids a layer-prior confound without relaxing matching | confirmed |
| One Phase-39 C2C runtime mismatch does not touch this W2C-only state population | `analysis/persistent_corrective_supervision/runtime_cohort_sensitivity.md` | Qualifies but does not invalidate the diagnostic | confirmed |
| Both routers lose held-out WHEN and bit discrimination | `analysis/4action_generalization_diagnostics/when_keep_vs_deviate.csv`; `read_write_bit_metrics.csv` | Identifies the primary deployed failure and broad mechanism collapse | confirmed |
| Online frozen state supports better fresh-probe discrimination than its trained router | `analysis/4action_generalization_diagnostics/representation_probe_results.json` | Supports trained head/objective/optimization use failure for online, not feature absence | confirmed |
| Joint online state shuffle leaves 83.6% of argmax actions unchanged and does not reduce validation AUROC | `analysis/4action_generalization_diagnostics/state_shuffle_results.json` | Shows online state changes are not used with generalizable discrimination | confirmed |
| Six of 14 selected cached-invalid states have a bounded execution-correct route | `analysis/4action_generalization_diagnostics/label_incompleteness_results.json` | Proves WHAT-label cache incompleteness in those cases | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-38 A2/B1 isolated remedies | Zero held-out W2C rescue | underlying cause unknown | `analysis/4action_collapse/decision_summary.md` | Diagnose the decision components, not another bundled retry | A2/B1 unchanged |
| Phase-39 persistent comparison | Small rescue and large train-to-validation boundary gap for both routers | underlying mechanism unknown | `analysis/persistent_corrective_supervision/decision_summary.md` | Separate timing, mechanism, representation, and labels | New full router training before diagnosis |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| WHEN/timing failure | Validation WHEN AUROC is near chance and deviations rarely occur at the boundary | Whether intervention detection is primary | completed | supported dominant deployed failure |
| WHAT or READ/WRITE/BOTH mechanism failure | Conditional validity, both bit decisions, and singleton mechanisms collapse | Whether factorized suppression fails selectively | completed | broad failure; no clean operation-specific winner |
| Layer-prior or objective/head failure | Online probe exceeds its trained router while shuffle does not reduce validation AUROC | Whether state signal exists but is unused | completed | online training-use failure supported; pure layer shortcut not sufficient |
| Representation/label smoothness or incompleteness failure | kNN mechanism purity is weak and bounded cached-invalid routes can execute correctly | Whether the target itself is poorly learnable | completed | exact labels weakly smooth; selected WHAT labels incomplete |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: replace the vague held-out-generalization
  failure with one evidence-backed dominant failure mode; the bottleneck is
  that current aggregate rescue does not distinguish WHEN from WHAT or
  representations from labels.
- Relevant memory item used: the 2026-08-30 promoted lesson that persistent
  supervision yields small rescue but no architecture advantage.
- Confirmed observation: both selected routers have near-chance validation
  WHEN AUROC and very low deviation recall; online frozen-state probes exceed
  the trained router; exact mechanism neighborhoods are weakly pure; and 6/14
  bounded cached-invalid actions execute correctly.
- Unverified interpretation: whether missing `FULL` continuations also make
  some supposedly mandatory WHEN boundaries incomplete, and whether a repaired
  label cache or a new two-stage head would improve free rollout.
- Diagnosis: supported WHEN generalization failure at deployment; supported
  trained-signal-use failure for online; supported bounded WHAT-label
  incompleteness; exact causal training subcomponent remains unknown.
- Viable alternatives considered for a future authorized phase: bounded
  WHEN-label completeness audit; broader route-cache repair; two-stage
  CONTINUE-vs-DEVIATE then mechanism objective/head; representation redesign.
- Chosen action: completed the complete bounded diagnostic plan on the two
  frozen selected checkpoints and stopped after answering Q1--Q9.
- Strongest objection to the future ranking: the bounded incompleteness result
  challenges WHAT labels, while the observed dominant failure is WHEN. An
  independent reviewer therefore promoted a direct WHEN-label audit above
  broad repair. The reconciled audit must insert `FULL`, not test non-`FULL`
  actions already known valid by boundary construction.
- How this differs from failed attempts: it does not retrain either router or
  change supervision; it decomposes the already observed failure and audits a
  small set of cached-invalid actions under execution.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request on 2026-08-30 to read and perform
  `plans/four_action_generalization_diagnostic_plan.md`.
- Stop condition: create every required table/result/figure and a dominant-
  failure decision, update compact research state, and do not execute the
  implied next method or another full training run.

## Latest Research-Action Result
- Action taken: froze 640 exact matched WHEN pairs; extracted POLAR/online
  selected-checkpoint representations and outputs on four GPUs; ran the
  prespecified WHEN/WHAT/bit/timing/layer/shuffle/probe/kNN analyses; then
  executed the 14-state/19-route bounded label audit on four GPUs.
- Result: POLAR/online validation WHEN AUROC is 0.542877/0.507751 and argmax
  deviation recall is 0.054688/0.148438. Conditional WHAT validity is
  0.285714/0.526316. IGNORE recall is zero for both. The online-state linear
  probe reaches 0.737976 WHEN AUROC, while joint state shuffle leaves 0.835938
  of argmax predictions unchanged. k=10 three-way mechanism purity is
  0.371--0.431. Six of 14 cached-invalid states have a correct bounded replay.
- Evidence saved: `analysis/4action_generalization_diagnostics/decision_summary.md`,
  compact tables/JSON/figures beside it, raw state tensor SHA-256
  `9293dbac61fa14da635fbcc0e3c8b2a25a71e188c8894254bf9af4c9aaa322fc`,
  and raw label-audit shards under the external output root.
- Verification: all 1,280 state tensors are finite, all 28 compact artifact
  checksums pass, all eight figures are nonempty and visually inspected, and
  the project test boundary passes 520/520 tests. Repository-root pytest also
  collects vendored reference tests and is not the project-suite boundary.
- Failure or issue: no execution failure. Scientific qualification: the
  label-audit population is selected and conditional, so 6/14 is not a global
  prevalence estimate and does not audit missing `FULL` validity.
- Lesson learned: the deployed bottleneck is predominantly WHEN, online state
  contains unused transferable signal, and cached exact WHAT targets cannot be
  assumed complete merely because they come from discovered correcting routes.
- Next implication: after one independent challenge, the smallest defensible
  next action is a prospective bounded WHEN-label completeness audit that
  inserts `FULL` at mandatory boundaries with compatible known suffixes. It
  requires explicit user approval and was not executed.
