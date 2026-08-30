# Phase 41: Selective CONTINUE/DEVIATE Memory

## Current Objective
Complete the user-authorized expanded selective CONTINUE/DEVIATE plan at its
prospective Phase-1 decision gate by auditing whether `FULL` is actually
invalid at the frozen held-out W2C mandatory boundaries.

## Active Constraints
- Follow `plans/selective_continue_deviate_expanded_plan.md` (SHA-256
  `20a7517dc61197c8d3914cf8cf45183af7438e514ccdb4cba1583f9b25da34e9`).
- Treat the complete 128-record held-out W2C split as the Phase-1 audit census;
  freeze exact UIDs and every deduplicated compatible known suffix before live
  execution.
- Insert `FULL` only at the mandatory boundary after the exact all-FULL prefix;
  retain every frozen compatible suffix and use the unchanged unified executor.
- Use 10,000 fixed-seed UID-group bootstrap draws and report overall,
  dataset-, depth-, and known-mechanism-specific bounded rescue rates.
- The prospective gate requires all 128 audit states to remain trusted: any
  bounded rescue or unresolved state leaves fewer than the plan's minimum 128
  trusted validation DEVIATE positives and stops gate training. This is a
  sample-contract rule, not a post-hoc rescue-rate threshold.
- If and only if Phase 1 passes, train the authorized linear and small MLP WHEN
  gates and evaluate learned-WHEN + oracle-WHAT. Never train Stage 2 or run
  external evaluation.
- Use all four local RTX 6000 Ada GPUs through direct `torchrun` for live model
  execution. This server has no Slurm.

## Current State
- Done: froze and executed all 252 unique FULL-insertion routes for the complete
  128-state held-out W2C census across four direct GPUs. The routes cover all
  256 compatible frozen source suffixes after removing four identical routes.
- Done: 39/128 boundaries are `FULL-cache-incomplete`, 89/128 are bounded
  `FULL-confirmed-invalid`, and zero are unresolved. Overall bounded rescue is
  0.304688 with 10,000-draw 95% UID-bootstrap CI [0.226562, 0.382812].
- Stopped: the prospective Phase-1 gate fails, so no linear/MLP gate dataset,
  training, threshold sweep, learned-WHEN + oracle-WHAT execution, Stage 2, or
  external evaluation was started.
- Blocked: none.
- Most recent useful observation: bounded rescue occurs across every dataset
  (15/43 ChartQA, 12/43 GQA, 12/42 TextVQA) and is especially frequent at early
  boundaries (25/48) and multi-valid boundaries (16/30).

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-40 deployed failure is predominantly WHEN | `analysis/4action_generalization_diagnostics/decision_summary.md` | Motivates a binary gate only after label validity is checked | confirmed |
| Selected cached-invalid non-FULL actions were incomplete in 6/14 bounded cases | `analysis/4action_generalization_diagnostics/label_incompleteness_results.json` | Makes the analogous missing-FULL audit necessary | confirmed |
| Frozen validation has exactly 128 W2C mandatory boundaries | `analysis/persistent_corrective_supervision/training_manifest.jsonl`; `boundary_manifest.jsonl` | Sets both the audit census and the plan's minimum clean validation-positive count | confirmed |
| Every frozen boundary index exactly enumerates compatible known suffixes | read-only Phase-41 preflight, 2026-08-30 | Supports exhaustive bounded replay without an arbitrary route cap | confirmed |
| Forced FULL has a correct compatible bounded continuation for 39/128 mandatory boundaries | `analysis/selective_continue_deviate/when_label_completeness_results.json` | Fails the prospective label-trust gate and forbids gate training | confirmed |
| All 252/252 deduplicated routes executed with zero unresolved states | `analysis/selective_continue_deviate/when_full_insertion_executions.jsonl`; external rank shards | Validates the Case-A interpretation | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-38 A2/B1 isolation | Zero held-out W2C rescue and near-all-FULL deployment | supported WHEN generalization failure; exact causal subcomponent unknown | `analysis/4action_collapse/decision_summary.md` | Validate labels before another router run | Another bundled four-action retry |
| Phase-39 persistent POLAR/online comparison | Only 7/128 and 6/128 W2C rescue with no architecture advantage | supported poor held-out WHEN behavior | `analysis/persistent_corrective_supervision/decision_summary.md` | Reduce the target to CONTINUE/DEVIATE only if DEVIATE labels are trustworthy | Select an architecture from the unsupported difference |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Exhaustive held-out FULL-insertion audit | Directly tests the assumed invalid action under every known compatible suffix | Whether binary WHEN targets are clean enough | medium | completed; label trust fails |
| Linear frozen-state WHEN gate | Phase-40 online probe reached 0.738 validation AUROC | Whether a simple head exposes a selective high-confidence region | medium | rejected by prospective Phase-1 gate |
| Small MLP frozen-state WHEN gate | Authorized limited nonlinear comparator | Whether modest capacity improves safe selectivity | medium | rejected by prospective Phase-1 gate |
| Label-cache repair | Required because FULL rescues invalidate mandatory boundaries | Restores a trustworthy WHEN target | high | future pivot; not authorized here |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: determine whether selective Stage 1 can be
  trained on a trustworthy DEVIATE target; the audit shows that the current
  target fails this prerequisite.
- Relevant memory item used: Phase 40 proved bounded WHAT-label incompleteness
  and ranked a direct missing-FULL audit as the smallest next discriminator.
- Confirmed observation: 39/128 states have a correct FULL-insertion bounded
  route; all 252 candidates executed and zero states are unresolved.
- Unverified interpretation: why cache discovery omitted these valid FULL
  continuations and how much broader continuation search would change the
  remaining 89 bounded-invalid states.
- Diagnosis: supported WHEN-label cache incompleteness; causal source unknown.
- Viable alternatives considered for a future authorized action: repair/expand
  continuation coverage and rebuild WHEN labels; relax the clean-validation
  minimum; train on known incomplete labels.
- Chosen action: apply the frozen Case-A stop and preserve the negative result;
  do not build or train the selective gate.
- Strongest objection: the 89 bounded-invalid states could still train a smaller
  gate. That would violate the prospectively required 128 trusted validation
  positives and select a weaker contract after seeing the outcome.
- How this differs from failed attempts: it changes no model or label and tests
  the key WHEN-label assumption directly under live execution.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request on 2026-08-30 to read and perform
  `plans/selective_continue_deviate_expanded_plan.md`.
- Stop condition: satisfied. Complete Phase-1 evidence exists and the action
  stopped before gate training after 39 bounded rescues.

## Latest Research-Action Result
- Action taken: froze the full 128-UID audit and 252 deduplicated routes, then
  executed every route on four direct GPUs and applied the frozen bootstrap and
  sample-contract decision.
- Result: 39/128 bounded FULL rescues (0.304688; 95% CI
  [0.226562, 0.382812]), 89 bounded invalid, zero unresolved; Case A.
- Evidence saved: `analysis/selective_continue_deviate/protocol.md`, exact
  subset/executions/results/report/decision/checksum files; raw rank shards at
  `/mnt/hyemin/qwen_train_eval/outputs/selective_continue_deviate_v1/full_insertion_audit/`.
- Verification: 128 unique UIDs, 252/252 routes, every frozen compatible suffix,
  all artifact checksums pass, and all four rank shards are checksum-bound to
  the same config/subset.
- Failure or issue: no runtime failure. Scientific failure: current DEVIATE
  labels do not meet the prospective completeness/trust contract.
- Lesson learned: a mandatory boundary derived from discovered correct routes
  cannot be assumed to make FULL invalid; compatible continuation coverage must
  be audited before binary gate training.
- Next implication: a continuation-cache repair, WHEN-label rebuild, and repeat
  audit is the smallest defensible next action, but it is a new action requiring
  explicit authorization.
