# Phase 41: Selective CONTINUE/DEVIATE Memory

## Current Objective
Execute the user-authorized expanded selective CONTINUE/DEVIATE plan, beginning
with an exhaustive bounded audit of whether `FULL` is actually invalid at the
frozen held-out W2C mandatory boundaries. Proceed to gate training only if the
prospective label-trust gate passes.

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
- Done: read the complete expanded plan and relevant Phase-40 evidence; verified
  the full held-out W2C census contains 128 unique states across all required
  datasets, depth bins, and mechanism groups.
- Done: verified that exhaustive suffix replay is bounded at 252 unique routes
  derived from all 256 compatible frozen suffixes (four duplicates removed),
  with 1--12 routes per state and median one.
- In progress: freeze and execute the Phase-1 FULL-insertion audit.
- Blocked: none.
- Most recent useful observation: the full census gives balanced dataset support
  (43 ChartQA, 43 GQA, 42 TextVQA) and covers 48 early, 43 middle, and 37 late
  boundaries plus 33 IGNORE, 33 READ_ONLY, 32 WRITE_ONLY, and 30 multi-valid
  mechanisms.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-40 deployed failure is predominantly WHEN | `analysis/4action_generalization_diagnostics/decision_summary.md` | Motivates a binary gate only after label validity is checked | confirmed |
| Selected cached-invalid non-FULL actions were incomplete in 6/14 bounded cases | `analysis/4action_generalization_diagnostics/label_incompleteness_results.json` | Makes the analogous missing-FULL audit necessary | confirmed |
| Frozen validation has exactly 128 W2C mandatory boundaries | `analysis/persistent_corrective_supervision/training_manifest.jsonl`; `boundary_manifest.jsonl` | Sets both the audit census and the plan's minimum clean validation-positive count | confirmed |
| Every frozen boundary index exactly enumerates compatible known suffixes | read-only Phase-41 preflight, 2026-08-30 | Supports exhaustive bounded replay without an arbitrary route cap | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Phase-38 A2/B1 isolation | Zero held-out W2C rescue and near-all-FULL deployment | supported WHEN generalization failure; exact causal subcomponent unknown | `analysis/4action_collapse/decision_summary.md` | Validate labels before another router run | Another bundled four-action retry |
| Phase-39 persistent POLAR/online comparison | Only 7/128 and 6/128 W2C rescue with no architecture advantage | supported poor held-out WHEN behavior | `analysis/persistent_corrective_supervision/decision_summary.md` | Reduce the target to CONTINUE/DEVIATE only if DEVIATE labels are trustworthy | Select an architecture from the unsupported difference |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Exhaustive held-out FULL-insertion audit | Directly tests the assumed invalid action under every known compatible suffix | Whether binary WHEN targets are clean enough | medium | testing |
| Linear frozen-state WHEN gate | Phase-40 online probe reached 0.738 validation AUROC | Whether a simple head exposes a selective high-confidence region | medium | conditional |
| Small MLP frozen-state WHEN gate | Authorized limited nonlinear comparator | Whether modest capacity improves safe selectivity | medium | conditional |
| Label-cache repair | Required if FULL rescues invalidate mandatory boundaries | Restores a trustworthy WHEN target | high | future pivot; not authorized here |

## Next-Step Decision
- Deliberation mode: deep.
- Active objective and bottleneck: test selective Stage 1 without training on an
  unverified DEVIATE target; current bottleneck is missing evidence that `FULL`
  is invalid at mandatory boundaries under compatible continuations.
- Relevant memory item used: Phase 40 proved bounded WHAT-label incompleteness
  and ranked a direct missing-FULL audit as the smallest next discriminator.
- Confirmed observation: all 128 validation W2C boundaries can be audited with
  252 deduplicated routes and complete frozen suffix provenance.
- Unverified interpretation: whether any forced-FULL boundary can still yield a
  correct answer under a known compatible continuation.
- Diagnosis: unknown until execution.
- Viable alternatives considered: audit a 64/96-state stratified subset; audit
  the complete 128-state held-out census; train the gate immediately.
- Chosen action: audit the complete 128-state census, because it is within the
  authorized range, avoids selection variance, and maximizes subgroup support.
- Strongest objection: requiring zero rescued states is conservative. It is
  fixed prospectively because the census contains exactly the plan's minimum
  128 validation DEVIATE positives; accepting any known incomplete state would
  weaken either the trusted-label or held-out-size requirement after outcomes.
- How this differs from failed attempts: it changes no model or label and tests
  the key WHEN-label assumption directly under live execution.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request on 2026-08-30 to read and perform
  `plans/selective_continue_deviate_expanded_plan.md`.
- Stop condition: if any state is rescued or unresolved, write the complete
  Phase-1 evidence and stop before gate training; otherwise complete the
  authorized gate/oracle stages and stop before Stage 2.

## Latest Research-Action Result
- Action taken: in progress.
- Result: pending.
- Evidence saved: helper implementation and tests; prospective protocol pending.
- Failure or issue: none.
- Lesson learned: pending.
- Next implication: pending Phase-1 gate.
