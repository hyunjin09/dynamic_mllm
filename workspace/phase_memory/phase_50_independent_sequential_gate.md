# Phase 50: Independent Layer-Wise Sequential Gate Memory

## Current Objective
Evaluate whether the existing frozen Phase-48 independent layer-wise failure probes can form a conservative sample-level sequential trigger gate, without retraining or connecting treatment.

## Active Constraints
- Reuse the exact Phase-48 28 checkpoints and frozen train/validation/test split; do not retrain any predictor.
- Calibrate only on the 800-row validation split and keep the 800-row test split unopened until thresholds and operating points are frozen.
- Use one shared alpha rule to derive separate thresholds for layers 0-27; do not fit 28 unrelated thresholds or dataset-specific thresholds.
- Evaluate sample-level preservation under the first-trigger trajectory policy, not independent per-layer preservation.
- Compare against validation-calibrated fixed gates at layers 14, 21, and 27.
- A trigger is admission only; do not execute four-action treatment, Stage 2, W-to-C repair, or external evaluation.
- Stop after this one sequential-gate analysis.

## Current State
- Done: Completed the frozen validation calibration, one-time untouched-test evaluation, fixed-layer comparison, dataset and trigger audits, figures, integrity verification, and phase closeout.
- In progress: None.
- Blocked: No.
- Most recent useful observation: The validation-conservative sequential union transferred to only 0.9575 test preservation and showed 0.7250 wrong-recall spread across datasets at the 99%/98% points.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Phase-48 has 28 provenance-bound independent checkpoints on a 6,399/800/800 image-group-disjoint split | `analysis/dense_failure_stage1/layerwise_failure_probe/` | Supplies the only predictors and split allowed by the plan | confirmed |
| Phase-48 test AUROC peaks at 0.8992 at layer 21 and is 0.8421 at layer 0 | Phase-48 analysis summary | Multiple layers may provide complementary detection, but AUROC alone does not establish a safe sequential union | confirmed |
| OOD source-selected thresholds were not calibration-stable across datasets | `analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md` | Requires dataset-wise reporting and prohibits silently adding task-specific thresholds | confirmed |
| Four RTX 6000 Ada GPUs are idle and this server uses direct execution | live `nvidia-smi`, 2026-08-31 | Supports four-way score extraction | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treating independently calibrated layer thresholds as if their preservation composed sequentially | Not executed here; explicitly identified by the plan as invalid | supported by policy union semantics | plan section 4 | Calibrate and select only from the full sample trajectory | Do not multiply or assume per-layer preservation |
| Four validation score workers with the host-wide default PyTorch thread count | All ranks remained in Phase-48 matrix composition for about 10 minutes while each consumed roughly 32 CPU threads; no score artifact was emitted | supported: CPU oversubscription in concurrent deterministic preprocessing | live process telemetry and interrupt tracebacks in `load_layer_matrices` / `compose_layer_features` | Bind each worker to the Phase-48-proven eight-thread setting, invalidate the preparation contract, and refreeze before rerun | Do not launch four unbounded-thread score workers |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Empirical higher quantile with an exact attainable tail-breakpoint grid | Conservative under strict `score > threshold` and honest about 400-correct validation resolution | Implements the shared-alpha V1 without interpolated pseudo-resolution | low | selected |
| Linear-interpolated quantiles on a hand-written log grid | Familiar default | Smoother curve | low | rejected because tiny nonzero alphas can admit a top-scoring correct sample at every layer and obscure finite resolution |
| Allocate a different false-positive budget to each layer | Could improve the sequential frontier | Tests a global risk-budget method | medium | rejected as explicitly outside V1 and the stop rule |

## Next-Step Decision
- Deliberation mode: standard
- Active objective and bottleneck: Determine whether the existing independent probes yield a useful validation-calibrated sequential gate at 99/98/95% sample-level correct preservation.
- Relevant memory item used: Phase 49 showed that calibration can shift strongly across datasets, so dataset-wise behavior must be reported without dataset-specific retuning.
- Confirmed observation: The plan fixes the shared-alpha quantile family but leaves empirical quantile interpolation and grid resolution unspecified.
- Unverified interpretation: Repeated layers may add complementary wrong coverage, but they may also accumulate false triggers too rapidly for conservative operation.
- Diagnosis: unknown
- Viable alternatives considered: Empirical higher quantiles over attainable breakpoints; linearly interpolated quantiles; per-layer risk allocation.
- Chosen action: Freeze empirical higher quantiles and sweep every attainable alpha breakpoint from 0 to 0.10 based only on the validation-correct sample count; choose the largest alpha meeting each preservation target, then evaluate test once.
- Strongest objection: The empirical grid can be coarse and may leave 99% with only the zero-trigger point. That is a real finite-sample limitation and must be reported rather than smoothed away.
- How this differs from failed attempts: It evaluates the union of first triggers at sample level and introduces no target- or dataset-specific calibration.
- Automatic execution authorized: yes
- Authorization basis: The user explicitly requested execution of `plans/independent_layerwise_sequential_gate_plan.md`.
- Stop condition: Required validation/test score matrices, frozen operating points, sequential/fixed comparisons, figures, decision summary, and integrity audit are complete, or a validity gate fails.

## Latest Research-Action Result
- Action taken: Rescored the 28 frozen Phase-48 independent probes on its exact validation and test cohorts; selected shared-alpha layer thresholds from validation-only full trajectories; opened test once; compared fixed L14/L21/L27 gates; audited dataset and trigger distributions.
- Result: Contract `6b1e4812a0a1e51f3dec822964b848a3a0410451dfe7a289d25c4b378653831d` completed. Validation selected alpha 0 for the 99%/98% targets (preservation 1.0000, wrong recall 0.4100) and alpha 0.00250627 for 95% (0.9675/0.4750). Untouched test yielded 0.9575/0.4175 for 99%/98% and 0.9550/0.4775 for 95%. Median test first-trigger layers were 3 and 2. Sequential recall gain versus the best fixed gate was +0.0200, -0.0225, and -0.0500 at 99%, 98%, and 95%. The common rule's test wrong-recall spread was 0.7250/0.7250/0.6650 across datasets.
- Evidence saved: `analysis/dense_failure_stage1/independent_sequential_gate/` contains both 800-row x 28-score matrices, selected thresholds, sweep/results/breakdowns, five inspected figures, decision summary, and a 17-file passing SHA-256 manifest.
- Failure or issue: The first validation launch used the host-wide PyTorch thread default and oversubscribed CPU preprocessing; no score artifact was emitted. Binding each rank to eight threads, refreezing the source-bound contract, and rerunning resolved it. No scientific failure or partial result was reused.
- Lesson learned: Independent per-layer validation tails do not compose into a reliably conservative sample-level gate on untouched data, and their wrong coverage is strongly dataset-dependent even in the known mixture.
- Next implication: Stop this authorized action. Do not connect the gate to treatment. A shared predictor or global sample-level risk-budget method would require a new plan and explicit authorization.
