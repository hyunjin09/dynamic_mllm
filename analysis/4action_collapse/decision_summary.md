# Four-Action Collapse Isolation Decision Summary

## Completion and validity

The ordered experiment in `plans/four_action_collapse.md` is complete. All
three production jobs ended `COMPLETED 0:0`:

| Stage | Slurm job | Elapsed | Completed evidence |
|---|---:|---:|---|
| A2 online guaranteed-boundary training | 1725 | 1:06:35 | 10 epochs, 480 optimizer steps, 8,660 validation executions |
| B1 POLAR C2C-all-FULL ablation and internal execution | 1729 | 0:14:13 | 10 epochs, 470 optimizer steps, 866 selected-checkpoint executions |
| Upfront-vs-online boundary probe | 1749 | 0:02:17 | 5,168 matched records, 664 validation records, 2,000 paired bootstrap draws |

The completion audit verified all 20 A2/B1 checkpoint hashes, exact validation
coverage, exactly 2,397 scheduled A2 mandatory-boundary exposures, all eight
probe feature shards, and all selected probe checkpoint hashes. External
evaluation was not run, as required.

## Question 1

> Can the unchanged online router recognize a mandatory deviation state when
> explicitly trained on it?

**Answer: YES, on the fixed overfit-capacity pilot.**

A1 passed at epoch 30. Boundary Valid-Action@1 and non-FULL recall were both
0.9583; singleton IGNORE/READ_ONLY/WRITE_ONLY recall was
0.9583/1.0000/0.9167. Free rollout departed from all-FULL for every pilot W2C sample,
W2C rescue was 0.8958, and C2C preservation was 0.9167.

This establishes local capacity under persistent direct exposure. It does not
establish held-out population generalization.

Evidence: `mandatory_boundary_overfit_report.md` and
`mandatory_boundary_overfit_history.jsonl`.

## Question 2

> Does guaranteed mandatory-boundary coverage break online all-FULL
> free-rollout collapse?

**Answer: NO, not under the frozen one-visit A2 intervention.**

A2 preserved all 61,440 balanced training visits and replaced exactly one
visit for each of 2,397 W2C samples with its latest mandatory-boundary route.
Those visits were front-loaded by the deterministic schedule:
2,274/114/8/1 in epochs 1–4 and zero thereafter.

All ten held-out validations had boundary Valid-Action@1 = 0 and W2C rescue =
0. The selected epoch 2 made 24,244 FULL decisions and four IGNORE decisions
across 24,248 layer decisions; C2C preservation remained 1.0. Therefore the
mechanism was active, but one mostly early exposure per W2C sample was not a
sufficient population-level remedy.

Evidence: `online_boundary_coverage_v2_report.md` and
`online_boundary_coverage_v2_history.jsonl`.

## Question 3

> Does removing the exact C2C all-FULL route break the upfront/POLAR all-FULL
> mode?

**Answer: NO.**

B1 removed 3,501 exact all-FULL routes from training C2C, explicitly excluded
the 35 training samples made route-empty, and changed zero validation labels.
The selected exact-set-NLL epoch 4 still predicted and executed one unique
all-FULL route for all 866 validation samples. W2C top-1 valid-route coverage
and actual W2C rescue were both 0; C2C preservation was 1.0. Later epochs
occasionally emitted a few non-FULL routes, but W2C top-1 coverage remained 0
while C2C coverage degraded. The universal C2C route was therefore not the
sole sufficient cause of POLAR collapse.

Evidence: `polar_c2c_no_allfull_report.md`,
`polar_c2c_no_allfull_history.jsonl`, and
`polar_c2c_no_allfull_execution_summary.json`.

## Question 4

> Are mandatory deviation states more predictable from current routed hidden
> states than from initial input features?

**Answer: NO under the frozen matched probe.**

The 955,777-parameter probes used 2,584 positive/negative pairs matched within
split, dataset, and target layer. Upfront validation AUROC was 0.576372 and
online validation AUROC was 0.575097. The paired online-minus-upfront AUROC was
-0.001275 with UID-group bootstrap 95% CI [-0.054773, 0.053437]. The frozen
online-advantage rule required the lower bound to exceed zero.

Both representations were only weakly predictive. The null online advantage
does not prove that no richer online representation can help, but it removes
the current empirical motivation for preferring this online state summary.

Evidence: `upfront_vs_online_boundary_probe_report.md` and
`upfront_vs_online_boundary_probe_summary.json`.

## Architecture decision

**Neither architecture is currently supported as the final four-action
router.** Neither produced any held-out W2C rescue after its isolated fix, and
the matched probe found no online representational advantage.

If a follow-up substrate had to be prioritized, the online architecture has
slightly stronger evidence because A1 is the only experiment demonstrating
that either four-action router can learn and execute corrective actions under
direct supervision. That is not enough to select it for full training or
external evaluation: A2 did not generalize, and its greater sequential runtime
complexity currently buys no measured representation advantage. POLAR is
operationally simpler, but three complete-route variants—including B1—have now
collapsed without a positive W2C capacity demonstration.

## Next action

Do not rerun either completed recipe, start external evaluation, or declare an
architecture winner. The smallest discriminating follow-up would be a matched,
low-budget comparison that gives both substrates the same persistent targeted
W2C/non-FULL supervision mass across epochs, then selects only on held-out W2C
rescue and C2C preservation. For online routing, this must avoid A2's single
front-loaded exposure; for upfront routing, it must test W2C corrective-route
capacity rather than overall route membership dominated by C2C.

That follow-up changes the training intervention and is not authorized by the
completed plan. It should be specified prospectively and requires explicit
user approval before implementation or execution.

## Decision confidence and independent review

Confidence is medium. A read-only independent review ranked: no architecture
selection before a matched discriminator > online follow-up > POLAR follow-up.
Its strongest objection was that A2's single diluted visit is not a fair test
of persistent online remediation; it nevertheless agreed that A1 is capacity
evidence, not evidence that online is the better population substrate.
