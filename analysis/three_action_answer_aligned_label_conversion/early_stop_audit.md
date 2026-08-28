# Three-Action Beam Pilot Early-Stop Audit

Date: 2026-08-25 (Asia/Seoul)

Jobs 1609 and 1610 were canceled after the user approved
`plans/4way_labeling_3.md`. Job 1609 had completed its 56-sample repeatability
calibration and 24 of 56 real-data pilot samples; job 1610 had never started.

The 24 completed pilot records are retained as historical provenance and are
not four-action training labels. They cover all five datasets and contain:

- 1,460 source routes;
- 1,417 replay-valid converted routes and 43 explicit replay failures;
- 516 W2C hard, 161 W2C soft, and 740 C2C converted routes;
- zero checksum, binary-parity, positive-correctness, C2C-gain, cache, or
  worker failures.

The prospectively frozen beam-stability gate was already impossible to pass:

- 322 of 1,417 route comparisons had different beam-8/beam-16 canonical
  routes;
- 167 of 1,417 had positive-set Jaccard below 0.50;
- the minimum Jaccard was 0.0.

Because the audit required every route comparison to pass, completing the
remaining samples could not change the launch decision. The supported
diagnosis is bounded-beam label instability, not executor failure. The approved
replacement uses exact sequential verified branching and no beam search.

Slurm provenance:

- 1609: `CANCELLED by 1003`, elapsed `04:22:15`, ended
  `2026-08-25T23:13:12`.
- 1610: `CANCELLED by 1003`, elapsed `00:00:00`, ended
  `2026-08-25T23:13:12`.
