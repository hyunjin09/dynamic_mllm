# Four-Action Collapse Isolation Tasks

## A0 — Boundary metadata

- [x] RED tests specify exact latest all-FULL-prefix boundaries.
- [x] Implement one deterministic record per W2C train sample.
- [x] Verify FULL is invalid and a non-FULL action is valid at every boundary.
- [x] Save checksum-bound manifest and human-readable audit.

## A1 — Online overfit pilot

- [x] Freeze 96 W2C and 24 C2C IDs before training.
- [x] Guarantee exact boundary-reaching trajectories for every selected W2C.
- [x] Preserve unchanged architecture, loss, optimizer, labels, and executor.
- [x] Save frequent checkpoints and all required boundary/free-rollout metrics.
- [x] Apply the prospective pass/fail gate without outcome-dependent tuning.

## A2 — Conditional full online retrain

- [x] If A1 passes, guarantee at least one boundary visit per W2C over 10 epochs.
- [x] Preserve ordinary valid-route sampling for remaining visits.
- [ ] Report internal routed behavior and do not start external evaluation.

## B1 — Upfront POLAR ablation

- [x] Remove only exact C2C all-FULL routes in a derived frozen manifest.
- [x] Exclude and report the 35 resulting route-empty C2C samples.
- [ ] Run matched ten-epoch exact-set NLL and internal routed execution.

## Cross-track probe and decision

- [ ] Compare matched-capacity upfront and current-state boundary probes.
- [ ] Save all required reports under `analysis/4action_collapse/`.
- [ ] Update phase memory, workflow state, decision log as warranted, and push
      portable handoff evidence.
