# Stage-2 observed-valid-set loss protocol

- Contract: `c3e6adfd54a58e5b30eea86757599be49a5dd9879e01a1f7b0c26ad692bd8421`
- Parent B contract: `b17a81d4749b842fd843a344d3799bffdfc0f41e2c516a835242b656324440f8`; diagnosis contract: `ffc2b02bbf7ab27fc0bf52399f4231813ab2ea4c936e4d11f8d8c0d11e23bbcf`.
- Exact identity: UID + layer + complete entering routed-action prefix; no approximate state matching.
- Population: 21,071 unique states from 34,253 successful route-state occurrences.
- Changed variable: plain single-label CE -> stable exact observed-valid-set logsumexp loss.
- Frozen: model/router, seed, optimizer, 3,144 updates, C:W and Single:MCTS draws, route/state schedule, Stage-1 thresholds, Historical-800 validation, executor, and LMMS scoring.
- Checkpoint: final global update only, matching Experiment B.
- Stop: B-vs-C oracle evaluation, C P98/P95/P90 rollout, and one unexecuted next decision; no test/search/threshold/architecture change.
