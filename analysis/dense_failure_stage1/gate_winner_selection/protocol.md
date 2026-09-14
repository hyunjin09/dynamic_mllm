# Stage-1 gate winner selection protocol

- Contract SHA-256: `f065d3728ebce2c95667a566e09ab49d5a4ead901b0334e5e7566df0f2288135`
- Phase-50 source contract: `6b1e4812a0a1e51f3dec822964b848a3a0410451dfe7a289d25c4b378653831d`
- Phase-51 source contract: `264a9407e5acb35e19bd4b53436ae8bf1175ad989f8c974cf7d8e859565e095b`
- Population: 800 validation records (400 current-dense correct, 400 current-dense wrong) and a separate 800-record test split.
- Target: current-runtime dense failure only.
- Candidate gates: independent sequential, shared All-28 sequential, shared Random-4 sequential, and shared Random-4 fixed layer 27.
- Utility: wrong samples triggered minus correct samples triggered, divided by all samples for utility rate.
- Validation eligibility: failure precision at least 90%.
- Winner: maximum validation utility rate; ties use preservation, recall, earlier median trigger, simplicity, then frozen candidate order.
- Independent sweep: every empirical shared-alpha breakpoint, converted into 28 per-layer `higher` quantile thresholds.
- Shared sweep: every attainable validation trajectory-maximum threshold plus an all-trigger terminal point.
- Test scores are not used by `select-validation`; the selected winner and all candidate controls are frozen first.

## Test-access qualification

The Phase-50/51 test trajectories already existed and had been evaluated in their source phases, so this is a Phase-52 selection-held-out confirmation, not a historically unopened test. During implementation, five leading Phase-51 test rows were printed before freeze to inspect format. No aggregate or candidate-selection statistic was computed from them. This deviation is retained in the frozen contract and limits any claim of strictly unopened test access.
