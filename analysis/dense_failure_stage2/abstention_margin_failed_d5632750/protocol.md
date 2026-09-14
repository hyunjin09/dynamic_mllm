# Stage-2 abstention-margin protocol

- Contract: `d5632750c38b7ca427e6943c85b5701f5785da73df1800fbf339fe342c33e823`
- Stage-1: robust ALL-source P90 `0.9061332901863008` with strict `>`.
- Stage-2: frozen Phase-66 Experiment A, checkpoint `48536bfbf6ebf62071898aea91952b6fe6a3f13b5723b09f6d814eb0ad9bbc7f`.
- Rule: choose the best non-FULL action only when `best_nonfull_logit - full_logit > delta`; otherwise choose FULL.
- Grid source: the already-frozen final Experiment-A training epoch (1048 sampler-weighted draws), never development outcomes.
- Development: frozen Historical-800, with actual sequential execution at every finite candidate margin and an exact dense `+inf` control.
- Selection: C→C preservation at least 99.500%; maximize Net, then fewer C→W, more W→C, fewer interventions, and larger delta. A positive margin must strictly improve Net over delta=0.
- Stability: five-fold image-group-disjoint cross-fit.
- External: one locked Phase-69 four-family rerun only if the selected development margin is positive; no external retuning.

The independent review found the design stable. Its grid-resolution caveat is retained as a limitation rather than addressed with outcome-dependent grid changes.
