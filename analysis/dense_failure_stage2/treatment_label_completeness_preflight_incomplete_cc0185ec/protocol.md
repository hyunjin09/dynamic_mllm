# Large-scale treatment-label completeness audit protocol

- Contract: `cc0185ec7d06dcfb5b6275a3c52bf7bc3a9eec3d623d108e7704d58f0ea9de64`
- Frozen census: 21,071 exact routed prefix states from 569 UIDs.
- Frozen audit sample: 1,200 states ({'INTERVENE_REQUIRED': 500, 'KEEP_REQUIRED': 500, 'MIXED': 200}); maximum four states per UID.
- Sampling inputs: old label, dataset, source regime, layer bin, route-source signature, UID/group identity, and deterministic hash only.
- Search per unobserved first action: forced action + FULL suffix; exhaustive one later non-FULL; then sequential MCTS capped at 200 from the actual post-action state.
- Every discovery and one existing successful continuation require exact replay. Errors or parity failures are quarantined.
- Labels remain bounded-search `AUDITED_KEEP`, `AUDITED_INTERVENE`, or `AUDITED_MIXED`; no structural necessity claim is permitted.
- Prospective interpretation: {"large_invalidation": "point estimate >=0.20 and UID-bootstrap 95% CI lower bound >=0.10", "material_probe_improvement": "RW AUROC gain >=0.05 over Phase 72 and audited RW AUROC >=0.60", "mcts_inconclusive": "fewer than 10 MCTS discoveries", "mcts_saturated": "at least 10 MCTS discoveries and <=10% first discovered in iterations 151-200", "mcts_unsaturated": "at least 10 MCTS discoveries and >10% first discovered in iterations 151-200", "mostly_stable": "both KEEP and INTERVENE UID-bootstrap 95% CI upper bounds <=0.10"}
- Probe recheck: frozen Phase-72 M0/M1/R/W/RW families, five UID/image-group-disjoint folds, audited MIXED excluded.
- Scope stops after audit, replay, label revision, probe recheck, and recommendation. No Stage-2 retraining or external evaluation.
