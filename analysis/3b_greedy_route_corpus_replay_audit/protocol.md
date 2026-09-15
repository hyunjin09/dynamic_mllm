User authorized plans/3b_greedy_route_corpus_audit_replay_filtering_plan.md,
eight GPUs via Slurm, submission only this turn. Preserve the canonical package.
One new derived corpus; source identities and labels remain immutable.
CPU inventory/census and relocation precede GPU gate, then deterministic unique
route replay. Gates fail closed. No geometry, raw visual capture, training,
pair-supervision rebuild, or automatic next experiment.
Gate rules and identities are in frozen_contract.json. Resume validates every
completed record's canonical hash, source hash, contract and route key; successful
routes are never replayed. Checkpoint atomically every16 routes and at sample end.
At most one attempt per uncompleted route per authorized run; errors retain UID,
route identity, type/message and attempt count. An interrupted uncommitted batch
may be retried; durable complete records are never duplicated. Error records are
not silently retried. Missing dense or any replay error prevents READY.
Per-benchmark reports and equal-family macro views accompany pooled counts.
Dense itself belongs to the enumerated route set and is replayed exactly once.
Readiness is a technical corpus check, not a claim of statistical adequacy.
Nonzero replay drift requires qualitative RS-category review against actual
rates; no post-outcome numerical threshold is invented by the batch reporter.
The job stops after derived reports with analysis_ready_for_interpretation;
the main agent must review RS/readiness conclusions when the user returns.
