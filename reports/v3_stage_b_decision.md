# v3 Stage B Decision

Date: 2026-08-06

## Decision basis

The complete four-action landscape is valid discovery evidence: all 3,200
matrices are complete, FULL parity is exact, and no values were imputed.
Heterogeneity is not explained entirely by numerical ties or a fixed schedule.
Outside layer 27, exact-epsilon ties are rare; practical `G > 0.05` occurs in
37.2% of GQA and 25.6% of TextVQA sample-layer pairs. Medians and 20%-trimmed
means preserve positive but much smaller suppression gains, confirming a
heavy-tailed rather than uniform landscape.

A single global action chooses FULL. The best per-layer and per-dataset/layer
schedules gain only `0.0045` and `0.0075` nats/token relative to FULL, compared
with the inspected sample-layer oracle's `0.0976`; their oracle regrets remain
`0.0931` and `0.0902`. Conditional sign reversals and exact independent-action
failures occur often enough to make interaction scientifically relevant, but
the near-zero interaction medians do not justify replacing the v3 four-action
objective with an interaction-only pivot.

The old Outcome B is the central caution: observed effects, including the new
best-of-actions/layers statistic, may reflect nonspecific residual
perturbations. This blocks confirmation, not a bounded preflight. Plan v3 can
compare the real statistic to isotropic, covariance/subspace, and real-residual
nulls receiving the identical frozen layer/action search budget. New
same-image candidate pools also exist (9,800 GQA groups metadata-eligible and
1,243 TextVQA groups technically valid after excluding inspected Stage C
images), although 17 Stage B TextVQA image identities require hash resolution.

## Candidate ranking and internal challenge

1. `PROCEED_TO_V3_PREFLIGHT`: resolves the decisive matched-null,
   query-invariance, overlap, and missing diagnostic questions without opening
   held-out outcomes.
2. `PIVOT_TO_INTERACTION_ANALYSIS`: viable only if the preflight shows that the
   suppression maximum is nonspecific while conditional sign variation remains
   valid against matched controls.
3. `PIVOT_TO_ANSWER_SILENT_REDUNDANCY`: warranted if search-matched nulls absorb
   the gains or prospective thresholds show mostly silent alternatives.
4. `STOP_V3_CAUSAL_DIRECTION`: unsupported while intervention validity remains
   intact and a bounded discriminator exists.

Strongest objection: the discovery oracle is optimized over inspected actions
and layers, most medians are small, and v2 Outcome B already demonstrates that
a replicating contrast can fail structured-null specificity. The answer is not
to treat the oracle as evidence of benefit; it is to make a search-budget-
matched null the first and decisive preflight gate.

An independent read-only research review returned the same ranking with medium
confidence. It emphasized that alternate sequence scoring is not independent
replication and that the missing matched null blocks confirmation but not the
bounded preflight.

## Next bounded action

With explicit approval, run one outcome-blind v3 preflight only: freeze a
prospective layer/action search statistic and practical tie rule, verify
same-image visual-prefix/WRITE equality numerically, resolve all inspected-image
identities, and exercise the complete real/null action-layer search on a tiny
discovery/calibration subset while collecting the missing residual/state
diagnostics. Do not freeze or open a new held-out confirmatory manifest until
that gate passes.

PROCEED_TO_V3_PREFLIGHT
