# Selective CONTINUE/DEVIATE Protocol

Frozen before any Phase-1 live execution outcome was observed.

## Authority and fixed inputs

- Source plan: `plans/selective_continue_deviate_expanded_plan.md`
- Source-plan SHA-256: `20a7517dc61197c8d3914cf8cf45183af7438e514ccdb4cba1583f9b25da34e9`
- Parent online config: `analysis/persistent_corrective_supervision/online_config.yaml`
- Parent config SHA-256: `e535873878ae3bdfcc8bdaf4c7580be674ede73e871c7f379a505870b16b69f0`
- Frozen source manifest: `analysis/persistent_corrective_supervision/training_manifest.jsonl`
- Frozen boundary manifest: `analysis/persistent_corrective_supervision/boundary_manifest.jsonl`
- Audit subset: `analysis/selective_continue_deviate/when_full_insertion_subset.json`
- Audit-subset SHA-256: `4e8677ad39bccd30ca805a868e607be310481130936a9112ec913daf945ba3e4`

No router or base-model parameter, source label, split, executor setting, or
external evaluation contract is changed.

## Phase-1 census

The cohort is the complete held-out W2C split: 128 unique UIDs,
which lies at the authorized upper bound of 128 and avoids a post-selection
subsample. It contains:

- datasets: {'chartqa': 43, 'gqa': 43, 'textvqa': 42}
- depth bins: {'early': 48, 'late': 37, 'middle': 43}
- known mechanisms: {'IGNORE': 33, 'MULTI': 30, 'READ_ONLY': 33, 'WRITE_ONLY': 32}

For each state, replay the exact all-FULL prefix, insert `FULL` at the frozen
mandatory boundary, and retain every suffix named by the frozen boundary route
indices. Identical complete routes are deduplicated while all source-route
provenance is retained. This yields 252 unique live
executions from 256 compatible source
suffixes (4 duplicates removed). No route
cap or outcome-dependent selection is used.

## Classification and uncertainty

- `FULL-cache-incomplete`: at least one tested continuation is correct.
- `FULL-confirmed-invalid`: every bounded tested continuation is incorrect and
  the complete known compatible suffix set was executed.
- `unresolved`: the compatible suffix set or execution coverage is incomplete.

`FULL-confirmed-invalid` is bounded evidence, not global proof of invalidity.
Report overall, dataset, depth, and known-mechanism rescue rates with
10,000 fixed-seed UID-group percentile bootstrap
draws (seed 20260831).

## Prospective Phase-1 decision

Gate training is admitted only if:

```text
rescued states == 0
unresolved states == 0
trusted validation DEVIATE positives == 128
```

This is not a post-hoc percentage cutoff. The complete census contains exactly
the plan's required minimum of 128
validation DEVIATE positives; one known incomplete or unresolved state makes
that clean held-out contract unattainable without changing the frozen split or
weakening the trusted-label requirement after seeing outcomes.

If the condition fails, write the complete audit report and Stage-1 decision,
then stop without gate training. If it passes, proceed to the plan's frozen
linear/MLP gate and learned-WHEN + oracle-WHAT stages. Never train Stage 2 and
never run external evaluation in this phase.
