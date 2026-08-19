# Phase 07: v3 Structured-Null Repair

## Current Objective

Complete one outcome-blind redesign of both v3 structured-null families using
a new independent geometry-only calibration pool, including the single
prospectively allowed enlargement.

## Active Constraints

- Preserve the seven-layer, three-action, maximum-over-21 real statistic.
- Do not score, load, or inspect held-out terminal action values.
- Do not freeze the proposed 1,600-record manifest.
- Do not use held-out answers, terminal action values, generated outputs, or
  correctness in calibration or selection.
- Exclude all discovery, v2 Stage C, candidate-confirmation, and reserved Stage
  C2 images from the new calibration pool.
- Evaluate the unchanged donor distance before considering any redesign; the
  donor count remains eight.
- Compare only the prospectively specified covariance representations: fixed
  32-row baseline, native-row-count strata, and native-row row-distribution.
- Stop if matching, covariance, Monte Carlo, reconstruction, or grounding gates
  cannot be frozen prospectively without weakening the approved claim.

## Current State

- Done: v3 discovery, preflight, Stage-B-only null repair, and the authorized
  independent-calibration redesign.
- Final calibration: 2,000 unique train images per dataset, 4,000 total;
  28,000 paired sample-layer residuals.
- Final decision: `STOP_V3_CONFIRMATION` because the empirical donor and all
  permitted covariance representations fail complete validity gates.
- No held-out answer outcome, final confirmation manifest, Stage C2 outcome,
  Stage D action, or training was opened.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| The real endpoint searches seven layers and three actions. | `workspace/v3_confirmatory_protocol.md` | Every null replicate must receive the same 21-cell maximum. | confirmed |
| The old v2 Stage C effect failed both structured-null comparisons. | `reports/stage_c_conclusion.md` | Specificity, not replication alone, is the bottleneck. | confirmed |
| Stage B has 400 inspected records with validated prompt/reconstruction machinery. | `reports/v3_stage_b_reanalysis.md`; `outputs/stage_a/` | It is the permitted outcome-blind calibration source. | confirmed |
| Independent READ/WRITE null insertion passed a bounded smoke. | `outputs/v3_preflight/null_preflight_manifest.json` | Paired insertion is implementable, but the subsequently fitted joint model fails final-native fidelity. | confirmed |
| Annotation-linked grounding subsets have 123 GQA and 130 TextVQA eligible records. | `outputs/v3_preflight/grounding_eligibility_audit_v1.json` | The minimum grounding control is technically feasible without heuristic regions. | confirmed |
| The completed geometry shards have exact READ/WRITE reconstruction. | `artifacts/v3_null_calibration/read_write_geometry_v1/shards/shard_00/`; `shard_01/`; `shard_02/`; `shard_03/` | The path-specific residual definitions remain valid on calibration records. | confirmed |
| Independent calibration has 4,000 unique train images and zero validation-universe overlap. | `outputs/v3_null_redesign/calibration_pool_manifest_v2.json` | Calibration is outcome-blind and image-disjoint from all proposed held-out pools. | confirmed |
| All 28,000 new residual pairs reconstruct within `5.96e-08`. | `artifacts/v3_null_redesign/read_write_geometry_combined_v3/manifest.json` | The negative result is not an extraction-validity failure. | confirmed |
| Enlarged donor coverage requires caliper `3.09375`. | `outputs/v3_null_redesign/donor_coverage_v2.json` | The empirical null is not completely well matched. | confirmed |
| Native-row rank 1,024 still fails cross-validated 0.50 fidelity in multiple strata. | `outputs/v3_null_redesign/covariance_representation_c_rank_extension.json` | The covariance family remains unrepaired without relaxing its validity gate. | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Provisional v3 null preflight | Joint covariance, donor geometry, and draw count remained unfrozen. | supported | `reports/v3_preflight_report.md` | Extract the complete paired Stage B geometry before selecting any null hyperparameter. | Freezing arbitrary ranks, calipers, or one-draw rules. |
| Five extraction shards landed on 16 GB A4000s | CUDA OOM during pinned-model load, before any record was extracted. | supported | `runs/v3_null_geometry_s03_20260806/slurm.log` through `s07_20260806/slurm.log` | Reallocate only untouched shards to 48 GB A6000s; the failure does not implicate the residual implementation. | Retrying the 7B model on a 16 GB A4000. |
| First independent-pool extraction launch | Direct script execution omitted the project root from `PYTHONPATH`; imports failed before model load. | supported | `runs/v3_null_redesign_geometry_20260807/shard_00.log` through `shard_07.log` | Preserve the logs and launch untouched shards with `PYTHONPATH=.`. | Reusing the original batch command. |
| Second independent-pool extraction launch | Deterministic CuBLAS rejected the first forward because `CUBLAS_WORKSPACE_CONFIG` was unset; zero records were extracted. | supported | `runs/v3_null_redesign_geometry_r2_20260807/shard_00.log` through `shard_07.log` | One focused relaunch sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization; stop if the equivalent failure recurs. | Further environment trial-and-error after the focused repair. |

## Candidate Outcomes

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Separate mapped READ/WRITE PCA plus correlated joint coefficients | Preserves native path structure while representing cross-path dependence in a shared low-dimensional score space. | Joint covariance null without concatenating incompatible native tensors. | high | tested; rejected under frozen native-fidelity gate |
| Joint flattened interpolation | Simple and previously proposed. | One covariance fit per layer. | medium | rejected provisionally: incompatible row semantics are mixed directly |
| Independent READ and WRITE draws | Already smoked. | Marginal covariance only. | low | rejected: loses paired geometry required by protocol |

## Next-Step Decision

- Deliberation mode: DEEP.
- Active objective and bottleneck: determine whether larger, independent
  geometry data repair donor coverage and native-row covariance fidelity.
- Confirmed observation / unverified interpretation: the 200-record pool was
  insufficient under the unchanged donor distance and the 32-row remap failed
  two GQA strata; whether this is sample-size versus representation failure is
  unverified.
- Diagnosis: donor sparsity and native-row remapping are supported failure
  modes; repairability remains unknown.
- Viable alternatives considered: (A) fixed 32-row baseline, (B) exact
  native-row-count strata when adequately populated, and (C) a common feature
  basis with direct reconstruction at the target native row count.
- Chosen action and strongest objection: build a deterministic train-split,
  image-disjoint 2,000-image pool, test the unchanged donor metric first, and
  compare A--C only by reconstruction, covariance, norm, and conditioning
  criteria. The strongest objection is that representation C may satisfy
  subspace reconstruction yet remain a weaker scientific approximation than
  real paired donors.
- How this differs from failed attempts: it removes calibration-pool overlap,
  increases each dataset pool fivefold, and avoids choosing covariance geometry
  from answer outcomes.
- Authorization and stop condition: explicitly authorized for one bounded
  repair attempt; stop before any held-out scoring, Q computation, final
  manifest freeze, Stage C2, or Stage D.

Final next-step decision: stop the current v3 causal-confirmation direction.
There is no authorized follow-up calibration or experiment.

### Authorized enlargement decision (2026-08-07)

- Confirmed result: the initial 1,000-image-per-dataset donor audit fails, with
  global eighth-neighbor caliper `2.625`; 58/14,000 target-layer rows exceed
  `1.5`, mainly because of rare image-token/WRITE-row geometry.
- Cheap decision-changing diagnostic: a fixed-target nested donor learning
  curve improves monotonically from 200 to 800 donors but retains a weak tail;
  evidence is `outputs/v3_null_redesign/donor_pool_size_curve.json`.
- Covariance result: A, B, and C all fail the complete gate at 1,000 images;
  C reaches its prospectively set rank-512 ceiling in most middle/late-layer
  WRITE strata while cross-validated errors are often only modestly above
  `0.50`.
- Chosen bounded repair: use the one enlargement explicitly allowed by the
  task (2,000 unique images per dataset), recompute donor coverage unchanged,
  and extend C's geometry-only rank search without relaxing any fidelity
  threshold. Do not add another representation or another enlargement.
- Strongest objection: expansion may add new rare shapes as quickly as it adds
  donors, and a higher-rank synthetic model may remain less scientifically
  direct than the empirical donor null. If either family still fails, stop v3
  confirmation rather than retune again.

## Latest Research-Action Result

- Action taken: completed the authorized independent-calibration redesign,
  including the single permitted pool enlargement and native-row rank
  extension.
- Result: 4,000 images/28,000 residual pairs pass reconstruction, but both
  structured-null families remain invalid. Final donor caliper is `3.09375`;
  A/B/C covariance representations fail complete cross-validated
  fidelity/coverage.
- Evidence saved: `reports/v3_null_redesign_v2.md` and all paths listed there.
- Failure or issue: **supported** persistent rare-shape donor mismatch and
  **supported** out-of-sample native-row covariance failure at rank 1,024.
- Lesson learned: scaling calibration improves bulk donor geometry but does not
  guarantee complete tail coverage, and in-subspace generated fidelity cannot
  replace cross-validated native reconstruction.
- Next implication: `STOP_V3_CONFIRMATION`; do not freeze a held-out manifest,
  retune the nulls, or enter Stage C2/Stage D.
