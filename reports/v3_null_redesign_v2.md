# Outcome-Blind v3 Null Redesign v2

## Decision

The bounded redesign did not construct a scientifically valid pair of
search-budget-matched structured nulls. No held-out confirmation manifest was
frozen, no held-out answer was scored, and no four-action Q value was computed
on the new calibration records.

## What was held fixed

- Active plan v3 and the maximum-over-21 real statistic.
- Layers `[0,4,8,12,16,20,24]` and non-FULL actions `IGNORE`, `READ_ONLY`, and
  `WRITE_ONLY`.
- Per-token accepted-reference utility for any future confirmation.
- Stage A READ and WRITE hooks, paired residual semantics, exact row-norm
  matching, donor count eight, matching distance, and the `0.50` final-native
  fidelity threshold.
- No training, router/probe/controller work, Stage C2, Stage D, or held-out
  terminal scoring.

## Calibration pool and integrity

The initial deterministic pool used 1,000 unique GQA train images and 1,000
unique TextVQA train images. The complete validation universes were reserved,
which guarantees image-disjointness from discovery, v2 Stage C, proposed v3
confirmation, and Stage C2. The manifest has no answer fields.

After the initial donor audit showed inadequate coverage, one enlargement was
performed under the task's prospective allowance. The final pool contains
2,000 unique images per dataset; its 2,000-record delta is image-disjoint from
the initial pool. Selection seed was `2026080701`.

Across 4,000 records and seven layers, extraction produced 28,000 paired
path-specific READ/visual-WRITE residuals. Every extraction used pinned
Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` in BF16. Maximum reconstruction
error was `5.960464477539063e-08`, below the `1e-4` gate. All shard manifests
state that answer/action outcomes were not loaded or used.

Two launch failures occurred before any record was extracted: a missing
`PYTHONPATH`, then a missing deterministic-CuBLAS environment variable. The
focused environment repair succeeded. A checksum merge was also changed from
whole-file to streaming SHA-256 to bound process memory. These repairs did not
change residual definitions or scientific settings.

## Empirical paired-donor null

The existing distance was evaluated before any redesign. The initial pool had
good bulk matching but failed complete coverage:

- global eighth-neighbor caliper: `2.625`;
- 58/14,000 target-layer rows above `1.5`.

A fixed-target donor-pool learning curve improved monotonically with donor
count, justifying the single allowed enlargement. On the final 4,000-record
pool:

| Dataset | Median range | 99th-percentile range | Worst eighth donor | Rows above 1.5 |
|---|---:|---:|---:|---:|
| GQA | 1.0734--1.1036 | 1.2583--1.3991 | 3.09375 | 38 |
| TextVQA | 1.0651--1.1005 | 1.1829--1.2962 | 1.6419 | 3 |

The GQA tail remained dominated by image-token/WRITE-row geometry. Although
the weak-row fraction fell, the exact global caliper worsened because the
larger pool introduced additional rare shapes. A caliper of `3.09375` is not a
minimal local repair under the unchanged `1.5`/`1.6` rule. The saved nearest-
eight donor index is audit evidence, not a confirmatory null freeze.

## Joint covariance/subspace null

Three prospective representations were compared without answer outcomes.

| Representation | Result |
|---|---|
| A: fixed 32-row path PCAs | Failed. Cross-validated native WRITE error exceeded 0.50 in many strata; joint covariance MC error was 0.1578--0.1689 versus tolerance 0.15. |
| B: exact native-row strata | Failed. Groups with at least 32 pairs covered only 48.2% of GQA and 50.5% of TextVQA. |
| C: native-row distribution | Failed. At rank cap 1,024 every stratum reached 85% sampled directional variance, but several cross-validated native errors remained above 0.50. |

For C, paired pooled-coordinate covariance, norm fidelity, conditioning,
deterministic serialization, and generated covariance fidelity passed. The
failure is specifically out-of-sample native geometry fidelity, supported by
the rank-1,024 extension: TextVQA layer-4 WRITE remained `0.6367`, TextVQA
layer-8 READ/WRITE remained `0.5589`/`0.5601`, and GQA layer-8 READ/WRITE
remained `0.5461`/`0.5227`. The 0.50 threshold was not relaxed.

## Null hierarchy recommendation

Option A (both covariance and empirical donor as required gates) is invalid
because both families fail. Option B is also invalid: a paired empirical donor
is the more direct test of the generic-real-perturbation alternative, but this
donor pool is not well matched under complete coverage and unchanged matching
rules. Treating its rare-shape caliper as acceptable would weaken specificity
after seeing geometry failures.

The only defensible recommendation is option C: stop the v3 causal
confirmation. The proposed real statistic remains a discovery object, not a
confirmed answer-misaligned dense-participation endpoint.

## Evidence and checksums

- Initial pool: `outputs/v3_null_redesign/calibration_pool_manifest.json`
  (`7c9988dc4d795a0e2d30728e46145caf26959bcda8e8a3c434d321dd1b0850fb`).
- Enlarged pool: `outputs/v3_null_redesign/calibration_pool_manifest_v2.json`
  (`3eeea7fbe7792b45481cb52fd8bb1b862533b795e16c265fd706470641977e03`).
- Combined geometry manifest:
  `artifacts/v3_null_redesign/read_write_geometry_combined_v3/manifest.json`
  (`f338d2bf73d6f888de47355e41eb5b18eef8ea90460b780e1d1ffb746e5bd723`).
- Final donor audit: `outputs/v3_null_redesign/donor_coverage_v2.json`
  (`64b7420de19687bff459cefabf778c20bb81a80789732a91e8105583d6e2042d`).
- A--C comparison:
  `outputs/v3_null_redesign/covariance_representation_comparison.json`
  (`647bde574a4db6075b8a08f0c73175496407bb6a33ed9265ea9501760bd0ae46`).
- C rank extension:
  `outputs/v3_null_redesign/covariance_representation_c_rank_extension.json`
  (`eeb13e5ea83879e543388a9e4a32a2779a22d217d606923c4d0a53077bfe54e4`).

No independent reviewer was invoked: after the authorized enlargement and
rank extension, every permitted hierarchy has an explicit failed validity gate
and the candidate ranking is not narrow or unresolved.

STOP_V3_CONFIRMATION
