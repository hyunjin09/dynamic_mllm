# v3 Search-Budget-Matched Structured-Null Specification v2

Status: calibration completed, but the specification is **not validly frozen**.
Two independent stop gates failed: final-native joint-covariance fidelity and
paired-donor coverage. No held-out scoring or manifest freeze is permitted.

## Preserved real statistic and search budget

The conditionally frozen real statistic is unchanged:

\[
S_{real}(x)=\max_{l\in[0,4,8,12,16,20,24],\ a\in
\{IGNORE,READ\_ONLY,WRITE\_ONLY\}}
[Q_l(a)-Q_l(FULL)].
\]

Utility is per-token accepted-reference log-likelihood. Every real and null
replicate has exactly seven layers, three actions per layer, 21 candidate
effects, and one maximum. Numerical tie tolerance remains `1e-6 nats/token`.

For a paired null residual `(R~, W~)`, apply it by subtraction from `FULL` at
the validated hooks:

- `WRITE_ONLY` analogue: subtract `R~` from current-layer text READ output;
- `READ_ONLY` analogue: subtract `W~` from current-layer visual block output;
- `IGNORE` analogue: subtract both members of the same pair;
- preserve visual READ rows, current-layer text WRITE rows, and the unchanged
  dense suffix.

This orientation supersedes the provisional replacement-style preflight null.
If a null residual equals the actual residual, subtraction reconstructs the
corresponding real suppression action.

## Calibration geometry

The only fitting pool is the 400 inspected Stage B records, 200 per dataset.
The sanitized calibration manifest contains IDs, images, and prompts but no
answers, correctness labels, likelihoods, or action values. Across 2,800
sample-layer pairs, the stored native residuals include READ text rows, WRITE
visual rows, row counts, token counts, Frobenius and row norms, RMSNorm scale
ratios, and paired row-energy correlation. Maximum exact reconstruction error
is `5.960464477539063e-08`.

Evidence: `artifacts/v3_null_calibration/read_write_geometry_v1/manifest.json`.

## Required hierarchy

The active v3 plan retains three search-budget-matched families.

### 1. Isotropic norm-matched baseline

Draw independent standard-normal native READ and WRITE tensors as one seeded
pair; match every target row norm exactly. Share the pair across the three
actions within a layer and use independent seeds across layers and draws.
Geometry-only Monte Carlo supports four draws per target. This family is a
required baseline but cannot rescue either failed structured family.

### 2. Joint covariance/subspace family

READ and WRITE are never flattened together in native coordinates. Each path
is linearly mapped to 32 rows, centered, and fitted with a separate sample-Gram
PCA. The smallest rank reaching the cross-validated explained-variance target
is retained. Only dimensionless standardized READ/WRITE PCA coordinates enter
one jointly shrunk covariance. One joint draw is decoded into both path
subspaces, mapped to native target rows, and matched to every target row norm.

Five deterministic sample-ID folds selected explained variance from
`[0.85,0.90,0.95]` and shared marginal/joint shrinkage from `[0.05,0.10,0.20]`
using reconstruction error, Mahalanobis calibration, rank fraction, and
conditioning—never answer outcomes.

| Dataset | Layer | READ rank | WRITE rank | Variance target | Shrinkage |
|---|---:|---:|---:|---:|---:|
| GQA | 0 | 111 | 183 | 0.95 | 0.10 |
| GQA | 4 | 114 | 175 | 0.95 | 0.10 |
| GQA | 8 | 102 | 154 | 0.85 | 0.05 |
| GQA | 12 | 110 | 154 | 0.85 | 0.05 |
| GQA | 16 | 138 | 167 | 0.90 | 0.05 |
| GQA | 20 | 112 | 146 | 0.85 | 0.05 |
| GQA | 24 | 161 | 183 | 0.95 | 0.05 |
| TextVQA | 0 | 102 | 184 | 0.95 | 0.10 |
| TextVQA | 4 | 120 | 176 | 0.95 | 0.10 |
| TextVQA | 8 | 118 | 170 | 0.90 | 0.05 |
| TextVQA | 12 | 104 | 157 | 0.85 | 0.05 |
| TextVQA | 16 | 112 | 155 | 0.85 | 0.05 |
| TextVQA | 20 | 113 | 149 | 0.85 | 0.05 |
| TextVQA | 24 | 151 | 182 | 0.95 | 0.10 |

Exact row norms, deterministic reload, conditioning, and Monte Carlo joint
score covariance pass. Final-native subspace fidelity fails the frozen `0.50`
relative-error gate at GQA layer 16 (`0.634965`) and layer 24 (`0.531477`).
Accordingly, these fitted parameters are calibration evidence, not an approved
confirmatory null. Geometry-only precision would otherwise freeze 16 draws.

### 3. Paired real-residual donor family

Keep READ and WRITE from the same Stage B sample-layer donor. Require identical
dataset and layer; exclude the target sample and image. The distance is the
maximum multiplicative ratio over READ/WRITE Frobenius norms, native row
counts, image-token and prompt-token counts, RMSNorm scale ratios, and row-norm
coefficients of variation. Choose the eight closest donors without replacement
and map/rescale both paths to exact target row norms. There is no fallback.

The prospectively defined tight cap is `1.5`. A local repair was permitted only
when the exact minimum stratum cap was at most `1.6` and no more than 1% of that
stratum exceeded `1.5`. Exact minimum caps are:

| Dataset | Layers 0, 4, 8, 12, 16, 20, 24 |
|---|---|
| GQA | `1.6116, 1.6511, 1.6404, 1.6116, 2.1872, 1.7612, 1.7346` |
| TextVQA | `1.5000, 1.7584, 1.5714, 1.4073, 1.4861, 1.4170, 1.5472` |

All GQA strata and TextVQA layer 4 fail; 59 GQA and 5 TextVQA target-layer
pairs exceed `1.5`. The worst target is `gqa:gqa_gh_02672825`, layer 16,
whose eighth donor distance is `2.187192118226601`. One allowed geometry-only
diagnostic found broad mismatches (READ norm and visual-row/image-token count
dominate), not a single defective covariate. The donor index is therefore an
audit artifact, not a valid frozen confirmatory index.

## Conditional draw and inference rules

The geometry-only Monte Carlo rule assessed repeated panels against a 128-draw
reference on 16 deterministic records per dataset. Tolerances were `0.01` for
null-mean error, `0.02` for the post-max upper-95% quantile, and `0.015` for
paired geometry-bootstrap CI endpoints. It supports 4 isotropic, 16 joint
covariance, and 8 real-donor draws. This does not prove terminal-score
stability and cannot be retuned after held-out outcomes.

If both structured families are validly repaired, define for family `f`:

\[
S_{null,f}(x)=B_f^{-1}\sum_b\max_{l,a} A^{null,b}_{l,a}(x),
\quad D_{i,f}=S_{real,i}-S_{null,f,i}.
\]

Use a 10,000-draw stratified image bootstrap, equal-weight the GQA and TextVQA
means, and report 95% percentile intervals. Positive `D` favors the real
intervention. The specificity gate requires every required family's mean `D`
CI entirely above zero. This is an intersection-union rule with no alpha split;
dataset-specific estimates are secondary. Primary replication and the
distributional/grounding gates in `workspace/v3_confirmatory_protocol.md`
remain unchanged.

## Current consequence

Neither failed family may be used for held-out confirmation. No final manifest,
held-out score, null technical sweep, or confirmatory conclusion is authorized
until a user-approved repair establishes both final-native covariance fidelity
and acceptable paired-donor coverage.
