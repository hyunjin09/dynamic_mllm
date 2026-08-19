# v3 Search-Budget-Matched Structured-Null Specification

Status: action analogues and comparison statistic are fixed; the full null
design is not frozen because the joint-path donor caliper and draw-count
seed-stability gate remain unresolved.

## Equal search opportunity

For every sample and null replicate, each family evaluates exactly the same 21
cells as the real statistic: seven layers `[0,4,8,12,16,20,24]` times the three
non-FULL action analogues. It then applies the identical maximum rule.

For a paired null READ residual `R~` and WRITE residual `W~`:

- null `WRITE_ONLY`: replace actual READ with `R~`; retain actual WRITE;
- null `READ_ONLY`: retain actual READ; replace actual WRITE with `W~`;
- null `IGNORE`: replace both with the same paired draw `(R~,W~)`;
- `FULL`: unchanged and shared with the real statistic.

READ replacement is inserted at the validated attention-output hook as
`READ_OFF + R~`, preserving visual rows and all non-visual paths. WRITE
replacement is inserted at the validated post-layer hook as
`pre_layer_visual + W~`, preserving the current layer's text rows. The dense
suffix is unchanged.

For family `f`, define

\[
S_{i,f}^{null}=\max_{l,a}[Q_{i,l,f}^{null}(a)-Q_{i,l}(FULL)],
\qquad D_{i,f}=S_i-S_{i,f}^{null}.
\]

Positive `D` favors specificity of the real intervention. Use a stratified
image bootstrap with 10,000 draws and fixed seeds derived by SHA-256 from
`family:dataset:sample_id:layer:path:draw`. Require the 95% CI for the joint
mean `D` to lie entirely above zero for every family. This is an
intersection-union conjunction; dataset-specific CIs are secondary.

## Calibration pool

Use exactly the 200 inspected Stage B images per dataset. Fit each dataset and
layer separately. No v3 held-out record, outcome, or Stage C2 record may enter
calibration. Geometry extraction may not score terminal held-out answers.

## Family 1: isotropic norm matched

Draw independent standard-normal READ and WRITE tensors at their native target
shapes. Match the exact target READ and WRITE Frobenius norms separately.
Reject degenerate or nonfinite draws. The seed is fixed before any held-out
terminal score.

## Family 2: joint covariance/subspace matched

Map READ postvisual rows and WRITE visual rows separately to 32 rows by linear
interpolation, concatenate the two flattened paths, center by the equal-weight
calibration mean, and fit the unbiased sample covariance through its sample
Gram matrix. Retain the smallest rank reaching 90% variance and shrink retained
eigenvalues 5% toward their retained mean. Draw one joint coefficient vector,
split READ/WRITE, map each path to its native target row count, and match each
target norm exactly. This joint fit preserves calibration cross-path
covariation better than independent path fits.

Degenerate fits, rank zero, nonfinite eigenvalues, or norm error above `1e-5`
fail closed. The bounded smoke validated separate-path covariance generation,
shape insertion, and suffix execution; the revised joint fit still requires a
calibration-only technical check.

## Family 3: same-layer real residual pair

Use one donor's READ and WRITE residuals jointly. Require identical dataset,
layer, hooks, and task; exclude the same sample and same image. Match on the
maximum multiplicative ratio over READ norm, WRITE norm, postvisual row count,
visual row count, and prompt length. Select closest donors without using any
answer, likelihood, correctness, or downstream effect. Map rows and match both
path norms exactly.

An arbitrary constant caliper such as `2.0` is rejected. The exact caliper must
be fitted outcome-blind from the complete Stage B joint-path geometry and then
checked against every prospective target before terminal scoring. The proposed
rule is the smallest dataset/layer-specific leave-one-image-out cap covering
the frozen donor count for every calibration target. No target outside the cap
may be replaced or silently assigned a wider donor; the entry gate stops.

## Unfrozen items and required repair

- **Null draws/donors per target:** provisional one. It may be frozen only if a
  small inspected-calibration seed-stability analysis shows that random-seed
  contribution to the full 21-cell maximum is negligible relative to the
  paired sampling uncertainty. Otherwise use the smallest prospectively
  justified multi-draw count.
- **Real-residual calipers and donor index:** not yet fitted for all seven
  layers and both paths. Computing them requires a new calibration activation
  extraction beyond this task's no-large-sweep boundary.
- **Joint covariance fit:** implementation and technical smoke remain required.

No held-out confirmation may start until these values, seeds, fitted parameters,
donor identities, coverage, and checksums are frozen.
