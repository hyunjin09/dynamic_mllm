# v3 Structured-Null Specification v3 Candidate

Status: **evaluated outcome-blind; invalid for confirmation; not frozen**.

This candidate preserves the active v3 real statistic and records the bounded
independent-calibration redesign. It does not authorize a confirmation sweep.

## Preserved real statistic

For each future held-out sample, the unchanged proposed statistic remains

\[
S_{real}(x)=\max_{l\in\{0,4,8,12,16,20,24\},\ a\in
\{IGNORE,READ\_ONLY,WRITE\_ONLY\}} [Q_l(a)-Q_l(FULL)].
\]

The utility remains per-token accepted-reference log-likelihood. Every valid
null replicate would need the identical seven-layer, three-action, 21-cell
maximum. No held-out answer score or four-action Q value was computed here.

## Independent calibration pool

- Deterministic selection seed: `2026080701`.
- Source splits: GQA `train_balanced_instructions`; TextVQA `train`.
- Selection: retain the minimum `SHA256(seed:dataset:record_id)` question per
  image, then take the minimum-ranked unique images.
- Entire GQA/TextVQA validation universes were excluded, in addition to all
  explicit v2/v3 Stage B, v2 Stage C, proposed confirmation, and reserved Stage
  C2 overlaps.
- Initial pool: 1,000 unique images per dataset. One prospectively authorized
  enlargement produced 2,000 unique images per dataset. The second 2,000-record
  delta is image-disjoint from the initial pool.
- Manifests contain no answer field. Geometry extraction did not invoke answer
  scoring, generation, correctness, or action-value computation.
- Pinned model: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, decoder eager attention.
- The 4,000 records yielded 28,000 paired READ/WRITE sample-layer tensors. The
  maximum reconstruction error was `5.960464477539063e-08`.

## Paired empirical residual candidate

The scientific definition was unchanged:

- same dataset, layer, and validated hooks;
- different sample and image;
- paired READ and WRITE residuals from the same donor;
- eight closest donors;
- target-specific exact row-norm matching;
- unchanged maximum multiplicative distance over READ/WRITE Frobenius norms,
  row counts, image/prompt token counts, RMS-scale ratios, and row-norm CVs;
- donor seed `2026080702`.

The initial 2,000-record audit required global caliper `2.625`. The one allowed
enlargement did not repair complete coverage: the 4,000-record pool required
`3.09375`. Final GQA medians were `1.0734`--`1.1036`, but 38 GQA target-layer
rows exceeded `1.5`; the tail was dominated by image-token and WRITE-row
geometry. Three TextVQA rows exceeded `1.5`, with worst distance `1.6419`.
Therefore the empirical donor is not a well-matched primary null under the
frozen complete-coverage rule. Its saved index is audit evidence only.

## Joint covariance candidates

All choices used geometry only, exact target-row norm matching, paired draws
shared across the three action analogues within a layer, 5% shrinkage, and
deterministic seeds.

### A — fixed 32-row baseline

Separate path-specific 32-row PCAs used 95% retained variance and a paired
standardized-score covariance. Native generated examples met the 0.50 bound,
but cross-validated WRITE reconstruction exceeded 0.50 in most middle/late
strata. With the prospectively fixed 32,768 fidelity draws, joint-covariance
relative errors were `0.1578`--`0.1689`, above the frozen `0.15` tolerance.
Candidate A fails.

### B — exact native-shape strata

Exact `(READ rows, WRITE rows)` groups were modeled only with at least 32
paired samples and used no row remapping. They covered only 48.2% of GQA and
50.5% of TextVQA samples per layer. Candidate B fails complete coverage.

### C — native-row distribution

Zero-centered row-direction feature bases reconstructed directly at target row
counts. Paired sample-pooled READ/WRITE coefficients supplied the joint
covariance; eight normalized-position bins supplied within-path row variance.
The fixed parameters were eight sampled rows per calibration sample, 85%
directional variance, and 5% shrinkage. A geometry-only rank extension from 512
to 1,024 made every stratum reach the variance target, but cross-validated
native-row error still exceeded 0.50 in multiple strata. Examples include:

- GQA layer 8: READ `0.5461`, WRITE `0.5227`;
- GQA layer 24: READ `0.5170`, WRITE `0.5267`;
- TextVQA layer 4 WRITE: `0.6367`;
- TextVQA layer 8: READ `0.5589`, WRITE `0.5601`;
- TextVQA layer 24 WRITE: `0.5493`.

Norm fidelity, determinism, conditioning, and generated joint-covariance
fidelity passed; cross-validated native fidelity did not. Candidate C fails.

## Search-matched comparison if a valid null had existed

For each null family and replicate, the planned statistic remained the maximum
over the same 21 layer/action cells. The frozen orientation remained
`S_real - S_null`, clustered at image/sample level with paired bootstrap
inference. A confirmed dense-participation claim would require the real
maximum to outperform every required primary structured null. This comparison
is not executable because no primary specificity null passed calibration.

## Scientific hierarchy decision

- Option A is unavailable: neither covariance representation nor empirical
  donor passed its complete gate.
- Option B is unavailable: paired real donors are scientifically more direct
  than a synthetic generic perturbation, but the actual design is not
  well-matched for every target under the unchanged distance and donor count.
- Option C is selected: stop v3 causal confirmation. Isotropic perturbations
  remain descriptive secondary controls and cannot substitute for a valid
  specificity null.

No further caliper, distance, donor-count, rank, representation, or pool-size
retuning is authorized under this candidate.
