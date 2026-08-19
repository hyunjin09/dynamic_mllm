# v3 Structured-Null Repair Report

## Decision

The bounded calibration action does not satisfy the v3 confirmation entry
gate. The real statistic, layer/action search, scoring rule, and proposed data
construction remain unchanged, but two mandatory structured-null families fail
independently. No held-out outcome was loaded, scored, or inspected, and the
1,600-record manifest was not frozen.

## Completed calibration evidence

- Extracted exact READ and WRITE residual geometry for 400 inspected Stage B
  records over layers `[0,4,8,12,16,20,24]`: 2,800 complete pairs, 200 records
  per dataset, maximum reconstruction error `5.96e-8`.
- Removed the unused inherited correct/wrong sampling-cell field before null
  fitting. Sample IDs, prompts, images, tensors, and tensor checksums were
  unchanged; no answer, likelihood, generated correctness, or action value was
  used in fitting.
- Fitted 14 path-specific PCA models and joint standardized-score covariance
  models by five-fold geometry-only cross-validation. Exact ranks, variance
  targets, and shrinkage are recorded in
  `workspace/v3_structured_null_spec_v2.md`.
- The joint score covariance Monte Carlo error is below `0.079` in every
  stratum (tolerance `0.15`); serialized reload is deterministic; condition
  numbers and exact row-norm matching pass.
- Geometry-only precision freezes 4 isotropic draws, 16 joint-covariance draws,
  and 8 real donors, conditional on repairing the invalid null families.
- Official-annotation grounding eligibility passes: 123 GQA and 130 TextVQA
  records meet the frozen unambiguous-target and matched-control rules (minimum
  100 per dataset). The perturbation and inference rule is specified but was
  not executed.
- A separate prospective Stage C2 amendment freezes common right-padding within
  image groups. It does not modify main Stage C and was not executed.

## Exact failed conditions

### Joint covariance/subspace null

Confirmed observation: after decoding, native-row remapping, and exact target
row-norm matching, final-native subspace relative error is `0.634965` at GQA
layer 16 and `0.531477` at GQA layer 24, above the prospectively frozen `0.50`
gate. All other strata pass that component; joint-coordinate covariance and
norm fidelity pass.

Diagnosis: **supported** that the currently frozen 32-row decode/remap/rescale
construction is invalid for two GQA strata. It is **unknown** whether a
different geometry representation can repair this without weakening the null.
Ruled out: nonfinite covariance, degenerate rank, poor conditioning,
nondeterministic serialization, joint-score covariance instability, and row
norm mismatch.

### Paired real-residual donor null

Confirmed observation: exact eighth-donor caps exceed the prospective
tight/local-repair rule in every GQA layer and TextVQA layer 4. The worst cap is
`2.187192118226601` for `gqa:gqa_gh_02672825` at layer 16. There are 64 weak
target-layer pairs (`59` GQA, `5` TextVQA) above `1.5`.

Diagnosis: **supported** that the 200-record-per-dataset Stage B pool does not
provide adequately matched eight-donor coverage under the frozen distance.
The one permitted geometry diagnostic found that READ norm dominates 29 weak
pairs and visual-row/image-token count dominates 23; this is not a single
pathological covariate that can be removed as a local bug. Ruled out:
same-sample/image leakage, outcome-based selection, tie nondeterminism, an
incorrect donor count, and one isolated tail record.

## Technical validation status

Residual extraction, exact reconstruction, tensor finiteness, model fitting,
conditioning, deterministic seeds, paired draw construction, 21-cell search
enumeration, serialization, and checksums passed. The code now implements the
approved null orientation by subtracting a paired null residual from `FULL` at
the validated READ/WRITE hooks, with unit coverage for visual/text preservation.

The full calibration suffix smoke was deliberately not run after the two stop
conditions became decisive. Therefore the repaired entry gate is not satisfied
and no claim is made that the invalid fitted nulls are ready for terminal
scoring.

## Grounding and Stage C2

The frozen grounding subsets use exactly one official scene-graph object for
GQA or one frequency-supported TextOCR word box for TextVQA, plus a
deterministic equal-area non-target match (source area ratio at most `1.25`,
target overlap IoU at most `0.05`). The future primary contrast compares the
absolute modulation of the unchanged maximum-over-21 statistic under target
versus matched-control Gaussian blur. No grounding image was perturbed here.

The Stage C2 amendment records the supported implementation fact that unequal
BF16 prompt shapes caused numerical WRITE divergence and common right-padding
restored exact visual-state/WRITE equality at all seven layers. It remains
separate and prospective.

## Integrity and principal artifacts

- Geometry manifest SHA-256:
  `998b48d5ac7c392dad6f5602d59463318937a9a12d1e8f9e07baca64f6a84d7f`
- Joint-model manifest SHA-256:
  `6a12a1e73bd9e6578eb5581be2a19d6500d20edc75eabe964208f77bdffb4e5e`
- Donor manifest SHA-256:
  `7aa635d7fb751c0dfc45c5cf9b4605e1dbe972272b4e7cd7d33062d76de5cc28`
- Coverage audit SHA-256:
  `bcd7ae1eee64cb505ea2609e9966331041c4946302518d87400717e9c91e5143`
- Draw-precision SHA-256:
  `6256bc35a17485fe88bd3d47bf8f97f1fae5e17ebdec7d6a32b700e1ae3361a6`
- Weak-tail diagnostic SHA-256:
  `c7a555f55de978bc9a67446626e8c266ea575bb4cc487c0971bcc310a4019102`
- Grounding audit SHA-256:
  `fad4057361a367609a852aa72cdef4d1b81f6d8005c5a652389624a06ed6110b`
- Active plan remains unchanged at SHA-256:
  `8612e1b22d76dcfcc8a5f63780493d9d29e1f9c0288d986ee21047890c129605`

Pinned runtime: Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, decoder eager attention
with query chunk 1024, vision SDPA, torch `2.6.0+cu124`, transformers `4.51.3`.

## Smallest viable follow-up options

1. Authorize a redesigned joint native-row representation and a new,
   image-disjoint residual-calibration pool large enough to improve paired
   donor density; re-run only outcome-blind geometry gates. This is the most
   coherent repair because both failures must be resolved together.
2. Authorize a revised donor-distance or donor-count rule plus a revised joint
   remapping tolerance/representation. This directly changes two null validity
   definitions and is scientifically weaker unless justified prospectively.
3. Stop the v3 confirmatory direction while preserving discovery results.

Accepting the observed wide caps or raising the `0.50` fidelity tolerance is
not a valid local repair. The strongest objection to option 1 is cost: it is a
new calibration sweep and method redesign, outside the current authorization,
with no guarantee that both gates will pass.

Independent review ranked a pre-confirmation pivot above naming only one repair
family because repairing either failed family alone cannot reopen the gate. I
adopt that ranking. Here, “pivot” is a decision to redesign or stop before
confirmation; it does not authorize a new scientific direction or any held-out
execution.

PIVOT_BEFORE_CONFIRMATION
