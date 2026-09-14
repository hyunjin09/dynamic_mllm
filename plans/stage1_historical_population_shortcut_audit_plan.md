# Stage-1 Historical Population & Shortcut Audit Plan

## 1. Goal

Before retraining or replacing Stage-1, audit two questions separately:

1. **Population construction**
   - How were the historical train/validation/test sets actually constructed?
   - Which sampling/selection rules made Dense-C and Dense-W nearly balanced?
   - How do those historical populations differ from the new canonical-source pool?

2. **Old-head behavior**
   - Which properties of the historical samples are encoded in the Stage-1 input features?
   - Which of those properties correlate with the frozen old Stage-1 score?
   - Can source/image-format/nuisance features explain a meaningful part of the old head's apparent failure-prediction performance?

This phase is retrospective diagnosis.

Do **not** retrain Stage-1 yet.
Do **not** change the threshold.
Do **not** train Stage-2.
Do **not** rerun corrective search.

The purpose is to understand whether the old Stage-1 success was driven by a genuine transferable failure signal, historical selection shortcuts, or both.

---

## 2. Scientific Questions

The report must answer:

### Q1 — Historical population construction

> Exactly how were the old 7,999 candidates selected, balanced, and split?

### Q2 — Historical split validity

> Were old train/val/test genuinely group-disjoint but still drawn from the same source/selection regime?

### Q3 — Old vs canonical shift

> Which observable properties changed between the old historical population and the new canonical-source population?

### Q4 — Source information in Stage-1 features

> Can the Stage-1 hidden-state features themselves predict whether a sample came from the historical or canonical source?

### Q5 — Nuisance information in Stage-1 features

> Can those features predict visual-token count, image format/size/aspect ratio, question length, or other available source properties?

### Q6 — Old-head score dependence

> Within the same correctness class, does the old Stage-1 score systematically vary with source/nuisance variables?

### Q7 — Historical shortcut sufficiency

> Can nuisance/source variables alone predict historical Dense-C vs Dense-W well enough to mimic the old Stage-1 behavior?

### Q8 — Matched evaluation

> Does the old Stage-1 AUROC remain high when historical C/W samples are matched or reweighted to remove obvious nuisance imbalances?

### Q9 — Feature geometry

> In Stage-1 feature space, do new canonical correct samples move toward the historical wrong region?

### Q10 — Decision

> Is the evidence most consistent with:
> - old-weight/source-specific overfitting,
> - normalization/statistics shift,
> - historical population selection bias,
> - a mixture of transferable failure signal and nuisance shortcut,
> - or an unresolved deeper representation problem?

---

## 3. Frozen Evidence Sources

Use existing artifacts only whenever possible.

Primary sources include:

```text
historical dense 7,999 manifests / split manifests
historical Stage-1 train/val/test feature shards
historical Stage-1 score trajectories
historical Stage-1 normalization artifact
historical Shared Random-4 checkpoint
historical trigger map

new canonical 4,000 candidate manifest
new dense outcomes
new Stage-1 score trajectories
new source metadata
new visual-token counts
new trigger map
```

Also inspect the scripts/configs that originally created the historical candidate pool.

Do not infer population construction from final counts alone.

Recover the actual:

```text
source dataset files
sampling code
correct/wrong filtering rule
per-dataset target counts
duplicate/group filtering
random seeds
split construction
```

and record them explicitly.

---

## 4. Phase A — Reconstruct the Historical Dataset Construction

### A1. Recover source provenance

For every historical sample, recover where possible:

```text
uid
dataset
raw source file / dataset split
original source index
image path / image hash
question ID
group ID
dense correctness
selection reason
```

Determine whether historical candidates were selected by:

```text
random sampling
correct/wrong quota filling
difficulty filtering
previous model outcomes
manual/metadata filtering
other eligibility rules
```

Do not assume.

### A2. Recover the exact balancing logic

Document the historical candidate-building logic that produced the approximately balanced old pool.

Report by dataset:

```text
requested C count
requested W count
actual C count
actual W count
candidate rejection/exhaustion rules
```

Explicitly distinguish:

```text
natural dataset prevalence
vs
historically selected prevalence
```

The old 50:50 overall C/W balance must not be treated as natural unless the source construction proves it.

### A3. Reconstruct train/val/test splitting

Verify:

```text
group-disjointness
image-content disjointness if applicable
UID disjointness
dataset proportions
C/W proportions
source-file proportions
```

Answer:

> Did train/val/test differ only by held-out identities while preserving essentially the same selection/source mechanism?

If yes, state that explicitly.

---

## 5. Phase B — Old vs New Population Shift Audit

Compare:

```text
historical train
historical validation
historical test
new canonical pool
```

both overall and separately for:

```text
GQA
ChartQA
TextVQA
```

Always condition important comparisons on:

```text
Dense-C
Dense-W
```

so source differences are not confused with correctness differences.

### B1. Observable metadata distributions

Audit every available non-label property, including:

```text
visual-token count
image width
image height
aspect ratio
image area
question length in tokens
prompt length
number of visual patches/tiles if available
answer length
source file / source subtype
question-type metadata if available
```

Do not invent unavailable metadata.

For each variable report:

```text
median / IQR
mean / std where useful
histogram / ECDF
old-vs-new standardized mean difference
```

by:

```text
dataset × correctness × population
```

### B2. Visual-token regime

Because visual-token count already showed a suspicious shift, treat it as a primary diagnostic.

For each dataset compare:

```text
old C
old W
new C
new W
```

Report:

```text
token-count histogram
median/IQR
fraction in major token-count modes
```

Also report old-head scores within token-count strata.

Do not conclude token count is causal from association alone.

---

## 6. Phase C — Frozen Old-Head Score Geometry

Use the exact old checkpoint + old normalization.

Do not change calibration.

For every available sample/layer preserve:

```text
Stage-1 raw logit
Stage-1 probability/score
Dense C/W
dataset
population/source
nuisance metadata
```

### C1. Score distributions by correctness and source

For each dataset and selected layers, especially:

```text
L0
L14
L21
L27
max-over-layers
```

plot/measure:

```text
old C vs old W
new C vs new W
old C vs new C
old W vs new W
```

Primary question:

> Does the new source shift the score independently of correctness?

Pay special attention to L0 because most new ChartQA/TextVQA false triggers already occur there.

### C2. Correctness-conditioned source effect

Within **Dense-C only**, test whether:

```text
old vs new source
```

strongly changes Stage-1 score.

Repeat within **Dense-W only**.

Report simple effect sizes and bootstrap confidence intervals.

### C3. Score vs nuisance variables

Within each:

```text
dataset × correctness × population
```

measure association between frozen Stage-1 score and:

```text
visual-token count
image dimensions/aspect ratio
question length
other available nuisance variables
```

Use:

```text
Spearman correlation
simple binned score curves
```

Do not use correlation as causal proof.

---

## 7. Phase D — Can Stage-1 Features Encode Source/Nuisance?

Use the exact same Stage-1 input features that were used by the Shared Random-4 head.

Do not use a richer representation.

Keep probes simple.

### D1. Historical-vs-canonical source-ID probe

Within each dataset, train a simple linear probe:

```text
Stage-1 feature → old vs new source
```

Use group-disjoint train/held-out splits.

Run separately on:

```text
Dense-C only
Dense-W only where support is sufficient
all samples with correctness balanced
```

Primary metric:

```text
AUROC
```

If source is highly predictable even within correctness class:

> source identity is strongly encoded in the Stage-1 feature space.

This does not yet prove the old failure head used it.

### D2. Nuisance probes

Using the same hidden features, predict available source properties such as:

```text
visual-token-count bin
aspect-ratio bin
image-size bin
question-length bin
```

Use simple linear/logistic probes.

Report cross-validated performance.

### D3. Layerwise source accessibility

Where feasible, run the source-ID probe across all 28 layers.

Plot:

```text
source-ID AUROC by layer
Dense-failure AUROC by layer
```

Question:

> Does source separability emerge at the same depths where historical failure prediction becomes strong?

This is descriptive, not proof of shared causal features.

---

## 8. Phase E — Nuisance-Only Correctness Baselines

Train models that **do not use Stage-1 hidden features**.

Inputs may include only available metadata:

```text
visual-token count
image dimensions
aspect ratio
question length
source subtype
dataset identity only where explicitly analyzed
```

Use simple models first:

```text
logistic regression
small shallow tree only as optional secondary diagnostic
```

Target:

```text
historical Dense-W vs Dense-C
```

Evaluate on:

```text
historical held-out
new canonical pool
```

Report:

```text
AUROC
AUPRC
ranking direction
```

Key diagnostic:

> Does a nuisance-only classifier perform well on the old population and then degrade/invert on the new source similarly to the frozen Stage-1 head?

If yes, historical shortcut opportunity was substantial.

Do not claim the Stage-1 head used exactly the same rule solely from this result.

---

## 9. Phase F — Matched / Reweighted Historical Evaluation

This is one of the strongest diagnostics that does not require retraining the old head.

Goal:

> Re-evaluate the frozen old head after reducing observable nuisance imbalance between historical C and W.

### F1. Visual-token-count matching

Within each dataset, create historical C/W subsets matched on visual-token count.

Prefer exact matching for discrete token-count modes.

If exact matching is impossible, use narrow bins.

Then recompute frozen old-head:

```text
AUROC
AUPRC
score distributions
```

Compare with the original historical metrics.

### F2. Multi-variable matching / weighting

If enough metadata exists, construct a conservative matched/reweighted analysis over:

```text
visual-token count
image aspect ratio / size
question length
```

Avoid highly flexible propensity models.

Report:

```text
pre-match balance
post-match balance
old-head AUROC before/after
effective sample size
```

### F3. Within-stratum AUROC

Where sample counts permit, compute old-head AUROC inside nuisance strata, e.g.:

```text
same visual-token-count mode
same dataset
similar question-length bin
```

If historical AUROC collapses only after nuisance balancing:

> a substantial portion of old performance depended on those observable correlations.

If it remains high:

> failure-relevant signal exists beyond those measured nuisances.

---

## 10. Phase G — Feature-Space Geometry

Use simple, interpretable geometry diagnostics.

Do not rely on UMAP/t-SNE as primary evidence.

### G1. Historical C/W direction

For each dataset/layer, estimate:

```text
historical correct centroid
historical wrong centroid
```

using normalized Stage-1 features.

Define a descriptive C→W direction:

```text
d = mean(W) - mean(C)
```

Project:

```text
old C
old W
new C
new W
```

onto `d`.

Question:

> Do new canonical correct samples move toward the historical wrong side of the old failure geometry?

### G2. Distance-to-old-class analysis

Compute simple standardized distances from new samples to:

```text
historical C centroid
historical W centroid
```

Condition on dataset and correctness.

Treat this as secondary descriptive evidence.

---

## 11. Phase H — Old-Head Use vs Mere Feature Availability

Maintain an evidence hierarchy.

### Level 1 — Feature availability

```text
source/nuisance property is predictable from Stage-1 features
```

This means the information is present.

### Level 2 — Score association

```text
frozen old-head score varies with that property within the same correctness class
```

This suggests the old head is aligned with it.

### Level 3 — Performance dependence

```text
old-head AUROC substantially decreases after matching/reweighting away that property
```

This is stronger evidence that historical performance relied on the correlation.

Even Level 3 is observational.

Do not claim a fully causal shortcut unless a later intervention changes the nuisance while preserving task semantics.

---

## 12. Normalization Diagnostic

Because the deployed artifact includes frozen normalization, inspect it separately.

Using existing features only:

```text
apply old normalization to old and new features
```

For each feature block/layer report:

```text
mean shift
std shift
fraction of dimensions >2 / >3 old-train std away
norm distribution
```

Compare:

```text
old train
old validation
new canonical
```

Question:

> Are new canonical states strongly out-of-distribution under the old normalization statistics?

Do not retrain normalization in this phase.

This helps separate:

```text
raw feature shift
vs
amplification by old normalization
```

---

## 13. Dataset-Construction Selection-Bias Diagnostic

Explicitly test whether the old candidate-selection procedure itself induced nuisance–correctness correlation.

For every major observable nuisance, compare:

```text
natural / pre-selection candidate distribution if recoverable
vs
historically selected C/W distribution
```

If pre-selection manifests exist, report:

```text
P(selected | nuisance, correctness)
```

or descriptive selection rates.

Main question:

> Did quota-filling or source harvesting preferentially select particular image/token regimes into C vs W?

If the pre-selection pool is unavailable, state that this causal step cannot be reconstructed.

---

## 14. Primary Comparison Tables

Produce one summary table per dataset:

| Population | Correctness | N | Trigger rate | Median L0 score | Median max score | Median visual tokens |
|---|---|---:|---:|---:|---:|---:|
| Historical train | C | | | | | |
| Historical train | W | | | | | |
| Historical val | C | | | | | |
| Historical val | W | | | | | |
| Canonical new | C | | | | | |
| Canonical new | W | | | | | |

And a shortcut-diagnostic table:

| Diagnostic | GQA | ChartQA | TextVQA | Interpretation |
|---|---:|---:|---:|---|
| Source-ID probe AUROC on C | | | | |
| Token-count probe | | | | |
| Nuisance-only old C/W AUROC | | | | |
| Frozen-head AUROC original old | | | | |
| Frozen-head AUROC token-matched old | | | | |
| Frozen-head AUROC new | | | | |

---

## 15. Decision Cases

### Case A — Source is encoded, score tracks source, matched AUROC collapses

Evidence supports:

> The historical Stage-1 head relied substantially on source/nuisance correlations induced by the historical population construction.

Next step:

```text
same architecture
+ source-balanced / canonical-inclusive retraining
```

### Case B — Source is encoded but matched AUROC stays high

Evidence supports:

> Nuisance/source information is present, but it does not explain most historical failure-prediction performance.

The canonical failure may be old-weight or normalization shift rather than a simple observable shortcut.

Next:

```text
same architecture retraining diagnostic
old normalization vs new normalization
```

### Case C — Source-ID probe weak, but new score ranking still collapses

The measured nuisance variables are not explaining the failure.

Next:

```text
same-head retraining on canonical labels
```

to test whether the failure signal itself remains learnable.

### Case D — New correct samples project strongly into historical wrong geometry

This supports:

> The old decision geometry is source-sensitive and new correct states occupy regions previously associated with failure.

Combine with matching/probe evidence before making a shortcut claim.

---

## 16. What Not to Do Yet

Do not:

```text
retrain Stage-1
change old normalization
retune threshold
train Stage-2 on expanded A+B
rerun MCTS
add layer embeddings
add new head architectures
use test data for model selection
```

This phase is diagnosis only.

---

## 17. Required Outputs

Use:

```text
analysis/dense_failure_stage1/historical_population_shortcut_audit/
```

Create:

```text
protocol.md

population/
    historical_construction.md
    historical_source_manifest.jsonl
    split_reconstruction.csv
    old_vs_new_population_summary.csv
    selection_bias_summary.csv

metrics/
    metadata_distribution.csv
    visual_token_distribution.csv
    old_head_score_by_population.csv
    score_nuisance_association.csv
    source_probe.csv
    nuisance_probes.csv
    layerwise_source_vs_failure_probe.csv
    nuisance_only_correctness_probe.csv
    matched_head_performance.csv
    within_stratum_head_performance.csv
    normalization_shift.csv
    feature_geometry.csv

figures/
    old_vs_new_visual_tokens.png
    old_vs_new_l0_scores.png
    old_vs_new_max_scores.png
    source_probe_by_layer.png
    source_vs_failure_predictability.png
    nuisance_only_generalization.png
    matched_vs_unmatched_head_auroc.png
    historical_failure_direction_projection.png
    normalization_shift.png

summaries/
    population_construction_summary.md
    old_head_shortcut_summary.md
    next_stage1_repair_recommendation.md

artifact_manifest.json
```

---

## 18. `population_construction_summary.md` Must Answer

1. What raw sources produced the historical 7,999 samples?
2. How were correct and wrong samples selected?
3. Why was the old population nearly 50:50 C/W?
4. Were train/val/test drawn from the same selection/source mechanism?
5. What exactly was group-disjoint?
6. How does the canonical 4K construction differ?
7. Which observable covariates shifted the most?
8. Can old validation success be interpreted as same-regime generalization rather than source-shift robustness?

---

## 19. `old_head_shortcut_summary.md` Must Answer

1. How strongly is historical-vs-canonical source identity encoded in Stage-1 features?
2. Which nuisance properties are most predictable from those features?
3. Does the frozen Stage-1 score vary with source within Dense-C and Dense-W separately?
4. How strongly does score depend on visual-token count or other observable properties?
5. Can nuisance-only models predict old correctness?
6. Do nuisance-only models fail/invert on canonical data in the same direction as Stage-1?
7. How much does frozen old-head AUROC change after nuisance matching/reweighting?
8. Do new correct samples move toward the historical wrong region in feature space?
9. How large is the old-normalization distribution shift?
10. What evidence supports a shortcut interpretation, and what remains only suspected?

---

## 20. `next_stage1_repair_recommendation.md`

Recommend the **smallest discriminating repair experiment** based on the audit.

Likely options:

```text
A. same head + old normalization + canonical training labels
B. same head + train-fold normalization + canonical training labels
C. source-balanced old+new mixed training
```

Do not execute them here.

Explain which hypothesis each experiment distinguishes.

---

## 21. Stop Rule

STOP after the historical-population reconstruction and old-head shortcut audit.

The next phase should begin only after deciding whether the dominant issue is:

```text
historical selection/source shortcut
normalization shift
old fitted decision boundary
or
deeper feature/head limitation
```

Only then should Stage-1 be retrained.

---

## 22. Interpretation Boundary

This audit may establish that:

```text
historical train/val/test shared a specific selection regime
new canonical data shifts observable and hidden-state distributions
source/nuisance information is encoded in Stage-1 features
the frozen head score is associated with those variables
historical AUROC depends partly on observable nuisance imbalance
```

It must not overclaim that:

```text
visual-token count is the sole cause
source identity is causally used by the head
Stage-1 architecture is fundamentally invalid
```

without stronger intervention or retraining evidence.

The goal is to replace:

> "Stage-1 unexpectedly failed OOD"

with a precise account of:

> **how the historical dataset was constructed, what shifted, what the old head could exploit, and which hypothesis should be tested next.**
