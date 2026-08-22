We have completed binary VISUAL_ON / VISUAL_OFF MCTS extraction for:

```text
GQA
TextVQA
ChartQA
WeMath2.0-Pro
```

We now want a **read-only cross-dataset analysis**.

Do not run new MCTS, new Qwen route execution, predictor training, or route generation.

The main scientific question is:

> **Do different multimodal task families require systematically different amounts or depth schedules of direct visual access, even though difficulty variation within WeMath2.0-Pro did not explain them?**

Do not assume in advance that:

```text
GQA = easy
WeMath2.0-Pro = hard
```

Instead treat the four datasets initially as distinct multimodal task families:

```text
GQA
→ general visual QA / object, attribute, relation, spatial questions

TextVQA
→ text/OCR-heavy visual QA

ChartQA
→ structured chart interpretation

WeMath2.0-Pro
→ multimodal mathematical reasoning
```

Any claim such as “more demanding tasks require more visual computation” must emerge from the data rather than be imposed beforehand.

---

# 1. Frozen executor semantics

Use the exact existing 28-bit binary route semantics for all datasets.

```text
VISUAL_ON at layer l:
- visual rows execute decoder layer l;
- text/control rows execute with visual tokens available;
- text can directly attend to visual K/V.

VISUAL_OFF at layer l:
- visual rows bypass layer l unchanged;
- text/control rows execute without visual tokens;
- text cannot directly attend to visual K/V at that layer.
```

Therefore:

```text
ALL-OFF
=
no decoder layer has direct access to encoded visual K/V.
```

Structural side channels may remain, so use the term:

```text
no direct visual access
```

rather than claiming literally “no image input.”

An ON count is therefore best interpreted as:

> **number of decoder layers with direct visual-token participation/access**

not generic model FLOPs or total vision computation.

---

# 2. Source caches

Use the existing authoritative raw MCTS caches for all four datasets.

For GQA / TextVQA / ChartQA, use the same raw cache underlying the previous binary MCTS label-geometry analysis.

For WeMath2.0-Pro, use the completed authoritative raw cache used by:

```text
reports/wemath2pro_visual_dependence_reanalysis_v1.md
reports/wemath2pro_visual_access_placement_v1.md
```

Do not use only max-50 predictor-training manifests for the primary comparison.

Verify for every dataset:

```text
number of source samples
number of evaluated routes
number of valid routes
FULL anchor presence
ALL-OFF anchor presence
route length = 28
correctness threshold
search-budget policy
```

Stop if route semantics differ across datasets.

---

# 3. Audit MCTS search budgets before scientific comparison

This is mandatory.

Different MCTS budgets can create artificial differences in:

```text
minimum discovered ON
number of valid routes
route-placement diversity
```

because a dataset searched longer has more opportunities to discover sparse or unusual valid routes.

For every dataset, report the exact search policy separately for:

```text
FULL-correct samples
FULL-wrong samples
```

including:

```text
number of MCTS simulations
anchor routes
early stopping if any
extensions if any
```

Then determine a **common matched search prefix** separately for the two relevant populations.

## FULL-correct comparison

Choose:

```text
B_correct
=
largest simulation count guaranteed to be available
for every FULL-correct sample across all four datasets.
```

Construct the primary comparison using:

```text
FULL anchor
ALL-OFF anchor
first B_correct MCTS simulations
```

for every dataset.

## FULL-wrong comparison

Likewise define:

```text
B_wrong
=
largest common simulation count guaranteed
for every FULL-wrong sample across all datasets.
```

Use the matched prefix for the secondary correction analysis.

Do not silently compare a 200-simulation dataset against a 400/600-simulation dataset.

Also repeat the main results using all available frozen search as a sensitivity analysis.

No new search is authorized.

---

# 4. Use the same population taxonomy everywhere

For each dataset, classify every sample using exact live cached FULL and ALL-OFF outcomes.

## FULL-correct side

### V0

```text
FULL correct
AND
ALL-OFF correct
```

Interpretation:

> A correct solution exists without any direct decoder access to encoded visual K/V.

### V+

```text
FULL correct
AND
ALL-OFF wrong
```

Interpretation:

> At least one direct VISUAL_ON layer is behaviorally necessary under the current executor.

## FULL-wrong side

### A0

```text
FULL wrong
AND
ALL-OFF correct
```

Interpretation:

> Removing all direct visual access itself changes the answer to correct.

### A+

```text
FULL wrong
AND
ALL-OFF wrong
AND
at least one nonzero-ON correcting route exists
```

### D

```text
FULL wrong
AND
no correcting route discovered
```

Use these definitions identically for:

```text
GQA
TextVQA
ChartQA
WeMath2.0-Pro
```

---

# 5. First comparison: visual dependence

Before comparing layer counts, ask whether the datasets differ in whether direct visual access is needed at all.

For every dataset report:

```text
eligible N
FULL accuracy

FULL-correct N
V0 N
V0 / FULL-correct %
V+ N
V+ / FULL-correct %

ALL-OFF accuracy over all eligible samples
```

Primary comparison:

```text
P(V+ | FULL correct, dataset)
=
P(ALL-OFF wrong | FULL correct, dataset)
```

Create:

```text
Dataset | FULL acc | V0 % | V+ %
```

This distinguishes:

```text
visual dependence
```

from:

```text
visual-access budget conditional on needing vision.
```

Do not mix the two.

---

# 6. Main amount comparison: V+ only

The primary cross-dataset visual-budget population is:

```text
V+ =
FULL correct
AND
ALL-OFF wrong.
```

For every V+ sample define:

```text
b_i
=
minimum positive VISUAL_ON count
among discovered valid routes.
```

Because ALL-OFF is wrong:

```text
1 <= b_i <= 28.
```

Report per dataset:

```text
V+ N

mean b_i
95% CI
median
SD
P10
Q25
Q75
P90

mean removable direct-visual layers
= 28 - b_i

fraction with cheaper-than-FULL route
```

Primary table:

```text
Dataset | V+ N | Mean min positive ON | Median | Q25-Q75 | Removable | Cheaper %
```

This is the cleanest test of:

> **Once direct vision is actually required, do task families differ in how many decoder layers need direct visual access?**

---

# 7. V+ budget-feasibility curves

For budgets:

```text
C = 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
```

compute:

```text
P(b_i <= C | V+, dataset).
```

Create one four-dataset curve.

Also report:

```text
C = 4
C = 8
C = 12
C = 16
C = 20
C = 24
C = 28
```

in a compact table.

Potentially interesting result:

```text
GQA curve rises earlier
WeMath curve rises later
```

but do not assume that ordering.

If curves overlap, report that directly.

---

# 8. Route-placement comparison within V+

The previous WeMath result showed that ON count alone may not capture route structure.

For each V+ sample, define the exact minimum-budget valid-route set:

```text
M_i(0)
=
all valid routes with ON_count = b_i.
```

Do not choose one route arbitrarily.

For every sample, average route-placement statistics across `M_i(0)` first, then average across samples.

Every sample receives equal weight.

Compute:

```text
first ON
last ON
normalized centroid
span

early ON fraction
middle ON fraction
late ON fraction

number of contiguous ON segments
number of OFF→ON reentries
late-reentry indicator
```

Freeze depth regions identically to the WeMath analysis:

```text
early  = layers 0–8
middle = layers 9–18
late   = layers 19–27
```

A late re-entry is a new ON segment beginning at layer >=19.

---

# 9. Primary cross-dataset placement table

Produce:

```text
Dataset
V+ N
normalized centroid
first ON
last ON
early fraction
middle fraction
late fraction
ON segments
late-reentry rate
```

The important question is:

> **Do task families with comparable ON counts nevertheless place direct visual access differently across depth?**

Possible observations include:

```text
GQA
→ front-loaded access

TextVQA / ChartQA
→ more sustained visual access

WeMath
→ more fragmented / late re-access
```

but these are hypotheses only.

---

# 10. Near-minimum route sensitivity

Exact minimum routes may be lucky finite-MCTS discoveries.

Repeat all main placement analyses using:

```text
M_i(+2)
=
valid routes with ON <= b_i + 2

M_i(+4)
=
valid routes with ON <= b_i + 4
```

subject to 28.

Report cross-dataset metrics for:

```text
min+0
min+2
min+4
```

A claimed task-family placement difference must be reasonably stable under these route-set definitions.

If exact layer identities change dramatically but coarse placement metrics remain stable, report both facts.

---

# 11. Explicit amount-control for placement

Different datasets may have different `b_i`.

Do not let ON-count differences automatically create apparent placement differences.

For V+ fit descriptive models such as:

```text
normalized_centroid
~
dataset
+
min_positive_ON
```

```text
late_fraction
~
dataset
+
min_positive_ON
```

```text
late_reentry
~
dataset
+
min_positive_ON
```

Use dataset-cluster-appropriate bootstrap uncertainty.

Also report placement within fixed ON-budget bins:

```text
1–8
9–12
13–16
17–20
21–27
28
```

Report cell sizes and do not interpret sparse cells.

This is descriptive adjustment, not causal inference.

---

# 12. Visual-token-count control

The datasets may differ dramatically in native visual-token count.

If authoritative visual-token counts are available, report per dataset:

```text
mean
median
P90
maximum
```

for:

```text
all samples
V+ only
```

Then fit descriptive controls such as:

```text
min_positive_ON
~
dataset
+
log1p(visual_token_count)
```

and:

```text
normalized_centroid
~
dataset
+
min_positive_ON
+
log1p(visual_token_count)
```

Also provide coarse visual-token-bin comparisons where practical.

Do not interpret dataset effects as task complexity if they disappear after controlling basic input scale.

---

# 13. Prompt / answer-format context

Audit obvious dataset-level differences that may affect route geometry:

```text
prompt/token length
visual-token count
answer length
open-ended vs structured answer format
scoring threshold
image count
```

Do not attempt to fully statistically “remove” all dataset differences.

The purpose is to document alternative explanations for any cross-dataset effect.

Cross-dataset comparisons are observational.

---

# 14. Per-layer access profiles

For each V+ sample and each layer compute:

```text
a_i,l
=
fraction of routes in M_i(Δ)
that have layer l ON.
```

Aggregate sample-balanced profiles:

```text
A_dataset,l
=
mean_i a_i,l.
```

Plot layers 0–27 for all four datasets.

Produce:

```text
exact-min profile
min+2 profile
min+4 profile
```

Look for broad reproducible depth bands rather than isolated single-layer spikes.

Do not claim a specific layer is causally required.

---

# 15. Cross-dataset profile distance

Quantify how different the aggregate access profiles are.

For each dataset pair report:

```text
L1 distance
L2 distance
cosine similarity
```

for the 28-dimensional sample-balanced access profile.

Dataset pairs:

```text
GQA ↔ TextVQA
GQA ↔ ChartQA
GQA ↔ WeMath2.0-Pro
TextVQA ↔ ChartQA
TextVQA ↔ WeMath2.0-Pro
ChartQA ↔ WeMath2.0-Pro
```

Repeat for min+0/+2/+4.

This is descriptive structure, not proof of input-specific causal need.

---

# 16. Secondary FULL-wrong A+ comparison

Analyze separately:

```text
A+ =
FULL wrong
AND
ALL-OFF wrong
AND
some nonzero-ON route correct.
```

For every dataset report:

```text
FULL-wrong N
A0 N / %
A+ N / %
no correction N / %

A+ correction rate
```

Then for A+ calculate:

```text
minimum correcting positive ON
```

and the same minimum-route placement metrics:

```text
centroid
last ON
late fraction
segments
late reentry
```

Do not pool A+ with V+.

The estimands differ:

```text
V+
→ correctness-preserving sparse visual program

A+
→ alternative visual program correcting FULL failure
```

---

# 17. Primary task-family hypotheses

Evaluate these hypotheses prospectively.

## H1 — visual dependence differs across task families

```text
P(V+ | FULL correct)
```

differs substantially.

## H2 — V+ visual-access amount differs

Among samples that genuinely require some visual access:

```text
min positive ON
```

differs by dataset.

## H3 — visual-access placement differs

Even when ON count is controlled, task families differ in:

```text
centroid
late fraction
fragmentation
late reentry
```

## H4 — task family does not explain routing

V0/V+, ON count, and placement become broadly similar after matched search-budget and input-scale controls.

Do not require all three H1/H2/H3 to be true.

---

# 18. Do not call the datasets easy/hard unless supported

Do not write conclusions such as:

```text
GQA is easy, so it uses fewer layers.
```

from dataset identity alone.

Use descriptive task-family terminology first.

If results reveal an ordering such as:

```text
GQA < TextVQA < ChartQA < WeMath
```

in multiple visual-access metrics, then discuss whether it is *consistent with* increasing visual/reasoning demand.

Do not claim dataset identity is a calibrated scalar difficulty measure.

---

# 19. Optional within-GQA complexity analysis

Only if reliable, authoritative GQA metadata already exists, inspect whether GQA contains categories corresponding to simpler versus more relational/compositional questions.

Examples might include:

```text
attribute/object
relation
spatial
compositional
```

but use only official metadata.

Do not infer categories from question text with an LLM.

This analysis is secondary and should be run only after the four-dataset comparison is complete.

Its purpose would be to test whether any cross-task trend has a compatible within-GQA analogue.

---

# 20. Strongest possible positive interpretation

A particularly interesting pattern would be:

```text
GQA
→ lower V+ min positive ON
→ earlier centroid
→ fewer late reentries

TextVQA / ChartQA
→ intermediate

WeMath
→ larger or more distributed direct visual access
```

stable under:

```text
matched MCTS budget
V+ conditioning
min+0/+2/+4
visual-token controls
```

Then a defensible conclusion would be:

> **Visual-computation demand is more strongly task-regime-dependent than difficulty-dependent within a single benchmark.**

This would motivate dynamic visual routing based on image-query/task properties rather than generic scalar difficulty.

Do not force this interpretation if the data disagree.

---

# 21. Important negative interpretation

If:

```text
V+ min positive ON ≈ same
placement ≈ same
```

across GQA/TextVQA/ChartQA/WeMath after controls, classify this as meaningful negative evidence:

> **Neither within-benchmark difficulty nor broad task family explains the observed visual-route heterogeneity.**

In that case, do not continue subdividing difficulty post hoc.

The next research question would instead be whether discovered route identities are genuinely input-specific or largely interchangeable across samples.

Do not launch that experiment automatically.

---

# 22. Required outcome classification

End with exactly one primary classification.

## Outcome A — task family strongly affects amount and placement

Both:

```text
V+ min positive ON
and
placement schedule
```

show robust cross-dataset differences.

## Outcome B — task family mainly affects visual dependence

V0/V+ prevalence differs strongly, but conditional V+ ON amount and placement are similar.

## Outcome C — task family affects visual-access amount, not placement

V+ min positive ON differs, but amount-controlled placement does not.

## Outcome D — task family affects placement, not amount

V+ ON counts are similar, but robust depth-schedule differences remain.

## Outcome E — no stable task-family relationship

After matched search and proper conditioning, neither amount nor placement differs robustly enough to explain route heterogeneity.

State whether visual-dependence prevalence itself still differs.

---

# 23. Required output directory

Create:

```text
outputs/cross_dataset_visual_access_v1/
```

including at least:

```text
source_integrity.csv
search_budget_audit.csv

full_alloff_taxonomy_by_dataset.csv
visual_dependence_by_dataset.csv

vplus_min_on_by_dataset.csv
vplus_budget_feasibility.csv

vplus_placement_by_dataset.csv
vplus_layer_profiles_min.csv
vplus_layer_profiles_plus2.csv
vplus_layer_profiles_plus4.csv

profile_distance_matrix.csv
amount_adjusted_placement_models.csv
visual_token_control.csv

aplus_taxonomy_by_dataset.csv
aplus_min_on_by_dataset.csv
aplus_placement_by_dataset.csv

analysis_summary.json
analysis_manifest.json
figures/
```

---

# 24. Required figures

At minimum:

```text
01_visual_dependence_v0_vplus_by_dataset
02_vplus_min_positive_on_by_dataset
03_vplus_budget_feasibility_by_dataset
04_vplus_layer_access_profile_exact_min
05_vplus_centroid_by_dataset
06_vplus_early_middle_late_fraction
07_vplus_late_reentry_by_dataset
08_min_plus2_plus4_sensitivity
09_visual_token_count_by_dataset
10_aplus_min_correcting_on_by_dataset
11_aplus_layer_access_profile_by_dataset
```

Keep figures sample-balanced.

---

# 25. Required final report

Create:

```text
reports/cross_dataset_visual_access_v1.md
```

The final report must directly answer:

1. How does direct visual dependence (`V+`) differ across GQA, TextVQA, ChartQA, and WeMath2.0-Pro?
2. Among V+ samples, do datasets differ in minimum positive VISUAL_ON count?
3. Do their visual-budget feasibility curves differ?
4. Do they place direct visual access differently across decoder depth?
5. Are any placement differences still present after controlling for ON count?
6. Are results stable for exact-min, min+2, and min+4 route sets?
7. Are apparent differences explained by unequal MCTS search budgets?
8. Are they explained by native visual-token count or other obvious input-scale differences?
9. How do A+ correction rates and correcting-route budgets differ across datasets?
10. Does the evidence support the statement that broad multimodal task regime explains visual-access requirements better than within-WeMath difficulty?
11. If not, what is the strongest remaining interpretation?
12. Which Outcome A–E best describes the evidence?

Do not launch another experiment automatically.

---

# Core execution principle

> **Compare GQA, TextVQA, ChartQA, and WeMath2.0-Pro under identical binary-route semantics and matched MCTS search opportunity. First separate whether direct vision is needed at all (V0 vs V+), then among V+ samples compare how much direct visual access is needed and where across decoder depth it occurs. Treat dataset identity as task family rather than a preassigned difficulty scale, and only infer a simple-to-complex computation trend if the observed amount and/or placement results actually support it.**