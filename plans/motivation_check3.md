We have completed the WeMath2.0-Pro visual-dependence reanalysis.

The previous apparent relationship between difficulty and the **number** of VISUAL_ON layers largely disappeared after separating ALL-OFF-solvable samples.

The corrected primary cohort is:

```text
V+ =
FULL correct
AND
ALL-OFF wrong
```

For these samples, at least one decoder layer with direct access to visual K/V is behaviorally necessary under the frozen executor.

The current result is approximately:

```text
degree 0: mean minimum positive ON ≈ 14.42
degree 1: ≈ 13.87
degree 2: ≈ 13.36
degree 3: ≈ 13.76

medians ≈ 12, 12, 11.5, 12
```

Thus there is no stable evidence that harder problems require **more VISUAL_ON layers**.

The next question is different:

> **Even if the amount of direct visual access is similar, does difficulty or reasoning type change *where across decoder depth* visual access is needed?**

Possible phenomena include:

```text
easy:
visual access concentrated early
→ visual information enters text state early
→ later reasoning proceeds mostly without direct vision

hard:
visual access persists deeper
or
vision is re-accessed after long OFF intervals
```

or the reverse.

Do not assume any direction.

The purpose of this task is to analyze the existing frozen WeMath2.0-Pro MCTS cache and determine whether **visual-access placement / schedule across depth** changes systematically with difficulty.

Do not run new MCTS, Qwen execution, predictor training, or route generation.

---

# 1. Frozen executor semantics

Use the exact binary executor semantics:

```text
VISUAL_ON at layer l:
- visual rows execute layer l;
- text/control rows execute with visual tokens available;
- text can attend directly to visual K/V.

VISUAL_OFF at layer l:
- visual rows bypass layer l unchanged;
- text/control rows execute without visual tokens;
- text cannot directly attend to visual K/V at that layer.
```

Therefore an ON layer represents a decoder layer with **direct visual participation/access**.

An OFF layer after an earlier ON does not imply that the text state has forgotten visual information. Earlier ON layers may already have transferred visual information into text hidden states.

Thus this task studies:

> **where direct visual access occurs across decoder depth**

not:

> where all visual information exists in the model.

---

# 2. Source data

Use exactly the same frozen raw WeMath2.0-Pro cache and source metadata as:

```text
reports/wemath2pro_visual_dependence_reanalysis_v1.md
```

Use all raw evaluated routes.

Do not use only the later max-50 predictor-supervision view.

Verify:

```text
eligible records
FULL / ALL-OFF anchors
V0 / V+ identities
route validity
difficulty metadata
question-family identities
image identities
```

against the previous frozen analysis.

Any mismatch is an integrity failure.

---

# 3. Primary population: V+ only

The primary population is:

```text
FULL correct
AND
ALL-OFF wrong
```

Expected size from the frozen reanalysis:

```text
V+ ≈ 428
```

but verify rather than hard-code.

This population is primary because:

```text
FULL → correct
ALL-OFF → wrong
```

so some nonzero direct visual access is behaviorally necessary.

Do not mix V0 samples into the primary schedule analysis.

V0 can be reported for context but must not influence placement statistics because it has no ON position.

---

# 4. Preserve all eight difficulty strata

Analyze separately:

```text
base
x
y
z
xy
xz
yz
xyz
```

before any aggregation.

Also retain coarse difficulty degree:

```text
degree 0 = base
degree 1 = x / y / z
degree 2 = xy / xz / yz
degree 3 = xyz
```

as a secondary summary.

Do not impose an ordering among x, y, and z.

---

# 5. Audit the semantic meaning of x / y / z

Before interpreting any axis-specific result, identify the authoritative semantics of:

```text
x
y
z
```

from the available WeMath2.0-Pro source metadata, dataset documentation, or repository documentation.

Do not infer the mapping from examples alone.

Report:

```text
x = ?
y = ?
z = ?
```

with exact provenance.

If authoritative mapping cannot be established, continue the quantitative analysis using only the literal names:

```text
x-axis
y-axis
z-axis
```

and explicitly avoid semantic labels such as “step complexity” or “visual complexity.”

Also inspect whether the source contains additional reliable reasoning-type metadata such as:

```text
problem category
knowledge type
reasoning type
geometry subtype
number of reasoning steps
difficulty-component metadata
```

Use such fields only if their semantics are authoritative and coverage is adequate.

Do not generate or infer reasoning-type labels with an LLM in this analysis.

---

# 6. Primary route set: minimum-budget valid routes

For each V+ sample `i`, let:

```text
b_i = minimum VISUAL_ON count
      among all discovered valid routes.
```

Define the primary minimum-budget route set:

```text
M_i(0)
=
{m :
    m is valid
    AND
    ON_count(m) = b_i
}
```

Do **not** arbitrarily choose one route when several minimum-budget routes exist.

Use all routes in `M_i(0)`.

For each sample/layer compute the sample-balanced inclusion probability:

```text
a_i,l
=
fraction of routes in M_i(0)
with layer l = VISUAL_ON.
```

Thus:

```text
a_i,l = 1
→ all discovered minimum-budget routes use visual access at layer l

a_i,l = 0
→ none use it

0 < a_i,l < 1
→ alternative minimum-budget routes disagree about that layer
```

Every sample must contribute equal total weight regardless of how many minimum-budget routes it has.

Do not let samples with many MCTS routes dominate aggregate statistics.

---

# 7. Near-minimum sensitivity sets

A single minimum-budget route can be a lucky or brittle MCTS discovery.

Therefore define two prospective sensitivity sets:

```text
M_i(+2)
=
valid routes with
ON_count <= b_i + 2

M_i(+4)
=
valid routes with
ON_count <= b_i + 4
```

subject to the maximum of 28.

Run all primary placement analyses for:

```text
Δ = 0
Δ = +2
Δ = +4
```

The scientific conclusion should be based only on patterns that are reasonably stable across these definitions.

If a pattern exists only for exact-min routes and vanishes at +2/+4, classify it as fragile.

---

# 8. Per-route placement metrics

For every route in each analysis set calculate:

## 8.1 Earliest visual-access layer

```text
first_on
=
minimum layer index with ON
```

## 8.2 Latest visual-access layer

```text
last_on
=
maximum layer index with ON
```

## 8.3 Access centroid

```text
centroid
=
sum_l (l * m_l)
/
sum_l m_l
```

This gives the center of direct visual access across depth.

## 8.4 Normalized centroid

```text
centroid / 27
```

for easier comparison.

## 8.5 Access span

```text
span = last_on - first_on
```

## 8.6 Early / middle / late ON counts

Freeze decoder thirds as:

```text
early  = layers 0–8
middle = layers 9–18
late   = layers 19–27
```

For every route report:

```text
early_ON
middle_ON
late_ON

early_fraction
middle_fraction
late_fraction
```

where fractions divide by total route ON count.

## 8.7 Late-access indicator

```text
late_access = 1
if any layer 19–27 is ON
```

## 8.8 Very-late-access indicator

```text
very_late_access = 1
if any layer 24–27 is ON
```

---

# 9. Visual re-access / fragmented-access metrics

A route may access vision early, stop, and then access it again later.

This may be more informative than simple ON count.

For every route compute contiguous ON runs.

Example:

```text
1111000001110000000011000000
```

has three visual-access segments.

Calculate:

```text
number_of_ON_segments
```

and:

```text
number_of_OFF_to_ON_reentries
=
number_of_ON_segments - 1
```

Also compute:

```text
maximum OFF gap between two ON segments
```

and indicators:

```text
has_reentry
has_late_reentry
```

Define `has_late_reentry` prospectively as:

```text
there exists an OFF→ON transition
whose new ON segment begins at layer >= 19
```

Interpretation:

```text
one early ON segment only
→ front-loaded visual access

multiple separated ON segments
→ direct visual re-access across depth

late reentry
→ visual information is directly revisited late
```

Use “re-access” descriptively.

Do not claim semantic re-grounding or causal revisitation beyond the executor behavior.

---

# 10. Sample-balanced summaries

Because each sample can contain multiple valid minimum/near-min routes:

1. compute each metric per route;
2. average within sample over `M_i(Δ)`;
3. then average across samples.

The statistical unit is the **sample / problem family**, not the route occurrence.

Report:

```text
sample-balanced mean
median
Q25/Q75
bootstrap CI
```

for all primary placement quantities.

Never pool all route occurrences naïvely.

---

# 11. Primary per-layer access profile

For each difficulty stratum calculate:

```text
A_d,l
=
mean over samples i in stratum d
of a_i,l
```

for layers:

```text
0 ... 27
```

This produces a sample-balanced visual-access profile across depth.

Create one figure with eight-stratum profiles if readable.

Also create:

```text
coarse degree 0 / 1 / 2 / 3 profiles
```

but interpret degree cautiously.

Key question:

> Given roughly similar minimum ON counts, do different difficulty groups place those ON layers at different depths?

---

# 12. Amount-controlled placement analysis

We already know minimum positive ON counts are similar but not mathematically identical across strata.

Do not allow a small difference in ON count to masquerade as a placement effect.

Use at least two controls.

## Control A — normalized placement quantities

Primary quantities such as:

```text
centroid
late_fraction
early_fraction
reentry rate
```

already separate placement from raw ON count better than absolute late-ON counts.

## Control B — explicit minimum-budget adjustment

Fit descriptive models on V+ such as:

```text
normalized_centroid
~
difficulty
+
min_positive_ON
```

```text
late_fraction
~
difficulty
+
min_positive_ON
```

```text
has_late_reentry
~
difficulty
+
min_positive_ON
```

Use family-cluster/bootstrap uncertainty.

This is descriptive adjustment, not causal inference.

Also report results within predeclared minimum-ON bins:

```text
1–8
9–12
13–16
17–20
21–27
28
```

Report cell counts.

Do not interpret sparse cells.

---

# 13. Test the simple “harder means later vision” hypothesis

One intuitive possibility is:

```text
low difficulty
→ direct visual access is mostly early

higher difficulty
→ direct visual access persists or reappears later
```

Test this explicitly but do not privilege it over alternatives.

Relevant metrics:

```text
latest ON
centroid
late fraction
late-access rate
late-reentry rate
number of ON segments
```

A credible “later vision” result should show concordant movement in multiple related quantities rather than one isolated significant statistic.

---

# 14. Test the opposite possibility

Also explicitly allow:

```text
harder reasoning
→ visual information is extracted earlier
→ later computation becomes increasingly language/reasoning dominated
```

This would appear as:

```text
earlier centroid
smaller latest ON
lower late fraction
fewer late reentries
```

Do not label either direction as failure.

---

# 15. Same-family paired analysis

Reuse the previously verified `question_id` / seed-family grouping.

For every supported difficulty transition where both endpoints are V+:

```text
base → x
base → y
base → z

x → xy
x → xz

y → xy
y → yz

z → xz
z → yz

xy → xyz
xz → xyz
yz → xyz
```

calculate paired differences in:

```text
normalized centroid
first ON
last ON
late fraction
number of ON segments
late-reentry indicator
```

Use sample-level metrics averaged across the corresponding `M_i(Δ)` route set.

Report:

```text
N paired families
mean delta
median delta
increase/equal/decrease fractions
family-bootstrap CI
```

Repeat for:

```text
Δ = 0
+2
+4
```

where practical.

This paired analysis is more important than a global cross-sectional correlation.

---

# 16. Same-image subset

Where exact same-image relationships are available, repeat the most important paired analyses on the same-image subset.

Prioritize:

```text
centroid
last ON
late fraction
late reentry
```

Do not force significance if sample sizes are small.

The purpose is to check whether placement differences survive stronger visual-content control.

---

# 17. Layerwise paired difference profiles

For each supported family transition compute:

```text
Δa_l
=
a_harder,l - a_easier,l
```

for every layer.

Average these sample-balanced paired differences across families.

Plot:

```text
layer index
vs
change in probability of visual access
```

This can reveal patterns such as:

```text
difficulty increase
→ less early visual access
→ more late visual access
```

even when total ON count is unchanged.

Use confidence bands or bootstrap intervals.

Do not infer a semantic mechanism from isolated layer spikes.

Look for contiguous / reproducible depth bands.

---

# 18. Route-shape similarity across difficulty

For paired V+ family endpoints, quantify how much the visual-access schedule changes.

Using the sample-level access profiles `a_i`:

```text
profile L1 distance
profile L2 distance
profile cosine similarity
```

Optionally calculate Jaccard/Hamming distance only for exact binary routes when there is a unique minimum-budget route.

If multiple minimum routes exist, do not arbitrarily choose one solely to obtain a binary distance.

Question:

> Even if two difficulty variants require the same number of ON layers, do they use the same locations?

---

# 19. Difficulty axis analysis

After reporting all eight strata, compare the presence/absence of each verified difficulty axis.

For each axis:

```text
axis absent
vs
axis present
```

report adjusted placement quantities.

More importantly, use family transitions that **add that axis**.

Examples:

```text
base → x
y → xy
z → xz
```

for x, etc., subject to the verified dataset construction.

Do not call an axis effect stable unless:

1. multiple transitions adding the same axis agree in direction;
2. the paired aggregate supports it;
3. the effect is not driven by one tiny cell;
4. it is reasonably stable under Δ = 0/+2/+4.

---

# 20. Reasoning-type analysis

If authoritative non-difficulty reasoning-type metadata exists, perform a secondary analysis:

```text
reasoning type
vs
visual-access placement
```

Potential metrics remain:

```text
centroid
late fraction
reentry
latest ON
```

But do not manufacture reasoning-type categories.

If metadata does not support this analysis, explicitly state:

```text
reasoning-type analysis unavailable from authoritative metadata
```

and keep the study difficulty-axis-based.

---

# 21. Secondary A+ correction analysis

Separately analyze:

```text
A+ =
FULL wrong
AND
ALL-OFF wrong
AND
at least one correcting route with ON >= 1
```

The previous analysis found that A+ minimum positive ON count is nearly constant across difficulty.

Now ask:

> Does the **location** of visual access in correcting routes change with difficulty even though their ON count does not?

For each A+ sample define:

```text
minimum correcting ON budget
```

and the corresponding minimum-budget correcting route set.

Repeat the core placement metrics:

```text
centroid
last ON
early/mid/late fractions
ON segments
late reentry
per-layer access profile
```

Keep A+ separate from V+ because:

```text
V+:
FULL is correct
→ correctness-preserving visual schedule

A+:
FULL is wrong
→ alternative visual schedule corrects failure
```

Also remember that FULL-wrong samples used a different search-budget regime.

Do not pool V+ and A+.

---

# 22. Do not rely on all-valid-route marginals as primary evidence

The raw MCTS cache contains many diverse valid routes.

Therefore:

```text
mean ON probability over every valid route
```

can reflect MCTS exploration density rather than necessary schedule structure.

Primary analysis must use:

```text
minimum-budget valid routes
```

with:

```text
Δ = 0/+2/+4
```

sensitivities.

All-valid-route profiles may be shown only as secondary context.

---

# 23. Important claim boundary

A route-location association does not prove:

```text
this exact layer is causally necessary
```

because the cache is finite and may not contain all valid alternatives.

Use language such as:

> discovered minimum-budget routes preferentially place direct visual access in...

not:

> the model requires layer 17.

Similarly:

```text
late ON
```

means direct visual access occurs late.

It does not by itself prove:

```text
semantic re-grounding
visual verification
backtracking
```

Those would require separate mechanistic experiments.

---

# 24. Required output tables

Create:

```text
outputs/wemath2pro_visual_access_placement_v1/
```

including at least:

```text
vplus_minroute_layer_profiles.csv
vplus_nearmin_layer_profiles_delta2.csv
vplus_nearmin_layer_profiles_delta4.csv

vplus_placement_metrics_by_sample.csv
vplus_placement_by_stratum.csv
vplus_placement_by_degree.csv

vplus_amount_adjusted_models.csv
vplus_budget_bin_analysis.csv

vplus_family_paired_transitions.csv
vplus_family_layerwise_deltas.csv
vplus_same_image_analysis.csv

axis_placement_summary.csv

aplus_placement_by_stratum.csv
aplus_layer_profiles.csv

analysis_summary.json
analysis_manifest.json
figures/
```

---

# 25. Required figures

At minimum create:

```text
01_vplus_layer_access_profile_by_degree
02_vplus_layer_access_profile_by_stratum
03_vplus_centroid_by_stratum
04_vplus_latest_on_by_stratum
05_vplus_early_mid_late_fraction
06_vplus_on_segment_count
07_vplus_late_reentry_rate
08_vplus_paired_centroid_delta
09_vplus_paired_late_fraction_delta
10_vplus_layerwise_paired_delta_profiles
11_delta0_delta2_delta4_sensitivity
12_aplus_layer_access_profile_by_stratum
```

Prefer interpretable figures over excessive plotting.

---

# 26. Required final report

Create:

```text
reports/wemath2pro_visual_access_placement_v1.md
```

The report must answer directly:

1. Among V+ samples, does difficulty change where direct visual access appears across decoder depth?
2. Are easier problems more front-loaded and harder problems more late-access/re-access heavy, or vice versa?
3. Does any apparent placement relationship remain after controlling for minimum ON count?
4. Does the relationship survive exact-min, min+2, and min+4 route-set definitions?
5. Does the relationship survive same-family paired analysis?
6. Does it survive same-image control?
7. Do specific difficulty axes produce reproducible placement shifts?
8. Is there authoritative reasoning-type metadata, and if so does reasoning type explain placement better than aggregate difficulty?
9. Among A+ corrections, does difficulty affect placement even though minimum positive ON count is stable?
10. Is there enough evidence to motivate a router that predicts **visual-access schedules**, rather than merely the number of active visual layers?

---

# 27. Final outcome categories

End with exactly one primary classification.

## Outcome A — later/deeper visual access with difficulty

Difficulty consistently shifts access later or increases late re-access while total ON count remains similar.

Evidence should include:

```text
centroid ↑
latest ON ↑
late fraction ↑
and/or late reentry ↑
```

with paired and sensitivity support.

Interpretation:

> Harder multimodal reasoning does not necessarily require more visual-access layers, but changes when across depth visual evidence is directly consulted.

---

## Outcome B — earlier/front-loaded visual access with difficulty

Difficulty consistently shifts access earlier while total ON count remains similar.

Interpretation:

> Harder reasoning increasingly relies on early visual extraction followed by predominantly nonvisual downstream computation.

---

## Outcome C — difficulty-axis-specific placement

No global trend exists, but one or more verified difficulty axes produce reproducible placement changes across multiple paired transitions.

Interpretation:

> Visual-access schedule depends on the type of difficulty rather than aggregate difficulty.

---

## Outcome D — route placement varies, but not with difficulty

Minimum-budget schedules are heterogeneous across samples, but no stable difficulty/axis relationship survives paired and amount-controlled analysis.

Interpretation:

> Input-specific visual-access schedules exist, but WeMath difficulty is not the variable explaining them.

This is still useful because it redirects predictor-conditioning analysis toward other input properties.

---

## Outcome E — no stable placement structure

Layer placement is too unstable across minimum/near-min routes, or apparent effects disappear under Δ=+2/+4 sensitivity.

Interpretation:

> The current finite MCTS cache does not support a robust claim about difficulty-dependent visual-access placement.

---

# Core execution principle

> **The previous WeMath analysis showed that difficulty does not robustly determine how many decoder layers require direct visual access once ALL-OFF-solvable samples are removed. Use the same frozen cache to ask the next orthogonal question: holding visual-access amount approximately fixed, does difficulty or difficulty type change *where across decoder depth* those visual accesses occur—early, late, continuously, or through separated re-access episodes? Analyze minimum-budget valid-route sets sample-balanced, verify the result under min+2/min+4 alternatives, and prioritize paired family/same-image evidence over unpaired route-frequency correlations.**