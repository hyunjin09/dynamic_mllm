We already completed MCTS extraction for WeMath2.0-Pro using the 28-bit binary visual-routing space:

- `1 = VISUAL_ON`: visual rows execute the decoder layer normally.
- `0 = TEXT_ONLY`: visual rows bypass that decoder layer while text/control rows still execute.
- `111...111 = FULL`, i.e. 28 VISUAL_ON layers.
- There is **no REPEAT action in the current cache**.

Do **not** run new MCTS, train a router, modify the route space, or introduce repeat/recurrent execution.

The purpose of this task is to analyze the existing frozen MCTS cache and answer:

> **As WeMath2.0-Pro problems become more difficult, does the minimum discovered visual-layer budget required for correctness increase?**

Equivalently:

> **Do easier multimodal math problems contain more removable visual computation, while harder problems require a larger fraction of the standard visual decoder depth?**

This is an exploratory diagnostic. Do not force the hypothesis to be true.

---

# 1. Use the existing frozen WeMath2.0-Pro cache

Use the completed hard-cap-400 WeMath2.0-Pro MCTS extraction and its frozen execution contract.

Expected integrity counts from the existing analysis are approximately:

```text
eligible samples                 4,544
current FULL correct               841
current FULL wrong               3,703

at least one valid route         2,266
zero-positive samples            2,278

FULL-wrong + correcting route    1,425   # Group A
FULL-correct + cheaper route       784   # Group B
FULL-correct + only FULL found      57   # Group C
FULL-wrong + no correction       2,278   # Group D
```

Verify these counts from the authoritative cache before analysis.

Use the **raw evaluated MCTS routes**, not only a predictor-training max-50 selection, for the primary difficulty analysis.

Validity must use the existing frozen WeMath scoring rule exactly.

Do not change correctness thresholds.

---

# 2. First audit the difficulty metadata

Identify the exact WeMath2.0-Pro difficulty field and enumerate all observed strata.

Expected labels are of the form:

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

Verify this from the actual source metadata rather than assuming it.

Report:

```text
difficulty stratum
total eligible samples
FULL-correct samples
FULL-wrong samples
positive-route samples
zero-positive samples
```

Do not impose an arbitrary total ordering such as:

```text
x < y < z
```

unless the dataset metadata explicitly defines one.

For a coarse ordinal analysis, define difficulty **degree** only if the observed labels support it:

```text
degree 0: base
degree 1: x, y, z
degree 2: xy, xz, yz
degree 3: xyz
```

Preserve the eight individual strata separately as well.

---

# 3. PRIMARY ANALYSIS: FULL-correct samples only

This is the cleanest test.

The current FULL route is already correct for these samples, so every sample has at least one known valid route: FULL.

Also, all FULL-correct samples were searched under the same MCTS simulation regime, so this avoids mixing the different FULL-correct/FULL-wrong search budgets.

For every FULL-correct sample `i`, calculate:

```text
min_valid_on_i
=
minimum VISUAL_ON count
among all discovered valid routes for sample i
```

Because FULL is valid:

```text
0 <= min_valid_on_i <= 28
```

Also calculate:

```text
removable_visual_layers_i
=
28 - min_valid_on_i
```

Interpretation:

```text
min_valid_on = 6
→ correctness was preserved with only 6 visual-active decoder layers
→ 22 visual-layer executions were removable

min_valid_on = 28
→ no cheaper valid visual route was discovered
```

This is the primary per-sample visual-depth requirement diagnostic.

Do **not** use the average ON count across all valid routes as the primary measure.

---

# 4. Difficulty vs minimum valid visual depth

For FULL-correct samples, report `min_valid_on` separately for:

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

and for coarse difficulty degree:

```text
0
1
2
3
```

For each group report:

```text
N
mean min_valid_on
median min_valid_on
std
Q25
Q75
P10
P90

mean removable layers
median removable layers
```

Create a distribution plot/boxplot of:

```text
difficulty degree
        vs
min_valid_on
```

and another of:

```text
difficulty degree
        vs
28 - min_valid_on
```

The hypothesized direction is:

```text
difficulty increases
→ min_valid_on increases
→ removable visual layers decrease
```

but do not assume this must occur.

Use confidence intervals, preferably sample- or family-cluster bootstrap where the grouping metadata permits it.

---

# 5. PRIMARY FIGURE: visual-budget feasibility curves

For each FULL-correct sample, `min_valid_on` gives the smallest discovered budget that preserved correctness.

For visual-layer budgets:

```text
C = 0, 2, 4, 6, ..., 28
```

compute for each difficulty group:

```text
P(min_valid_on <= C | FULL-correct, difficulty)
```

This means:

> fraction of originally correct problems for which MCTS discovered a correctness-preserving route using at most C VISUAL_ON layers.

Plot one curve per coarse difficulty degree:

```text
degree 0
degree 1
degree 2
degree 3
```

Optionally show the eight individual difficulty strata in a secondary figure.

The most important qualitative pattern to test is:

```text
low difficulty:
curve rises earlier
→ many problems remain correct under aggressive visual-compute reduction

high difficulty:
curve shifts right
→ larger visual-depth budget is required
```

Do not hide non-monotonic results.

Also report a compact table at:

```text
C = 8
C = 12
C = 16
C = 18
C = 20
C = 22
C = 24
C = 28
```

---

# 6. Cheaper-route existence among FULL-correct samples

For every difficulty stratum compute:

```text
P(min_valid_on < 28 | FULL-correct, difficulty)
```

This is:

> probability that a correctness-preserving route cheaper than FULL was discovered.

Report:

```text
difficulty
FULL-correct N
cheaper-route N
cheaper-route %
FULL-only N
FULL-only %
```

This is another clean redundancy measure.

Question:

> Does the probability of finding removable visual computation decrease as difficulty increases?

---

# 7. SECONDARY ANALYSIS: FULL-wrong samples only

Do not combine FULL-wrong samples with the FULL-correct primary analysis.

For FULL-wrong samples calculate, separately by difficulty:

```text
correction_found_rate
=
# FULL-wrong samples with >=1 correcting valid route
/
# all FULL-wrong samples
```

Report:

```text
difficulty
FULL-wrong N
correction found N
correction found %
no correction N
no correction %
```

This directly addresses:

> Are harder problems increasingly likely to be among the samples for which the current SKIP/KEEP search cannot find any correcting route?

---

# 8. Analyze the zero-positive / no-correction population

There are approximately 2,278 FULL-wrong samples for which no valid correcting route was discovered.

Analyze them in two complementary ways.

## A. Composition

Among all zero-positive samples:

```text
what fraction belongs to each difficulty stratum?
```

Report:

```text
difficulty
zero-positive count
% of all zero-positive samples
```

## B. Failure rate within difficulty

More importantly, for every difficulty:

```text
P(zero-positive | FULL-wrong, difficulty)
```

This controls for different population sizes.

Do not claim that zero-positive means:

```text
"this sample requires >28 visual layers"
```

That conclusion is unsupported.

Zero-positive only means:

> no correcting route was discovered within the frozen finite MCTS search and current SKIP/KEEP route space.

Possible explanations include:

```text
search incompleteness
lack of a useful SKIP/KEEP route
base model reasoning failure
need for computation outside the current action space
```

Keep these interpretations separate.

---

# 9. Corrected FULL-wrong samples: minimum correcting visual budget

For Group A only:

```text
FULL wrong
+
at least one correcting route found
```

calculate:

```text
min_correcting_on
=
minimum ON count among correcting routes
```

Analyze this by difficulty.

Report the same:

```text
N
mean
median
Q25/Q75
distribution
budget-feasibility curve
```

But label this explicitly as:

> **minimum discovered visual budget among alternative correcting programs**

not:

> required visual computation for ordinary correctness.

The Group A and FULL-correct interpretations are different and must not be pooled.

---

# 10. Same-family / same-image paired analysis

Inspect the source metadata and determine whether reliable identifiers exist for:

```text
same seed problem family
same underlying image
progressive variants of the same problem
```

Do not infer family identity from an arbitrary ID unless verified.

If reliable grouping exists, perform a paired analysis.

This is potentially stronger than the global difficulty comparison because it controls some image/problem-family variation.

For groups containing multiple difficulty levels, ask whether:

```text
difficulty increases within the same family/image
        ↓
min_valid_on also tends to increase
```

For example, if a family contains:

```text
base
x
xy
xyz
```

compare the paired `min_valid_on` values.

Report:

```text
number of usable families
number of paired transitions
fraction of transitions with min_ON increase
fraction equal
fraction decrease

median paired delta
mean paired delta
cluster-bootstrap CI
```

Where meaningful, examine partial-order transitions such as:

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

Only run transitions that are actually supported by the dataset's difficulty construction.

Do not invent a linear ordering among x/y/z.

---

# 11. Visual-token-count confound check

ON-layer count measures **visual decoder depth**, not exact total FLOPs.

WeMath2.0-Pro has highly variable visual-token counts, so check whether apparent difficulty effects are merely caused by visual input size.

If the existing cache/source metadata contains visual-token count, report by difficulty:

```text
median visual tokens
mean visual tokens
P90 visual tokens
```

Then test whether the difficulty–`min_valid_on` relationship remains after accounting for visual-token count.

At minimum:

```text
difficulty degree
vs
min_valid_on
within coarse visual-token-count bins
```

Preferably also fit a simple descriptive model such as:

```text
min_valid_on
~ difficulty_degree
+ log(visual_token_count)
```

for FULL-correct samples.

Do not overinterpret this as causal inference.

The purpose is only to determine whether difficulty has information beyond raw input size.

---

# 12. Secondary route-geometry metrics

For completeness, report by difficulty:

```text
mean number of valid routes
median number of valid routes

mean ON across valid routes
median ON across valid routes

mean pairwise Hamming distance
bit entropy
```

These are secondary diagnostics only.

Do not use:

```text
mean ON across all valid routes
```

as the primary estimate of required visual computation, because it depends on which parts of route space MCTS happened to explore.

The primary quantity remains:

```text
min_valid_on
```

for FULL-correct samples.

---

# 13. Statistical trend tests

For coarse degree `0/1/2/3`, report a descriptive trend test between:

```text
difficulty degree
and
min_valid_on
```

such as Spearman correlation.

Also report a trend for:

```text
difficulty degree
and
removable layers
```

Do not rely only on a p-value.

Always provide:

```text
effect size
confidence interval
group distributions
sample counts
```

If same-family grouping exists, prioritize clustered/paired uncertainty estimates over naive IID tests.

---

# 14. Main interpretation matrix

Classify the result into one of the following.

## Outcome A — strong visual-depth scaling with difficulty

Evidence looks like:

```text
difficulty ↑
→ min valid ON ↑
→ removable layers ↓
→ low-budget feasibility ↓
```

and preferably the same direction appears in same-family comparisons.

Interpretation:

> Harder multimodal math problems exhibit less removable visual-decoder computation within the current SKIP/KEEP search space.

This would strongly motivate input-adaptive visual-computation budgets.

---

## Outcome B — difficulty affects route discovery but not required visual depth

Example:

```text
difficulty ↑
→ zero-positive rate ↑

BUT

difficulty ↛ min valid ON
```

Interpretation:

> Harder problems are more difficult for the existing route search/base model, but the current evidence does not show that they require more visual decoder depth.

Do not claim visual-compute scaling.

---

## Outcome C — visual depth scales within positives, but coverage collapses

Example:

```text
hard positive samples have higher min_ON

AND

hard samples have much lower positive-route coverage
```

Interpretation:

> Conditional evidence supports higher visual-depth demand among solvable samples, but survivorship/search-selection limits the population-level claim.

State the selection caveat clearly.

---

## Outcome D — no meaningful difficulty relationship

Example:

```text
min_ON roughly constant
budget curves overlap
same-family deltas near zero
```

Interpretation:

> WeMath2.0-Pro difficulty does not appear to determine required visual decoder depth under the current route space.

Do not rescue the hypothesis with post-hoc subgroup search.

---

## Outcome E — axis-specific effect

Example:

```text
x-related difficulty
→ strong min_ON increase

y/z
→ little or opposite effect
```

Interpretation:

> Different forms of multimodal mathematical difficulty have different visual-computation demands; aggregate difficulty is too coarse.

This may be more interesting than a simple monotonic difficulty result.

---

# 15. Important claim boundary

The current action space only contains:

```text
VISUAL_ON
TEXT_ONLY
```

Therefore this analysis can test:

> whether harder problems require **more of the standard 0–28 visual decoder depth**.

It cannot test:

> whether difficult problems benefit from **more visual computation than FULL**.

Do not make any claim involving:

```text
repeat
recurrence
>28 visual executions
additional visual refinement beyond FULL
```

Those require a future expanded search space.

---

# 16. No new experimental intervention

For this task do not:

```text
rerun MCTS
increase simulation budgets
train BCE/NLL predictors
apply Pareto filtering
apply 18/20/22/24 training caps
generate new routes
add REPEAT
change validity thresholds
modify the frozen cache
execute new Qwen routes
```

This is a read-only analysis of the completed WeMath2.0-Pro MCTS results.

---

# 17. Required outputs

Create:

```text
outputs/wemath2pro_visual_compute_difficulty_v1/
```

with at least:

```text
difficulty_population.csv
full_correct_min_on_by_difficulty.csv
full_correct_budget_feasibility.csv
full_wrong_correction_by_difficulty.csv
zero_positive_by_difficulty.csv
group_a_min_correcting_on.csv
paired_family_analysis.csv          # if grouping is valid
visual_token_control.csv            # if token counts available
analysis_summary.json
figures/
```

and a final report:

```text
reports/wemath2pro_visual_compute_difficulty_v1.md
```

---

# 18. Required final report structure

The report should answer these questions directly:

1. Does minimum correctness-preserving VISUAL_ON depth increase with difficulty among FULL-correct samples?
2. Does the fraction of problems solvable under small visual-layer budgets decrease with difficulty?
3. Does cheaper-route availability decrease with difficulty?
4. Among FULL-wrong samples, does correcting-route discovery decrease with difficulty?
5. Which difficulty strata dominate the zero-positive population, and what are their within-stratum failure rates?
6. Does the difficulty relationship survive same-family/same-image paired analysis?
7. Is the relationship explained simply by visual-token count?
8. Which Outcome A–E best describes the evidence?
9. Does the current evidence justify expanding the route space later to include additional visual computation such as REPEAT?

Finish with a concise scientific conclusion, but **do not launch the next experiment automatically**.

---

# Core execution principle

> **Use the existing WeMath2.0-Pro VISUAL_ON/TEXT_ONLY MCTS cache to test whether progressively more difficult multimodal math problems require a larger minimum visual-decoder depth for correctness. Treat FULL-correct samples as the primary clean analysis, use minimum valid ON count and visual-budget feasibility as the main metrics, analyze FULL-wrong correction failure separately, and preserve the distinction between “no route found” and “more visual computation required.”**