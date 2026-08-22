We have completed the first WeMath2.0-Pro difficulty analysis using the frozen binary MCTS cache.

That analysis found that, among samples where FULL is correct, the discovered minimum VISUAL_ON count decreases rather than increases with coarse difficulty. However, the interpretation must now be revisited because the exact binary-executor semantics are:

```text
VISUAL_ON at a layer:
- visual hidden states execute that decoder layer;
- text/control tokens execute with visual tokens present;
- text can attend to visual K/V.

VISUAL_OFF at a layer:
- visual hidden states bypass that decoder layer unchanged;
- text/control tokens execute the layer without visual tokens;
- text cannot attend to visual K/V at that layer.
```

Critically:

```text
ALL-OFF:
- visual states never execute any decoder layer;
- text never directly reads visual K/V at any decoder layer;
- no earlier ON layer exists from which visual content could have been transferred into text hidden states.
```

There may still be limited non-content structural side channels such as image placeholders, positional indices, or visual-token-count/layout information. Therefore do not call ALL-OFF strictly “no image whatsoever.”

But for the visual features controlled by this executor:

> `min_valid_on = 0` means that a correct route exists with **no direct decoder access to encoded visual content**.

This changes how the previous `min_valid_on` result should be interpreted.

The purpose of this task is to determine:

> **Is the previously observed low minimum-ON count caused primarily by an increased prevalence of ALL-OFF-solvable samples, or does a genuine difference in required visual-access budget remain even among samples that behaviorally require visual access?**

Analyze **all FULL-correct WeMath2.0-Pro samples and all difficulty strata**, not only the previously notable `x`-containing strata.

Do not run new MCTS, new Qwen execution, predictor training, or route generation.

---

# 1. Frozen evidence

Use exactly the existing completed WeMath2.0-Pro raw MCTS cache and source metadata used by:

```text
reports/wemath2pro_visual_compute_difficulty_v1.md
```

Use all raw evaluated routes.

Do not use only the max-50 predictor-training view.

Verify all integrity counts and hashes before analysis.

The primary cohort should reproduce:

```text
FULL-correct samples = 841
```

Do not hard-code this count if the authoritative cache disagrees; verify it.

Preserve the eight official difficulty strata:

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

and the previously defined coarse degree only as a secondary aggregation:

```text
degree 0 = base
degree 1 = x / y / z
degree 2 = xy / xz / yz
degree 3 = xyz
```

Do not invent an ordering among x, y, and z.

---

# 2. First establish the exact ALL-OFF outcome

Identify the exact 28-bit:

```text
0000000000000000000000000000
```

route for every sample from the authoritative evaluated-route cache.

For each sample determine:

```text
FULL correctness
ALL-OFF correctness
```

Use the same frozen benchmark scorer and correctness threshold.

Do not infer ALL-OFF correctness merely from `min_valid_on == 0` if the exact ALL-OFF route outcome is directly available; verify it from the route record.

Confirm that:

```text
FULL correct + ALL-OFF correct
```

corresponds exactly to FULL-correct samples with:

```text
min_valid_on = 0
```

Any mismatch is an integrity failure that must be investigated before continuing.

---

# 3. Primary FULL-correct decomposition

For every FULL-correct sample, assign exactly one of two groups.

## V0 — no-direct-vision solution exists

```text
FULL correct
AND
ALL-OFF correct
```

Interpretation:

> Correctness does not require direct access to encoded visual K/V under the current executor.

Do not claim that the model literally received zero image-related side information.

## V+ — direct visual access is behaviorally necessary

```text
FULL correct
AND
ALL-OFF wrong
```

Interpretation:

> Removing direct visual-token access from every decoder layer changes this sample from correct to incorrect, so at least one VISUAL_ON layer is behaviorally necessary under the tested executor.

For V+:

```text
min_valid_on >= 1
```

must hold.

---

# 4. Report this decomposition for every difficulty

For each of:

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

report:

```text
eligible N
FULL-correct N

V0 N
V0 fraction among FULL-correct

V+ N
V+ fraction among FULL-correct
```

Also report the same for coarse degrees 0–3.

Primary quantity:

```text
P(ALL-OFF correct | FULL correct, difficulty)
```

This determines whether the original low mean min-ON values were driven by an unusually high prevalence of no-direct-vision solutions.

Produce a table such as:

```text
Stratum | FULL-correct N | V0 N | V0 % | V+ N | V+ %
```

and a bar plot of:

```text
P(V0 | FULL correct)
```

across all eight strata.

Do not focus the plot only on x-containing strata.

---

# 5. Explicitly decompose the original mean min-ON

Because every V0 sample has:

```text
min_valid_on = 0
```

the original FULL-correct mean satisfies:

```text
E[min_ON | FULL-correct, d]
=
P(V+ | FULL-correct, d)
*
E[min_ON | V+, d]
```

for difficulty group `d`.

Verify this identity numerically for every stratum and degree.

For every difficulty report:

```text
original mean min ON over all FULL-correct

V0 fraction

mean min ON over V+ only

reconstructed original mean
=
V+ fraction * V+-conditional mean
```

The reconstructed and original means should agree up to numerical precision.

This is a central analysis.

It allows us to distinguish:

### Mechanism A — zero-mass / visual-independence composition

```text
difficulty changes
→ V0 fraction changes strongly

BUT
E[min_ON | V+] is approximately stable
```

Then the original min-ON trend is mostly caused by different rates of ALL-OFF-solvable samples.

### Mechanism B — genuine conditional visual-budget difference

```text
difficulty changes
→ E[min_ON | V+] also changes materially
```

Then a visual-access-budget difference remains even after removing ALL-OFF-solvable samples.

Both may coexist.

Quantify their relative contribution rather than choosing one prematurely.

---

# 6. Recompute the main min-ON analysis on V+ only

This is the new primary visual-budget analysis.

For every sample satisfying:

```text
FULL correct
AND
ALL-OFF wrong
```

calculate:

```text
min_positive_valid_on
=
minimum ON count among all discovered valid routes
```

Since ALL-OFF is wrong:

```text
1 <= min_positive_valid_on <= 28
```

For every individual difficulty stratum report:

```text
N
mean
median
SD
P10
Q25
Q75
P90
95% clustered/bootstrap CI
```

Repeat for coarse degree 0–3.

Create direct side-by-side comparisons:

```text
ALL FULL-correct min ON
vs
V+-only min ON
```

for each difficulty.

This is necessary to show how much the original pattern changes after separating V0.

---

# 7. Recompute visual-budget feasibility curves on V+ only

The previous feasibility curve was:

```text
P(min_valid_on <= C | FULL correct, difficulty)
```

which mixes:

```text
V0 samples
+
vision-dependent V+ samples.
```

Now compute:

```text
P(min_positive_valid_on <= C | FULL correct, ALL-OFF wrong, difficulty)
```

for:

```text
C = 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
```

Show:

1. eight-stratum curves;
2. coarse degree curves;
3. a compact table at:

```text
C = 4, 8, 12, 16, 20, 24, 28
```

Interpretation:

> Among problems that actually need some direct visual access, how much decoder-depth visual access is sufficient?

This curve is much closer to the scientific quantity we originally intended to measure.

---

# 8. Compare the old and new difficulty relationships

Explicitly compare:

## Original

```text
difficulty
vs
min_valid_on
among all FULL-correct samples
```

## Conditional

```text
difficulty
vs
min_positive_valid_on
among V+ samples only
```

Report for both:

```text
Spearman rho
family-cluster/bootstrap 95% CI
mean trend
median trend
```

Classify what happens after V0 removal.

Possible patterns:

### Pattern 1 — relationship disappears

Example:

```text
old rho = strongly negative
new rho ≈ 0
```

Interpretation:

> The previous negative depth trend was mostly a change in the prevalence of ALL-OFF-solvable cases, not a visual-access-budget difference among vision-dependent problems.

### Pattern 2 — relationship reverses

Example:

```text
old rho < 0
new rho > 0
```

Interpretation:

> ALL-OFF-solvable cases masked an opposite relationship among problems that genuinely require vision.

This would be especially important.

### Pattern 3 — negative relationship remains

Interpretation:

> Even among samples that behaviorally require direct visual access, harder or particular difficulty strata remain solvable using fewer visual-access layers.

This would be a genuine conditional visual-budget phenomenon requiring further interpretation.

### Pattern 4 — axis-specific only

Interpretation:

> Aggregate difficulty is inappropriate; different difficulty axes affect visual dependence and/or visual-access depth differently.

---

# 9. Analyze all eight strata before any x-specific summary

Do not start from:

```text
x vs non-x
```

The primary report must first show:

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

independently.

Only after those results are complete may you include secondary summaries such as:

```text
x-containing
vs
non-x-containing
```

if useful.

Likewise inspect y-containing and z-containing summaries if symmetry is scientifically informative.

Do not single out x simply because the previous analysis found an x-associated pattern.

---

# 10. Revisit cheaper-route availability within V+

For V+ samples calculate:

```text
P(min_positive_valid_on < 28 | V+, difficulty)
```

and:

```text
removable_direct_visual_layers
=
28 - min_positive_valid_on
```

Report per stratum:

```text
V+ N
cheaper positive route N
cheaper positive route %
mean removable layers
median removable layers
```

This asks:

> Given that the problem truly needs some direct visual access, how much of the 28-layer visual-access schedule can still be removed?

This should replace the previous interpretation of cheaper-route availability that mixed V0 and V+.

---

# 11. Same-family paired reanalysis

Reuse the previously verified seed-family key.

First reproduce the old paired result as an audit.

Then perform a new paired analysis requiring both endpoints to satisfy:

```text
FULL correct
AND
ALL-OFF wrong
```

That is, compare difficulty transitions only when both variants are V+.

For every supported transition:

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

where valid under the dataset construction, report:

```text
number of paired families
mean delta in min_positive_valid_on
median delta
increase %
equal %
decrease %
family-bootstrap CI
```

If sample size becomes too small, report it descriptively rather than forcing significance.

Also reproduce same-image subsets where available.

This determines whether any conditional visual-budget pattern survives paired control once ALL-OFF-solvable examples are removed.

---

# 12. Optional but useful: full FULL/OFF behavioral table

For context, produce the complete 2×2 outcome table by difficulty:

```text
FULL correct / ALL-OFF correct
FULL correct / ALL-OFF wrong
FULL wrong   / ALL-OFF correct
FULL wrong   / ALL-OFF wrong
```

This is not the primary min-ON analysis, but it reveals whether difficulty changes visual dependence at the population level.

Report counts and percentages with the entire eligible stratum as denominator.

This also prevents the FULL-correct analysis from being interpreted without its survivorship context.

---

# 13. Secondary FULL-wrong decomposition

Do not mix this with the primary FULL-correct analysis, but perform a compact secondary audit.

Split previously corrected Group A into:

## A0

```text
FULL wrong
ALL-OFF correct
```

Interpretation:

> Removing all direct visual access itself produces a correct answer.

## A+

```text
FULL wrong
ALL-OFF wrong
some route with ON >= 1 is correct
```

Interpretation:

> A genuinely nonzero visual-access program corrects the FULL failure.

For every difficulty report:

```text
FULL-wrong N
A0 N / %
A+ N / %
no correcting route N / %
```

For A+ only, report:

```text
minimum correcting positive ON
```

by difficulty.

This prevents ALL-OFF corrections from being mistaken for successful sparse visual routing.

---

# 14. Do not yet analyze REPEAT or >FULL compute

The current task is entirely about understanding the existing binary cache correctly.

Do not:

```text
add REPEAT
rerun MCTS
increase MCTS budgets
train a router
change route labels
apply predictor supervision filtering
change correctness criteria
execute new routes
```

No follow-up experiment should be launched automatically.

---

# 15. Terminology

Avoid using `min_valid_on` as an unqualified synonym for:

```text
required visual computation
```

Use more precise language:

```text
minimum discovered direct-visual-access layer count
```

or:

```text
minimum discovered VISUAL_ON budget
```

For the V+ subset, it is acceptable to describe it as:

> minimum discovered visual-access budget among samples for which ALL-OFF is behaviorally insufficient.

Do not claim causal necessity beyond the tested route space/search.

---

# 16. Required figures

At minimum create:

```text
01_all_off_rate_by_stratum
02_full_correct_v0_vplus_composition
03_original_vs_vplus_min_on_by_stratum
04_vplus_min_on_by_degree
05_vplus_budget_feasibility_by_degree
06_vplus_budget_feasibility_by_stratum
07_original_vs_conditional_difficulty_trend
08_full_vs_alloff_2x2_by_stratum
09_paired_vplus_transition_deltas
```

Use the same plotting conventions as the previous report where possible.

---

# 17. Required output files

Create a new read-only analysis directory:

```text
outputs/wemath2pro_visual_dependence_reanalysis_v1/
```

including at least:

```text
full_correct_v0_vplus_by_difficulty.csv
mean_min_on_decomposition.csv
vplus_min_on_by_difficulty.csv
vplus_budget_feasibility.csv
full_alloff_contingency.csv
paired_vplus_transitions.csv
group_a0_aplus_by_difficulty.csv
analysis_summary.json
figures/
```

and final report:

```text
reports/wemath2pro_visual_dependence_reanalysis_v1.md
```

---

# 18. Required final answers

The final report must directly answer:

1. Among FULL-correct samples, how often is ALL-OFF also correct for each difficulty stratum?
2. How much of the previously observed low mean min-ON is mathematically explained by an increased V0/ALL-OFF-solvable fraction?
3. After excluding V0 and restricting to `FULL correct + ALL-OFF wrong`, how does minimum positive VISUAL_ON budget vary across all eight difficulty strata?
4. Does the previously negative coarse difficulty–min-ON relationship disappear, persist, or reverse?
5. Do budget-feasibility curves still differ by difficulty among vision-dependent V+ samples?
6. Does any axis-specific relationship remain after this conditioning?
7. Does the relationship survive same-family/same-image paired analysis among V+ samples?
8. Among FULL-wrong examples, how many corrections are actually ALL-OFF corrections versus corrections requiring nonzero visual access?
9. What interpretation of the previous x-associated result remains defensible?
10. Does the evidence now support a genuine difficulty-dependent visual-access-budget phenomenon, or was the original trend mainly a visual-dependence/survivorship composition effect?

---

# 19. Final decision categories

End with exactly one primary classification.

## Outcome A — mostly ALL-OFF composition

The original difficulty trend largely disappears after V0 removal.

Conclusion:

> The apparent min-ON trend was primarily caused by changing prevalence of samples solvable without direct visual access.

## Outcome B — genuine conditional visual-budget trend

A substantial difficulty relationship remains among V+ samples.

Conclusion:

> Beyond visual-dependence differences, vision-dependent problems themselves require systematically different numbers of direct visual-access layers.

Specify direction and axes.

## Outcome C — reversal after conditioning

The original negative relationship becomes positive among V+.

Conclusion:

> ALL-OFF-solvable cases masked a qualitatively different visual-budget relationship among genuinely vision-dependent samples.

## Outcome D — axis-specific conditional structure

No meaningful global degree trend exists, but one or more difficulty axes show reproducible differences after V0 removal and paired controls.

Conclusion:

> Visual-computation demand is difficulty-type-specific rather than determined by aggregate difficulty.

## Outcome E — no stable visual-budget relationship

After proper decomposition, no robust difficulty or axis relationship remains.

Conclusion:

> Difficulty affects correctness/visual dependence, but the current MCTS cache does not support a systematic difficulty-dependent visual-access budget.

---

# Core principle

> **Do not treat `min_valid_on = 0` as merely an extremely sparse visual program. It is a qualitatively different behavioral regime in which the decoder never directly accesses encoded visual K/V. First separate whether direct vision is needed at all; only then, among samples where ALL-OFF fails but FULL succeeds, analyze how much direct visual access is needed across all WeMath2.0-Pro difficulty strata.**