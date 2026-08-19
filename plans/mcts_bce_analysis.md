We have completed full predictor training/evaluation with both exact valid-set NLL and POLAR-style duplicated BCE.

Before changing the loss, correct/wrong sampling ratio, MCTS search, or predictor architecture, perform a **label-only analysis** of the existing binary MCTS supervision.

This action must answer two questions:

> **Q1. What structure does the current MCTS valid-route supervision actually contain?**

and

> **Q2. If duplicated BCE learned the existing labels perfectly, what complete 28-bit masks would the labels themselves imply?**

The purpose is to separate:

```text
MCTS/search problem
vs
valid-route selection problem
vs
duplicated-BCE objective problem
vs
predictor/generalization problem
```

Do not train a new predictor in this action.

Do not regenerate MCTS labels unless the analysis later establishes that the existing labels are structurally inadequate.

---

# 1. Starting evidence

Current full external evaluation showed:

## Exact valid-set NLL

The selected predictors essentially collapsed to ALL-ON:

```text
Question-only:
22,263 / 22,307 ALL-ON

Image+Question:
22,307 / 22,307 ALL-ON
```

Thus exact set-NLL found the global ALL-ON mode instead of useful sample-specific routes.

## Duplicated BCE

Duplicated BCE escaped the total collapse more than NLL, but predicted masks remained relatively dense and low-diversity.

It produced non-ALL-ON masks, but many benchmarks still used roughly 23–28 VISUAL_ON layers and some showed substantially more FULL-correct→wrong harm than FULL-wrong→correct rescue.

Therefore:

> Do not assume the predictor is the primary failure source.

The existing label geometry may already imply dense or ambiguous BCE targets.

---

# 2. Scope

Analyze the exact frozen MCTS route data used for P11–P13/full10 training.

Use:

```text
GQA
TextVQA
ChartQA
```

and the existing image-group-disjoint train/validation split.

Where available, analyze both:

```text
A. raw MCTS discovered/evaluated routes
B. final selected valid routes used for supervision
```

The current training supervision uses at most:

```text
50 valid masks per positive input
```

If raw MCTS routes are unavailable for some records, state this explicitly and perform the available analyses on selected supervision only.

Do not silently substitute regenerated routes.

---

# 3. First verify route semantics

Before any statistics, verify:

```text
mask length = 28
1 = VISUAL_ON
0 = TEXT_ONLY
ALL-ON = 111...111
```

and confirm exactly what qualifies a route as `valid` in the frozen label files.

Document whether validity is based on:

```text
task correctness
task score threshold
answer likelihood
or another stored criterion
```

Do not infer this from memory; inspect the frozen label-generation artifacts/code/config.

Also verify:

```text
FULL correctness label
route correctness/utility fields
ON count
selected-vs-raw route identity
route weights
```

where available.

---

# 4. Population taxonomy

Do not analyze all positive inputs as one homogeneous population.

Partition every input into at least the following groups.

## Group A — FULL-wrong, correction exists

```text
ALL-ON / FULL is wrong
at least one discovered valid route is correct
```

These are the most important correction examples.

The desired router behavior is necessarily non-ALL-ON.

---

## Group B — FULL-correct, cheaper valid route exists

```text
ALL-ON / FULL is correct
AND
at least one valid route has ON count < 28
```

These are `sparsifiable-correct` examples.

The desired behavior may preserve correctness while reducing computation.

---

## Group C — FULL-correct, no cheaper valid route found

```text
ALL-ON / FULL is correct
AND
no valid selected/discovered route has ON count < 28
```

These are `must-stay-full under current search` examples.

Do not assume this proves that no sparse route exists globally; it only means the current MCTS search did not find one.

---

## Optional Group D — FULL-wrong, no correction route found

If such records exist in the raw dataset but were excluded from positive-route training, count them separately.

These are important for understanding dataset/search coverage even if they are not part of the current predictor objective.

---

# 5. Audit current correct/wrong training composition

The current training construction was approximately balanced between FULL-correct and FULL-wrong examples.

Verify the actual frozen counts.

Report:

```text
FULL-correct count
FULL-wrong count

Group A count
Group B count
Group C count
Group D count if available
```

for:

```text
train
validation
each dataset
overall
```

Also report the proportions.

This will later determine whether the current 5:5 correct/wrong balance is creating excessive ALL-ON supervision.

Do not modify the ratio in this action.

---

# 6. Route-count geometry

For each sample compute:

```text
number of raw evaluated MCTS routes
number of raw valid routes
number of selected valid routes
```

Report:

```text
mean
median
p10
p25
p75
p90
min
max
```

overall, per dataset, and separately for Groups A/B/C.

Important:

> Route count alone is not sufficient evidence of useful diversity.

The remaining analyses must determine whether 50 routes represent 50 genuinely different computation modes or only near-duplicates around ALL-ON.

---

# 7. ON-count geometry

For every selected valid route compute:

```text
ON(m) = number of VISUAL_ON layers
```

Report route-level and sample-level distributions.

For each sample calculate:

```text
min ON
max ON
mean ON
median ON
p25 ON
p75 ON
```

Report overall/per-dataset/per-group.

Especially report:

```text
fraction with a valid route <= 4 ON
<= 8 ON
<= 12 ON
<= 16 ON
<= 20 ON
<= 24 ON
27 ON
28 ON
```

Also report:

```text
fraction where minimum-ON valid route is ALL-ON
fraction where minimum-ON <= 14
fraction where minimum-ON < 28
```

This determines whether current MCTS supervision actually contains substantial compute reduction opportunities.

---

# 8. ALL-ON geometry

Report:

```text
fraction of samples where ALL-ON is valid
fraction where ALL-ON is selected into max-50 supervision
fraction where ALL-ON is valid AND a cheaper valid route exists
```

separately for Groups A/B/C and datasets.

For Group B specifically report:

```text
ON-count gap:
28 - min_valid_ON
```

If ALL-ON coexists with many cheaper equally-valid routes, quantify how frequently this occurs.

This is directly relevant to the NLL shortcut.

---

# 9. Pairwise route diversity within a sample

For each sample with at least two valid routes, compute pairwise Hamming distances:

```text
d_H(m_a, m_b)
```

Do not materialize all O(K^2) pairs if unnecessary; K <= 50 makes exact computation acceptable.

Report per sample:

```text
mean pairwise Hamming
median pairwise Hamming
minimum nonzero Hamming
maximum Hamming
```

Then aggregate overall/per-dataset/per-group.

Also report normalized Hamming:

```text
d_H / 28
```

Useful summary fractions:

```text
mean pairwise Hamming <= 1
<= 2
<= 4
<= 7
>= 10
```

This determines whether the current valid set contains genuinely different routes or many near-duplicates.

---

# 10. Distance from ALL-ON

For every route:

```text
d_FULL(m) = Hamming(m, ALL-ON)
          = 28 - ON(m)
```

Report distributions.

For every sample report:

```text
minimum d_FULL among non-FULL routes
median d_FULL
maximum d_FULL
```

This identifies whether MCTS labels are concentrated in a narrow shell around FULL.

Example undesirable geometry:

```text
28 ON
27 ON
27 ON
26 ON
27 ON
...
```

even if nominal route count is large.

---

# 11. Per-layer ON marginals

For sample `i` and layer `l`, compute the unweighted marginal:

```text
q_i,l
=
(1 / |V_i|)
Σ_{m in V_i} m_l
```

where `m_l ∈ {0,1}`.

Report the distribution of `q_i,l` over:

```text
samples
layers
datasets
Groups A/B/C
```

Also compute global layer marginals:

```text
q_l = mean_i q_i,l
```

Report all 28 layers.

We specifically want to know whether most labels imply:

```text
q_i,l > 0.5
```

for almost every layer.

---

# 12. Weighted BCE marginals matching actual duplicated-BCE training

The duplicated BCE training may apply route weights, including the existing POLAR-style ALL-ON downweight.

Therefore compute a second target that matches the actual training objective.

For each sample `i`, let route-example weights be `alpha_i,m`.

Compute:

```text
q_weighted(i,l)
=
Σ_m alpha_i,m * m_l
/
Σ_m alpha_i,m
```

Use the exact same route weights and per-input normalization as the actual duplicated-BCE training.

Verify this from code/config.

Do not assume it.

Report both:

```text
unweighted marginal q
weighted-training marginal q_weighted
```

and quantify how much ALL-ON downweighting actually changes them.

---

# 13. Per-bit label entropy

For each sample/layer compute:

```text
H_i,l
=
- q_i,l log q_i,l
- (1-q_i,l) log(1-q_i,l)
```

and the weighted-training analogue.

This measures how intrinsically ambiguous the duplicated-BCE target is.

Report:

```text
mean entropy per layer
mean entropy per sample
fraction of bits with q in [0.45, 0.55]
fraction with q >= 0.9
fraction with q <= 0.1
```

overall/per-dataset/per-group.

Interpretation:

```text
q ≈ 0 or 1
→ BCE target is consistent

q ≈ 0.5
→ valid routes disagree strongly on that layer
```

This is important because highly multimodal route sets may be fundamentally poorly represented by independent per-bit BCE.

---

# 14. BCE LABEL ORACLE — primary diagnostic

This is the most important part of the analysis.

Construct the complete mask that **duplicated BCE itself implies**, assuming unlimited capacity and perfect per-sample fitting.

For the exact actual training-weighted target:

```text
p*_i,l = q_weighted(i,l)
```

Then decode using the same deployed threshold rule:

```text
m_BCE_oracle(i,l)
=
1 if p*_i,l > 0.5
0 otherwise
```

Handle exact ties exactly as the current predictor decoder does.

Document the tie rule.

Call this:

```text
BCE label oracle
```

This is not an MCTS oracle.

It is the ideal deterministic threshold prediction implied by the duplicated-BCE supervision.

---

# 15. Evaluate the BCE label oracle

For every training and validation positive input, report:

```text
BCE-oracle ALL-ON fraction
BCE-oracle ALL-OFF fraction
mean ON layers
number of unique complete masks
mask-frequency entropy
```

and, because the valid set is known:

```text
valid-set Hit@1:
m_BCE_oracle ∈ V_i

nearest-valid Hamming:
min_{m in V_i} d_H(m_BCE_oracle, m)
```

Report overall/per-dataset/per-group.

This result directly answers:

> If duplicated BCE learned the labels perfectly, would it still produce dense, low-diversity, possibly invalid hybrid masks?

---

# 16. Unweighted BCE oracle as a secondary diagnostic

Also construct:

```text
m_unweighted(i,l)
=
1[q_i,l > 0.5]
```

Compare:

```text
unweighted BCE oracle
vs
actual-training-weighted BCE oracle
```

This isolates the effect of the ALL-ON `0.3` weighting scheme.

Report:

```text
mask disagreement rate
ON-count difference
ALL-ON-rate difference
Hit@1 difference
Hamming difference
```

If weighting barely changes the oracle geometry, explicitly say so.

---

# 17. Compare BCE label oracle against actual trained BCE predictor

Use the frozen full10 BCE predictions.

For the same records where both are available, compare:

```text
actual predictor mask
vs
BCE label-oracle mask
```

Report:

```text
exact-mask agreement
mean Hamming(actual, oracle)
per-layer agreement
actual mean ON
oracle mean ON
actual ALL-ON %
oracle ALL-ON %
actual unique masks
oracle unique masks
actual valid-set Hit@1
oracle valid-set Hit@1
```

Interpretation:

## Case 1

```text
BCE oracle itself is dense / ALL-ON-like
AND
actual predictor resembles it
```

Then the primary problem is label geometry + duplicated-BCE objective, not predictor optimization.

## Case 2

```text
BCE oracle is sparse/diverse/good
BUT
actual predictor is dense
```

Then predictor/generalization/optimization is the main bottleneck.

## Case 3

```text
BCE oracle is diverse
BUT
oracle Hit@1 is poor
```

Then bitwise BCE is creating invalid hybrid masks from a multimodal valid set.

This would be direct evidence of objective-label mismatch.

---

# 18. Majority-mask validity failure analysis

For every sample where:

```text
m_BCE_oracle ∉ V_i
```

analyze why.

Report:

```text
distance to nearest valid route
number of disagreeing bits
ON count
number of valid-route modes
per-bit entropy
```

Compare oracle-valid versus oracle-invalid samples.

We want to know whether invalid BCE hybrids occur especially when valid routes are highly diverse.

---

# 19. Route clustering / effective number of modes

Route count is insufficient.

Cluster each sample's selected valid routes using Hamming distance.

Do not over-engineer the clustering algorithm.

Use at least one deterministic analysis such as:

```text
connected components under Hamming radius r
```

for several fixed radii:

```text
r = 1, 2, 4
```

or an equivalently transparent deterministic clustering.

Report per sample:

```text
number of clusters
largest cluster fraction
effective route-mode count
```

overall/per-dataset/per-group.

If route count is high but cluster count is approximately one, explicitly classify that as redundant supervision.

---

# 20. Deduplication analysis

Verify whether the selected max-50 routes contain exact duplicates.

Report:

```text
raw route count
unique route count
duplicate fraction
```

before and after the current selection/canonicalization.

If duplicates exist and duplicated BCE treats them as repeated training examples, quantify the implicit reweighting they create.

---

# 21. Raw MCTS vs selected max-50 geometry

If raw route candidates are available, this comparison is critical.

For both:

```text
raw valid routes
selected max-50 routes
```

compute:

```text
route count
mean/min/median ON
pairwise Hamming
distance from ALL-ON
per-layer ON marginals
entropy
cluster count
ALL-ON presence
```

Then answer:

> Does the current max-50 selection preserve or destroy route diversity?

Possible outcomes:

### A. Raw MCTS is diverse, selected-50 is dense/redundant

Then the label-selection policy is the bottleneck.

### B. Raw MCTS is already dense/redundant

Then the MCTS search/reward itself is the bottleneck.

### C. Both are diverse but BCE oracle is poor

Then duplicated BCE is the bottleneck.

---

# 22. Pareto-dominance audit

For each input, define route compute:

```text
C(m) = ON count
```

and use the stored route utility/correctness criterion.

At minimum, under binary correctness:

A route `a` is dominated by route `b` if:

```text
utility(b) >= utility(a)
AND
C(b) < C(a)
```

with at least one strict improvement.

For the current valid route set, report:

```text
fraction of routes Pareto-dominated
fraction of ALL-ON routes dominated
number of Pareto-efficient routes per sample
ON-count distribution of Pareto-efficient routes
```

If richer task score/utility exists, use it carefully and document the exact dominance definition.

Do not invent continuous utility if the labels only contain correctness.

---

# 23. Counterfactual Pareto-filtered label audit

Do not train on it yet.

Construct a **diagnostic-only** alternative label set:

```text
remove Pareto-dominated valid routes
```

For Group B, this will often remove ALL-ON whenever a cheaper equally-valid route exists.

Then recompute:

```text
route counts
ON-count geometry
pairwise Hamming
BCE weighted marginals
BCE label oracle
ALL-ON rate
mean ON
unique masks
Hit@1
nearest Hamming
```

This tells us whether dominated routes are causing the current dense supervision.

Do not launch predictor training from this alternative set.

---

# 24. Counterfactual diversity-balanced label audit

Again, analysis only.

Using the existing valid routes, construct one simple deterministic route-diversity selection alternative.

For example:

```text
start from the lowest-ON valid route
then greedily add the route maximizing minimum Hamming distance
until K representatives are selected
```

Test small representative values such as:

```text
K = 4
K = 8
K = 16
```

Do not claim this is the final label-generation method.

The purpose is only to answer:

> If we select a small number of structurally distinct valid modes instead of up to 50 redundant routes, how does the implied BCE target change?

Recompute the BCE oracle for these diagnostic subsets.

---

# 25. Correct/wrong balance counterfactual — label-level only

Do not retrain.

Using the existing per-sample losses/label statistics, quantify how much the current training population is dominated by Groups A/B/C.

Then construct simple sample-weight scenarios:

```text
FULL-wrong correction examples : FULL-correct examples

1 : 1
2 : 1
3 : 1
```

This cannot create a new per-sample BCE oracle, but it changes the global training pressure.

Compute global weighted layer marginals under each scenario and report:

```text
global per-layer ON marginal
mean global ON probability
expected ALL-ON pressure
```

Do not over-interpret this as predictor performance.

This analysis should answer whether merely oversampling FULL-wrong cases plausibly counteracts the global ON prior.

---

# 26. Cross-sample route diversity

Because the research goal is input-conditioned routing, analyze diversity across samples.

Report:

```text
number of unique selected routes globally
frequency of top-1 / top-5 / top-10 / top-50 most common routes
coverage by ALL-ON
coverage by top-5 masks
coverage by top-50 masks
```

per dataset and overall.

Also report the same for:

```text
minimum-ON valid route per sample
Pareto-efficient representative routes
```

This determines whether there are only a few global route templates or genuinely sample-specific programs.

---

# 27. Cross-sample route interchangeability if cached executions permit

Only do this if existing cached route executions make it inexpensive and exact.

Do not launch a large new execution campaign.

For a bounded set of samples, evaluate whether valid routes from one sample tend to remain valid on another sample.

Examples:

```text
own-sample route utility
vs
same-dataset donor route utility
```

The goal is to estimate:

```text
cross-sample transfer regret
```

If route identity differs across samples but routes are freely interchangeable, visual diversity alone does not justify dynamic routing.

If this requires new expensive Qwen execution, report it as a future diagnostic rather than executing it.

---

# 28. Required decision matrix

The final report must distinguish these cases.

## Outcome A — MCTS search geometry is poor

Evidence:

```text
raw valid routes are already heavily concentrated near ALL-ON,
low Hamming diversity,
few effective route modes,
little sparse/Pareto-efficient structure.
```

Interpretation:

> Fix MCTS search/reward before changing predictor loss.

---

## Outcome B — MCTS is diverse but current max-50 selection is poor

Evidence:

```text
raw route set is diverse,
selected max-50 collapses toward dense/near-duplicate routes.
```

Interpretation:

> Fix supervision selection, not MCTS itself.

---

## Outcome C — Labels are good, but duplicated BCE geometry is intrinsically bad

Evidence:

```text
valid routes contain distinct useful modes,
but per-bit majority/BCE oracle becomes dense or invalid hybrid,
with poor valid-set Hit@1.
```

Interpretation:

> Independent duplicated BCE is mismatched to multimodal complete-route supervision.

---

## Outcome D — BCE label oracle is good but learned predictor is poor

Evidence:

```text
BCE oracle is sparse/diverse/high-validity,
actual trained predictor is substantially denser/worse.
```

Interpretation:

> Predictor optimization/generalization is the primary bottleneck.

---

## Outcome E — Dominated FULL routes are the main issue

Evidence:

```text
Pareto filtering / removing dominated ALL-ON routes
dramatically improves BCE oracle geometry.
```

Interpretation:

> Supervision should preferentially retain Pareto-efficient routes.

---

# 29. Explicit non-goals

Do not:

```text
train a new predictor
change predictor architecture
run another full10 training
regenerate MCTS labels
change MCTS search
add RL
implement a new route scorer
tune route thresholds
decide on a new correct/wrong sampling ratio from intuition alone
```

This task is an analysis task.

Use existing frozen artifacts wherever possible.

---

# 30. Required figures

Create compact plots/tables that make the geometry immediately visible.

At minimum:

1. histogram of valid-route count per sample;
2. histogram of minimum/median route ON count;
3. histogram of pairwise Hamming distance;
4. per-layer ON marginal heatmap or equivalent matrix summary;
5. per-layer global ON probability for Groups A/B/C;
6. sample-level BCE-oracle ON-count histogram;
7. actual BCE predictor vs BCE-label-oracle ON-count comparison;
8. actual vs oracle ALL-ON fraction;
9. BCE-oracle nearest-valid Hamming distribution;
10. raw vs selected max-50 comparison if raw routes exist;
11. Pareto-filtered vs original BCE-oracle comparison.

Keep plots diagnostic and reproducible.

---

# 31. Required final report

Create:

```text
reports/binary_mcts_label_geometry_and_bce_oracle_report.md
```

The report must contain:

1. exact source artifacts analyzed;
2. validity definition;
3. population taxonomy and counts;
4. current correct/wrong composition;
5. route-count geometry;
6. ON-count geometry;
7. ALL-ON geometry;
8. pairwise Hamming diversity;
9. distance-to-FULL analysis;
10. per-layer ON marginals;
11. label entropy;
12. weighted duplicated-BCE oracle definition;
13. BCE-oracle results;
14. BCE oracle vs actual full10 BCE predictor;
15. invalid-hybrid analysis;
16. route-cluster/effective-mode analysis;
17. raw-vs-selected comparison;
18. Pareto-dominance audit;
19. diagnostic Pareto-filtered oracle;
20. diagnostic diversity-balanced oracle;
21. correct/wrong ratio pressure analysis;
22. cross-sample route diversity;
23. explicit Outcome A/B/C/D/E or combination;
24. concrete recommendation for what should be changed next.

---

# 32. Most important interpretation rule

Do not conclude:

```text
"we need more routes"
```

merely because route count is small.

First distinguish:

```text
number of routes
vs
number of genuinely distinct route modes
```

Likewise, do not conclude:

```text
"sparser labels are always better"
```

The desired labels are:

```text
high utility
+
compute efficient
+
non-redundant
+
representative of distinct valid modes
```

---

# 33. Core BCE-oracle interpretation

The central diagnostic is:

```text
existing valid routes
        ↓
exact weighted per-bit BCE optimum
        ↓
threshold at the deployed decoder threshold
        ↓
BCE label-oracle complete mask
```

If this oracle is already ALL-ON-like, dense, low-diversity, or frequently invalid, then no amount of perfect BCE optimization can solve that underlying label/objective mismatch.

If this oracle is good but the learned predictor is not, then optimization/generalization becomes the next target.

---

# 34. One-sentence execution principle

> **Before changing the router again, determine whether the existing MCTS supervision itself contains sparse, distinct, Pareto-efficient, input-specific route modes, and compute the exact complete masks that duplicated BCE would ideally produce from those labels so that label failure, objective failure, and predictor failure can be cleanly separated.**
