# READ-Harm Structure & Learnability Phase
## Is harmful visual READ a structured, learnable problem?

## 0. Goal

We already know that disabling READ can improve final answers under controlled intervention. This phase therefore does **not** re-test whether READ can ever be harmful.

The new question is:

> **Does READ-induced harm have repeatable structure, and is that structure learnable?**

Focus only on READ. WRITE will be analyzed later as a separate problem.

Use the frozen controlled pair:

```text
FULL       = READ1 WRITE1
WRITE_ONLY = READ0 WRITE1
```

Define the intuitive READ-harm score:

```text
H_R = q_WO - q_F
```

so:

```text
H_R > 0 : removing READ helps → READ is harmful
H_R < 0 : removing READ hurts → READ is beneficial
```

Primary population:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 dense post-trigger states
```

Secondary routed population:

```text
35,565 exact routed states
569 UIDs
```

No small scientific subset.

---

# 1. Strong behavioral cohorts

Define four exact behavioral categories from the frozen Step-A branch results.

### READ-harmful flip

```text
FULL        → Wrong
WRITE_ONLY  → Correct
```

### READ-beneficial flip

```text
FULL        → Correct
WRITE_ONLY  → Wrong
```

### Stable-wrong

```text
FULL        → Wrong
WRITE_ONLY  → Wrong
```

### Stable-correct

```text
FULL        → Correct
WRITE_ONLY  → Correct
```

Keep continuous `H_R` for all states. The flip cohorts are only high-confidence behavioral subsets.

---

# 2. Phase questions

The READ phase has three sequential questions.

## Q1 — Structure

Are harmful READ events random layer-wise points, or do they form repeatable patterns across:

```text
adjacent layers
contiguous spans
trigger-relative depth
absolute depth
sample-level READ fragility
dataset/source regime
Dense-C vs Dense-W
```

?

## Q2 — Mechanism

When READ is harmful, is the actual text→visual READ operation systematically different?

## Q3 — Learnability

Can READ-specific structural/mechanistic features predict:

```text
continuous H_R
```

and identify strong READ-harmful flips on unseen image groups?

---

# Part I — Structural analysis

## 3. Per-UID READ harm maps

For every triggered UID construct the ordered post-trigger sequence:

```text
(layer, H_R, FULL correctness, WO correctness, flip type)
```

for:

```text
l = L* ... 27
```

Store one harm map per UID.

---

## 4. Adjacent-layer persistence

Compute:

```text
P(H_R(l+1) > 0 | H_R(l) > 0)
P(H_R(l+1) < 0 | H_R(l) < 0)
```

Compare against a UID-preserving null:

```text
shuffle layer order of H_R values within each UID
```

Use repeated shuffles to obtain null distributions.

Question:

> Does harmful READ persist across nearby layers more than expected from each UID's overall harm rate?

---

## 5. Sign-transition structure

Report the within-UID transition matrix:

```text
harmful    → harmful
harmful    → beneficial
beneficial → harmful
beneficial → beneficial
```

Treat exact-zero separately.

Compare observed transitions with the within-UID shuffled null.

---

## 6. Harmful spans

Define maximal contiguous runs with:

```text
H_R > 0
```

Report:

```text
# spans per UID
span-length distribution
longest harmful span
fraction of harmful layers in spans length >=2
fraction in spans length >=3
```

Compare to the UID-shuffled null.

Do the same for beneficial spans as a control.

---

## 7. Strong-flip neighborhoods

For every READ-harmful flip at layer `l`, inspect:

```text
l-3 ... l+3
```

within valid bounds.

Plot mean/median `H_R` aligned to the flip layer.

Question:

> Are W→C READ-off rescues isolated, or embedded inside broader harmful READ regions?

---

## 8. Trigger-relative structure

For each state define:

```text
d = l - L*
```

Report:

```text
mean/median H_R
harmful prevalence
harmful-flip prevalence
```

for:

```text
d = 0
d = 1-2
d = 3-5
d >= 6
```

---

## 9. Sample-level READ burden

For every triggered UID compute:

```text
fraction of post-trigger layers with H_R > 0
mean positive H_R
max H_R
first harmful layer
# harmful flips
```

Report separately for:

```text
Dense-W
Dense-C
```

Question:

> Are some samples globally READ-fragile over many layers?

---

## 10. Dataset/source structure

Report all structural statistics separately for:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

Do not interpret prevalence differences as mechanisms before nuisance matching.

---

# Part II — READ-specific mechanism features

## 11. Principle

Generic current-state prediction already failed.

Do not repeat:

```text
generic hidden state → H_R
```

as the main experiment.

Instead extract features tied directly to the READ operation:

```text
how text/control queries access visual evidence
and
what update READ injects into text/control representations
```

---

## 12. Controlled READ pair

For every exact state use:

```text
FULL       = READ ON
WRITE_ONLY = READ OFF
```

from the same pre-action state.

Reuse immediate one-layer branch states from the counterfactual-identifiability experiment.

---

## 13. READ-induced text update

Let:

```text
t_pre = pre-layer text/control state
t_F   = post-layer text/control state under FULL
t_WO  = post-layer text/control state under WRITE_ONLY
```

Define:

```text
delta_t_READ = t_F - t_WO
```

This is the immediate text/control effect attributable to READ while WRITE stays ON.

---

## 14. Feature groups

Freeze the following scalar/mechanistic feature families before inspecting labels.

### F1 — READ update magnitude

```text
||delta_t_READ||
||delta_t_READ|| / ||t_pre||
||delta_t_READ|| / ||t_WO||
```

### F2 — READ update direction

```text
cos(delta_t_READ, t_pre)
cos(delta_t_READ, t_WO)
cos(delta_t_READ, t_F)
```

### F3 — Token-level update concentration

If multiple text/control positions receive READ effects:

```text
max update / mean update
top-1 update share
top-k update share
entropy of normalized update magnitude
```

### F4 — Text→visual attention structure

From the FULL READ operation:

```text
total text→visual attention mass
visual-attention entropy
top-1 attention weight
top-5 cumulative mass
effective # visual tokens
head-wise entropy mean/std
```

### F5 — Query–visual-key compatibility

Using the actual pre-WRITE visual keys:

```text
max q·k
mean top-k q·k
top1-top2 gap
top1-top5 gap
cosine to top visual key
```

### F6 — READ output/value statistics

```text
READ output norm
visual-value aggregation norm
READ output / residual norm
head-wise output-norm statistics
```

### F7 — Visual concentration/spatial structure

Where patch/token coordinates exist:

```text
# tokens carrying top X% attention
attention spread
attention center/spatial concentration
```

Otherwise retain token-index concentration only.

Define:

```text
F_ALL = concat(F1 ... F7)
```

---

# Part III — Strong-effect matched mechanism study

## 15. Primary matched cohort

Compare:

```text
READ-harmful flip
vs
READ-beneficial flip
```

Match prospectively on:

```text
dataset
source regime
absolute layer
trigger-relative depth bin
visual-token-count bin
text-token-count bin
```

Do not match on READ-specific mechanism features.

---

## 16. Secondary controls

Also compare harmful flips against:

```text
stable-wrong
stable-correct
```

using the same nuisance-matching strategy where support allows.

This checks whether a feature is specific to harmful READ rather than generic correctness/failure.

---

## 17. Matched feature effects

For every preregistered scalar feature report:

```text
matched mean/median difference
standardized effect size
image-group bootstrap 95% CI
```

Do not search over arbitrary hidden dimensions post hoc.

---

## 18. Strong-effect separability

On the matched harmful-vs-beneficial flip cohort, run:

```text
linear logistic probe
2-layer MLP
```

using 5-fold image-group-disjoint OOF.

Report:

```text
AUROC
AUPRC
Precision@Top10%
```

This is secondary to the full continuous task.

---

# Part IV — Full READ-harm learnability

## 19. Primary target

Use all 15,185 dense states.

Target:

```text
H_R = q_WO - q_F
```

Train predictors from:

```text
F1
F2
F3
F4
F5
F6
F7
F_ALL
```

---

## 20. Baselines

Compare against:

```text
B0 nuisance-only
B1 Step-B generic pre-state predictor
B2 one-step delta predictor from the counterfactual phase
R1 READ-specific linear predictor
R2 READ-specific 2-layer MLP
```

The scientific question is:

> Do READ-specific features reveal structure that generic state representations missed?

---

## 21. OOF protocol

Reuse the exact Step-B:

```text
5-fold image-group-disjoint registry
```

Use:

```text
fold-local normalization
UID-balanced state weighting
same folds for every feature group
```

No state-level random split.

---

## 22. Continuous metrics

Report:

```text
Spearman
Pearson
MAE
RMSE
```

Primary:

```text
Spearman
```

---

## 23. Harmful READ ranking

Define:

```text
harmful iff H_R > 0
```

Exact zero is neutral.

Report:

```text
AUROC
AUPRC
Precision@Top5%
Precision@Top10%
Precision@Top20%
Recall@90% precision
Recall@95% precision
```

---

## 24. Strong-flip ranking

Without retraining on flip labels, evaluate whether predicted `H_R` ranks:

```text
FULL wrong → WO correct
```

states highly.

Report:

```text
harmful-flip AUROC
harmful-flip AUPRC
predicted H_R by behavioral category
```

---

## 25. Dense-W-only analysis

Mandatory.

Repeat the main READ learnability evaluation only on:

```text
Dense-W states
```

Report:

```text
Spearman
harmful AUROC
Precision@Top10%
harmful-flip ranking
```

A predictor that works only by identifying globally correct/wrong samples is not enough.

---

## 26. Layer/trigger-relative learnability

Report by:

```text
Early / Middle / Late
```

and:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers
```

Question:

> Is READ harm solvable only in restricted computational phases?

---

## 27. Feature-group ablation

Train/evaluate:

```text
F1 only
F2 only
F3 only
F4 only
F5 only
F6 only
F7 only
F_ALL
```

Report which feature groups carry stable OOF signal.

Do not add new feature families after seeing results except as explicitly exploratory follow-up.

---

## 28. Generic + READ-specific fusion

Secondary test:

```text
[generic pre-state summary ; F_ALL]
→ H_R
```

Question:

> Do READ-specific diagnostics add information beyond the generic state?

---

# Part V — Generalization of READ structure

## 29. Semantic similarity

Reuse the frozen Step-C question encoder and bins.

Report the READ-specific predictor on:

```text
least-similar Q1
most-similar Q5
```

Metrics:

```text
Spearman
harmful AUROC
```

---

## 30. Question-family OOD

Reuse the frozen Step-C K=100 semantic-cluster split.

Train on train clusters and evaluate on unseen clusters using the same READ-specific model/hyperparameters.

No OOD retuning.

---

## 31. Historical ↔ Canonical

Run:

```text
Historical → Canonical
Canonical → Historical
```

with the preregistered READ-specific predictor.

Report:

```text
Spearman
harmful AUROC
Precision@Top10%
```

---

## 32. Dataset LODO

Run:

```text
Train ChartQA + TextVQA → Test GQA
Train GQA + TextVQA     → Test ChartQA
Train GQA + ChartQA     → Test TextVQA
```

using fixed READ-specific features/model.

---

## 33. External-transfer gate

Do not automatically launch another full external counterfactual study.

Proceed only if the best preregistered READ-specific model shows material ID improvement over the generic state-only baseline.

Frozen gate:

```text
ID Spearman improves by >= +0.10 absolute
OR
harmful AUROC improves by >= +0.08 absolute

AND

group-bootstrap 95% CI for the improvement is > 0
```

If the gate fails:

```text
do not spend another full external branch-measurement run
```

Conclude that local READ-specific structure is not sufficiently predictive under this feature family.

---

# Part VI — Secondary routed-state study

## 34. Routed-state READ structure

Repeat the strongest preregistered READ-specific feature set on:

```text
35,565 exact routed states
```

Questions:

```text
Does READ-harm structure persist after prior routing changes the state?
Does a dense-trained READ predictor transfer to routed states?
```

Keep this secondary because of selection bias.

---

# 35. Decision categories

## R-STRUCT-A — Structured

Evidence:

```text
adjacent harmful persistence exceeds UID-shuffled null
harmful spans are enriched
strong flips lie in broader harmful regions
```

## R-STRUCT-B — Mostly isolated

Evidence:

```text
transition/span statistics resemble shuffled null
strong flips are mostly isolated
```

---

## R-MECH-A — Mechanistic signature

Evidence:

```text
harmful vs beneficial strong flips differ after nuisance matching
and the differences are consistent across folds/sources
```

## R-MECH-B — No robust local signature

Evidence:

```text
matched feature effects are small/inconsistent
strong-effect separability is near chance
```

---

## R-LEARN-A — Solvable with READ-specific features

Evidence:

```text
READ-specific OOF materially beats generic state-only baseline
works within Dense-W
useful high-precision harmful subset exists
```

Interpretation:

> READ harm has learnable structure, but generic hidden-state representations obscured it.

## R-LEARN-B — Restricted niche

Evidence:

```text
signal exists only in particular layers/datasets/regimes
```

## R-LEARN-C — Not locally solvable

Evidence:

```text
generic state weak
one-step counterfactual weak
READ-specific features also weak
```

Interpretation:

> Downstream READ harm is not readily inferable from local READ mechanics.

---

# 36. What this phase can establish

This phase can establish:

```text
whether harmful READ has within-sample layer structure
whether W→C READ-off flips form broader harmful regions
whether harmful READ has a repeatable text→visual attention/update signature
whether READ-specific features recover learnability missed by generic hidden states
whether the signal survives Dense-W-only evaluation
whether any READ-harm signal transfers across semantic/source/dataset shift
```

---

# 37. What this phase cannot establish

Even a positive result does not prove:

```text
the model literally attends to the wrong object
the discovered feature is causal
a READ-only router will improve final benchmark accuracy
WRITE is unimportant
READ and WRITE are independent
```

Mechanism claims must stay at the level supported by measured READ-operation statistics.

---

# 38. No WRITE analysis yet

Do not run the symmetric WRITE study in this phase.

First complete READ and determine:

```text
Is there structure?
Is there a READ-operation signature?
Is it learnable?
```

Then design WRITE separately using what was learned here.

---

# 39. No deployment READ router yet

Do not train/evaluate a final READ intervention policy here.

A READ-only router is justified only after `H_R` shows material learnability.

---

# 40. Output directory

Use:

```text
analysis/read_harm_structure_learnability/
```

---

# 41. Required artifacts

Create:

```text
protocol.md
frozen_contract.json

population/
    dense_read_state_manifest.jsonl
    routed_read_state_manifest.jsonl
    cohort_counts.csv
    group_registry.jsonl

structure/
    per_uid_read_harm_maps.jsonl
    layer_summary.csv
    trigger_relative_summary.csv
    sign_transition_matrix.csv
    adjacent_persistence.csv
    harmful_span_statistics.csv
    shuffled_null_statistics.csv
    strong_flip_neighborhood.csv
    sample_read_burden.csv
    dataset_source_structure.csv

features/
    read_feature_contract.md
    read_operation_features.jsonl
    feature_group_manifest.json
    feature_distribution_summary.csv

matched_mechanism/
    harmful_vs_beneficial_matches.jsonl
    harmful_vs_stable_wrong_matches.jsonl
    harmful_vs_stable_correct_matches.jsonl
    feature_effect_sizes.csv
    matched_probe_metrics.csv

learnability/
    inherited_stepB_fold_registry.jsonl
    nuisance_baseline.csv
    generic_prestate_baseline.csv
    one_step_delta_baseline.csv
    read_linear_metrics.csv
    read_mlp_metrics.csv
    feature_group_ablation.csv
    generic_plus_read_features.csv
    dense_w_only_metrics.csv
    strong_flip_ranking.csv
    layer_breakdown.csv
    trigger_relative_breakdown.csv
    dataset_source_breakdown.csv

generalization/
    semantic_similarity_metrics.csv
    question_cluster_ood.csv
    historical_to_canonical.csv
    canonical_to_historical.csv
    dataset_lodo.csv
    external_transfer_gate.md

routed_secondary/
    structure_summary.csv
    learnability_metrics.csv
    dense_to_routed_transfer.csv

statistics/
    group_bootstrap_ci.csv
    pairwise_model_differences.csv

figures/
    read_harm_map_examples.png
    read_harm_by_layer.png
    read_harm_by_trigger_depth.png
    harmful_span_length.png
    strong_flip_neighborhood.png
    read_attention_entropy_harmful_vs_beneficial.png
    read_update_norm_harmful_vs_beneficial.png
    read_feature_effect_sizes.png
    read_learnability_comparison.png
    read_high_precision_harmful_subset.png
    read_generalization_ladder.png

summaries/
    read_harm_structure_summary.md
    read_harm_mechanism_summary.md
    read_harm_learnability_summary.md
    next_read_method_recommendation.md

artifact_manifest.json
```

---

# 42. `read_harm_structure_summary.md` must answer

1. How many primary UIDs/states were analyzed?
2. How many harmful/beneficial/zero READ states exist?
3. How many harmful/beneficial strong flips exist?
4. Does harmful READ persist across adjacent layers beyond the shuffled null?
5. What is the harmful-span distribution?
6. Are strong harmful flips inside broader harmful neighborhoods?
7. How far after Stage-1 trigger does READ harm occur?
8. Are some UIDs globally READ-fragile?
9. How does structure vary by dataset/source?
10. Is R-STRUCT-A or R-STRUCT-B supported?

---

# 43. `read_harm_mechanism_summary.md` must answer

1. Which READ feature groups differ between harmful and beneficial strong flips?
2. Do differences remain after nuisance matching?
3. Is harmful READ associated with larger text/control updates?
4. Is visual attention more concentrated or diffuse?
5. Does query-key compatibility differ?
6. Are READ effects concentrated on a few text tokens?
7. Which feature family has the largest matched effect?
8. Can harmful vs beneficial flips be separated OOF?
9. Are signatures consistent across datasets/sources?
10. Is R-MECH-A or R-MECH-B supported?
11. What mechanism claims remain unjustified?

---

# 44. `read_harm_learnability_summary.md` must answer

1. What is the generic state-only baseline?
2. What is the one-step delta baseline?
3. How well does nuisance-only perform?
4. How well does each READ feature group predict `H_R`?
5. How well does F_ALL perform?
6. Does F_ALL materially beat generic state-only?
7. Does it beat one-step counterfactual delta?
8. Is there a useful high-precision harmful READ subset?
9. Does predictability survive Dense-W-only?
10. Which layers/trigger-relative regions are most predictable?
11. Is there a dataset/source niche?
12. Does generic + READ-specific fusion help?
13. How does semantic/source/LODO generalization behave?
14. Does the external-transfer gate pass?
15. Is R-LEARN-A/B/C supported?
16. What does this phase not establish?

---

# 45. `next_read_method_recommendation.md`

Recommend exactly one next step.

If R-LEARN-A:

```text
train one minimal READ-specific critic/router
using the identified mechanism features
```

with WRITE always ON.

If R-LEARN-B:

```text
restrict READ control to the predictable regime
```

and test selective intervention.

If R-LEARN-C:

```text
short-horizon READ effect propagation / planning
```

instead of another local classifier.

Do not return immediately to a four-action router.

---

# 46. Stop rule

STOP after:

```text
1. full 15,185-state READ structure census
2. UID-shuffled structural null analysis
3. full READ-operation feature extraction
4. strong-effect nuisance-matched mechanism analysis
5. 5-fold group-disjoint READ-specific learnability
6. Dense-W-only learnability
7. feature-group ablations
8. semantic/source/LODO generalization
9. external-transfer gate decision
10. secondary routed-state analysis
11. exactly one READ-specific next-method recommendation
```

Do not start WRITE or deploy a READ router inside this phase.

---

# 47. Core principle

The project already knows:

```text
READ can hurt.
```

The unresolved question is:

\[
oxed{
	ext{Does harmful READ have repeatable structure
that makes it a solvable prediction problem?}
}
\]

Answer that in three layers:

```text
STRUCTURE:
    where harmful READ occurs

MECHANISM:
    what is different about the READ operation

LEARNABILITY:
    whether those READ-specific differences predict harm
    on unseen samples
```

Only after READ is understood in isolation should WRITE be studied separately.
