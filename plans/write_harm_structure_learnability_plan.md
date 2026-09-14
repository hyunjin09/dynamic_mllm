# WRITE-Harm Structure, Mechanism & Learnability Plan
## Is harmful visual WRITE a structured and learnable problem?

## 0. Research question

The project has already established that changing visual computation can alter final correctness. We therefore do **not** need another existence study for WRITE.

The WRITE-only question is:

> **When updating the visual tokens is harmful, does that failure have repeatable structure that makes it solvable?**

This phase isolates WRITE from READ.

The controlled action pair is:

```text
FULL      = READ1 WRITE1
READ_ONLY = READ1 WRITE0
```

READ is held ON in both branches.

Define an intuitive WRITE-harm score:

```text
H_W = q_READ_ONLY - q_FULL
```

so:

```text
H_W > 0:
    suppressing WRITE improves downstream answer quality
    → WRITE is harmful

H_W < 0:
    suppressing WRITE hurts
    → WRITE is beneficial
```

Reuse the exact frozen Step-A branch labels.

Do not regenerate q targets.

---

# 1. Why WRITE must be studied separately from READ

READ and WRITE have different causal roles.

```text
READ:
    text/control stream accesses current visual evidence

WRITE:
    visual rows are updated/refined and carried into future layers
```

Under the frozen model semantics:

```text
same-layer READ uses pre-WRITE visual K/V
```

Therefore disabling WRITE at layer `l` should primarily change the **visual state leaving layer l**, while the text/control stream at that same layer should remain identical or nearly identical up to numerical tolerance.

The effect of WRITE should then influence later computation when future layers read or further transform the modified visual state.

This creates a distinct hypothesis:

> WRITE harm may be difficult to identify immediately, but its visual-state effect may develop into a predictable downstream signal after future READ/WRITE computation.

This is why the WRITE plan includes both local visual-update analysis and a gated propagation analysis.

---

# 2. Primary population

Use the same complete dense post-trigger population as the READ studies:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 exact dense post-trigger states
```

Primary population:

```text
S2-Dense
```

Secondary routed population:

```text
35,565 exact routed states
569 UIDs
```

Keep routed evidence secondary because the routed corpus is selection-biased.

Reuse the exact Step-B image-group-disjoint fold registry.

---

# 3. Frozen WRITE outcomes

For every state `s_l` reuse:

```text
FULL branch:
    WRITE ON at l
    FULL suffix afterward

READ_ONLY branch:
    WRITE OFF at l
    FULL suffix afterward
```

Targets:

```text
q_F
q_RO
correct_F
correct_RO
H_W = q_RO - q_F
```

No MCTS.

No trajectory search.

No new local-validity labels.

---

# 4. Strong behavioral cohorts

Define:

## WRITE-harmful flip

```text
FULL      → Wrong
READ_ONLY → Correct
```

## WRITE-beneficial flip

```text
FULL      → Correct
READ_ONLY → Wrong
```

## Stable-wrong

```text
FULL      → Wrong
READ_ONLY → Wrong
```

## Stable-correct

```text
FULL      → Correct
READ_ONLY → Correct
```

Verify exact counts from the frozen branch corpus.

Do not infer directional counts from earlier aggregate flip totals.

---

# 5. Staged execution philosophy

Do **not** immediately launch another expensive trajectory search.

Execute this plan in stages:

```text
W0  contract/parity audit
W1  WRITE structural census
W2  immediate visual-update mechanism + ID learnability
    ↓
    if local WRITE signal is materially useful:
        STOP and recommend a WRITE-specific critic/router study
    else:
W3  short-horizon WRITE-effect propagation
    ↓
    if propagation becomes predictive:
        recommend a short-lookahead WRITE critic
    else:
        close the local WRITE-identifiability line
```

External evaluation is gated and not automatic.

---

# Stage W0 — Semantics, cache, and parity

## 6. Immediate WRITE isolation check

At every exact state `s_l`, compare:

```text
FULL      = READ1 WRITE1
READ_ONLY = READ1 WRITE0
```

after executing layer `l`.

Let:

```text
T_F(l)   = post-layer text/control state under FULL
T_RO(l)  = post-layer text/control state under READ_ONLY

V_F(l)   = post-layer visual-token states under FULL
V_RO(l)  = post-layer visual-token states under READ_ONLY
```

Primary semantic expectation:

```text
T_F(l) ≈ T_RO(l)
V_F(l) != V_RO(l)
```

because same-layer READ uses pre-WRITE visual K/V.

Quantify:

```text
||T_F - T_RO||
max absolute text difference
||V_F - V_RO||
```

If text-state equality fails beyond expected numerical tolerance, stop and audit implementation semantics before interpreting WRITE effects.

This parity check is mandatory.

---

## 7. Cache reuse

Reuse existing FULL / READ_ONLY one-layer branch states if available.

Before regenerating anything, inspect:

```text
counterfactual-effect cache
Step-A branch cache
short-horizon branch cache
routed-state cache
```

Only regenerate missing representations.

No new full-suffix utility generation is needed because `H_W` is already frozen.

---

# Stage W1 — Structural organization of WRITE harm

## 8. Per-UID WRITE-harm maps

For each UID construct:

```text
(layer,
 H_W,
 FULL correctness,
 READ_ONLY correctness,
 behavioral cohort)
```

for all post-trigger layers.

Preserve continuous `H_W`.

Also define descriptive sign:

```text
harmful     iff H_W > 0
beneficial  iff H_W < 0
zero        iff H_W = 0
```

Do not threshold away small values in the primary census.

---

## 9. Adjacent persistence

Compute:

```text
P(H_W(l+1)>0 | H_W(l)>0)
P(H_W(l+1)<0 | H_W(l)<0)
```

Compare against a UID-preserving shuffled-layer null.

Repeat enough shuffles to obtain:

```text
null mean
95% / 97.5% interval
enrichment ratio
```

Question:

> Does harmful WRITE persist across adjacent layers more than expected from each UID's marginal harm rate?

---

## 10. Harmful WRITE spans

Define maximal contiguous runs with:

```text
H_W > 0
```

Report:

```text
span count
mean/median span length
maximum span length
fraction of harmful states in spans >=2
fraction in spans >=3
```

Compare to the same UID-shuffled null.

Repeat for beneficial spans as a control.

---

## 11. Strong-flip neighborhoods

For each WRITE-harmful correctness flip at layer `l`, inspect:

```text
l-3 ... l+3
```

within valid bounds.

Report:

```text
mean/median H_W by relative offset
harmful prevalence by offset
```

Question:

> Are strong WRITE-off rescues isolated or embedded in a broader harmful-WRITE region?

---

## 12. Trigger-relative / absolute-depth structure

Report:

```text
mean H_W
median H_W
harmful prevalence
harmful-flip prevalence
```

by:

```text
absolute layer
Early / Middle / Late
trigger-relative depth:
    d=0
    d=1-2
    d=3-5
    d>=6
```

---

## 13. Sample-level WRITE burden

For every triggered UID compute:

```text
fraction of post-trigger layers with H_W > 0
mean positive H_W
max H_W
# harmful WRITE flips
first harmful layer
first harmful-flip layer
```

Report separately for:

```text
Dense-W
Dense-C
```

Question:

> Are some samples globally WRITE-fragile even if population-level adjacency is weak?

---

## 14. Dataset/source structure

Report the full structural census separately for:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

Do not turn descriptive prevalence differences into mechanism claims.

---

# Stage W2 — Immediate visual-update mechanism

## 15. Direct WRITE effect

Define the immediate visual update attributable to WRITE:

```text
delta_V_WRITE
=
V_F(l) - V_RO(l)
```

Because READ is identical in the two branches, this is the primary local WRITE-effect object.

Also keep:

```text
V_pre
V_F
V_RO
```

for geometry analyses.

Do not use `delta_T` as the main WRITE signal at the intervention layer; text-state equality is primarily a semantic parity check.

---

## 16. Feature family F1 — Update magnitude

Compute:

```text
mean-token ||delta_V||
max-token ||delta_V||
global Frobenius norm
||delta_V|| / ||V_pre||
||delta_V|| / ||V_RO||
```

and robust log versions where needed.

Question:

> Is harmful WRITE associated with unusually large or small visual updates?

---

## 17. Feature family F2 — Update direction

For each visual token and pooled summary, compute alignment:

```text
cos(delta_V, V_pre)
cos(delta_V, V_RO)
cos(delta_V, V_F)
```

Aggregate:

```text
mean
std
min/max
quantiles
```

Question:

> Does harmful WRITE move visual states in systematically different directions?

---

## 18. Feature family F3 — Token heterogeneity / concentration

Compute per-token update magnitudes and:

```text
top-1 update share
top-k update share
max / mean
entropy of normalized update magnitude
Gini-like concentration if already implemented
```

Question:

> Is harmful WRITE concentrated on a few visual tokens or broadly distributed?

---

## 19. Feature family F4 — Visual diversity collapse / expansion

Measure before and after WRITE:

```text
mean pairwise cosine similarity
token variance
effective rank of visual-token covariance
spectral entropy
average distance from visual-token centroid
```

Then construct changes:

```text
FULL minus READ_ONLY
post-FULL minus pre-state
```

Question:

> Does harmful WRITE over-homogenize, collapse, or excessively disperse visual representations?

This is a central WRITE-specific mechanism hypothesis.

---

## 20. Feature family F5 — Query/text alignment change

Using the current frozen text/query summary `q_text`, measure:

```text
mean visual↔query cosine before WRITE
mean visual↔query cosine after FULL
mean visual↔query cosine after READ_ONLY
top-k query-aligned visual tokens
change in max/mean query alignment
```

Primary counterfactual feature:

```text
alignment(V_F, q_text) - alignment(V_RO, q_text)
```

Question:

> Does harmful WRITE move visual states toward or away from the current textual query in a systematic but potentially maladaptive way?

Do not assume more query alignment is necessarily better.

---

## 21. Feature family F6 — Visual-row attention/update source

If the implementation exposes the visual-row attention decomposition without changing the model, extract:

```text
visual←visual attention mass
visual←text attention mass
entropy over attended sources
head-wise source mass
```

and where available:

```text
attention contribution norm
MLP contribution norm
```

Treat this family as optional if hooks would require invasive model changes.

Do not delay the primary phase for F6.

---

## 22. Feature family F7 — Spatial structure

Where visual token coordinates exist:

```text
spatial concentration of large WRITE updates
update center-of-mass
spatial spread
# patches carrying top X% update norm
```

If coordinates are unavailable, retain token-index concentration only.

---

## 23. Frozen feature groups

Define before training:

```text
F1 magnitude
F2 direction
F3 token concentration
F4 visual diversity/collapse
F5 query alignment
F6 source-attention/update decomposition (optional)
F7 spatial structure
F_ALL = concat available preregistered families
```

Do not add arbitrary feature families after seeing labels except as explicitly exploratory analysis.

---

## 24. Matched strong-effect mechanism study

Primary matched contrast:

```text
WRITE-harmful flip
vs
WRITE-beneficial flip
```

Match prospectively on:

```text
dataset
source
absolute layer
trigger-relative depth bin
visual-token-count bin
text-token-count bin
```

Do not match on WRITE-specific features.

Secondary comparisons:

```text
harmful flip vs stable-wrong
harmful flip vs stable-correct
```

Report for each preregistered scalar feature:

```text
matched mean/median difference
standardized effect size
image-group bootstrap 95% CI
```

---

## 25. Strong-effect separability

On the matched harmful-vs-beneficial flip cohort run:

```text
linear logistic probe
2-layer MLP
```

with inherited 5-fold image-group-disjoint OOF.

Report:

```text
AUROC
AUPRC
Precision@Top10%
```

This is a mechanism diagnostic, not the main full-population learnability task.

---

# Stage W2B — Full ID learnability

## 26. Target

Use all 15,185 states.

Continuous target:

```text
H_W = q_RO - q_F
```

Positive means WRITE is harmful.

---

## 27. Baselines

Compare:

```text
B0 nuisance-only
B1 generic pre-action state predictor from Step B
B2 previous one-step WRITE counterfactual delta predictor
W1 linear WRITE-specific features
W2 2-layer MLP WRITE-specific features
```

Historical references to verify from artifacts:

```text
WRITE generic-state Spearman ≈ 0.0347
harmful AUROC ≈ 0.5115
WRITE one-step delta Spearman ≈ 0.038
```

Do not hardcode these if source artifacts disagree.

---

## 28. Training protocol

Reuse exactly:

```text
5-fold image-group-disjoint OOF
UID-balanced state weighting
fold-local normalization
same fixed model capacities
same seed policy
```

Primary nonlinear model:

```text
2-layer MLP
```

No architecture search.

---

## 29. Metrics

Continuous:

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

Harmful-WRITE ranking:

```text
harmful iff H_W > 0
```

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

## 30. Strong harmful-flip ranking

Without retraining on flip labels, evaluate whether predicted `H_W` prioritizes:

```text
FULL wrong
READ_ONLY correct
```

Report:

```text
harmful-flip AUROC
harmful-flip AUPRC
median predicted H_W by behavioral cohort
```

---

## 31. Dense-W-only analysis

Mandatory.

Repeat the main WRITE-specific learnability evaluation on:

```text
Dense-W states only
```

A predictor that only exploits global C/W differences is insufficient.

---

## 32. Layer / trigger-relative / source breakdown

Report primary model performance by:

```text
Early / Middle / Late
trigger-relative depth
dataset/source cell
```

Do not select a favorable niche post hoc as the global result.

---

## 33. Generic + WRITE-specific fusion

Secondary control:

```text
[generic pre-state summary ; F_ALL]
→ H_W
```

Question:

> Do explicit WRITE mechanics add information beyond the generic hidden state?

---

## 34. Local WRITE decision gate

After W1/W2, make one of two decisions.

### W-LOCAL-POSITIVE

Proceed no further in this phase if:

```text
WRITE-specific features materially improve over both:
    generic pre-state baseline
    one-step WRITE delta baseline

AND

the gain survives Dense-W-only analysis
AND

a useful high-precision harmful subset exists.
```

Frozen materiality guide:

```text
Spearman improvement >= +0.10 absolute
OR
harmful AUROC improvement >= +0.08 absolute

with image-group bootstrap 95% CI > 0
```

If positive:

```text
STOP.
Recommend a minimal WRITE-specific critic/router experiment.
```

Do not spend compute on propagation merely because it was preplanned.

### W-LOCAL-WEAK

If local structure/mechanism/learnability remains weak:

```text
proceed to Stage W3
```

This is scientifically justified because WRITE affects future visual states and may require propagation before its harm becomes observable.

---

# Stage W3 — Short-horizon WRITE-effect propagation

## 35. Propagation setup

At exact state `s_l`, create:

### WRITE ON branch

```text
layer l:
    FULL

layers > l:
    FULL
```

### WRITE OFF branch

```text
layer l:
    READ_ONLY

layers > l:
    FULL
```

After intervention, both branches use identical FULL continuation.

No later routing.

No search.

---

## 36. WRITE-specific horizon convention

Use:

```text
H0 = immediately after intervention layer l
H1 = after one common FULL layer (through l+1)
H2 = after two common FULL layers
H4 = after four common FULL layers
H8 = after eight common FULL layers
```

At H0:

```text
visual difference should exist
text difference should be approximately zero by semantics
```

At H1 and later:

```text
future READ can convert the visual-state difference into text/control divergence
```

This is a central WRITE hypothesis.

---

## 37. Common-support requirement

As horizon increases, late-layer states become ineligible.

Run:

```text
native-support analysis
```

and:

```text
H8-common-support analysis
```

The common-support comparison is primary for horizon emergence.

Do not attribute gains to propagation if they disappear on common support.

---

## 38. Propagated representations

At every horizon store:

```text
T_F(H), T_RO(H)
V_F(H), V_RO(H)
```

Construct:

```text
delta_T(H) = T_F(H) - T_RO(H)
delta_V(H) = V_F(H) - V_RO(H)
```

Evaluate:

```text
FULL state only
READ_ONLY state only
visual delta only
text delta only
[text delta ; visual delta]
ordered pair
```

Use the same fixed-capacity predictor at every horizon.

---

## 39. Propagation mechanism questions

Before prediction, measure:

```text
||delta_V(H)||
||delta_T(H)||
```

by horizon.

Key mechanistic quantity:

> At what horizon does a WRITE-induced visual difference first create a measurable text/control difference?

Report separately for:

```text
harmful WRITE
beneficial WRITE
strong harmful flips
strong beneficial flips
```

This directly tests whether WRITE harm is mediated through later visual→text READ.

---

## 40. Horizon predictability

At each H0/H1/H2/H4/H8 predict the same frozen target:

```text
H_W
```

Primary metrics:

```text
Spearman
harmful AUROC
Precision@Top10%
strong harmful-flip AUROC
```

Same-capacity controls:

```text
single branch
delta
pair
```

No horizon-specific tuning.

---

## 41. Propagation decision categories

### W-PROP-A — WRITE harm emerges after propagation

Pattern:

```text
H0 weak
later horizon materially stronger
delta/pair beats single-branch controls
Dense-W result survives
```

Interpretation:

> WRITE harm is not readable from the immediate visual update but becomes identifiable after that update influences later computation.

Next:

```text
minimal short-lookahead WRITE critic
```

using the smallest successful horizon.

### W-PROP-B — Later state, not counterfactual difference, is enough

Pattern:

```text
single branch improves with depth
delta/pair adds little
```

Next:

```text
delayed WRITE verification / rollback-style experiment
```

### W-PROP-C — No local or short-horizon WRITE signal

Pattern:

```text
H0/H1/H2/H4/H8 all weak
no useful high-precision harmful subset
```

Interpretation:

> WRITE downstream harm is not locally or short-horizon identifiable under the tested information family.

STOP the local WRITE line.

Do not automatically launch WRITE MCTS/search inside this phase.

---

## 42. Optional frozen Stage-1 branch-critic control

Run only if essentially free from cached states.

At each eligible WRITE ON/OFF post-state compute frozen Stage-1 scores:

```text
p_F
p_RO
```

and test:

```text
p_F - p_RO
```

against branch preference.

This is **not** a primary experiment because the READ branch-critic study already showed that absolute Stage-1 failure signal need not imply useful relative action ranking.

Do not build the WRITE phase around this control.

---

# 43. Semantic/source generalization gate

Do not automatically repeat the entire Step-C generalization suite.

Run semantic/source/LODO generalization only if either:

```text
W-LOCAL-POSITIVE
or
W-PROP-A/B
```

is supported.

Then freeze the selected WRITE predictor and evaluate:

```text
least-similar semantic Q1 vs Q5
question-cluster OOD
Historical → Canonical
Canonical → Historical
dataset LODO
```

No OOD retuning.

---

# 44. External benchmark gate

Do not launch external ChartQA/TextVQA/MMU-Pro/POPE WRITE branch measurement unless:

```text
ID WRITE signal is materially positive
AND
source/dataset transfer is not collapsed
```

The current purpose is solvability characterization, not another expensive deployment run.

---

# 45. Secondary routed-state analysis

Only after a positive dense WRITE signal.

Evaluate the selected frozen WRITE predictor on exact routed states.

Questions:

```text
Does the WRITE signal survive after prior routing changes the state?
Does dense-trained WRITE prediction transfer to routed states?
```

Keep this secondary.

---

# 46. Final WRITE characterization categories

At the end, choose the strongest supported statement.

### WRITE-S1 — Locally structured and learnable

```text
WRITE-specific local mechanics materially predict H_W.
```

### WRITE-S2 — Propagation-structured

```text
Immediate WRITE mechanics are weak,
but short-horizon propagation reveals harm.
```

### WRITE-S3 — Restricted niche

```text
Signal exists only in a narrow layer/task/source regime.
```

### WRITE-S4 — Locally non-identifiable

```text
WRITE harm exists,
but local mechanics and H<=8 propagation remain weak.
```

Do not claim impossibility beyond the tested information family.

---

# 47. READ-vs-WRITE comparison

After WRITE finishes, create a compact comparison using the same conceptual axes:

| Axis | READ | WRITE |
|---|---|---|
| Harm exists? | yes | verify frozen evidence only |
| Layer/span structure | mostly weak/isolated | |
| Local mechanism signature | weak | |
| Generic-state learnability | weak | |
| Operation-specific learnability | weak | |
| One-step counterfactual predictability | weak | |
| Short-horizon propagation | weak through H8 | |
| Stage-1 relative branch critic | weak | optional |
| Best supported characterization | locally non-identifiable | |

The purpose is to answer:

> Is the difficulty specific to READ, or is harmful visual computation generally a long-horizon/nonlocal phenomenon?

---

# 48. What this phase can establish

It can establish:

```text
whether harmful WRITE has layer/sample structure
whether harmful WRITE has a visual-update geometry signature
whether visual diversity collapse/expansion is associated with WRITE harm
whether query alignment changes are predictive
whether WRITE-specific features outperform generic hidden states
whether WRITE effects become predictable only after later propagation
whether any signal survives Dense-W and source/task shifts
```

---

# 49. What this phase cannot establish

Even a positive result does not prove:

```text
causal mechanism
semantic corruption of a specific visual object
deployment benefit
net compute savings
joint READ/WRITE optimality
that WRITE and READ are independent
```

---

# 50. No joint 4-action router in this phase

Do not return to:

```text
FULL / READ_ONLY / WRITE_ONLY / IGNORE
```

joint policy learning during this phase.

WRITE must first be understood in isolation.

---

# 51. Output directory

Use:

```text
analysis/write_harm_structure_learnability/
```

Required artifacts:

```text
protocol.md
frozen_contract.json

population/
    dense_write_state_manifest.jsonl
    routed_write_state_manifest.jsonl
    cohort_counts.csv
    fold_registry.jsonl

parity/
    write_semantics_contract.md
    text_stream_parity.csv
    visual_stream_difference.csv
    smoke_report.md

structure/
    per_uid_write_harm_maps.jsonl
    adjacent_persistence.csv
    sign_transition_matrix.csv
    harmful_span_statistics.csv
    shuffled_null_statistics.csv
    strong_flip_neighborhood.csv
    trigger_relative_summary.csv
    sample_write_burden.csv
    dataset_source_structure.csv

features/
    write_feature_contract.md
    write_operation_features.jsonl
    feature_group_manifest.json
    feature_distribution_summary.csv

matched_mechanism/
    harmful_vs_beneficial_matches.jsonl
    harmful_vs_stable_wrong_matches.jsonl
    harmful_vs_stable_correct_matches.jsonl
    feature_effect_sizes.csv
    matched_probe_metrics.csv

learnability/
    nuisance_baseline.csv
    generic_prestate_baseline.csv
    one_step_write_delta_baseline.csv
    write_linear_metrics.csv
    write_mlp_metrics.csv
    feature_group_ablation.csv
    generic_plus_write_features.csv
    dense_w_only_metrics.csv
    strong_flip_ranking.csv
    layer_breakdown.csv
    trigger_relative_breakdown.csv
    dataset_source_breakdown.csv
    local_decision_gate.md

propagation/
    horizon_support.csv
    h8_common_support_manifest.jsonl
    rollout_manifest.jsonl
    action_trace_validation.csv
    propagated_features.jsonl
    effect_growth_statistics.csv
    horizon_metrics.csv
    dense_w_horizon_metrics.csv
    strong_flip_horizon_metrics.csv
    single_vs_counterfactual_controls.csv
    propagation_decision.md

generalization/
    semantic_similarity_metrics.csv
    question_cluster_ood.csv
    historical_to_canonical.csv
    canonical_to_historical.csv
    dataset_lodo.csv
    external_transfer_gate.md

routed_secondary/
    learnability_metrics.csv
    dense_to_routed_transfer.csv

statistics/
    group_bootstrap_ci.csv
    pairwise_model_differences.csv
    seed_metrics.csv

figures/
    write_harm_by_layer.png
    write_harm_span_length.png
    write_flip_neighborhood.png
    write_update_norm.png
    write_visual_diversity_change.png
    write_query_alignment_change.png
    write_feature_effect_sizes.png
    write_learnability_comparison.png
    write_horizon_spearman.png
    write_horizon_harmful_auroc.png
    write_text_vs_visual_divergence_by_horizon.png
    read_vs_write_summary.png

summaries/
    write_structure_summary.md
    write_mechanism_summary.md
    write_learnability_summary.md
    write_propagation_summary.md
    read_vs_write_characterization.md
    next_research_direction.md

artifact_manifest.json
```

---

# 52. Summary requirements

## `write_structure_summary.md`

Must answer:

1. How many WRITE states/UIDs were analyzed?
2. How many harmful/beneficial/zero WRITE states exist?
3. How many harmful/beneficial correctness flips exist?
4. Does harmful WRITE persist across adjacent layers beyond the shuffled null?
5. What is the harmful-span distribution?
6. Are strong harmful flips inside broader harmful neighborhoods?
7. How does WRITE harm vary by trigger-relative depth?
8. Are some UIDs globally WRITE-fragile?
9. How does prevalence vary by dataset/source?
10. Is WRITE structurally organized or mostly isolated?

## `write_mechanism_summary.md`

Must answer:

1. Did same-layer text-state parity pass between FULL and READ_ONLY?
2. How large is the immediate visual WRITE update?
3. Does harmful WRITE have larger/smaller update magnitude?
4. Does update direction differ?
5. Is harmful WRITE more token-concentrated?
6. Does harmful WRITE collapse or expand visual-token diversity?
7. Does query/text alignment change systematically?
8. Do matched harmful vs beneficial flips show robust feature differences?
9. Which feature family has the largest stable effect?
10. What mechanism claims remain unjustified?

## `write_learnability_summary.md`

Must answer:

1. What is the generic pre-state baseline?
2. What is the one-step WRITE delta baseline?
3. How well does each WRITE-specific feature family predict H_W?
4. How well does F_ALL perform?
5. Does F_ALL materially beat both baselines?
6. Does performance survive Dense-W-only?
7. Is there a useful high-precision harmful-WRITE subset?
8. Are strong harmful flips prioritized?
9. Is there a layer/task/source niche?
10. Does generic + WRITE-specific fusion help?
11. Is W-LOCAL-POSITIVE or W-LOCAL-WEAK supported?
12. What does the local analysis not establish?

## `write_propagation_summary.md`

Create only if W3 runs.

Must answer:

1. How many states are eligible at H0/H1/H2/H4/H8?
2. Does H0 text parity hold?
3. At what horizon does text/control divergence emerge?
4. How does visual divergence grow/decay?
5. How do harmful and beneficial WRITE states differ in propagation?
6. What is single-branch predictability at each horizon?
7. What is visual-delta predictability?
8. What is text-delta predictability?
9. What is combined/pair predictability?
10. Does any horizon materially beat H0?
11. Does any gain survive H8 common support?
12. Does any gain survive Dense-W-only?
13. Are strong harmful flips better identified at longer horizon?
14. Is W-PROP-A/B/C supported?
15. What does propagation not establish?

---

# 53. READ-vs-WRITE synthesis

After the WRITE phase, create:

```text
summaries/read_vs_write_characterization.md
```

Compare:

```text
existence
structural organization
local mechanism
local learnability
counterfactual identifiability
short-horizon propagation
source/task transfer
```

Do not force symmetry if the evidence differs.

The purpose is to decide whether the next research direction should be:

```text
READ-specific
WRITE-specific
joint trajectory-level
or a broader conclusion that harmful visual computation is intrinsically nonlocal.
```

---

# 54. `next_research_direction.md`

Recommend exactly one next move.

If W-LOCAL-POSITIVE:

```text
minimal WRITE-only critic/router
```

with READ always ON.

If W-PROP-A:

```text
short-lookahead WRITE critic
```

using the smallest horizon with material signal.

If W-PROP-B:

```text
delayed WRITE verification / rollback experiment
```

If W-PROP-C / WRITE-S4:

```text
use the READ-vs-WRITE comparison to decide between:
    bounded joint trajectory reasoning
    or
    closing the local routing hypothesis and reframing around nonlocal intervention structure
```

Do not automatically return to the original four-action classifier.

---

# 55. Stop rule

STOP at the earliest scientifically decisive point.

Minimum required:

```text
1. semantics/cache/parity audit
2. full WRITE structure census
3. immediate visual-update mechanism extraction
4. matched strong-effect analysis
5. full 5-fold ID WRITE learnability
6. Dense-W-only analysis
7. local decision gate
```

If W-LOCAL-POSITIVE:

```text
STOP.
Do not run propagation.
```

If W-LOCAL-WEAK:

```text
8. run H0/H1/H2/H4/H8 propagation
9. make W-PROP-A/B/C decision
```

Only after a positive WRITE signal:

```text
10. run semantic/source/LODO generalization
11. optionally run routed-state transfer
```

Do not run external deployment or a joint four-action router inside this phase.

---

# 56. Core principle

For READ, the project found:

```text
harm exists,
but local state, local mechanism,
one-step counterfactuals, and H<=8 propagation
do not provide a strong action-selection signal.
```

WRITE must now be tested independently because its causal pathway is different:

```text
WRITE changes the visual state first,
and only later can that changed visual state affect text reasoning.
```

The decisive WRITE question is:

> **Does harmful visual-state updating have a learnable local signature, or does its harm only become visible after downstream propagation?**

This plan answers that question without reintroducing the four-action routing confound.
