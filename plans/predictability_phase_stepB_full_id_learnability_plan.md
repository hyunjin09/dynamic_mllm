# Predictability & Generalization Phase — Step B
## Full In-Domain Learnability of Stage-1 Failure and Stage-2 READ/WRITE Utility

## 0. Executive purpose

Step A established a validated full-scale measurement corpus:

```text
Internal samples                         = 10,399
Stage-1 (sample, layer) states          = 291,172

P90-triggered internal samples          = 1,413
Primary dense post-trigger states       = 15,185
Primary dense four-action branches      = 60,740

Secondary exact routed states           = 35,565
Secondary routed four-action branches   = 142,260
```

All Step-A parity, action-semantic, reproducibility, completeness, and utility-algebra checks passed.

Step B asks the first actual learnability question:

> **Given only the current model state, are eventual dense failure and current READ/WRITE utility predictable in-domain on unseen UIDs/image groups?**

This step is intentionally **in-domain only**.

Do not yet test:

```text
nearest-question dependence
question-cluster OOD
leave-one-dataset-out transfer
external ChartQA/TextVQA/MMMU-Pro/POPE transfer
```

Those belong to Step C/D.

The purpose of Step B is to establish whether a genuine training signal exists before asking whether it generalizes farther.

---

# 1. Scientific questions

Step B must answer four questions.

## Q1 — Stage-1 failure learnability

Can the current dense hidden state predict:

```text
eventual all-FULL final failure
```

on unseen UID/image groups from the same internal distribution?

## Q2 — Stage-2 READ utility learnability

For the primary dense-state task:

```text
U_READ_PRIMARY
=
q_F - q_WO
```

because:

```text
FULL       = READ on,  WRITE on
WRITE_ONLY = READ off, WRITE on
```

Interpretation:

```text
U_READ_PRIMARY > 0
→ READ helps

U_READ_PRIMARY < 0
→ READ hurts
```

## Q3 — Stage-2 WRITE utility learnability

Primary WRITE target:

```text
U_WRITE_PRIMARY
=
q_F - q_RO
```

because:

```text
FULL      = READ on, WRITE on
READ_ONLY = READ on, WRITE off
```

Interpretation:

```text
U_WRITE_PRIMARY > 0
→ WRITE helps

U_WRITE_PRIMARY < 0
→ WRITE hurts
```

## Q4 — Is predictability genuinely state-dependent?

If a predictor works, determine whether it is already explained by:

```text
dataset/source
layer
trigger depth
token counts
Dense C/W
text/query state only
visual state only
```

or whether the full multimodal current state adds material information.

---

# 2. Why these are the primary Stage-2 targets

Step A also constructed:

```text
U_R|W=0
U_W|R=0
U_READ factorial main effect
U_WRITE factorial main effect
U_INT interaction
```

Keep all of them.

However, the primary Step-B targets should be the one-bit deviations from dense FULL:

```text
READ  : q_F - q_WO
WRITE : q_F - q_RO
```

because they map most directly to the original motivation:

> Turning off a specific visual computation while leaving the other visual computation unchanged can improve a wrong dense answer.

This avoids averaging away important READ×WRITE interactions.

Factorial main effects and opposite-context effects are secondary analyses.

---

# 3. Frozen Step-A artifacts

Use the Step-A frozen contract and artifacts exactly.

Primary inputs:

```text
stage1/
    dense_state_manifest.jsonl
    dense_outcome_labels.csv
    state_feature_manifest.jsonl

stage2_dense/
    exact_state_manifest.jsonl
    utility_labels.csv
    correctness_flip_labels.csv
    state_feature_manifest.jsonl

stage2_routed/
    exact_state_manifest.jsonl
    utility_labels.csv
    correctness_flip_labels.csv
    state_feature_manifest.jsonl

manifests/
    internal_sample_manifest.jsonl
    group_registry.jsonl
    p90_trigger_manifest.jsonl
```

Do not regenerate or change labels during Step B.

---

# 4. No label leakage

Predictors may use only state information available at the current layer.

Never feed:

```text
final correctness
q_F/q_RO/q_WO/q_I
utility labels
future hidden states
future actions
previous search success
MCTS metadata
correctness-flip labels
```

as model inputs.

Metadata such as dataset/source/layer may only be used in explicitly named nuisance-control baselines.

---

# 5. Primary evaluation design: 5-fold group-disjoint OOF

All main estimates must be out-of-fold.

Use:

```text
5-fold group-disjoint cross-validation
```

Grouping priority:

```text
image/content group
then UID identity
```

No states from the same image/content group may occur in both train and held-out fold.

Because each sample contributes many layer states, **state-level random splitting is forbidden**.

---

# 6. Shared fold registry

Construct one frozen fold registry at the base sample/group level.

Use the same group assignment for:

```text
Stage-1
Stage-2 dense
Stage-2 routed where the UID/group exists
```

Approximate stratification should preserve, where possible:

```text
dataset
source regime
Dense C/W
P90 trigger status
```

without violating group-disjointness.

Freeze fold IDs before training any predictor.

---

# 7. Fold-support audit

Before model training, report per fold:

```text
# image groups
# UIDs
# Stage-1 C/W
# P90-triggered UIDs
# Stage-2 dense states
# READ harmful/beneficial signs
# WRITE harmful/beneficial signs
# correctness-flip support
```

Do not modify folds based on model performance.

Only repair a fold if a required target class is literally absent, and document the repair before any predictor fit.

---

# 8. Training weights

Many states come from one UID, especially for early triggers.

Primary training weighting:

```text
each UID contributes equal total weight
```

Within a UID:

```text
weight each state by 1 / (# eligible states for that task)
```

For Stage-1:

```text
each sample contributes equal total weight across its 28 layers
```

For Stage-2:

```text
each triggered UID contributes equal total weight across its post-trigger states
```

Do not let early-triggered samples dominate simply because they contribute more layer states.

---

# 9. Fold-local preprocessing

Any normalization, PCA, standardization, or target scaling must be fit on the outer training fold only.

For each fold:

```text
fit normalization on train
apply unchanged to held-out fold
```

Do not compute global mean/std before cross-validation.

---

# 10. Step-B model ladder

The capacity ladder is predefined.

Do not perform open-ended architecture search.

## M0 — Nuisance-only baseline

Inputs may include:

```text
dataset
source regime
layer
trigger layer / trigger-relative depth for Stage-2
visual-token count
text-token count
question length
Dense C/W only when explicitly evaluating conditional utility controls
```

Use a simple linear/logistic or small tabular model.

Purpose:

> How much performance is available from obvious population/position shortcuts?

This is a control, not a candidate method.

## M1 — Linear state probe

Use a frozen low-dimensional state summary.

Stage-1:

```text
established current Stage-1 raw/state summary
→ linear predictor
```

Stage-2 text-only:

```text
current query/text state
→ linear predictor
```

Stage-2 visual-only:

```text
mean/attention-free pooled current visual-token state
→ linear predictor
```

Stage-2 simple multimodal:

```text
[text summary ; visual summary]
→ linear predictor
```

No learned cross-attention in M1.

Purpose:

> Is the signal already linearly readable from the current representation?

## M2 — Two-layer MLP

Use the exact same frozen summary inputs as M1.

Architecture:

```text
input
→ Linear
→ GELU
→ optional standard dropout
→ Linear
→ scalar output
```

Keep hidden width modest and frozen before experiments.

Do not tune a large hyperparameter grid.

Purpose:

> Is the signal present behind a simple nonlinear boundary?

## M3 — Current router-style cross-modal predictor

Use the actual cached current text and visual-token states.

Stage-2:

```text
READ branch:
    current text/query state cross-attends current visual states
    → z_R

WRITE branch:
    learned WRITE pooling query attends current visual states
    → z_W
```

Then predict the target from:

```text
z_R only
z_W only
[z_R ; z_W]
```

This is the primary method-level representation test.

For Stage-1, use the current robust Stage-1 architecture/input contract as the M3 reference predictor.

---

# 11. No layer/source IDs in primary state models

Primary M1/M2/M3 models must not receive:

```text
dataset ID
source ID
layer embedding
trigger depth
```

unless already structurally part of the frozen current model contract.

Keep these variables in M0 nuisance controls.

Reason:

> We want to know whether the hidden state itself carries the signal, not whether the model memorizes layer/source priors.

---

# 12. Stage-1 target and loss

Target:

```text
y_fail ∈ {0,1}
```

Use binary cross-entropy / logistic loss.

Training should be sample-balanced as above.

Do not rebalance the held-out OOF distribution.

---

# 13. Stage-1 primary metrics

Report pooled OOF:

```text
AUROC
AUPRC
```

Also report:

```text
per-layer AUROC
per-layer AUPRC
```

from layer 0 through 27.

This asks:

> When does eventual failure become decodable?

---

# 14. Stage-1 high-preservation operating points

For each outer fold and predictor:

1. fit only on outer training groups;
2. select score thresholds using training-fold predictions only to target:

```text
90% C preservation
95% C preservation
98% C preservation
99% C preservation
```

3. apply those frozen thresholds to the held-out fold.

Report:

```text
held-out C preservation
held-out W recall
precision
median trigger layer if sequentially applied
```

These are secondary to AUROC/AUPRC but connect learnability to the actual Stage-1 use case.

---

# 15. Stage-1 layer-emergence analysis

Plot:

```text
layer
vs
OOF AUROC
```

for:

```text
M0 nuisance
M1 linear
M2 MLP
M3 current Stage-1 head
```

Interpretation examples:

```text
low early → high late
→ failure information emerges during computation

high from layer 0
→ static question/source cues may dominate

M0 close to M3
→ nuisance structure explains much of apparent predictability
```

Do not make causal claims from this plot alone.

---

# 16. Stage-2 primary regression tasks

Train separate predictors for:

```text
READ target:
U_READ_PRIMARY = q_F - q_WO

WRITE target:
U_WRITE_PRIMARY = q_F - q_RO
```

Use a robust scalar regression loss.

Primary default:

```text
Huber loss
```

with fold-local target scale derived from the training fold.

Do not globally clip utility labels.

---

# 17. Why continuous regression is primary

Continuous utility retains information even when final correctness does not flip.

Therefore continuous utility is the primary target.

Binary harmful/helpful and strong correctness-flip metrics are derived from the same OOF predictions.

---

# 18. Stage-2 primary continuous metrics

For held-out OOF predictions report:

```text
Spearman correlation
Pearson correlation
MAE
RMSE
```

Primary metric:

```text
Spearman
```

because ranking harmful versus helpful states is more important than perfect score calibration.

Report:

```text
state-micro metrics
UID-macro metrics
```

For UID-macro Spearman, include only UIDs with enough nonconstant states and report support.

---

# 19. Harmful-computation ranking

Define for evaluation only:

```text
READ harmful:
U_READ_PRIMARY < 0

WRITE harmful:
U_WRITE_PRIMARY < 0
```

Exact-zero states are neutral and excluded from sign AUROC/AUPRC.

Use:

```text
harmfulness score = - predicted utility
```

Report:

```text
AUROC
AUPRC
```

Do not train a separate classifier in the primary experiment.

---

# 20. High-precision harmful subset metrics

For READ and WRITE separately report:

```text
Precision among top 5% most predicted-harmful states
Precision among top 10%
Precision among top 20%

Recall at >=90% precision
Recall at >=95% precision
```

Use natural held-out prevalence.

These metrics matter because a router may only need a conservative high-confidence harmful subset.

---

# 21. Strong correctness-flip evaluation

Use controlled Step-A flip labels as a secondary behavioral test.

Dense-baseline-aligned events:

```text
READ harmful flip:
    c_F=0 and c_WO=1

READ beneficial flip:
    c_F=1 and c_WO=0

WRITE harmful flip:
    c_F=0 and c_RO=1

WRITE beneficial flip:
    c_F=1 and c_RO=0
```

Evaluate whether the continuous prediction ranks these events correctly.

Report:

```text
harmful-flip AUROC
beneficial-flip AUROC
median predicted utility by flip type
```

Do not train directly on flip labels in the primary study.

---

# 22. READ/WRITE representation specificity matrix

For M3, train/evaluate:

| Predictor input | READ utility | WRITE utility |
|---|---:|---:|
| `z_R` | primary diagnostic | cross-target control |
| `z_W` | cross-target control | primary diagnostic |
| `[z_R;z_W]` | joint reference | joint reference |

Key questions:

```text
Does z_R predict READ utility better than z_W?
Does z_W predict WRITE utility better than z_R?
Does concatenation materially improve both?
```

This is the main mechanism-oriented Step-B analysis.

---

# 23. Do not overinterpret branch specificity

If:

```text
z_R > z_W on READ
z_W > z_R on WRITE
```

that supports representation specialization.

It does not prove causality or unique necessity.

If specialization is absent, utility may still be predictable from joint state.

---

# 24. Secondary factorial/context targets

After primary tasks are complete, evaluate the same model ladder on:

```text
U_R|W=0 = q_RO - q_I
U_W|R=0 = q_WO - q_I
U_READ factorial main effect
U_WRITE factorial main effect
U_INT
```

These are secondary.

Purpose:

```text
Does predictability depend on the state of the other READ/WRITE bit?
Is READ×WRITE interaction itself predictable?
```

Do not select the main conclusion from whichever target looks best.

---

# 25. Dense-C versus Dense-W conditional analysis

A utility predictor could appear strong by merely detecting whether the dense sample is globally correct or wrong.

Therefore report OOF metrics separately for:

```text
Dense-C states
Dense-W states
```

for READ and WRITE.

Also report a control:

```text
Dense C/W label only
→ utility prediction
```

If pooled performance is strong but within-C and within-W performance collapses, do not claim fine-grained local utility predictability.

---

# 26. Layerwise Stage-2 predictability

Report READ/WRITE predictability by exact layer where support permits and by:

```text
Early  : 0-8
Middle : 9-18
Late   : 19-27
```

Metrics:

```text
Spearman
harmful AUROC
support
```

Question:

> At what depths does local visual-computation utility become predictable?

---

# 27. Trigger-relative analysis

Also report by:

```text
d = l - L*
```

or bins:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers after trigger
```

Do not feed trigger-relative depth to primary state predictors.

---

# 28. Dataset/source breakdown

For each primary task report OOF performance separately for:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

where support permits.

This is still in-domain because each fold contains the same overall source-family mixture.

Do not interpret this as leave-one-dataset-out OOD.

---

# 29. Nuisance-only comparison

For each task compare:

```text
M0 nuisance-only
M1 linear state
M2 MLP state
M3 full/current state predictor
```

Report absolute gains.

Flag any task where:

```text
M3 - M0
```

is small.

---

# 30. Text-only and visual-only controls

For Stage-2 compare:

```text
text/query summary only
visual pooled summary only
simple concatenation
router-style cross-modal representation
```

Question:

> Is utility predictability primarily in language state, visual state, or their interaction?

Do not use an external text encoder or nearest-question retrieval in Step B.

Those belong to Step C.

---

# 31. Secondary S2-Routed learnability

After the primary S2-Dense study, run the same 5-fold group-disjoint protocol on:

```text
35,565 exact routed states
```

Keep results separate.

Important limitation:

```text
S2-Routed comes from the existing 569-UID successful/preservation route corpus
```

and is selection-biased.

Therefore:

```text
S2-Dense = primary learnability evidence
S2-Routed = secondary state-regime robustness evidence
```

---

# 32. Dense-to-routed state-regime transfer

Optional but valuable if a valid group-disjoint comparison can be built:

```text
train on S2-Dense
test on S2-Routed
```

and, if support permits:

```text
train on S2-Routed
test on S2-Dense
```

This is state-regime transfer, not source OOD.

If support is insufficient, record `not estimable` rather than relaxing group constraints.

---

# 33. No model selection from the capacity ladder

The models have predefined scientific roles:

```text
M1 linear:
    direct decodability

M2 MLP:
    simple nonlinear decodability

M3 router-style:
    method-level learnability
```

Do not invent a new architecture because one OOF result is weak.

The prospectively defined M3 remains the reference formulation for Step C unless an implementation/optimization failure is found.

---

# 34. Optimization protocol

For each outer fold:

```text
train on outer-train groups only
use a small group-disjoint calibration subset from outer-train
for early stopping / training stability
evaluate once on outer-test
```

The calibration subset never overlaps outer-test.

Use identical optimization settings across folds.

Avoid broad hyperparameter sweeps.

---

# 35. Seed policy

Use:

```text
3 seeds
```

for M2/M3 if compute permits.

Report:

```text
seed mean
seed standard deviation
```

for primary metrics.

M1 may be deterministic.

Do not turn Step B into a large model-search exercise.

---

# 36. Statistical uncertainty

Use group/UID-level bootstrap on concatenated OOF predictions.

Report 95% CIs for:

```text
Stage-1 AUROC
Stage-1 AUPRC
Stage-2 Spearman
Stage-2 harmful AUROC
Precision@top10%
```

Bootstrap at the group/UID level, never the individual state-row level.

---

# 37. Evidence categories

Do not define success/failure from one arbitrary threshold.

Classify the evidence pattern.

## Case A — Strong direct learnability

```text
linear materially above nuisance
MLP/router improve further
useful OOF Spearman / harmful AUROC
useful high-precision harmful subset
consistent folds
```

Interpretation:

> Current hidden states contain genuine in-domain signal.

## Case B — Nonlinear/cross-modal learnability

```text
linear weak
MLP/router clearly stronger
```

Interpretation:

> Signal exists but is not directly linearly exposed.

## Case C — Shortcut-conditioned apparent learnability

```text
pooled performance looks good
but nuisance is nearly as good
or Dense-C/Dense-W conditional performance collapses
```

Interpretation:

> Apparent signal is substantially explained by global population structure.

## Case D — Stage-1 strong, Stage-2 weak

Interpretation:

> Eventual failure is predictable, but current visual-computation utility is not readily predictable from the current state.

This challenges the state-only Stage-2 assumption.

## Case E — Both weak

Interpretation:

> Hidden-state predictability is weak even in-domain.

---

# 38. What Step B can establish

Step B can establish:

```text
whether Stage-1 failure is learnable in-domain
whether current-layer READ utility is learnable in-domain
whether current-layer WRITE utility is learnable in-domain
whether signal is linearly or nonlinearly accessible
whether READ/WRITE branches show target-specific specialization
whether full-state signal exceeds nuisance/text-only/visual-only controls
```

---

# 39. What Step B cannot establish

Step B does not establish:

```text
question/template-independent generalization
leave-one-dataset-out robustness
external benchmark transfer
causal representation mechanism
deployable routing improvement
```

A high Step-B score means only:

> The signal is learnable on unseen groups drawn from the same internal population.

---

# 40. Step-C readiness logic

Step C proceeds after Step B regardless of positive or negative findings.

If Step B is strong:

```text
Step C asks:
Does the signal survive semantic/source shift?
```

If Step B is weak:

```text
Step C becomes a confirmation/control:
there is little in-domain signal to generalize.
```

Do not redesign targets between B and C based on performance.

---

# 41. Output directory

Use:

```text
analysis/predictability_generalization/stepB_id_learnability/
```

---

# 42. Required artifacts

Create:

```text
protocol.md
frozen_contract.json

splits/
    group_fold_registry.jsonl
    fold_support.csv
    fold_validation.md

stage1/
    model_configs/
        nuisance.json
        linear.json
        mlp.json
        current_head.json
    oof_predictions.jsonl
    overall_metrics.csv
    layer_metrics.csv
    high_preservation_operating_points.csv
    source_breakdown.csv
    seed_metrics.csv

stage2_dense/
    targets/
        primary_target_definition.md
    models/
        nuisance/
        linear/
        mlp/
        router_style/
    oof_predictions_read.jsonl
    oof_predictions_write.jsonl
    continuous_metrics.csv
    harmful_ranking_metrics.csv
    high_precision_metrics.csv
    strong_flip_metrics.csv
    representation_specificity.csv
    dense_cw_conditional.csv
    layer_metrics.csv
    trigger_relative_metrics.csv
    dataset_source_breakdown.csv
    nuisance_control_comparison.csv
    text_visual_control_comparison.csv
    secondary_factorial_targets.csv
    seed_metrics.csv

stage2_routed/
    oof_predictions_read.jsonl
    oof_predictions_write.jsonl
    continuous_metrics.csv
    harmful_ranking_metrics.csv
    representation_specificity.csv
    layer_metrics.csv
    state_regime_transfer.csv

statistics/
    uid_bootstrap_ci.csv
    primary_comparison_table.csv

figures/
    stage1_auroc_by_layer.png
    stage1_capacity_ladder.png
    stage2_read_capacity_ladder.png
    stage2_write_capacity_ladder.png
    stage2_read_write_specificity.png
    stage2_harmful_precision_coverage.png
    stage2_predictability_by_layer.png
    stage2_predictability_by_trigger_depth.png
    nuisance_vs_state_predictor.png
    dense_vs_routed_predictability.png
    dense_c_vs_dense_w.png

summaries/
    stepB_id_learnability_summary.md
    stepC_readiness.md

artifact_manifest.json
```

---

# 43. Main Stage-1 table

| Model | AUROC | AUPRC | W Recall @95% C-Preserve | W Recall @98% C-Preserve |
|---|---:|---:|---:|---:|
| Nuisance only | | | | |
| Linear state | | | | |
| 2-layer MLP | | | | |
| Current Stage-1 head | | | | |

Also provide the layerwise curves.

---

# 44. Main Stage-2 READ table

Primary target:

```text
q_F - q_WO
```

| Model/Input | Spearman | Harmful AUROC | Harmful AUPRC | Precision@Top5% | Precision@Top10% |
|---|---:|---:|---:|---:|---:|
| Nuisance only | | | | | |
| Text-only linear | | | | | |
| Visual-only linear | | | | | |
| Simple multimodal linear | | | | | |
| Simple multimodal MLP | | | | | |
| Router `z_R` | | | | | |
| Router `z_W` | | | | | |
| Router `[z_R;z_W]` | | | | | |

---

# 45. Main Stage-2 WRITE table

Primary target:

```text
q_F - q_RO
```

Use the same model/input rows and metrics as the READ table.

---

# 46. Mechanism-specific table

| Representation | READ utility | WRITE utility |
|---|---:|---:|
| `z_R` | | |
| `z_W` | | |
| `[z_R;z_W]` | | |

Each cell should include:

```text
Spearman
harmful AUROC
```

This is the main READ/WRITE specialization diagnostic.

---

# 47. `stepB_id_learnability_summary.md` must answer

1. How many groups/UIDs/states were evaluated OOF for Stage-1?
2. How many groups/UIDs/states were evaluated OOF for Stage-2 dense?
3. Did all five folds remain group-disjoint and adequately supported?
4. How predictable is eventual Dense failure with nuisance, linear, MLP, and current Stage-1 head?
5. At which layers does Stage-1 failure predictability emerge?
6. How well does Stage-1 operate at 95/98/99% C preservation?
7. How predictable is primary READ utility `q_F-q_WO`?
8. How predictable is primary WRITE utility `q_F-q_RO`?
9. Is there a useful high-precision harmful-computation subset?
10. Does `z_R` specialize for READ relative to `z_W`?
11. Does `z_W` specialize for WRITE relative to `z_R`?
12. Does `[z_R;z_W]` materially improve over either branch alone?
13. Does utility predictability survive within Dense-C and Dense-W separately?
14. How much performance is explained by nuisance-only controls?
15. Is text-only or visual-only already sufficient?
16. How does predictability vary by layer and trigger-relative depth?
17. How do conditional/factorial secondary targets compare?
18. How predictable are exact routed-state utilities?
19. Is dense-to-routed state-regime transfer measurable, and how strong is it?
20. Does the evidence support genuine in-domain hidden-state learnability?
21. Is Stage-1 materially more predictable than Stage-2?
22. What does Step B not justify concluding?

---

# 48. `stepC_readiness.md`

State:

```text
READY_FOR_STEP_C = true / false
```

Readiness requires:

```text
all OOF predictions complete
no group leakage
fold-local preprocessing verified
primary metrics reproducible
nuisance/text/visual controls complete
primary targets unchanged
```

Also record one evidence category:

```text
A strong direct learnability
B nonlinear/cross-modal learnability
C shortcut-conditioned apparent learnability
D Stage-1 strong / Stage-2 weak
E both weak
```

Do not redesign Step-C targets based on the category.

---

# 49. Stop rule

STOP after:

```text
1. frozen 5-fold group registry
2. full Stage-1 OOF capacity ladder
3. full primary Stage-2 dense READ OOF
4. full primary Stage-2 dense WRITE OOF
5. nuisance/text/visual controls
6. READ-vs-WRITE representation specificity
7. Dense-C / Dense-W conditional analysis
8. layer and trigger-relative analysis
9. secondary routed-state learnability
10. bootstrap uncertainty
11. Step-B summary
12. Step-C readiness decision
```

Do not run:

```text
question-similarity analysis
question clustering
dataset LODO
external four-benchmark transfer
new router deployment evaluation
```

during Step B.

---

# 50. Core principle

Step A established that local visual-computation effects exist and can be measured cleanly.

Step B asks the narrower learnability question:

> **On unseen examples from the same internal population, is the current hidden state sufficient to predict eventual failure and the immediate utility of READ/WRITE computation?**

Only after this is answered should the project ask whether the learned signal survives semantic, source, and external distribution shift.
