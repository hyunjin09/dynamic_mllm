# READ Short-Horizon Counterfactual Propagation Plan
## At what computational horizon does harmful READ become identifiable?

## 0. Motivation

We already know three things:

```text
1. READ harm is real:
   disabling READ can improve q and can produce W→C flips.

2. Current-state identifiability is weak:
   generic pre-state READ Spearman ≈ 0.04.

3. One-step/local READ diagnostics are still weak:
   one-step delta Spearman ≈ 0.077;
   READ-specific F_ALL/MLP ≈ 0.070;
   local mechanism features do not provide a robust signature.
```

So the next question is:

> **Does READ harm become identifiable only after the READ ON/OFF difference is allowed to propagate through several downstream layers?**

This phase studies READ only. No WRITE study and no deployment router are included.

---

# 1. Frozen target

Use the exact existing READ-harm target:

```text
H_R = q_WRITE_ONLY - q_FULL
```

Interpretation:

```text
H_R > 0 → removing READ improves the final answer → READ is harmful
H_R < 0 → removing READ hurts the final answer → READ is beneficial
```

Do not regenerate or redesign q.

Primary population:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 dense post-trigger states
```

Reuse the exact Step-B 5-fold image-group-disjoint registry.

---

# 2. Counterfactual branches

At each exact dense pre-action state `s_l` create two branches.

## READ ON

```text
layer l:
    FULL = READ1 WRITE1

layers l+1 onward:
    FULL
```

## READ OFF

```text
layer l:
    WRITE_ONLY = READ0 WRITE1

layers l+1 onward:
    FULL
```

The two branches differ **only at layer l**.

No later routing, MCTS, search, or additional intervention.

---

# 3. Propagation horizons

Freeze these horizons:

```text
H = 1, 2, 4, 8
```

Definition:

```text
H=1:
    immediately after executing intervention layer l

H=2:
    after one additional common FULL layer

H=4:
    after three additional common FULL layers

H=8:
    after seven additional common FULL layers
```

So `H=k` means the initial READ ON/OFF difference has propagated through `k` decoder-layer executions counting layer `l`.

Do not fabricate states beyond layer 27.

---

# 4. Native support and common support

Larger horizons naturally have fewer eligible late-layer states.

Therefore run both:

## Native-support analysis

Use all states eligible at each horizon.

## Common-support analysis

Restrict all PRE/H1/H2/H4/H8 comparisons to the states eligible for H=8.

This is the **primary horizon-emergence comparison**, because the sample population is identical across horizons.

Report at each horizon:

```text
# states
# UIDs
# image groups
layer distribution
Dense-C / Dense-W
dataset/source composition
```

---

# 5. Representations at each horizon

For READ-ON/OFF branches at horizon `k`, preserve:

```text
text/control hidden state
visual-token hidden states
UID/layer metadata
branch ID
state hash
```

Define simple fixed-length representations:

```text
t_ON(k)  = current/last text-control vector
t_OFF(k) = current/last text-control vector

v_ON(k)  = mean-pooled visual tokens
v_OFF(k) = mean-pooled visual tokens

r_ON(k)  = [t_ON(k);  v_ON(k)]
r_OFF(k) = [t_OFF(k); v_OFF(k)]
```

---

# 6. Feature conditions

At every horizon evaluate the same inputs.

## Single READ-ON state

```text
r_ON(k)
```

## Single READ-OFF state

```text
r_OFF(k)
```

## Counterfactual pair

```text
[r_ON(k); r_OFF(k)]
```

## Signed counterfactual delta

```text
delta_r(k) = r_ON(k) - r_OFF(k)
```

## Pair + delta

```text
[r_ON(k); r_OFF(k); delta_r(k)]
```

Always use:

```text
ON - OFF
```

as the delta sign convention.

---

# 7. Text/visual decomposition

For every horizon also test:

```text
delta_t(k) = t_ON(k) - t_OFF(k)
delta_v(k) = v_ON(k) - v_OFF(k)
```

Inputs:

```text
text delta only
visual delta only
[text delta ; visual delta]
```

Question:

> Does harmfulness emerge first in the propagated text/control stream, visual stream, or both?

---

# 8. Effect-growth analysis

Before predictor training, characterize how the intervention difference evolves.

For each horizon compute:

```text
||delta_t(k)||
||delta_v(k)||
||delta_r(k)||

||delta(k)|| / ||delta(1)||
cos(delta_t(k), delta_t(1))
cos(delta_v(k), delta_v(1))
```

Report separately for:

```text
harmful states
beneficial states
READ-harmful flips
READ-beneficial flips
```

This is descriptive propagation evidence, not a predictability claim.

---

# 9. Predictor capacity

Keep model capacity fixed across horizons.

Primary scientific comparison:

```text
2-layer MLP
```

Secondary:

```text
linear predictor
```

Optional upper reference:

```text
one fixed token-aware ON/OFF comparator
```

The token comparator architecture must be identical at H=1/2/4/8.

Do not use a deeper model for longer horizons.

---

# 10. Training protocol

Reuse:

```text
Step-B 5-fold image-group-disjoint OOF
UID-balanced state weighting
fold-local normalization
Huber regression
same seed policy
```

No state-level random split.

No horizon-specific hyperparameter tuning.

---

# 11. Primary metrics

For every horizon and feature condition report:

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

Also evaluate harmful READ ranking:

```text
harmful iff H_R > 0
```

with:

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

# 12. Strong behavioral flips

Use the frozen behavioral cohorts:

```text
READ harmful flip:
FULL wrong, WRITE_ONLY correct

READ beneficial flip:
FULL correct, WRITE_ONLY wrong
```

At each horizon report whether predicted `H_R` ranks harmful flips highly:

```text
harmful-flip AUROC
harmful-flip AUPRC
median predicted H_R by cohort
```

---

# 13. Dense-W-only analysis

Mandatory.

Repeat the complete horizon comparison on Dense-W states only.

Report at H=1/2/4/8:

```text
Spearman
harmful AUROC
Precision@Top10%
harmful-flip ranking
```

A signal driven by globally correct samples is not sufficient for correction.

---

# 14. Same-branch depth control

At every horizon compare:

```text
ON-only
OFF-only
PAIR
DELTA
```

This is essential.

If:

```text
ON-only ≈ OFF-only ≈ DELTA
```

then deeper computation itself reveals more information.

If:

```text
DELTA / PAIR >> both single states
```

then propagated counterfactual difference is uniquely informative.

---

# 15. Random-pair control

At H=1, H=4, and H=8 construct a negative control:

```text
ON state from one sample
OFF state from another sample
```

matched where feasible on:

```text
reached layer
intervention layer
dataset/source
Dense C/W
```

Train the same pair/delta predictor.

True within-state counterfactual information should outperform this random-pair control.

---

# 16. Permuted-target control

At H=1 and H=8:

```text
permute H_R across training UIDs
```

and train the same delta predictor.

Held-out performance must collapse to chance/zero correlation.

This is a leakage sanity check.

---

# 17. Layer and trigger-relative analysis

Report horizon curves by intervention-layer bins:

```text
Early
Middle
Late
```

and by:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers after trigger
```

Question:

> Does harmful READ become identifiable faster near the Stage-1 failure boundary?

---

# 18. Dataset/source analysis

Report H=1/2/4/8 curves separately for:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

where support permits.

Do not select the best cell as the global result.

---

# 19. Horizon-emergence statistics

On H=8 common support define:

```text
DeltaSpearman(k)
=
Spearman(H=k) - Spearman(H=1)

DeltaAUROC(k)
=
AUROC(H=k) - AUROC(H=1)
```

for:

```text
k = 2, 4, 8
```

Use image-group bootstrap 95% CIs.

Also report whether performance shows an approximately monotonic trend with horizon.

---

# 20. Prospective materiality gate

A short horizon is considered materially informative only if, relative to H=1 on common support:

```text
Spearman improves by >= +0.10 absolute
OR
harmful AUROC improves by >= +0.08 absolute
```

AND:

```text
the group-bootstrap 95% CI of the improvement is > 0
```

This gate determines whether a short-lookahead method is justified.

---

# 21. Decision categories

## H-READ-A — Counterfactual propagation emerges

Requirements:

```text
some H<=8 passes materiality gate
PAIR/DELTA materially beats both single-branch controls
random-pair control remains weak
```

Interpretation:

> Harmful READ becomes identifiable after its effect propagates.

Next step:

```text
minimal short-lookahead counterfactual READ critic
```

using the smallest successful horizon.

---

## H-READ-B — Delayed state information emerges

Requirements:

```text
single ON/OFF states improve materially with depth
PAIR/DELTA provide little extra gain
```

Interpretation:

> Later states expose useful information, but explicit counterfactual comparison is unnecessary.

Next step:

```text
delayed verification / rollback-style READ control
```

---

## H-READ-C — Token-level propagation signal

Requirements:

```text
pooled features remain weak
token-aware comparator passes materiality gate
```

Interpretation:

> Propagated harm information exists but is token-local.

Next step:

```text
token-aware short-lookahead READ critic
```

---

## H-READ-D — No short-horizon identifiability

Requirements:

```text
H=1/2/4/8 all fail materiality gate
no useful high-precision harmful subset
token comparator remains weak
```

Interpretation:

> READ downstream harm is not locally or short-horizon identifiable under this information family.

Next hypothesis:

```text
bounded longer-horizon planning / search-conditioned READ decision
```

Do not return to another one-state classifier.

---

# 22. Compute strategy

The frozen target already exists, so branches only need to run to H=8.

For each state:

```text
READ ON rollout
READ OFF rollout
```

and save intermediate states at:

```text
H=1,2,4,8
```

One rollout yields all horizons.

No full-suffix label regeneration is required.

Maximum primary work is approximately:

```text
2 short rollouts × 15,185 states
```

with fewer layers near the end.

---

# 23. Parity requirements

Require:

```text
ON H=1 == previous FULL one-step post-state
OFF H=1 == previous WRITE_ONLY one-step post-state
```

For H>1:

```text
both branches must follow identical FULL actions
after the intervention layer
```

Record exact action traces and state hashes.

No unexplained mismatch is allowed.

---

# 24. Secondary routed-state analysis

If primary dense results show material emergence, repeat the selected successful horizon(s) on the existing:

```text
35,565 exact routed states
```

If primary result is H-READ-D, routed analysis may be limited to H=1 and the strongest preregistered short horizon as confirmation.

Keep routed evidence secondary due to selection bias.

---

# 25. No deployment evaluation yet

Do not train a final READ router or run external benchmark deployment in this phase.

A deployment experiment is justified only for H-READ-A/B/C.

---

# 26. Output directory

Use:

```text
analysis/read_harm_short_horizon_propagation/
```

Create:

```text
protocol.md
frozen_contract.json

population/
    state_manifest.jsonl
    horizon_support.csv
    common_support_h8_manifest.jsonl

branches/
    rollout_manifest.jsonl
    action_trace_validation.csv
    state_hash_parity.csv
    horizon_state_manifest.jsonl

features/
    pooled_horizon_features.jsonl
    delta_horizon_features.jsonl
    token_horizon_features.jsonl
    effect_growth_statistics.csv

splits/
    inherited_stepB_fold_registry.jsonl
    fold_support_by_horizon.csv

metrics/
    native_support_metrics.csv
    common_support_metrics.csv
    harmful_ranking_metrics.csv
    strong_flip_metrics.csv
    dense_w_only_metrics.csv
    layer_breakdown.csv
    trigger_relative_breakdown.csv
    dataset_source_breakdown.csv
    horizon_emergence.csv
    monotonicity.csv

controls/
    single_branch_depth_control.csv
    random_pair_control.csv
    permuted_target_control.csv
    text_visual_delta_ablation.csv

statistics/
    group_bootstrap_ci.csv
    horizon_pairwise_differences.csv
    seed_metrics.csv

figures/
    read_horizon_spearman.png
    read_horizon_harmful_auroc.png
    read_single_vs_counterfactual_by_horizon.png
    read_dense_w_horizon_curve.png
    read_harmful_precision_by_horizon.png
    read_delta_norm_growth.png
    read_horizon_by_trigger_depth.png

summaries/
    read_short_horizon_propagation_summary.md
    next_read_method_recommendation.md

artifact_manifest.json
```

---

# 27. Main result table

On common H=8 support produce:

| Input | PRE | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| ON-only Spearman | | | | | |
| OFF-only Spearman | | | | | |
| Pair Spearman | | | | | |
| Delta Spearman | | | | | |
| Token comparator Spearman | | | | | |

Also produce the same table for:

```text
harmful AUROC
Precision@Top10%
```

---

# 28. Summary must answer

`read_short_horizon_propagation_summary.md` must answer:

1. How many states/UIDs are eligible at H=1/2/4/8?
2. Did H=1 reproduce the previous one-step experiment?
3. Did both branches follow identical FULL continuation after intervention?
4. How does ON/OFF representation distance grow with horizon?
5. Do harmful and beneficial states propagate differently?
6. What is ON-only predictability at H=1/2/4/8?
7. What is OFF-only predictability at H=1/2/4/8?
8. What is PAIR predictability at H=1/2/4/8?
9. What is DELTA predictability at H=1/2/4/8?
10. Does the token comparator improve with horizon?
11. On common support, does predictive signal emerge monotonically?
12. Does any H<=8 pass the frozen materiality gate?
13. Does PAIR/DELTA beat both single-branch controls?
14. Does random-pair performance collapse?
15. Does permuted-target performance collapse?
16. Does any gain survive Dense-W-only analysis?
17. Are strong harmful flips identified better at longer horizons?
18. Does emergence depend on layer or trigger-relative depth?
19. Is any result stable across datasets/sources?
20. Which H-READ-A/B/C/D category is supported?
21. What does this phase not establish?

---

# 29. Next recommendation

`next_read_method_recommendation.md` must recommend exactly one next step.

```text
H-READ-A:
    minimal short-lookahead counterfactual READ critic

H-READ-B:
    delayed verification / rollback READ controller

H-READ-C:
    token-aware short-lookahead READ critic

H-READ-D:
    bounded longer-horizon READ planning/search study
```

State:

```text
why the result supports it
what a positive next result would mean
what a negative next result would mean
what remains unproven
```

---

# 30. Stop rule

STOP after:

```text
1. full ON/OFF rollout extraction
2. H=1 parity
3. H=1/2/4/8 feature construction
4. native-support analysis
5. H=8 common-support analysis
6. same-capacity ON/OFF/PAIR/DELTA comparison
7. token-aware comparator
8. Dense-W-only analysis
9. strong-flip analysis
10. layer/trigger-relative analysis
11. random-pair + permuted-target controls
12. frozen materiality decision
13. exactly one READ-specific next recommendation
```

Do not start WRITE or a deployment router inside this phase.

---

# 31. Core principle

READ harm has already failed every local identifiability test:

```text
current state           → weak
one-step counterfactual → weak
local READ mechanics    → weak
```

The next discriminating question is:

> **Does the consequence of READ need to propagate through multiple layers before harmfulness becomes identifiable?**

This phase measures that emergence while keeping the intervention, target, model capacity, and evaluation split fixed.
