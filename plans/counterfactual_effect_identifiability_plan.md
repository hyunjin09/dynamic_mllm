# Counterfactual Effect Identifiability
## Can READ/WRITE utility be predicted from one-step post-action effects when it is not predictable from the pre-action state?

## 0. Purpose

The previous Predictability A→D phase established:

```text
Stage-1:
eventual failure is learnable in-domain but source/task specific

Stage-2:
local READ/WRITE utility exists,
but is barely predictable from the pre-action current-state snapshot
```

Reference Step-B Stage-2 results:

```text
READ  target = q_F - q_WO
READ  M3 Spearman = 0.0416

WRITE target = q_F - q_RO
WRITE M3 Spearman = 0.0347
```

The next question is:

> Does utility become identifiable after candidate visual computation is executed for exactly one layer?

This experiment compares:

```text
pre-action state
vs
single post-action state
vs
counterfactual post-state pair
vs
counterfactual difference
```

This is an identifiability experiment, not yet a deployment-router experiment.

---

## 1. Main hypotheses

### H1 — Single post-state sufficiency

```text
pre weak
single post-state strong
pair/delta adds little
```

Interpretation:

> Utility becomes observable after one layer, but explicit counterfactual comparison is not necessary.

### H2 — Counterfactual-difference identifiability

```text
pre weak
single post-states weak
pair/delta strong
```

Interpretation:

> Utility is encoded primarily in the action-induced difference between the two one-step outcomes.

This would justify a future speculative probe-and-route controller.

### H3 — One-step lookahead still insufficient

```text
pre/post/pair/delta all weak
```

Interpretation:

> Downstream utility is not exposed even after one-step local probing; a longer-horizon probe/search becomes the next candidate.

---

## 2. Frozen population

Use the complete primary Step-A S2-Dense population:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 exact dense post-trigger states
```

No small scientific subset.

Reuse the exact Step-B five-fold image-group-disjoint OOF registry.

---

## 3. Frozen utility targets

Do not redesign targets.

### READ

```text
U_READ = q_F - q_WO
```

where:

```text
FULL       = READ1 WRITE1
WRITE_ONLY = READ0 WRITE1
```

### WRITE

```text
U_WRITE = q_F - q_RO
```

where:

```text
FULL      = READ1 WRITE1
READ_ONLY = READ1 WRITE0
```

The Step-A full-suffix branch measurements remain the frozen ground truth.

No MCTS or new suffix search is required.

---

## 4. One-step branch construction

For every exact pre-action state `s_l`:

### READ comparison

```text
s_l
├─ FULL        → one-layer post-state s_{l+1}^F
└─ WRITE_ONLY  → one-layer post-state s_{l+1}^WO
```

These branches differ only in READ.

### WRITE comparison

```text
s_l
├─ FULL       → one-layer post-state s_{l+1}^F
└─ READ_ONLY  → one-layer post-state s_{l+1}^RO
```

These branches differ only in WRITE.

Feature construction stops immediately after the current layer.

Do not execute any future decoder layer for the new input features.

---

## 5. Post-action state definition

For each branch preserve:

```text
post-layer text/control hidden state
post-layer visual-token hidden states
layer
UID
image/content group
branch action
pre-state hash
post-state hash
```

No final-answer logits or future hidden states may be used as predictor inputs.

---

## 6. Primary fixed-length representation

For branch `a` define:

```text
t_a = post-layer current/last text-control hidden state
v_a = mean-pool(post-layer visual-token hidden states)
r_a = [t_a ; v_a]
```

Use the same pooling rule across every fold/action/target.

Also preserve raw token tensors so a later token-aware comparator can be evaluated without rerunning Qwen.

---

## 7. READ feature conditions

Evaluate the same READ target under:

```text
R0 PRE:
    pre-action state representation

R1 FULL_POST:
    r_F

R2 WO_POST:
    r_WO

R3 PAIR:
    [r_F ; r_WO]

R4 DELTA:
    r_F - r_WO

R5 PAIR_PLUS_DELTA:
    [r_F ; r_WO ; r_F-r_WO]
```

`R4 DELTA` is the primary counterfactual-effect feature.

---

## 8. WRITE feature conditions

Evaluate the WRITE target under:

```text
W0 PRE:
    pre-action state representation

W1 FULL_POST:
    r_F

W2 RO_POST:
    r_RO

W3 PAIR:
    [r_F ; r_RO]

W4 DELTA:
    r_F - r_RO

W5 PAIR_PLUS_DELTA:
    [r_F ; r_RO ; r_F-r_RO]
```

`W4 DELTA` is the primary WRITE counterfactual-effect feature.

---

## 9. Why single post-state controls are mandatory

If:

```text
PRE        rho = 0.04
FULL_POST  rho = 0.35
OFF_POST   rho = 0.34
DELTA      rho = 0.36
```

then the correct conclusion is:

> One extra layer exposes utility information.

Not:

> Counterfactual comparison is essential.

In contrast:

```text
PRE        rho = 0.04
FULL_POST  rho = 0.07
OFF_POST   rho = 0.08
PAIR       rho = 0.45
DELTA      rho = 0.43
```

supports:

> The within-state action-induced difference contains information unavailable in either individual post-state.

---

## 10. Same-capacity predictor ladder

This experiment is about information availability, not architecture search.

Use:

### C1 Linear

```text
input feature → scalar utility
```

### C2 Two-layer MLP

```text
input
→ Linear
→ GELU
→ Linear
→ scalar utility
```

Use one frozen hidden width for every feature condition.

### C3 Token-aware comparator

Only after C1/C2.

Use raw paired token states:

READ:

```text
FULL tokens + WRITE_ONLY tokens
→ small token-aware comparator
→ U_READ
```

WRITE:

```text
FULL tokens + READ_ONLY tokens
→ small token-aware comparator
→ U_WRITE
```

Keep this comparator lightweight and fixed before training.

---

## 11. Primary scientific comparison

The main information comparison uses **C2 MLP for every input condition**:

```text
PRE
FULL_POST
OFF_POST
PAIR
DELTA
PAIR_PLUS_DELTA
```

This isolates the information change while keeping model capacity fixed.

C1 tests direct linear decodability.

C3 tests whether pooled vectors discard useful token-local information.

Do not compare a linear PRE baseline against a deep counterfactual comparator and attribute the whole gain to counterfactual information.

---

## 12. Training/evaluation contract

Reuse Step-B:

```text
5-fold image-group-disjoint OOF
UID-balanced state weighting
fold-local feature normalization
Huber regression loss
same seed policy
```

For each triggered UID:

```text
total loss weight = constant
```

so early-triggered UIDs do not dominate by contributing more states.

---

## 13. Primary metrics

For each target × feature × predictor report OOF:

```text
Spearman
Pearson
MAE
RMSE
```

Primary metric:

```text
Spearman
```

Also report UID-macro metrics where estimable.

---

## 14. Harmful-computation ranking

From the same continuous predictions:

```text
READ harmful  iff U_READ < 0
WRITE harmful iff U_WRITE < 0
```

Exact zeros are neutral.

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

Do not train a separate harmful classifier in the primary experiment.

---

## 15. Strong correctness-flip analysis

Use the frozen Step-A behavioral labels.

READ:

```text
harmful flip:
c_F=0, c_WO=1

beneficial flip:
c_F=1, c_WO=0
```

WRITE:

```text
harmful flip:
c_F=0, c_RO=1

beneficial flip:
c_F=1, c_RO=0
```

Evaluate whether each representation ranks these states appropriately.

No direct training on flip labels.

---

## 16. Text-vs-visual delta ablation

For READ:

```text
Δtext_READ   = t_F - t_WO
Δvisual_READ = v_F - v_WO
```

For WRITE:

```text
Δtext_WRITE   = t_F - t_RO
Δvisual_WRITE = v_F - v_RO
```

Evaluate:

```text
text delta only
visual delta only
[text delta ; visual delta]
```

This tests the original semantic intuition:

```text
READ effect may appear more strongly in the text/control stream
WRITE effect may appear more strongly in the visual stream
```

Treat this as a hypothesis, not an assumption.

---

## 17. Dense-C / Dense-W conditional analysis

Report primary OOF metrics separately for:

```text
Dense-C states
Dense-W states
```

Especially report:

```text
READ Spearman on Dense-W
WRITE Spearman on Dense-W
```

because corrective routing matters most there.

This also tests whether counterfactual predictability is merely another form of global failure detection.

---

## 18. Layerwise and trigger-relative analysis

Report by:

```text
exact layer where supported
Early / Middle / Late
```

and trigger-relative:

```text
at trigger
1-2 layers after trigger
3-5 layers after trigger
6+ layers
```

Question:

> Does counterfactual identifiability emerge only at specific depths?

---

## 19. Random-pair negative control

Construct a matched negative control:

```text
FULL post-state from one sample
paired with OFF post-state from another sample
```

match, where feasible, on:

```text
layer
dataset/source
Dense C/W
```

Train the same PAIR/DELTA model.

If true within-state counterfactual correspondence matters, random-pair performance should collapse.

This is a diagnostic control, not a method candidate.

---

## 20. Branch-order control

Primary READ pair order:

```text
[FULL ; WRITE_ONLY]
```

Primary WRITE pair order:

```text
[FULL ; READ_ONLY]
```

Run one diagnostic swapped-order test.

Do not average orders in the primary experiment.

The model must have a consistent sign convention for action effects.

---

## 21. Layer-27 handling

At layer 27 there is no next decoder layer.

Use the final layer output immediately after action execution as the post-action representation.

Do not fabricate an `s_28`.

If WRITE has zero downstream utility by semantics at layer 27, preserve those samples for completeness and report them separately.

---

## 22. Secondary routed-state experiment

After the full S2-Dense experiment, repeat the strongest predefined feature conditions on the existing:

```text
35,565 exact routed states
```

Keep this secondary because the routed corpus is selection-biased.

Use exact routed-prefix identity.

Do not merge routed states by UID+layer alone.

---

## 23. Dense-to-routed transfer

If group-disjoint support permits:

```text
train the counterfactual predictor on S2-Dense
test on S2-Routed
```

using the same feature definition.

Question:

> Is one-step effect identifiability robust when the entering hidden state has already been changed by earlier routing actions?

Do not relax group-disjointness.

---

## 24. Compute plan

If Step A already stored immediate branch states, reuse them.

Otherwise generate at most these one-layer branches per dense state:

```text
FULL
WRITE_ONLY
READ_ONLY
```

Total maximum:

```text
15,185 × 3 = 45,555 one-layer branch executions
```

IGNORE is unnecessary for the primary READ/WRITE targets.

This is far cheaper than Step-A full-suffix branch execution.

---

## 25. Branch parity checks

For every state verify:

```text
same exact pre-state feeds all candidate branches
FULL post-state matches canonical Dense next-layer state
WRITE_ONLY differs only in READ bit
READ_ONLY differs only in WRITE bit
```

Run repeated deterministic parity on a frozen stratified set.

No unexplained mismatch is allowed.

---

## 26. Primary READ result table

| Input feature | Linear rho | MLP rho | Harmful AUROC | Precision@Top10% |
|---|---:|---:|---:|---:|
| PRE | | | | |
| FULL post | | | | |
| WRITE_ONLY post | | | | |
| Pair [F;WO] | | | | |
| Delta F-WO | | | | |
| Pair+Delta | | | | |
| Token comparator | | | | |

Reference Step-B pre-state result:

```text
M3 READ rho = 0.0416
```

Also report the same-capacity PRE MLP result for the fair information comparison.

---

## 27. Primary WRITE result table

| Input feature | Linear rho | MLP rho | Harmful AUROC | Precision@Top10% |
|---|---:|---:|---:|---:|
| PRE | | | | |
| FULL post | | | | |
| READ_ONLY post | | | | |
| Pair [F;RO] | | | | |
| Delta F-RO | | | | |
| Pair+Delta | | | | |
| Token comparator | | | | |

Reference Step-B pre-state result:

```text
M3 WRITE rho = 0.0347
```

---

## 28. Decision categories

### Case A — Counterfactual difference clearly wins

```text
PRE weak
single post-states weak
PAIR/DELTA materially stronger
high-precision harmful subset improves
random-pair control collapses
```

Interpretation:

> Utility is not identifiable from the pre-state but becomes identifiable from the immediate counterfactual effect.

Next method:

```text
one-layer speculative probe-and-route Stage-2
```

### Case B — Single post-state is enough

```text
FULL_POST or OFF_POST
≈ PAIR/DELTA
>> PRE
```

Interpretation:

> One layer of execution reveals utility, but explicit comparison is unnecessary.

Next method:

```text
post-action verification / delayed-routing controller
```

### Case C — Token-aware pair works, pooled vectors do not

```text
pooled pair/delta weak
token comparator strong
```

Interpretation:

> Useful effect information is token-local and lost by simple pooling.

Next method:

```text
small token-aware one-layer comparator
```

### Case D — Everything remains weak

```text
PRE
POST
PAIR
DELTA
TOKEN COMPARATOR
all weak
```

Interpretation:

> One-step action effects still do not expose downstream utility.

Next experiment:

```text
2-layer / short-rollout counterfactual identifiability
```

Do not jump directly back to unrestricted MCTS.

---

## 29. What this experiment can establish

It can establish whether local READ/WRITE utility is more identifiable from:

```text
pre-action state
one post-action state
or
within-state counterfactual effect
```

It can also test whether the relevant effect is concentrated in:

```text
text/control changes
visual-state changes
or both
```

---

## 30. What this experiment cannot establish

Even a positive result does not establish:

```text
positive routed benchmark gain
net compute savings
external transfer
causal correctness of the comparator
global optimality of one-step probing
```

Those require a later deployment experiment.

---

## 31. No deployment training yet

Do not train:

```text
final 4-action routing policy
beam controller
new Stage-1
MCTS policy
on-policy controller
```

Only utility-identifiability predictors are trained here.

---

## 32. Output directory

Use:

```text
analysis/dense_failure_stage2/counterfactual_effect_identifiability/
```

---

## 33. Required artifacts

Create:

```text
protocol.md
frozen_contract.json

states/
    dense_state_manifest.jsonl
    post_state_manifest.jsonl
    branch_execution_manifest.jsonl
    state_hash_parity.csv

features/
    representation_contract.md
    pooled_post_features_manifest.jsonl
    delta_features_manifest.jsonl
    token_feature_manifest.jsonl

splits/
    inherited_stepB_fold_registry.jsonl
    fold_support.csv
    fold_validation.md

read/
    pre_state/
    full_post/
    write_only_post/
    pair/
    delta/
    pair_plus_delta/
    token_comparator/
    text_visual_delta_ablation.csv
    metrics.csv

write/
    pre_state/
    full_post/
    read_only_post/
    pair/
    delta/
    pair_plus_delta/
    token_comparator/
    text_visual_delta_ablation.csv
    metrics.csv

controls/
    random_pair_control.csv
    branch_order_control.csv
    dense_cw_conditional.csv
    layer_breakdown.csv
    trigger_relative_breakdown.csv

routed_secondary/
    metrics_read.csv
    metrics_write.csv
    dense_to_routed_transfer.csv

statistics/
    uid_bootstrap_ci.csv
    pairwise_model_differences.csv
    seed_metrics.csv

figures/
    read_identifiability_comparison.png
    write_identifiability_comparison.png
    pre_vs_post_vs_counterfactual.png
    harmful_precision_coverage_read.png
    harmful_precision_coverage_write.png
    text_vs_visual_delta.png
    layerwise_counterfactual_identifiability.png

summaries/
    counterfactual_effect_identifiability_summary.md
    next_method_recommendation.md

artifact_manifest.json
```

---

## 34. `counterfactual_effect_identifiability_summary.md` must answer

1. How many UIDs/states were used?
2. Did exact branch parity pass?
3. Does FULL post-state reproduce canonical Dense next-state features?
4. What is READ predictability from PRE?
5. What is READ predictability from FULL post-state?
6. What is READ predictability from WRITE_ONLY post-state?
7. What is READ predictability from the F/WO pair?
8. What is READ predictability from F-WO delta?
9. Does pair+delta improve further?
10. What are the corresponding WRITE F/RO results?
11. Does one post-state alone explain most of any gain?
12. Does explicit counterfactual comparison provide unique gain?
13. Is linear decoding enough?
14. Does the token-aware comparator improve over pooled features?
15. Is READ more predictable from text delta or visual delta?
16. Is WRITE more predictable from visual delta or text delta?
17. Does the result survive within Dense-W only?
18. How does it vary by layer?
19. How does it vary by trigger-relative depth?
20. Does the random-pair control collapse?
21. Does branch-order behavior follow the intended sign convention?
22. How predictable are exact routed-state counterfactual effects?
23. Does dense-trained prediction transfer to routed states?
24. Which Case A/B/C/D is supported?
25. What does the experiment not establish?

---

## 35. `next_method_recommendation.md`

Recommend exactly one next experiment.

If Case A:

```text
one-layer counterfactual probe-and-route Stage-2
```

If Case B:

```text
one-layer post-action verification / delayed-routing controller
```

If Case C:

```text
token-aware one-layer counterfactual comparator
```

If Case D:

```text
2-layer / short-horizon counterfactual identifiability audit
```

For the recommendation state:

```text
why it is the smallest discriminating next experiment
what positive evidence would mean
what negative evidence would mean
what remains unproven
```

---

## 36. Stop rule

STOP after:

```text
1. complete 15,185-state one-step branch construction
2. exact parity/reproducibility checks
3. inherited 5-fold group-disjoint OOF
4. READ PRE/POST/PAIR/DELTA comparison
5. WRITE PRE/POST/PAIR/DELTA comparison
6. fixed-capacity linear + MLP comparison
7. token-aware comparator
8. text-vs-visual delta ablation
9. Dense-C/Dense-W conditional analysis
10. layer/trigger-relative analysis
11. random-pair and branch-order controls
12. secondary routed-state evaluation
13. exactly one next-method recommendation
```

Do not run external deployment evaluation or retrain the full routing system in this phase.

---

## 37. Core principle

The old Stage-2 asked:

```text
"Before doing READ/WRITE, can the current state tell whether it will help?"
```

The full-scale answer was:

```text
mostly no
```

This experiment asks:

```text
"If I execute the relevant alternatives for one layer,
does their immediate effect reveal which visual computation is helpful?"
```

The decisive comparison is:

```text
PRE
vs
single POST
vs
counterfactual PAIR
vs
counterfactual DELTA
```

Only if counterfactual information materially improves identifiability should the project invest in a speculative probe-and-route Stage-2.
