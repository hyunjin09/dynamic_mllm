# Predictability & Generalization Phase — Step A
## Full-Scale Measurement Construction for Stage-1 Failure and Stage-2 Visual-Computation Utility

## 0. Purpose

This step does **not** train a new router and does **not** evaluate predictability yet.

Its only purpose is to construct the full-scale measurement datasets needed to ask later:

```text
Stage-1:
Can the current hidden state predict whether the all-FULL trajectory will eventually fail?

Stage-2:
Can the current hidden state predict whether READ or WRITE at the current layer
will help or hurt the final answer?
```

The key distinction is:

```text
oracle opportunity exists
≠
oracle opportunity is predictable
```

No small scientific subset is used. Small runs are allowed only for implementation/parity checks.

---

## 1. Scope

Step A has two tracks.

### A1 — Stage-1 failure measurement

Construct a full dense state census for the internal Historical + Canonical population.

Target:

```text
current dense hidden state
→ eventual all-FULL final correctness
```

### A2 — Stage-2 visual-computation utility measurement

For every post-trigger state in the full internal P90 Stage-2 decision domain, measure the controlled effect of:

```text
READ
WRITE
READ × WRITE interaction
```

by executing all four current-layer actions from the **same exact entering state** and then using the **same all-FULL suffix**.

No MCTS is used to define this utility.

---

## 2. Frozen execution contract

Freeze:

```text
Qwen2.5-VL-7B-Instruct
28 decoder layers
robust ALL-source Stage-1
P90 trigger threshold
four-action semantics
same-layer READ semantics
existing prompts / decoding / correctness contracts
```

Actions:

```text
FULL       = READ1 WRITE1
READ_ONLY  = READ1 WRITE0
WRITE_ONLY = READ0 WRITE1
IGNORE     = READ0 WRITE0
```

Do not change the model, trigger, action semantics, or evaluator during Step A.

---

## 3. Internal population

Use the full currently established internal population from:

```text
GQA
ChartQA
TextVQA
```

across the eligible Historical + Canonical sources/manifests.

For Stage-2, do **not** restrict to:

```text
the 569 successful-route UIDs
MCTS-success samples
single-fixable samples
```

The primary Stage-2 population is:

> **all internal samples for which robust Stage-1 P90 triggers, regardless of whether previous corrective search succeeded.**

This avoids conditioning the predictability study on prior route-search success.

---

## 4. Stable identity and metadata

Every sample/state must preserve:

```text
UID
image/content group ID
dataset
source regime
question text / question ID
image ID
Dense correctness
P90 trigger status
P90 first-trigger layer
```

Also retain nuisance metadata for later controls:

```text
layer
trigger-relative depth
visual-token count
text-token count
question length
answer length if applicable
```

These fields are metadata only, not primary model inputs.

---

# Track A1 — Stage-1 measurement

## 5. Full dense state census

For every internal sample, execute the canonical all-FULL trajectory.

For every layer:

```text
l = 0 ... 27
```

save/reference the exact current state representation needed to reproduce the existing Stage-1 input and to train later probes.

Unit:

```text
(sample, layer)
```

---

## 6. Stage-1 target

For sample `i`:

```text
y_fail(i) = 1
if the final all-FULL answer is wrong

y_fail(i) = 0
if the final all-FULL answer is correct
```

Every layer-state of that sample inherits the same eventual-failure label.

Do not use the Stage-1 trigger prediction itself as the target.

The scientific question later is:

> At what depth does the current model state begin to encode eventual dense failure?

---

## 7. Stage-1 feature storage

At minimum preserve:

```text
current text/control hidden state used by the Stage-1 head
the established Stage-1 input representation needed for exact reproduction
layer identity as metadata
```

Prefer raw frozen-backbone features/reference over only storing a trained head's final score.

This keeps Step B able to compare:

```text
linear probe
small MLP
current Stage-1 head
```

without regenerating the full corpus.

---

## 8. Stage-1 parity

Require:

```text
final dense answer == frozen dense baseline answer
Dense C/W counts == frozen manifest counts
existing robust Stage-1 score reproducible where applicable
```

Report every mismatch.

Step A is not ready if unexplained parity mismatches remain.

---

# Track A2 — Stage-2 measurement

## 9. Primary dense post-trigger state domain

For every internal sample where P90 first triggers at `L*`, construct the dense pre-action state at each layer:

```text
l = L* ... 27
```

with:

```text
all previous layers = FULL
```

This is the primary Stage-2 measurement domain.

The exact state is the state **before** choosing the current layer action.

---

## 10. Why dense post-trigger states are primary

At one exact dense state `s_l` we can compare four branches with:

```text
same input
same prefix
same current state
same future continuation
```

and only the current READ/WRITE choice changes.

This gives a controlled local-effect measurement.

It avoids the MCTS confound:

```text
different current action
+
different future corrective suffix
```

---

## 11. Four-branch counterfactual execution

For exact state `s_l`, execute:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

at layer `l`.

For each branch:

```text
layers l+1 ... 27 = FULL
```

Define:

```text
q_F(s_l)
q_RO(s_l)
q_WO(s_l)
q_I(s_l)
```

No later router.
No MCTS.
No suffix search.

---

## 12. Continuous answer-quality score

For every branch record a continuous ground-truth answer score.

Primary recommendation:

```text
token-normalized log-likelihood of the evaluator-compatible gold answer
```

For multiple acceptable references/aliases, use a frozen documented rule, such as:

```text
max token-normalized log-likelihood
over evaluator-normalized acceptable references
```

Do not silently vary the scoring rule across datasets.

Also record:

```text
exp(delta_q)
```

where useful so later analysis can express the effect as a multiplicative change in average per-token answer probability.

---

## 13. Discrete final correctness

Under the same frozen generation/evaluator contract also record:

```text
c_F  ∈ {0,1}
c_RO ∈ {0,1}
c_WO ∈ {0,1}
c_I  ∈ {0,1}
```

Continuous utility is the primary dense measurement.

Correctness flips are secondary, directly interpretable behavioral events.

---

## 14. Controlled READ effects

With WRITE on:

```text
U_R|W=1 = q_F - q_WO
```

because:

```text
FULL       = R1W1
WRITE_ONLY = R0W1
```

With WRITE off:

```text
U_R|W=0 = q_RO - q_I
```

Sign:

```text
positive → enabling READ helps
negative → enabling READ hurts
```

---

## 15. Controlled WRITE effects

With READ on:

```text
U_W|R=1 = q_F - q_RO
```

With READ off:

```text
U_W|R=0 = q_WO - q_I
```

Sign:

```text
positive → enabling WRITE helps
negative → enabling WRITE hurts
```

---

## 16. 2×2 factorial decomposition

Store context-averaged main effects:

```text
U_READ
=
0.5 * [
    (q_F  - q_WO)
  + (q_RO - q_I)
]
```

```text
U_WRITE
=
0.5 * [
    (q_F  - q_RO)
  + (q_WO - q_I)
]
```

and interaction:

```text
U_INT
=
q_F - q_RO - q_WO + q_I
```

Interpretation:

```text
U_INT ≈ 0
→ READ and WRITE effects are approximately additive

large |U_INT|
→ utility of one operation depends strongly on whether the other is enabled
```

Do not discard the conditional effects; store both conditional and main-effect forms.

---

## 17. Strong correctness-flip labels

Record controlled flips.

### READ with WRITE=1

```text
READ_HARMFUL_FLIP_W1:
    c_F=0 and c_WO=1

READ_BENEFICIAL_FLIP_W1:
    c_F=1 and c_WO=0
```

### READ with WRITE=0

```text
READ_HARMFUL_FLIP_W0:
    c_RO=0 and c_I=1

READ_BENEFICIAL_FLIP_W0:
    c_RO=1 and c_I=0
```

### WRITE with READ=1

```text
WRITE_HARMFUL_FLIP_R1:
    c_F=0 and c_RO=1

WRITE_BENEFICIAL_FLIP_R1:
    c_F=1 and c_RO=0
```

### WRITE with READ=0

```text
WRITE_HARMFUL_FLIP_R0:
    c_WO=0 and c_I=1

WRITE_BENEFICIAL_FLIP_R0:
    c_WO=1 and c_I=0
```

---

## 18. Additional descriptive state labels

For each exact state record:

```text
best_action_by_q
best_q
FULL_gap = max(q_F,q_RO,q_WO,q_I) - q_F

local_rescue_exists =
    c_F=0 and any(c_RO,c_WO,c_I)=1

local_regression_exists =
    c_F=1 and any(c_RO,c_WO,c_I)=0

all_four_correct
all_four_wrong
# correct actions among four
```

These are descriptive only.

Do not define the Step-B prediction task from `best_action_by_q` yet.

---

## 19. Stage-2 feature storage

For every primary dense state preserve/reference the raw current state needed later.

At minimum:

```text
current text/query hidden state
current visual-token hidden states
visual token span/index metadata
layer
trigger layer
trigger-relative depth
```

Where practical, also cache reproducible:

```text
z_R
z_W
[z_R ; z_W]
```

but preserve the raw state so Step B is not forced to use a previously trained router representation.

---

# Secondary routed-state extension

## 20. Full exact routed-state utility corpus

After primary dense-state parity passes, perform the same four-branch utility measurement on the full existing exact routed-state corpus.

Current available scale is approximately:

```text
35,565 unique exact routed states
```

For each routed state:

```text
preserve its exact routed prefix
force one of the four current actions
then FULL for all later layers
```

Measure the same:

```text
q_F, q_RO, q_WO, q_I
U_READ
U_WRITE
U_INT
correctness flips
```

This asks:

> After routing has already altered the hidden state, is local visual-computation utility still measurable/predictable?

Keep this corpus separate from the primary dense-state corpus.

---

## 21. Exact routed-state identity

Identify a routed state by:

```text
UID
layer
exact action prefix
state hash/reference
```

Do not merge states with the same UID/layer if previous actions differ.

If the exact state appears in several trajectories, evaluate its four branches once.

---

# Validation

## 22. FULL-branch dense parity

For every primary dense Stage-2 state:

```text
FULL at current layer
+ FULL suffix
```

must reproduce the canonical dense baseline.

Require parity for:

```text
decoded answer
correctness
continuous answer score within frozen tolerance
```

This is the strongest branch-execution sanity check.

---

## 23. Action-semantic checks

Verify exactly:

```text
FULL       → R1W1
READ_ONLY  → R1W0
WRITE_ONLY → R0W1
IGNORE     → R0W0
```

At layer 27 there is no later suffix; evaluate the four current-layer actions directly.

---

## 24. Algebra checks

Recompute every derived utility from raw q-values and require exact/tolerance-bounded equality:

```text
U_R|W=1
U_R|W=0
U_W|R=1
U_W|R=0
U_READ
U_WRITE
U_INT
```

Do not trust precomputed derived columns without this check.

---

## 25. Reproducibility

On a frozen deterministic parity set, rerun all four branches twice.

Require:

```text
same decoded answer
same correctness
same q within frozen numerical tolerance
```

No stochastic decoding is allowed in measurement construction.

---

# Execution order

## 26. Run Step A sequentially

```text
A1. freeze protocol/manifests
A2. implementation smoke + parity
A3. full Stage-1 dense-state census
A4. full P90-trigger manifest
A5. full primary dense post-trigger Stage-2 utility generation
A6. parity + distribution audit
A7. full routed-state utility extension
A8. freeze Step-A artifacts
```

Do not begin Step B predictor training before Step A is frozen.

---

## 27. No outcome-driven filtering

Keep states even if:

```text
utility ≈ 0
all four actions are correct
all four actions are wrong
no correctness flip occurs
previous MCTS failed
previous MCTS succeeded
```

Step A must preserve the raw full distribution.

---

## 28. Label-distribution summary

Before any predictor training, report Stage-1:

```text
# samples
# layer states
Dense C/W prevalence
per-dataset/source prevalence
```

For Stage-2 report:

```text
# triggered UIDs
# exact dense states
# four-action branch evaluations

U_READ mean/median/std/quantiles
U_WRITE mean/median/std/quantiles
U_INT mean/median/std/quantiles
```

Also report:

```text
READ positive/negative sign counts
WRITE positive/negative sign counts
correctness-flip counts
local rescue counts
local regression counts
all-four-correct
all-four-wrong
```

Break down by:

```text
dataset
source regime
layer
Dense C/W
trigger-relative depth
```

---

## 29. Do not threshold utility yet

Do not choose a data-driven epsilon during Step A.

Store continuous utility exactly.

For descriptive summaries, sign is enough:

```text
U > 0
U < 0
```

Any later high-confidence threshold must be defined prospectively in Step B using training/development data.

---

## 30. Compute logging

Record:

```text
Stage-1 dense forwards
Stage-1 layer-state count
Stage-2 dense exact states
Stage-2 routed exact states
four-action branches
suffix-layer executions
generation/scoring calls
GPU-hours
wall time
cache/storage size
```

Report dense and routed Stage-2 costs separately.

---

# Required datasets

## 31. S1 — Stage-1 full failure census

Unit:

```text
(sample, layer)
```

Target:

```text
eventual all-FULL failure
```

## 32. S2-Dense — Primary Stage-2 utility census

Unit:

```text
(P90-triggered sample, layer, exact dense state)
```

Targets:

```text
q_F/q_RO/q_WO/q_I
conditional READ/WRITE effects
U_READ
U_WRITE
U_INT
correctness flips
```

## 33. S2-Routed — Secondary exact routed-state utility census

Unit:

```text
exact routed state
```

Same utility targets as S2-Dense.

---

# Required artifacts

## 34. Output tree

Use:

```text
analysis/predictability_generalization/stepA_measurement/
```

Create:

```text
protocol.md
frozen_contract.json

manifests/
    internal_sample_manifest.jsonl
    group_registry.jsonl
    p90_trigger_manifest.jsonl

stage1/
    dense_state_manifest.jsonl
    dense_outcome_labels.csv
    state_feature_manifest.jsonl
    parity_report.md
    stage1_census_summary.csv

stage2_dense/
    exact_state_manifest.jsonl
    four_branch_results.jsonl
    utility_labels.csv
    correctness_flip_labels.csv
    state_feature_manifest.jsonl
    parity_report.md
    utility_distribution_summary.csv
    layer_breakdown.csv
    dataset_source_breakdown.csv

stage2_routed/
    exact_state_manifest.jsonl
    routed_state_dedup.csv
    four_branch_results.jsonl
    utility_labels.csv
    correctness_flip_labels.csv
    state_feature_manifest.jsonl
    parity_report.md
    utility_distribution_summary.csv

validation/
    action_semantics_tests.md
    full_branch_dense_parity.csv
    repeated_execution_reproducibility.csv
    utility_algebra_checks.csv

compute/
    compute_summary.csv
    storage_summary.csv

summaries/
    stepA_measurement_summary.md
    stepB_readiness.md

artifact_manifest.json
```

---

## 35. `stepA_measurement_summary.md` must answer

1. How many internal samples were included?
2. How many Stage-1 `(sample, layer)` states were constructed?
3. Did dense baseline parity pass?
4. How many samples P90-triggered?
5. How many primary dense post-trigger Stage-2 states were measured?
6. How many four-action branches were executed?
7. What are the distributions of `U_READ`, `U_WRITE`, and `U_INT`?
8. How often is READ beneficial versus harmful by sign?
9. How often is WRITE beneficial versus harmful by sign?
10. How many strong correctness flips exist for READ and WRITE?
11. How many local single-layer rescue opportunities exist?
12. How do utility distributions vary by layer?
13. How do they vary by dataset/source?
14. How do they differ between Dense-C and Dense-W?
15. How many exact routed states were additionally measured?
16. Does routed-state utility distribution differ from dense-state utility?
17. Did action semantics, FULL parity, reproducibility, and algebra checks all pass?
18. What measurement limitations remain?

Do not make a predictability claim in Step A.

---

## 36. `stepB_readiness.md`

Step B may begin only if:

```text
Stage-1 dense parity passed
Stage-2 FULL-branch dense parity passed
four-action semantics passed
continuous utility is reproducible
all required sample/group identities are preserved
no unexplained large missing population exists
```

State explicitly:

```text
READY_FOR_STEP_B = true / false
```

If false, list only measurement/construction issues.

Do not propose router changes here.

---

## 37. Stop rule

STOP after:

```text
1. full Stage-1 measurement census
2. full primary dense post-trigger Stage-2 counterfactual census
3. full secondary exact routed-state utility census
4. parity/reproducibility validation
5. full label-distribution report
6. Step-B readiness decision
```

Do not train:

```text
linear probes
MLPs
router heads
OOD predictors
```

during Step A.

---

## 38. Core principle

The project motivation is:

> **Oracle search shows that suppressing some visual computation can turn a wrong answer into a correct one.**

Step A converts that observation into controlled full-scale measurements.

Stage-1 measures:

```text
Does the current dense state precede eventual failure?
```

Stage-2 measures:

```text
Holding the current state and future continuation fixed,
what is the effect of enabling READ or WRITE right now?
```

Only after these measurements are frozen should Step B ask whether those signals are actually predictable from the hidden state.
