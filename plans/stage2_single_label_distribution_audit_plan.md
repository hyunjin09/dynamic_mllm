# Stage-2 Single-Label Distribution Audit Plan

## 1. Objective

Before training the first Stage-2 router, audit the **single-intervention corrective supervision** generated in Phase 56.

The goal is to understand exactly what Stage-2 V1 would learn from:

```text
Corpus A: triggered Dense-C FULL preservation trajectories
+
Corpus B: SINGLE_FIXABLE Dense-W single-intervention trajectories
```

This phase is analysis only.

Do **not** train Stage-2 yet.

Main question:

> **What is the structure, balance, ambiguity, and layer/action distribution of the single-intervention labels, and can they support a simple shared Stage-2 V1 without obvious collapse risks?**

---

## 2. Frozen Inputs

Use the authoritative Phase-56 outputs only.

Relevant populations:

```text
Triggered Dense-C preservation samples: 39
SINGLE_FIXABLE Dense-W samples: 698
Successful single-intervention routes: 7,628
```

Use the exact replay-validated routes and routed states already stored under:

```text
analysis/dense_failure_stage2/full_corrective_labels/
```

Do not:

```text
rerun Qwen inference
rerun corrective search
rerun MCTS
change Stage-1 trigger layers
change the Stage-1 threshold
add validation/test labels
train any Stage-2 model
```

---

## 3. Audit Units

Keep three units separate.

### 3.1 Sample-level

One triggered Dense-W sample may have many successful single routes.

Example:

```text
UID A
trigger L3

successful routes:
    L7  READ_ONLY
    L11 WRITE_ONLY
    L17 IGNORE
```

This is still one sample.

### 3.2 Route-level

Each successful single route contains exactly one non-FULL action after the trigger and FULL elsewhere.

### 3.3 State-level

A route provides one action label at every suffix layer.

Example:

```text
trigger L3
intervention L11 = WRITE_ONLY

L3  FULL
L4  FULL
...
L10 FULL
L11 WRITE_ONLY
L12 FULL
...
L27 FULL
```

Naively using every state equally can create severe FULL imbalance.

All reports must therefore distinguish:

```text
sample distribution
route distribution
state/action distribution
```

---

## 4. Successful Route Multiplicity

For each of the 698 SINGLE_FIXABLE W samples, count:

```text
# successful single routes
```

Report:

```text
mean
median
IQR
min
max
P90
P95
```

Histogram bins:

```text
1
2-3
4-7
8-15
16+
```

Main question:

> How ambiguous is the corrective target at the sample level?

A future loader must not automatically weight a sample more simply because more successful routes were discovered for it.

---

## 5. Unique Successful Intervention Positions

For each SINGLE_FIXABLE W sample, identify all successful:

```text
(intervention_layer, action)
```

pairs.

Report:

```text
# unique successful layers
# unique successful actions
# unique successful (layer, action) pairs
```

This reveals whether correction is:

```text
highly localized
or
available at many alternative layers
```

---

## 6. Action Distribution

Across successful single routes, report:

```text
READ_ONLY
WRITE_ONLY
IGNORE
```

at two levels.

### Route-weighted

Every successful route counts once.

### Sample-weighted

Each sample contributes total weight 1, divided across its successful single routes.

Primary interpretation should use the **sample-weighted** version.

---

## 7. Intervention-Layer Distribution

Report the successful intervention layer over:

```text
L0 ... L27
```

and grouped bins:

```text
Early:  0-8
Middle: 9-18
Late:   19-27
```

Produce:

```text
route-weighted layer histogram
sample-weighted layer histogram
```

Also stratify by action:

```text
READ_ONLY by layer
WRITE_ONLY by layer
IGNORE by layer
```

Question:

> Are successful one-shot corrections concentrated at specific depths, or broadly distributed?

---

## 8. Trigger-to-Intervention Delay

For every successful single route compute:

```text
delay = intervention_layer - trigger_layer
```

Report bins:

```text
0
1
2-4
5-9
10+
```

and:

```text
mean
median
IQR
P90
```

Also report by:

```text
dataset
action type
trigger-depth bin
```

This is important because Stage-2 may need to continue FULL for several layers after Stage-1 triggers before making the one corrective intervention.

---

## 9. Immediate vs Delayed Correction

For each SINGLE_FIXABLE W sample classify:

```text
IMMEDIATE_FIXABLE:
    at least one successful single intervention at the trigger layer

DELAYED_ONLY_FIXABLE:
    no successful action at the trigger layer,
    but at least one successful intervention later
```

Report counts and fractions.

This is a key architectural diagnostic.

If delayed-only is common, Stage-2 must learn:

```text
triggered does not mean intervene immediately
```

and FULL must remain a normal action after handoff.

---

## 10. Multi-Action Ambiguity at the Same Layer

For each sample/layer, check whether multiple non-FULL actions independently rescue the sample.

Observed successful sets may include:

```text
{RO}
{WO}
{IGNORE}
{RO, WO}
{RO, IGNORE}
{WO, IGNORE}
{RO, WO, IGNORE}
```

Report:

```text
single successful action at successful layer
multiple successful actions at same layer
```

Use wording:

```text
observed successful actions
```

not exhaustive validity.

---

## 11. FULL-Class Imbalance Under Naive Teacher Forcing

Construct the hypothetical state-level dataset if every layer of every successful single route were used once.

Report counts:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Compute:

```text
FULL fraction
non-FULL fraction
FULL : non-FULL ratio
```

Do this for:

```text
Corpus B alone
Corpus A alone
Corpus A + Corpus B
```

This quantifies how strongly a naive CE learner would be encouraged to predict FULL.

---

## 12. Candidate Simple Sampling Schemes — Analysis Only

Do not train.

Simulate the state/action distribution produced by:

### S0 — Naive all-state

```text
all states from chosen route
```

### S1 — 1 corrective + 2 pre-FULL + 2 post-FULL

For each W route:

```text
1 mandatory non-FULL corrective state
up to 2 randomly sampled FULL states before intervention
up to 2 randomly sampled FULL states after intervention
```

### S2 — 1 corrective + K random FULL

Test:

```text
K = 2
K = 4
K = 6
```

### S3 — Sample-balanced route sampling

For each W sample:

```text
choose one successful route uniformly
then sample states according to S1 or S2
```

Report expected class composition for each scheme.

Do not train or over-optimize a sampler in this phase.

---

## 13. Preservation-Sample Balance

Future V1 has:

```text
39 triggered-C preservation samples
698 SINGLE_FIXABLE W samples
```

For the 39 C trajectories report:

```text
trigger-layer distribution
suffix length
total available FULL states
```

Simulate simple future sample-level mixtures:

```text
C:W = 1:1
C:W = 1:2
natural frequency
```

Question:

> How much oversampling of triggered-C would be required to make preservation supervision visible during V1 training?

Do not introduce weighted losses yet.

---

## 14. Dataset Breakdown

For:

```text
GQA
ChartQA
TextVQA
```

report at minimum:

```text
# SINGLE_FIXABLE samples
successful routes/sample
sample-weighted action distribution
intervention-layer distribution
trigger-to-intervention delay
immediate vs delayed-only
```

Do not create dataset-specific Stage-2 policies.

---

## 15. Trigger-Depth Breakdown

Stratify by first Stage-1 trigger:

```text
L0
L1-L8
L9-L18
L19-L27
```

For each group report:

```text
# samples
routes/sample
action distribution
intervention layer
trigger-to-intervention delay
immediate-fixable rate
delayed-only rate
```

---

## 16. State Redundancy Audit

Because many successful routes for the same sample share long FULL prefixes/suffixes, quantify duplicated training states where exact identity or hashes are available.

Report:

```text
total stored state rows
unique state IDs/hashes
duplicate rows caused by route multiplicity
```

Do not delete anything in this phase.

This is only to quantify how much naive route expansion would over-repeat FULL supervision.

---

## 17. Candidate V1 Label Representation

Prepare, but do not train, the simplest V1 format:

```text
input:
    current routed state feature
    current layer index

target:
    FULL
    READ_ONLY
    WRITE_ONLY
    IGNORE
```

For W examples, preserve:

```text
sample ID
route ID
trigger layer
intervention layer
```

For C examples:

```text
target = FULL
```

along the safe dense suffix.

Do not yet add:

```text
MCTS Corpus C
failure latent z_fail
expected-utility heads
EMA/persistence logic
RL
```

---

## 18. Key Decision Questions

The audit must answer:

1. How many successful single routes exist per SINGLE_FIXABLE sample?
2. Are successful corrections localized to one layer or available at many layers?
3. Which action is most commonly successful: READ_ONLY, WRITE_ONLY, or IGNORE?
4. At which layers do successful single interventions occur?
5. How far after Stage-1 trigger does intervention usually happen?
6. What fraction are fixable immediately at the trigger layer?
7. What fraction are delayed-only?
8. How often are multiple actions successful at the same layer?
9. How severe is FULL imbalance under naive state-level training?
10. How severe is route-multiplicity/sample-weight imbalance?
11. How different are GQA, ChartQA, and TextVQA?
12. How different are labels across trigger-depth groups?
13. What simple sampling scheme gives a reasonable V1 state distribution?
14. How much triggered-C oversampling appears necessary?
15. Is the single-intervention corpus structurally suitable for a simple shared Stage-2 V1?

---

## 19. Required Metrics

Create:

```text
sample_route_multiplicity.csv
successful_pair_multiplicity.csv
action_distribution_route_weighted.csv
action_distribution_sample_weighted.csv
intervention_layer_distribution.csv
action_by_layer.csv
trigger_to_intervention_delay.csv
immediate_vs_delayed.csv
same_layer_action_ambiguity.csv
naive_state_class_balance.csv
sampling_scheme_simulation.csv
preservation_balance_simulation.csv
dataset_breakdown.csv
trigger_depth_breakdown.csv
state_redundancy.csv
```

---

## 20. Required Figures

Create:

```text
routes_per_sample_histogram.png
single_intervention_layer_distribution.png
single_action_by_layer.png
trigger_to_intervention_delay.png
immediate_vs_delayed.png
naive_action_imbalance.png
sampling_scheme_balance.png
dataset_action_distribution.png
dataset_delay_distribution.png
trigger_depth_delay_distribution.png
```

---

## 21. Required Outputs

Use:

```text
analysis/dense_failure_stage2/single_label_audit/
```

Create:

```text
protocol.md

metrics/
    sample_route_multiplicity.csv
    successful_pair_multiplicity.csv
    action_distribution_route_weighted.csv
    action_distribution_sample_weighted.csv
    intervention_layer_distribution.csv
    action_by_layer.csv
    trigger_to_intervention_delay.csv
    immediate_vs_delayed.csv
    same_layer_action_ambiguity.csv
    naive_state_class_balance.csv
    sampling_scheme_simulation.csv
    preservation_balance_simulation.csv
    dataset_breakdown.csv
    trigger_depth_breakdown.csv
    state_redundancy.csv

figures/
    routes_per_sample_histogram.png
    single_intervention_layer_distribution.png
    single_action_by_layer.png
    trigger_to_intervention_delay.png
    immediate_vs_delayed.png
    naive_action_imbalance.png
    sampling_scheme_balance.png
    dataset_action_distribution.png
    dataset_delay_distribution.png
    trigger_depth_delay_distribution.png

summaries/
    single_label_audit_summary.md
    stage2_v1_data_recommendation.md

artifact_manifest.json
```

---

## 22. `stage2_v1_data_recommendation.md`

At the end, recommend only the **simplest reasonable V1 data-loading contract** justified by the audit.

It may specify:

```text
how to sample one successful route per W sample
how many FULL states to sample around the corrective state
how to mix triggered-C preservation samples
whether multi-valid actions should be retained now or deferred
```

Prefer simplicity unless the audit shows a clear blocking issue.

Do not recommend a complex architecture.

Do not train anything.

---

## 23. Stop Rule

STOP after the single-label distribution audit and V1 data recommendation.

Do not:

```text
train Stage-2
add MCTS Corpus C
run free rollout
change Stage-1
rerun search
retune thresholds
use validation/test supervision
```

The next separately authorized phase should be:

```text
Stage-2 V1 training pilot
=
Corpus A preservation
+
Corpus B single-intervention correction
```

with a simple shared four-action head and a sampling scheme justified by this audit.

---

## 24. Interpretation Boundary

This audit can establish:

```text
what the discovered single-intervention supervision looks like
how imbalanced it is
how ambiguous it is
how delayed correction is relative to trigger
whether simple V1 training is structurally plausible
```

It cannot establish:

```text
that Stage-2 will generalize
that a learned router will preserve C→C
that single supervision is sufficient
that MCTS supervision is unnecessary
```

Those require actual Stage-2 training and free rollout.
