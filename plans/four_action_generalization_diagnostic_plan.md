# Four-Action Router Generalization Failure: Mechanism Diagnostic Plan

## 1. Goal

Both current four-action routing substrates can produce non-`FULL` actions under stronger supervision, but held-out W2C rescue remains very low.

The next phase should **not** immediately modify the router architecture or training objective.

The goal is to determine **what exactly fails to generalize**:

1. Does the router fail to recognize **when it must leave `FULL`**?
2. Once it decides to leave `FULL`, does it fail to choose the correct **READ / WRITE / BOTH suppression mechanism**?
3. Is one mechanism specifically harder to generalize?
   - READ suppression
   - WRITE suppression
   - BOTH suppression
4. Is the router mostly using **layer/depth priors** rather than sample-specific hidden-state information?
5. Are the current representations actually predictive of the corrective mechanism?
6. Are the four-action oracle labels themselves too multimodal, sample-specific, or incomplete to define a stable learned mapping?

The analysis should proceed in the order:

```text
WHEN
→ WHAT
→ WHY
```

where:

```text
WHEN = should the model KEEP FULL or DEVIATE now?
WHAT = if it must deviate, which mechanism should be suppressed?
WHY  = representation failure, layer-prior shortcut, label inconsistency, or search incompleteness?
```

No new large training run should begin until this diagnostic is complete.

---

## 2. Frozen Inputs

Use the existing completed matched persistent-supervision checkpoints and frozen splits.

Analyze both:

```text
POLAR-style upfront router
Online state-conditioned router
```

Primary selected checkpoints:

```text
POLAR selected epoch: 15
Online selected epoch: 14
```

Also retain access to all epoch checkpoints for supporting trend analysis if needed.

Use the same held-out validation population already used in the matched comparison.

Do not run external evaluation.

---

## 3. Core Diagnostic Decomposition

The current four-action space is:

```text
FULL        = READ ON,  WRITE ON
READ_ONLY   = READ ON,  WRITE OFF
WRITE_ONLY  = READ OFF, WRITE ON
IGNORE      = READ OFF, WRITE OFF
```

The analysis should explicitly separate:

### Stage 1 — WHEN

Binary decision:

```text
KEEP_FULL
vs
DEVIATE
```

This asks:

> Is `FULL` still safe/valid at the current state, or must visual computation be modified now?

### Stage 2 — WHAT

Conditional on deviation being required:

```text
WRITE_OFF only  -> READ_ONLY
READ_OFF only   -> WRITE_ONLY
BOTH_OFF        -> IGNORE
```

This asks:

> If `FULL` must stop, which visual operation is answer-unaligned?

Do not conflate Stage 1 and Stage 2.

---

## 4. Diagnostic Dataset Construction

### 4.1 Positive WHEN states

Use W2C mandatory-deviation boundary nodes.

At each such node:

```text
FULL is invalid by construction.
```

Label:

```text
DEVIATE = 1
```

Store:

```text
uid
dataset
layer
prefix
valid action set
singleton/multi-valid status
router logits/probabilities
```

### 4.2 Negative WHEN states

Construct clean negative nodes from W2C correcting trajectories where:

```text
FULL is the unique valid next action
```

Label:

```text
DEVIATE = 0
```

Prefer matching positives and negatives by:

```text
dataset
layer
split
```

so that a trivial layer prior cannot solve the task.

If exact matching is impossible, record imbalance and use stratified metrics.

### 4.3 WHAT states

Restrict initially to mandatory-deviation states.

Create two views.

#### Singleton-only clean subset

Nodes where exactly one next action is valid:

```text
READ_ONLY only
WRITE_ONLY only
IGNORE only
```

These are the cleanest mechanism labels.

#### Multi-valid subset

Nodes where multiple non-`FULL` actions are valid.

Keep these separate.

Do not force a single arbitrary mechanism label.

---

## 5. Analysis A — WHEN: Can the Router Detect That FULL Must Stop?

This is the first and most important analysis.

For each router, evaluate:

```text
P(DEVIATE) = 1 - P(FULL)
```

at matched positive/negative states.

Report:

```text
AUROC
AUPRC
accuracy
balanced accuracy
precision
recall
F1
false-positive rate
false-negative rate
```

Break down by:

```text
train vs validation
dataset
layer
early/middle/late depth
```

Main question:

> Is the dominant failure that the router cannot generalize the timing of deviation?

---

## 6. Analysis B — WHAT: Which Suppression Mechanism Is Missed?

At mandatory-deviation states, analyze the predicted action conditioned on:

```text
predicted action != FULL
```

Report:

```text
Conditional Valid-Action@1
=
P(predicted action is valid | predicted action != FULL)
```

Also report unconditional Valid-Action@1.

Interpretation:

```text
low DEVIATE recall
high conditional Valid-Action@1
```

means the router usually knows what to do once it intervenes, but fails to know when.

Conversely:

```text
high DEVIATE recall
low conditional Valid-Action@1
```

means timing is learned but READ/WRITE mechanism selection is poor.

---

## 7. Analysis C — Singleton Four-Way Confusion Matrix

Use only nodes with a single valid action.

Ground-truth classes:

```text
FULL only
READ_ONLY only
WRITE_ONLY only
IGNORE only
```

Produce train and validation confusion matrices for:

```text
POLAR
Online
```

Report per-class:

```text
support
precision
recall
F1
predicted FULL fraction
```

Interpret classes mechanistically:

```text
READ_ONLY only
= WRITE must be suppressed

WRITE_ONLY only
= READ must be suppressed

IGNORE only
= both READ and WRITE must be suppressed
```

This should directly answer whether the router specifically fails on READ suppression, WRITE suppression, BOTH suppression, or FULL preservation.

---

## 8. Analysis D — READ-OFF and WRITE-OFF Bit Metrics

Convert four-action probabilities into bit-level probabilities.

Definitions:

```text
P(READ_OFF)
=
P(WRITE_ONLY) + P(IGNORE)

P(WRITE_OFF)
=
P(READ_ONLY) + P(IGNORE)
```

Report separately:

```text
READ_OFF AUROC
READ_OFF AUPRC
READ_OFF recall
READ_OFF precision

WRITE_OFF AUROC
WRITE_OFF AUPRC
WRITE_OFF recall
WRITE_OFF precision
```

Report train and validation.

Primary question:

> Is one visual operation systematically less predictable across inputs?

---

## 9. Analysis E — BOTH-OFF / Interaction Failure

`IGNORE-only` nodes are special because both visual operations must be suppressed.

For singleton `IGNORE-only` states, break wrong predictions into:

```text
FULL
READ_ONLY
WRITE_ONLY
```

Interpretation:

- `FULL`: failed to detect either harmful operation.
- `READ_ONLY`: detected WRITE suppression but missed READ suppression.
- `WRITE_ONLY`: detected READ suppression but missed WRITE suppression.

Report:

```text
IGNORE-only recall
partial-suppression rate
FULL-error rate
```

where:

```text
partial suppression
=
P(pred in {READ_ONLY, WRITE_ONLY} | target IGNORE-only)
```

---

## 10. Analysis F — Train-to-Validation Generalization Gap by Mechanism

Produce:

| Decision | Train | Validation | Gap |
|---|---:|---:|---:|
| KEEP vs DEVIATE AUROC | | | |
| READ_OFF AUROC | | | |
| WRITE_OFF AUROC | | | |
| READ_ONLY-only recall | | | |
| WRITE_ONLY-only recall | | | |
| IGNORE-only recall | | | |
| Conditional valid-action given deviation | | | |

Do this separately for:

```text
POLAR
Online
```

If one mechanism has a much larger train→validation collapse, that mechanism is the main generalization bottleneck.

If all mechanisms collapse similarly, the issue is likely broader than READ-vs-WRITE specialization.

---

## 11. Analysis G — First-Deviation Timing Error

For W2C free rollout, compare:

```text
l_pred
=
first predicted non-FULL layer

l_boundary
=
mandatory all-FULL-prefix boundary
```

Categorize each sample:

```text
too early
exact
within +/-1
within +/-2
too late
never deviates
```

Cross-tabulate timing errors with eventual W2C rescue.

Questions:

1. Are rescued samples mostly those whose first deviation is near the boundary?
2. Does Online fail because it deviates too early?
3. Does POLAR fail because it never deviates?
4. Are wrong mechanisms concentrated in early or late deviations?

---

## 12. Analysis H — Does the Router Use Sample-Specific State or Mostly Layer Identity?

This is critical because both architectures contain strong layer-specific learned parameters.

### 12.1 Layer-only baseline

Build a simple train-only baseline using:

```text
layer index
```

to predict:

```text
KEEP/DEVIATE
singleton four-action class
READ_OFF
WRITE_OFF
```

Evaluate on validation.

Compare against the full router.

If the full router barely exceeds this baseline, it is likely relying heavily on depth priors.

### 12.2 Within-layer state shuffle

For the online router, within each layer and dataset, shuffle:

```text
q_l
V_l
```

across samples while preserving:

```text
layer identity
```

Then recompute router predictions.

Measure:

```text
fraction of predictions unchanged
KL divergence between original and shuffled action distributions
drop in WHEN AUROC
drop in READ_OFF AUROC
drop in WRITE_OFF AUROC
drop in singleton Valid-Action@1
```

If shuffling sample states barely changes outputs:

> the router is not meaningfully using sample-specific hidden-state information.

Use deterministic shuffle seeds.

---

## 13. Analysis I — Representation Probes

Use frozen representations from the trained online router.

### 13.1 READ representation

Use:

```text
z_R
```

to predict:

```text
READ_OFF required?
WRITE_OFF required?
KEEP vs DEVIATE?
```

### 13.2 WRITE representation

Use:

```text
z_W
```

to predict:

```text
WRITE_OFF required?
READ_OFF required?
KEEP vs DEVIATE?
```

Use:

```text
linear probe
small MLP probe
```

with matched capacity.

Do not fine-tune the backbone/router.

Report validation AUROC/AUPRC.

Desired architecture-consistent pattern:

```text
z_R -> READ_OFF
better than
z_R -> WRITE_OFF

z_W -> WRITE_OFF
better than
z_W -> READ_OFF
```

If this specialization does not appear, the intended READ/WRITE factorization may not be reflected in the learned representations.

---

## 14. Analysis J — Hidden-State Predictability vs Head/Objective Failure

This separates representation failure from classifier/head failure.

For the frozen features used by each architecture:

1. train a fresh diagnostic binary probe for:
   - KEEP vs DEVIATE
   - READ_OFF
   - WRITE_OFF
2. train a fresh singleton 3-way mechanism probe:
   - READ_ONLY
   - WRITE_ONLY
   - IGNORE

Compare probe performance against the trained router's own outputs.

Interpretation:

### Probe also near chance

> The representation does not contain a transferable corrective signal.

### Probe substantially better than router

> The signal exists, but the router head/objective/training fails to use it.

This distinction is required before changing architecture.

---

## 15. Analysis K — Label Consistency / Learnability

Oracle-valid routes may be highly multimodal and sample-specific.

Test whether similar states receive consistent corrective labels.

### 15.1 kNN label agreement

Choose representation spaces:

```text
upfront feature
online current hidden-state feature
z_R
z_W
```

For each validation state:

1. find k nearest training states, preferably matched by dataset and layer;
2. calculate agreement for:
   - KEEP/DEVIATE
   - READ_OFF
   - WRITE_OFF
   - singleton mechanism

Recommended:

```text
k = 5, 10, 20
```

Report:

```text
neighbor label purity
majority-vote accuracy
distance-vs-label-agreement relationship
```

If nearest states have inconsistent labels, exact corrective actions may not define a smooth/generalizable function.

### 15.2 Same-layer mechanism entropy

For each layer, compute:

```text
H(action | layer)
H(READ_OFF | layer)
H(WRITE_OFF | layer)
```

and compare with:

```text
H(action | layer, dataset)
```

This measures how much structure is explained by depth alone.

---

## 16. Analysis L — Label Incompleteness Audit

The valid action set comes from discovered correcting routes.

Therefore:

```text
label-invalid
```

may mean:

```text
truly invalid
```

or:

```text
not discovered by search/conversion
```

Target held-out W2C states where:

```text
router predicted non-FULL
prediction is not in the cached valid action set
```

Stratify by:

```text
READ_ONLY prediction
WRITE_ONLY prediction
IGNORE prediction
```

For a bounded subset:

1. execute the predicted action at the exact state;
2. test the known suffix where meaningful;
3. if necessary, run a small bounded continuation search;
4. determine whether the supposedly invalid action can actually lead to a correct final answer.

Report:

```text
cached-invalid but execution-correct fraction
by action type
```

Keep this audit small and targeted.

---

## 17. Priority Order

### Priority 1 — No new training required

Using current checkpoints and labels:

```text
A. WHEN: KEEP vs DEVIATE
B. conditional WHAT given deviation
C. singleton confusion matrix
D. READ_OFF / WRITE_OFF bit metrics
E. BOTH_OFF error decomposition
F. train-vs-validation mechanism gaps
G. first-deviation timing
```

Complete these first.

### Priority 2 — Cheap diagnostic probes

Then:

```text
H. layer-only baseline
I. within-layer state shuffle
J. representation probes
K. kNN label consistency
```

### Priority 3 — Bounded execution audit

Only afterward:

```text
L. label incompleteness audit
```

Do not start a new full training run before Priority 1 and Priority 2 are complete.

---

## 18. Main Decision Tree

### Case 1 — WHEN is the dominant failure

Pattern:

```text
KEEP/DEVIATE validation poor
conditional WHAT relatively strong
```

Interpretation:

> The main problem is identifying when intervention is necessary.

Possible future direction:

```text
two-stage router:
CONTINUE vs DEVIATE
then mechanism selection
```

Do not implement yet.

### Case 2 — WHEN is good, WHAT is poor

Pattern:

```text
high deviation recall
low conditional valid-action accuracy
```

Interpretation:

> The router knows intervention is needed but cannot identify READ vs WRITE vs BOTH.

### Case 3 — WRITE suppression specifically fails

Pattern:

```text
READ_OFF generalizes
WRITE_OFF does not
```

Inspect later:

```text
WRITE visual pooling
WRITE-specific layer query
features predictive of visual-state degradation
```

### Case 4 — READ suppression specifically fails

Pattern:

```text
WRITE_OFF generalizes
READ_OFF does not
```

Inspect later:

```text
READ query representation
text state choice
cross-attention summary
```

### Case 5 — BOTH_OFF specifically fails

Pattern:

```text
single-component suppression works
IGNORE-only recall poor
partial suppression high
```

Inspect the joint interaction modeling.

### Case 6 — Layer-only baseline nearly matches full router

Interpretation:

> The router is learning depth priors rather than sample-specific corrective signals.

### Case 7 — Probes succeed but trained router fails

Interpretation:

> Information exists in the representation, but the objective/head/training does not exploit it.

### Case 8 — Probes and kNN both fail

Interpretation:

> Corrective route labels may be highly sample-specific or non-smooth in the current feature space.

Potential later directions:

```text
coarser routing targets
route clustering
latent program families
confidence-based routing
search-at-inference for ambiguous cases
```

### Case 9 — Label incompleteness is substantial

Interpretation:

> The router may be penalized for valid actions absent from the discovered route cache.

Repair supervision before further architecture comparisons.

---

## 19. Required Tables

### Table 1 — Main failure decomposition

| Metric | POLAR Train | POLAR Val | Online Train | Online Val |
|---|---:|---:|---:|---:|
| KEEP vs DEVIATE AUROC | | | | |
| DEVIATE recall | | | | |
| Conditional Valid-Action@1 | | | | |
| READ_OFF AUROC | | | | |
| WRITE_OFF AUROC | | | | |
| READ_ONLY-only recall | | | | |
| WRITE_ONLY-only recall | | | | |
| IGNORE-only recall | | | | |

### Table 2 — Singleton confusion matrix

Produce separately for POLAR and Online.

### Table 3 — Timing

| Metric | POLAR | Online |
|---|---:|---:|
| exact first deviation | | |
| within +/-1 | | |
| within +/-2 | | |
| too early | | |
| too late | | |
| never deviates | | |
| rescue given near-boundary deviation | | |

### Table 4 — Representation usage

| Diagnostic | POLAR | Online |
|---|---:|---:|
| Layer-only KEEP/DEVIATE AUROC | | |
| Full-router KEEP/DEVIATE AUROC | | |
| State-shuffle prediction unchanged | N/A | |
| State-shuffle AUROC drop | N/A | |
| READ probe READ_OFF AUROC | N/A | |
| WRITE probe WRITE_OFF AUROC | N/A | |

### Table 5 — Label learnability

| Representation | kNN KEEP/DEVIATE agreement | READ_OFF agreement | WRITE_OFF agreement | mechanism agreement |
|---|---:|---:|---:|---:|
| upfront | | | | |
| online | | | | |
| z_R | | | | |
| z_W | | | | |

---

## 20. Required Figures

Create only figures that answer a diagnostic question:

1. Train vs validation mechanism performance.
2. READ_OFF and WRITE_OFF AUROC by layer.
3. Four-action singleton confusion matrix.
4. First-deviation error histogram.
5. W2C rescue probability vs deviation-layer error.
6. Layer-only vs full-state performance.
7. State-shuffle effect on online predictions.
8. kNN distance vs corrective-label agreement.

---

## 21. Required Outputs

Suggested root:

```text
analysis/4action_generalization_diagnostics/
```

Create:

```text
diagnostic_protocol.md
state_manifest.jsonl

when_keep_vs_deviate.csv
what_conditional_mechanism.csv
singleton_confusion_polar.csv
singleton_confusion_online.csv
read_write_bit_metrics.csv
both_off_error_breakdown.csv
train_val_gap.csv
first_deviation_analysis.csv

layer_only_baseline.json
state_shuffle_results.json
representation_probe_results.json
knn_label_consistency.json

label_incompleteness_subset.jsonl
label_incompleteness_results.json

figures/

diagnostic_summary.md
decision_summary.md
```

---

## 22. Final Questions the Report Must Answer

### Q1 — WHEN

> Is the dominant generalization failure recognizing when `FULL` must stop?

### Q2 — READ

> Can the model generalize when READ must be suppressed?

### Q3 — WRITE

> Can the model generalize when WRITE must be suppressed?

### Q4 — BOTH

> Can the model recognize cases where both READ and WRITE must be suppressed?

### Q5 — PRESERVE

> Can the model recognize states where `FULL` should remain unchanged?

### Q6 — REPRESENTATION

> Does the router actually use sample-specific state information beyond layer/depth priors?

### Q7 — OBJECTIVE

> Is corrective information present in the representation but lost by the trained router head/objective?

### Q8 — LABEL LEARNABILITY

> Are nearby/similar states assigned consistent corrective mechanisms?

### Q9 — LABEL COMPLETENESS

> Are supposedly invalid actions actually invalid under execution, or merely absent from the discovered route set?

---

## 23. Stop Rule

Do not start another full router training run until the report can identify the dominant failure mode.

The next method should be chosen from evidence:

```text
WHEN failure
WHAT failure
READ-specific failure
WRITE-specific failure
BOTH-interaction failure
layer-prior shortcut
objective/head failure
representation failure
label inconsistency
label incompleteness
```

The purpose of this phase is to convert the vague statement:

```text
"the router does not generalize"
```

into a precise statement of:

```text
what decision does not generalize
and why.
```
