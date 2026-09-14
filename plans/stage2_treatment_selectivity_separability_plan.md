# Stage-2 Treatment-Selectivity Separability Diagnostic Plan
## Does the current READ/WRITE representation know when to KEEP FULL vs INTERVENE?

## 1. Why this experiment matters

The full-benchmark audit showed that the current Stage-2 problem is not simply "too few interventions."

Under the frozen P90 + Stage-2-A pipeline:

```text
P(non-FULL | triggered W) = 23.99%
P(non-FULL | triggered C) = 22.72%
```

but:

```text
W→C among treated W = 3 / 119  = 2.52%
C→W among treated C = 19 / 92 = 20.65%
```

A post-hoc non-FULL-vs-FULL margin did not improve the rescue/regression trade-off.

The next important question is therefore:

> **Does the current Stage-2 representation contain a generalizable signal for whether a routed state should KEEP FULL or INTERVENE?**

This is the smallest experiment that tells us whether the next method change should target:

```text
the decision head / objective
```

or:

```text
the representation / training-state coverage
```

Do not train a new router yet.

---

## 2. Use a large diagnostic population

Do **not** run this on a few dozen states.

Use the largest clean training-side replay-valid population available from the robust-gate Stage-2 corpus.

Current available scale is approximately:

```text
569 union UIDs
34,253 routed state rows
2,519 exact-replay-valid routes
```

Primary policy:

> **Use all eligible exact routed states after clean-label filtering.**

Target support should be:

```text
hundreds of unique base UIDs
thousands to tens of thousands of exact routed states
```

If clean labeling leaves fewer states, use all of them and report exact support.

Do not artificially cap the analysis at 50–100 samples.

---

## 3. Frozen components

Freeze:

```text
Qwen2.5-VL backbone
robust ALL-source Stage-1
P98 / P95 / P90 trigger maps
Stage-2 Experiment-A router
READ branch
WRITE branch
4-way action head
existing replay-valid route/state store
```

Only lightweight diagnostic probes may be trained.

Do not:

```text
retrain Stage-2
change Stage-1
rerun search
use external full-benchmark transition labels for training
```

---

## 4. Exact-state labeling

For every exact routed state `s`, reconstruct the observed replay-valid successful action set:

```text
A_obs(s) ⊆ {FULL, READ_ONLY, WRITE_ONLY, IGNORE}
```

Only actions observed to produce a replay-valid correct continuation from the **exact same entering routed state** belong in `A_obs`.

State identity must include:

```text
UID
current layer
exact routed-prefix action sequence
stored state ID/hash/reference
```

Do not merge states merely because they have the same UID and layer.

---

## 5. Primary clean labels

Use a conservative binary target.

### INTERVENE_REQUIRED

```text
FULL ∉ A_obs(s)
AND
at least one non-FULL action ∈ A_obs(s)
```

Meaning:

> In the observed successful continuations, leaving FULL is required at this state.

### KEEP_REQUIRED

```text
A_obs(s) = {FULL}
```

Meaning:

> FULL is the only observed replay-valid successful action at this state.

### MIXED

```text
FULL ∈ A_obs(s)
AND
at least one non-FULL action ∈ A_obs(s)
```

Exclude MIXED states from the primary binary probe.

Analyze them separately.

### UNRESOLVED

If no replay-valid successful action is known from the exact state, exclude it.

Do **not** label "not found by search" as KEEP.

---

## 6. Why this target is the right one

The target is not:

```text
Dense-C vs Dense-W
```

and not:

```text
Stage-1 trigger vs no trigger
```

It is:

> **At this routed state, is a non-FULL treatment actually required by the observed successful continuation, or should FULL be preserved?**

This directly addresses the current full-benchmark selectivity failure.

---

## 7. Dataset construction audit

Before any probe fitting, report:

```text
# total exact states
# unique UIDs
# KEEP_REQUIRED
# INTERVENE_REQUIRED
# MIXED
# excluded/unresolved
```

Break down by:

```text
dataset
source regime
layer
trigger-depth bin
route source:
    preservation
    single
    MCTS
```

Also report:

```text
states per UID
KEEP/INTERVENE counts per UID
```

This checks whether a few route-rich samples dominate the state pool.

---

## 8. Group-disjoint evaluation

Use:

```text
5-fold group-disjoint cross-validation
```

with the strongest existing base grouping:

```text
UID / image-content group
```

No states from the same base problem may appear in both probe train and held-out fold.

This is mandatory.

A random state-level split would overestimate generalization because many states from one trajectory are highly correlated.

Approximate fold stratification:

```text
KEEP vs INTERVENE
dataset
source regime
```

without violating group-disjointness.

---

## 9. UID-balanced probe training

One UID may contribute many more states than another.

Primary probe weighting:

```text
each UID contributes equal total training weight
```

Within a UID, distribute its weight over its eligible states.

Also balance KEEP/INTERVENE on the probe-training fold.

Do not rebalance the held-out fold.

---

## 10. Probe inputs

Compare the following frozen representations.

### M0 — Current margin baseline

No learned probe.

```text
m(s) = max(non-FULL logit) - FULL logit
```

This is the scalar abstention score that already failed as a deployment margin.

Evaluate its ranking anyway as the baseline.

### M1 — Full 4-way logits

Input:

```text
[z_FULL, z_RO, z_WO, z_IGNORE]
```

Train a linear binary probe.

Question:

> Is treatment selectivity present in the full logit pattern even though the scalar margin loses it?

### R — READ representation

Input:

```text
z_R
```

Train a linear probe.

### W — WRITE representation

Input:

```text
z_W
```

Train a linear probe.

### RW — READ + WRITE representation

Input:

```text
[z_R ; z_W]
```

Train a linear probe.

This is the primary representation diagnostic.

Do not add:

```text
layer embeddings
dataset ID
source ID
Stage-1 features
visual-token count
```

to the learned representation probes.

---

## 11. Optional nonlinear probe

Only if RW-linear is weak-but-nonrandom, train one small MLP:

```text
[z_R ; z_W]
→ one small hidden layer
→ KEEP / INTERVENE
```

This is secondary.

Do not tune a large model.

---

## 12. Fold-local normalization

For every fold:

```text
fit probe mean/std on training states only
apply to held-out states
```

No held-out normalization statistics may leak into probe fitting.

---

## 13. Primary metrics

For each probe report OOF:

```text
AUROC
AUPRC
```

But because the actual method needs **selective intervention**, also report:

```text
INTERVENE precision at top:
    5% coverage
    10% coverage
    20% coverage

Recall at:
    90% precision
    95% precision
```

where support allows.

A modest-recall / high-precision signal is useful.

Perfect classification is not required.

---

## 14. Main comparison table

| Probe | Input | AUROC | AUPRC | Precision@5% | Precision@10% | Recall@90%Prec |
|---|---|---:|---:|---:|---:|---:|
| M0 | nonFULL-FULL margin | | | | | |
| M1 | 4-way logits | | | | | |
| R | z_R | | | | | |
| W | z_W | | | | | |
| RW | z_R + z_W | | | | | |
| RW-MLP optional | z_R + z_W | | | | | |

Key comparison:

```text
M0 vs RW
```

---

## 15. READ vs WRITE analysis

Because READ and WRITE are semantically different, compare:

```text
R-only
W-only
RW
```

Questions:

- Is treatment selectivity primarily present in READ?
- Is it primarily present in WRITE?
- Is combining them necessary?

This may later guide architecture design, but do not modify the router in this phase.

---

## 16. Dataset/source generalization

Report OOF metrics separately where support permits for:

```text
Historical GQA
Historical ChartQA
Historical TextVQA
Canonical GQA
Canonical ChartQA
Canonical TextVQA
```

Do not train dataset-specific probes.

For sparse cells, report support and wide uncertainty rather than forcing conclusions.

---

## 17. Layerwise analysis

Report separability by:

```text
Early: 0-8
Middle: 9-18
Late: 19-27
```

and exact layer where support is sufficient.

Do not use layer as an input feature.

Question:

> Is KEEP-vs-INTERVENE information available only at certain depths?

This matters because the robust Stage-1 often hands off late.

---

## 18. Route-source analysis

Evaluate separately on:

```text
single-route states
MCTS-route states
preservation states
```

Questions:

- Is single correction easier to separate?
- Are MCTS states intrinsically harder?
- Does one representation generalize across route complexity?

---

## 19. MIXED-state secondary analysis

After the clean binary probe is trained, evaluate MIXED states without using them for fitting.

Report their score distribution.

Interpretation:

```text
MIXED near KEEP
→ probe may represent intervention necessity

MIXED near INTERVENE
→ probe may represent intervention opportunity
```

Do not force a binary ground truth for these states.

---

## 20. Shortcut controls

Because Stage-1 previously exploited source/selection structure, include simple nuisance baselines using only:

```text
dataset identity
source regime
layer bin
route source
```

These are diagnostic controls, not method inputs.

Report their AUROC.

If nuisance-only performance is high, run matched sensitivity before interpreting RW as genuine treatment signal.

---

## 21. Matched sensitivity analysis

Create a secondary matched/reweighted evaluation balancing KEEP and INTERVENE across:

```text
dataset
source regime
layer bin
route source
```

as support permits.

Re-evaluate at minimum:

```text
M0
RW
```

Report:

```text
matched AUROC/AUPRC
effective UID count
effective state count
```

Question:

> Does RW retain useful treatment-selectivity signal after obvious dataset/source/layer priors are reduced?

Do not destroy most support merely to achieve perfect matching.

---

## 22. Decision cases

### Case A — RW clearly beats margin and has a high-precision region

Interpretation:

> Current READ/WRITE representations already contain treatment-selectivity information, but the existing 4-way decision head does not use it well.

This justifies the next minimal method:

```text
KEEP vs INTERVENE head
        ↓ if INTERVENE
existing action selection
```

Do not train that deployment head here.

### Case B — 4-way logits work, RW gives little additional gain

Interpretation:

> Treatment selectivity is already encoded near the action output, but the scalar margin is a poor readout.

Next step may be a lightweight learned abstention readout over logits.

### Case C — R or W clearly dominates

Interpretation:

> Treatment selectivity is concentrated in one branch.

Useful mechanism evidence; do not remove the other branch yet.

### Case D — All representation probes remain weak

Interpretation:

> The current Stage-2 representation does not provide a readily generalizable treatment-need signal.

Next work should target representation or training-state diversity rather than another confidence threshold.

This does **not** falsify corrective routing.

### Case E — Natural performance is high but matched performance collapses

Interpretation:

> Apparent treatment selectivity is largely explained by dataset/source/layer priors.

Do not build a learned treatment gate from that signal yet.

---

## 23. What a negative result means

A weak probe result means only:

> **The current frozen Stage-2 representation does not linearly/generalizably separate clean KEEP_REQUIRED from INTERVENE_REQUIRED states under this corpus.**

It does not mean:

```text
dynamic routing cannot work
READ/WRITE control is invalid
a richer state representation cannot help
additional diverse training data cannot help
```

---

## 24. No external-result fitting

Do not use the full external ChartQA/TextVQA/MMMU-Pro/POPE rescue/regression labels for probe fitting or threshold selection.

Those results motivate the diagnostic only.

The probe must be developed entirely from replay-valid training-side supervision.

---

## 25. Required outputs

Use:

```text
analysis/dense_failure_stage2/treatment_selectivity_separability/
```

Create:

```text
protocol.md

dataset/
    exact_state_manifest.jsonl
    clean_binary_states.jsonl
    mixed_states.jsonl
    dataset_summary.csv
    uid_state_counts.csv
    fold_manifest.jsonl

features/
    representation_manifest.json
    feature_schema.md

probes/
    margin_baseline/
    logits_linear/
    read_linear/
    write_linear/
    read_write_linear/
    read_write_mlp_optional/

metrics/
    probe_summary.csv
    fold_metrics.csv
    precision_recall_operating_points.csv
    dataset_source_breakdown.csv
    layer_breakdown.csv
    route_source_breakdown.csv
    matched_sensitivity.csv
    nuisance_controls.csv
    mixed_state_analysis.csv

figures/
    probe_roc_comparison.png
    probe_pr_comparison.png
    high_precision_tradeoff.png
    dataset_source_auroc.png
    layerwise_separability.png
    read_vs_write_separability.png
    mixed_state_score_distribution.png
    matched_vs_unmatched.png

summaries/
    treatment_selectivity_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 26. `treatment_selectivity_summary.md` must answer

1. How many unique UIDs were used?
2. How many exact routed states were used?
3. How many are KEEP_REQUIRED?
4. How many are INTERVENE_REQUIRED?
5. How many are MIXED?
6. Does the current scalar margin rank treatment need?
7. Do the 4-way logits contain more signal than the margin?
8. How predictive is z_R?
9. How predictive is z_W?
10. How predictive is [z_R; z_W]?
11. Is there a useful high-precision INTERVENE subset?
12. Does the signal survive group-disjoint evaluation?
13. Does it survive matching for dataset/source/layer priors?
14. Is the signal consistent across route types?
15. Is a learned KEEP-vs-INTERVENE head justified?
16. What does the result not justify concluding?

---

## 27. `next_stage2_recommendation.md`

Recommend exactly one next step.

If Case A:

```text
train a minimal KEEP-vs-INTERVENE head
on frozen z_R/z_W,
then invoke action selection only after INTERVENE.
```

If Case B:

```text
train a minimal learned abstention readout over frozen 4-way logits.
```

If Case D:

```text
do not add another head;
diagnose representation/training-state diversity next.
```

For the chosen recommendation state:

```text
why it matters
which hypothesis it tests
what a positive result means
what a negative result means
what should not be overinterpreted
```

---

## 28. Stop rule

STOP after:

```text
large clean state-dataset construction
5-fold group-disjoint probe evaluation
high-precision operating-point analysis
matched sensitivity analysis
one next-step recommendation
```

Do not:

```text
train a deployment KEEP/INTERVENE head
change Stage-1
change P90
rerun search
rerun external full evaluation
```

---

## 29. Core principle

This experiment does not ask:

> "Can Stage-2 perfectly predict every corrective action?"

It asks the smaller and more useful question:

> **Does the current Stage-2 representation contain enough information to identify a reliable subset of states where intervention is worth attempting?**

A modest-recall but high-precision signal is enough to justify the next method step because the ultimate objective remains:

```text
W→C > C→W
```

not exhaustive routing coverage.
