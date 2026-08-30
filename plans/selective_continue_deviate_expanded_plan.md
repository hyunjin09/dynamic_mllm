# Selective CONTINUE/DEVIATE Gate: Expanded Analysis Plan

## 1. Goal

The next step is to isolate and validate **Stage 1** of a two-stage selective visual-intervention policy:

```text
Stage 1:
CONTINUE FULL
vs
DEVIATE

Stage 2:
if DEVIATE,
choose READ_OFF / WRITE_OFF / BOTH_OFF
```

Do **not** train Stage 2 yet.

The immediate question is:

> **Can we reliably detect high-confidence states where continuing `FULL` is unsafe, while preserving C2C by defaulting to `FULL` whenever the gate is uncertain?**

The policy principle is asymmetric:

```text
uncertain
-> CONTINUE FULL

high-confidence evidence that FULL is harmful
-> DEVIATE
```

The primary objective is therefore not ordinary binary accuracy.

The main quantity of interest is:

> **How much W2C can be safely rescued at a fixed C2C preservation level?**

---

## 2. Why We Focus on Stage 1 First

Previous diagnostics showed:

- the dominant deployment failure is `WHEN` rather than only `WHAT`;
- the trained online router has poor held-out KEEP-vs-DEVIATE performance;
- however, a fresh linear probe on the frozen online representation reaches materially higher held-out WHEN AUROC;
- therefore some transferable `CONTINUE/DEVIATE` signal exists in the online state representation;
- exact READ/WRITE/BOTH mechanism labels remain much less smooth and are partially incomplete.

This motivates testing whether a **simple selective gate** can exploit the available WHEN signal before introducing a second-stage mechanism predictor.

---

## 3. Expanded Sample-Size Requirement

Do not base the next analysis on 14 or similarly tiny conditional examples.

Use larger, prespecified cohorts.

### 3.1 Mandatory-boundary FULL-insertion audit

Target:

```text
minimum: 64 W2C mandatory-boundary states
preferred: 96-128 if runtime is affordable
```

Stratify across:

```text
GQA
TextVQA
ChartQA
```

and across:

```text
early layers
middle layers
late layers
```

Also preserve diversity in the known non-FULL boundary mechanism:

```text
READ_ONLY-valid
WRITE_ONLY-valid
IGNORE-valid
multi-valid
```

If some strata contain fewer than 40 eligible samples, use all available samples and report the exact support.

For any major subgroup comparison, aim for:

```text
>= 40 samples per subgroup
```

whenever the frozen population permits it.

### 3.2 Gate evaluation cohort

For threshold / selective-risk analysis, use at least:

```text
128 held-out W2C
128 held-out C2C
```

Prefer the full existing internal validation split if execution cost is manageable.

Do not reduce this evaluation to the 64-sample audit subset.

### 3.3 Mechanism-conditioned follow-up

If Stage 1 passes and Stage 2 is later analyzed, use:

```text
>= 40 singleton examples per mechanism
```

where available:

```text
READ_ONLY only
WRITE_ONLY only
IGNORE only
```

If a class has fewer than 40 available examples, use all available and report that limitation explicitly.

---

## 4. Phase 1 — WHEN-Label Completeness Audit

Before training a new binary gate, verify that the current `DEVIATE` target is sufficiently trustworthy.

### 4.1 Question

At a frozen mandatory boundary `l*`, the current route cache says:

```text
FULL is invalid
```

But route-cache incompleteness has already been observed for non-FULL actions.

Therefore directly test:

> **If `FULL` is inserted at the supposed mandatory boundary, can a compatible known continuation still produce the correct answer?**

### 4.2 Audit cohort

Select:

```text
64-128 W2C mandatory-boundary states
```

with deterministic stratification by:

```text
dataset
boundary depth
known valid mechanism type
```

Write the exact selected UIDs before execution.

Suggested file:

```text
analysis/selective_gate/when_full_insertion_subset.json
```

### 4.3 FULL-insertion execution

For each audited W2C state:

1. replay the exact all-FULL prefix up to the mandatory boundary;
2. at `l*`, force:
   ```text
   FULL
   ```
3. attach every compatible known suffix that can be tested under the frozen executor contract;
4. execute the complete resulting route;
5. record final answer and correctness.

Do not test only one suffix if several compatible known suffixes exist.

Deduplicate identical routes.

Cache all executions.

### 4.4 Audit outcomes

For every audited boundary classify:

```text
A. FULL-confirmed-invalid
   no tested FULL-insertion continuation is correct

B. FULL-cache-incomplete
   at least one FULL-insertion continuation is correct

C. unresolved
   bounded compatible suffix set is insufficient to decide
```

Important:

```text
A != global proof of invalidity
```

It only means no bounded tested continuation rescued the sample.

### 4.5 Required audit statistics

Report:

```text
overall FULL-insertion bounded rescue rate
95% bootstrap confidence interval
per-dataset rate
per-depth-bin rate
per-known-mechanism rate
number of suffixes tested per state
```

Use:

```text
10,000 UID-group bootstrap draws
fixed seed
```

---

## 5. Phase-1 Decision Rule

### Case A — substantial WHEN-label incompleteness

If a nontrivial fraction of supposed mandatory boundaries admits a correct `FULL` continuation, then:

> `DEVIATE` labels are not sufficiently trustworthy for a clean binary gate.

Do not train the selective gate yet.

Next action:

```text
repair/expand route-cache continuation coverage
rebuild WHEN labels
re-audit
```

Do not use an arbitrary percentage as a post-hoc threshold.

Report the measured rate and confidence interval.

### Case B — bounded FULL rescue is rare

If the audit supports that most mandatory boundaries are genuinely unsafe for continued `FULL`, proceed to the selective gate.

The exact observed completeness rate must still be reported.

---

## 6. Phase 2 — Build a Clean CONTINUE/DEVIATE Dataset

The binary gate target should be as clean as possible.

### 6.1 Positive class: DEVIATE

Use audited / trusted mandatory-boundary states:

```text
DEVIATE = 1
```

Definition:

```text
continuing FULL is unsupported / boundedly unsafe at this state
```

Prefer states that passed the FULL-insertion audit.

Do not silently include known cache-incomplete positives.

### 6.2 Negative class: CONTINUE

Use two complementary negative sources.

#### A. W2C FULL-safe states

Use trajectory states where:

```text
FULL is the unique valid next action
```

This is the strongest within-W2C negative.

#### B. C2C preservation states

Use C2C states from the normal all-FULL trajectory.

These represent:

```text
default FULL computation already yields a correct final answer
```

and are necessary because the gate must preserve original model correctness.

### 6.3 Matching

For diagnostic training/evaluation, match or stratify by:

```text
dataset
layer
split
```

so that the gate cannot solve the task by depth alone.

Record:

```text
positive/negative counts by layer
positive/negative counts by dataset
C2C vs W2C-negative composition
```

---

## 7. Phase 3 — Train Only a Simple Stage-1 Gate

Do not reuse the full 4-action head.

The purpose is to test whether the available online state signal can support selective intervention.

### 7.1 Input

Start from the frozen online representation that previously supported the stronger linear probe.

Test in this order:

```text
1. Linear gate
2. Small 2-layer MLP gate
```

Do not build a large new router.

### 7.2 Output

Single scalar:

```text
s_l = P(DEVIATE | current state)
```

Loss:

```text
binary cross entropy
```

or equivalent 2-class cross-entropy.

No READ/WRITE/BOTH supervision is used in this phase.

### 7.3 Training population

Use a substantially larger training population than the earlier tiny diagnostics.

Recommended:

```text
train positives: >= 512 DEVIATE states if available
train negatives: matched 512+ CONTINUE states
```

Prefer more if already cached and cheap.

Validation:

```text
>= 128 DEVIATE states
>= 128 CONTINUE states
```

Use fixed train/validation splits before training.

---

## 8. Phase 4 — Evaluate the Gate as a Selective Decision Rule

Do not default to threshold `0.5`.

The intended policy is:

```text
if s_l <= tau:
    CONTINUE FULL

if s_l > tau:
    DEVIATE
```

Sweep `tau` across the validation score range.

---

## 9. Core Selective-Risk Curves

Report as functions of threshold:

### 9.1 C2C preservation

```text
fraction of C2C samples that remain correct
```

### 9.2 W2C deviation recall

At trusted mandatory boundaries:

```text
fraction detected as DEVIATE
```

### 9.3 Deviation precision

Among states where the gate fires:

```text
fraction that are actually DEVIATE states
```

### 9.4 False-deviation rate

On CONTINUE states:

```text
P(gate fires | CONTINUE)
```

---

## 10. Mandatory Operating Points

Report results at fixed preservation targets:

```text
C2C preservation >= 99%
C2C preservation >= 98%
C2C preservation >= 95%
```

For each operating point report:

```text
threshold
W2C boundary recall
deviation precision
false-deviation rate
coverage
```

The most important question is:

> **At 99% / 98% C2C preservation, does the gate still identify a meaningful subset of W2C deviation states?**

This is more important than aggregate AUROC.

---

## 11. Confidence / Uncertainty Analysis

Report:

```text
AUROC
AUPRC
reliability / calibration curve
score histograms for CONTINUE vs DEVIATE
precision-recall curve
```

Use bootstrap intervals for the main operating-point metrics:

```text
10,000 UID-group draws
```

Do not claim usefulness from a point estimate alone.

---

## 12. Phase 5 — Stage-1 + Oracle-WHAT Execution

If the selective gate has a meaningful high-confidence operating region, evaluate Stage 1 independently from Stage 2.

Do **not** train a mechanism predictor yet.

When the gate says:

```text
CONTINUE
```

execute:

```text
FULL
```

When the gate says:

```text
DEVIATE
```

use an **oracle known-valid non-FULL action** from the trusted route labels.

This yields:

```text
learned WHEN
+
oracle WHAT
```

---

## 13. Oracle-WHAT Evaluation Cohort

Use at least:

```text
128 held-out W2C
128 held-out C2C
```

Prefer the full internal held-out set if affordable.

For W2C samples with multiple known-valid actions, use a deterministic prespecified oracle rule or an explicitly labeled upper-bound protocol.

Do not choose per-sample oracle actions after seeing final outcomes unless explicitly defining an oracle upper bound.

---

## 14. Oracle-WHAT Behavioral Metrics

At each fixed C2C preservation operating point report:

```text
W2C rescue
C2C preservation
rescues
regressions
net accuracy change
mean first-deviation layer
fraction never deviating
```

Key quantity:

> **How much W2C rescue is achievable if WHEN is learned but WHAT is assumed correct?**

---

## 15. Stage-1 Success Criteria

Stage 1 is promising if there exists a conservative operating point with:

```text
high C2C preservation
+
nontrivial W2C rescue under oracle WHAT
```

Do not require rescue of all W2C samples.

A useful result could be:

```text
C2C preservation ~99%
with measurable positive W2C rescue
```

even if W2C coverage is modest.

---

## 16. Decision Cases

### Case 1 — High-preservation tail exists

If:

```text
C2C preservation >= 99%
W2C boundary recall clearly > 0
oracle-WHAT W2C rescue materially > 0
```

then selective CONTINUE/DEVIATE gating is viable.

Proceed later to Stage 2.

### Case 2 — AUROC is moderate but high-preservation tail is useless

If aggregate discrimination is decent but:

```text
at 99% C2C preservation
W2C recall ~0
```

then the signal is not useful for conservative selective intervention.

Do not proceed to Stage 2.

### Case 3 — Gate fires but oracle-WHAT still gives little rescue

Then timing/WHEN remains insufficient even with perfect mechanism selection.

Do not train Stage 2.

### Case 4 — Gate + oracle-WHAT works

Then Stage 1 is viable and the remaining bottleneck is mechanism selection.

Only then analyze/train:

```text
READ_OFF
WRITE_OFF
BOTH_OFF
```

### Case 5 — WHEN-label audit reveals substantial incompleteness

Repair supervision before selective-gate training.

---

## 17. Stage-2 Preparation Only If Stage 1 Passes

Do not execute Stage 2 yet.

If Stage 1 passes, prepare a later plan with at least:

```text
>= 40-50 clean singleton examples per mechanism
```

preferably:

```text
>= 64 per mechanism
```

for:

```text
READ_OFF
WRITE_OFF
BOTH_OFF
```

where available.

Before training Stage 2, expand the non-FULL label-completeness audit beyond the previous tiny conditional subset.

Target:

```text
minimum 50 cached-invalid non-FULL decisions
preferred 64+
```

if enough eligible examples exist.

---

## 18. Required Tables

### Table 1 — WHEN-label completeness

| Group | States | FULL bounded rescue | Rescue rate | 95% CI |
|---|---:|---:|---:|---:|
| Overall | | | | |
| GQA | | | | |
| TextVQA | | | | |
| ChartQA | | | | |
| Early | | | | |
| Middle | | | | |
| Late | | | | |

### Table 2 — Binary gate discrimination

| Metric | Linear | MLP |
|---|---:|---:|
| AUROC | | |
| AUPRC | | |
| Balanced accuracy | | |
| DEVIATE recall | | |
| CONTINUE recall | | |

### Table 3 — Selective operating points

| C2C preservation target | Threshold | W2C boundary recall | Deviation precision | False-deviation rate |
|---:|---:|---:|---:|---:|
| >=99% | | | | |
| >=98% | | | | |
| >=95% | | | | |

### Table 4 — Learned WHEN + Oracle WHAT

| Operating point | W2C rescue | C2C preservation | Rescues | Regressions | Net change |
|---|---:|---:|---:|---:|---:|
| >=99% C2C | | | | | |
| >=98% C2C | | | | | |
| >=95% C2C | | | | | |

---

## 19. Required Figures

Create:

1. CONTINUE vs DEVIATE score distribution.
2. ROC curve.
3. Precision-recall curve.
4. W2C recall vs C2C preservation.
5. Oracle-WHAT W2C rescue vs C2C preservation.
6. Optional reliability/calibration plot.

The key figure should be:

```text
x-axis: C2C preservation
y-axis: W2C rescue
```

---

## 20. Required Outputs

Suggested root:

```text
analysis/selective_continue_deviate/
```

Create:

```text
protocol.md

when_full_insertion_subset.json
when_full_insertion_executions.jsonl
when_label_completeness_report.md

gate_train_manifest.jsonl
gate_val_manifest.jsonl

linear_gate_config.yaml
linear_gate_history.jsonl
linear_gate_results.json

mlp_gate_config.yaml
mlp_gate_history.jsonl
mlp_gate_results.json

threshold_sweep.csv
selective_operating_points.json

oracle_what_execution.jsonl
oracle_what_results.json

figures/

stage1_decision_summary.md
```

---

## 21. Questions the Final Report Must Answer

### Q1 — Label validity

> Are mandatory-boundary `DEVIATE` labels sufficiently complete/trustworthy under an expanded 64-128-state FULL-insertion audit?

### Q2 — Signal

> Can a simple online-state binary gate generalize CONTINUE vs DEVIATE?

### Q3 — Selectivity

> Is there a high-confidence operating region where C2C preservation stays at 99% / 98% / 95% while still identifying meaningful W2C deviations?

### Q4 — Behavioral value

> With oracle WHAT, does the learned WHEN gate produce actual W2C rescue?

### Q5 — Next step

> Is Stage 1 good enough to justify training a READ_OFF / WRITE_OFF / BOTH_OFF Stage 2?

---

## 22. Stop Rule

Do not train Stage 2 until:

1. the expanded WHEN-label audit is complete;
2. the binary gate is evaluated on a sufficiently large held-out cohort;
3. the C2C-preservation/W2C-rescue curve is available;
4. learned-WHEN + oracle-WHAT execution demonstrates whether Stage 1 has real behavioral value.

The next method decision should be based on:

```text
safe rescue at fixed preservation
```

not on raw four-action classification accuracy.
