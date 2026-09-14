# Full-Benchmark Exhaustive Regression/Rescue Audit Plan
## C/W → Stage-1 Trigger → Stage-2 Intervention → Final Outcome

## 1. Why This Experiment Matters

The full benchmark evaluation produced a clear method-level negative result for the current frozen candidate:

```text
ChartQA:   W→C 1,  C→W 3,  Net -2
TextVQA:   W→C 2,  C→W 12, Net -10
MMMU-Pro:  W→C 0,  C→W 4,  Net -4
POPE:      W→C 0,  C→W 0,  Net  0

Overall:
    W→C = 3
    C→W = 19
    Net  = -16
```

However, this result does **not** tell us yet whether the dominant failure is:

```text
A. Stage-1 admission failure
   - too few Dense-W samples reach Stage-2
   - or the wrong Dense-C samples are admitted

B. Stage-2 intervention selectivity failure
   - Stage-2 intervenes more harmfully on C than usefully on W

C. Stage-2 action/timing failure
   - non-FULL is used, but at the wrong layer or with the wrong action

D. benchmark-specific generalization failure
   - especially TextVQA / MMMU-Pro / POPE
```

The next experiment should therefore **fully decompose the current frozen pipeline before changing training or thresholds**.

This is a high-value diagnostic because the number of correctness-changing samples is only:

```text
3 rescues + 19 regressions = 22 samples
```

which makes an exhaustive sample-level audit practical.

---

## 2. Main Questions

The audit must answer:

### Q1 — Stage-1 bottleneck

> Among full-benchmark Dense-W samples, how many are admitted by Stage-1?

### Q2 — Stage-2 intervention bottleneck

> Among triggered W and triggered C, how often does Stage-2 actually use a non-FULL action?

### Q3 — Treatment quality

> Conditional on Stage-2 intervention, is the intervention more likely to rescue W or regress C?

### Q4 — Regression mechanism

> What exact trigger/action/timing patterns produce the 19 C→W regressions?

### Q5 — Rescue mechanism

> What exact trigger/action/timing patterns produce the 3 W→C rescues?

### Q6 — Benchmark heterogeneity

> Why is TextVQA responsible for most regressions, MMMU-Pro purely negative, and POPE completely inactive?

### Q7 — Next decision

> Is the next improvement primarily a Stage-1 admission problem, a Stage-2 selectivity problem, or an action/timing generalization problem?

---

## 3. Frozen Inputs

Use only the completed full-benchmark evaluation artifacts.

Do not rerun training.

Do not change:

```text
robust ALL-source Stage-1 head
P90 threshold
Stage-2 Experiment A checkpoint
four-action executor
generation contract
LMMS-Eval correctness
```

Use the exact full-evaluation samples already executed for:

```text
ChartQA
TextVQA
MMMU-Pro
POPE
```

No new search in this phase.

No new inference unless a missing diagnostic field cannot be reconstructed from stored logs.

---

## 4. Phase A — Full Pipeline Funnel

For every benchmark family construct the following two funnels.

### A1. Dense-W funnel

```text
Dense-W total
    ↓
Stage-1 triggered?
    ↓
Stage-2 used any non-FULL?
    ↓
Final outcome:
    W→C
    W→W
```

Report:

```text
# Dense-W
# triggered W
P(trigger | W)

# triggered-W with >=1 non-FULL
P(non-FULL | trigger, W)

# W→C
P(W→C | W)
P(W→C | trigger, W)
P(W→C | trigger, W, non-FULL)
```

### A2. Dense-C funnel

```text
Dense-C total
    ↓
Stage-1 triggered?
    ↓
Stage-2 used any non-FULL?
    ↓
Final outcome:
    C→C
    C→W
```

Report:

```text
# Dense-C
# triggered C
P(trigger | C)

# triggered-C with >=1 non-FULL
P(non-FULL | trigger, C)

# C→W
P(C→W | C)
P(C→W | trigger, C)
P(C→W | trigger, C, non-FULL)
```

---

## 5. Primary Funnel Table

Produce one table per family and a pooled summary.

| Family | P(trigger|W) | P(trigger|C) | P(nonFULL|trig,W) | P(nonFULL|trig,C) | W→C | C→W | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | | | | | 1 | 3 | -2 |
| TextVQA | | | | | 2 | 12 | -10 |
| MMMU-Pro | | | | | 0 | 4 | -4 |
| POPE | | | | | 0 | 0 | 0 |

This table should make the dominant failure stage immediately visible.

---

## 6. Phase B — Exhaustive Audit of All 22 Answer-Change Samples

Audit every:

```text
W→C sample: 3
C→W sample: 19
```

No sampling.

For every sample save:

```text
uid
benchmark
dense correctness
dense answer
routed correctness
routed answer

Stage-1:
    trigger layer
    trigger score
    max score
    score trajectory if available

Stage-2:
    first non-FULL layer
    first non-FULL action
    all non-FULL actions
    number of non-FULL actions
    full post-trigger action sequence

timing:
    trigger → first non-FULL delay

final transition:
    W→C or C→W
```

---

## 7. Regression Audit — 19 C→W Samples

For all 19 regressions classify:

```text
R1. Single harmful intervention
R2. Multiple harmful interventions
R3. Immediate intervention after trigger
R4. Delayed intervention after trigger
R5. READ-related pattern
R6. WRITE-related pattern
R7. IGNORE-dominated pattern
R8. Other / mixed
```

These categories are descriptive and may overlap.

Report:

```text
action frequency among regressions
first harmful action distribution
trigger layer distribution
first non-FULL layer distribution
delay distribution
number of non-FULL actions
```

Key question:

> Is one action or timing pattern disproportionately responsible for regressions?

---

## 8. Rescue Audit — 3 W→C Samples

For all 3 rescues document the exact same fields.

For each rescue ask:

```text
Was correction caused by:
    READ_ONLY?
    WRITE_ONLY?
    IGNORE?
    multiple actions?

Was the action immediate or delayed?

Was the trajectory close to known training single routes?
```

Do not make statistical claims from N=3.

Use these as concrete positive mechanism examples.

---

## 9. Rescue-vs-Regression Contrast

Compare the 3 W→C and 19 C→W samples on:

```text
trigger score
trigger layer
first non-FULL layer
trigger→intervention delay
first action type
number of non-FULL actions
```

Do not overfit a classifier with 22 samples.

Use:

```text
descriptive distributions
rank comparisons
simple contingency tables
```

Question:

> Are rescues and regressions behaviorally distinguishable with obvious signals already available to the router or Stage-1?

If yes, that may suggest a simple selectivity improvement later.

---

## 10. Phase C — Action Selectivity on All Triggered Samples

Do not limit to answer-changing samples.

For every triggered sample, partition by:

```text
Dense-C
Dense-W
```

and report Stage-2 action behavior:

```text
never non-FULL
READ_ONLY used
WRITE_ONLY used
IGNORE used
multiple action types
```

Also report:

```text
first non-FULL action
first non-FULL layer
non-FULL count
delay from trigger
```

Compare C vs W.

Key question:

> Does Stage-2 behave differently on triggered W than triggered C?

If action behavior is almost identical:

```text
Stage-2 has weak treatment selectivity
```

If behavior differs but outcomes are still poor:

```text
action quality/timing is the issue
```

---

## 11. Phase D — Stage-1 Admission Generalization

For each benchmark report:

```text
Dense-W count
triggered-W count
W recall

Dense-C count
triggered-C count
C false-admission rate
```

Compare with the development/calibration expectation for P90.

Do not treat mismatch alone as proof that Stage-1 must be retrained.

Question:

> Is Stage-1 admission under full evaluation much weaker or differently calibrated than expected?

Special attention:

```text
POPE:
    Stage-1 triggered 0 samples
```

This must be interpreted as:

> the current method is inactive on POPE under the frozen P90 gate.

Do not call this "successful preservation."

---

## 12. Phase E — TextVQA Deep Audit

TextVQA accounts for:

```text
2 W→C
12 C→W
Net = -10
```

Perform a benchmark-specific decomposition:

```text
Dense-C / Dense-W totals
triggered C / W
non-FULL on triggered C / W
action distributions
trigger/intervention timing
all 14 answer-changing trajectories
```

Questions:

1. Does Stage-1 disproportionately trigger problematic TextVQA C samples?
2. Does Stage-2 intervene more often on C than W?
3. Is one action responsible for most regressions?
4. Are regressions concentrated at specific layers?
5. Are the 2 rescues behaviorally different from the 12 regressions?

---

## 13. Phase F — MMMU-Pro Audit

MMMU-Pro has:

```text
W→C = 0
C→W = 4
```

Determine whether failure is due to:

```text
very low Stage-1 W admission
Stage-2 rarely using non-FULL on W
Stage-2 using non-FULL mainly on C
action/timing mismatch
```

Document all 4 regressions exhaustively.

Do not conclude from four regressions that MMMU-Pro can never benefit.

---

## 14. Phase G — POPE Inactivity Audit

POPE has:

```text
W→C = 0
C→W = 0
Stage-1 triggers = 0
```

Therefore Stage-2 was never activated.

Audit only:

```text
Stage-1 score distribution
distance to P90 threshold
Dense-C vs Dense-W score ranges
```

Question:

> Is POPE entirely outside the Stage-1 learned risk regime?

Do not lower the threshold in this phase.

Do not make claims about Stage-2 performance on POPE because Stage-2 was never called.

---

## 15. Phase H — "Where Is the Bottleneck?" Accounting

For each benchmark compute:

```text
Stage-1 opportunity:
    triggered-W / total-W

Stage-2 activation:
    nonFULL-triggered-W / triggered-W

Treatment success:
    W→C / nonFULL-triggered-W

Preservation risk:
    C→W / nonFULL-triggered-C
```

Label the dominant bottleneck per benchmark as one of:

```text
ADMISSION_LIMITED
INTERVENTION_LIMITED
TREATMENT_QUALITY_LIMITED
PRESERVATION_LIMITED
INACTIVE
MIXED
```

Use quantitative justification.

---

## 16. Important Interpretation Boundaries

Do not conclude:

```text
Stage-1 is bad because W recall is low
```

without checking whether more admission would likely worsen C→W.

Do not conclude:

```text
Stage-2 is bad because W→C is low
```

without checking whether Stage-2 was rarely activated on W.

Do not conclude:

```text
POPE is solved
```

when the method never triggered.

Do not conclude:

```text
the whole direction fails
```

from the current candidate.

This audit should identify **which stage limits the current implementation**.

---

## 17. Decision Logic

### Case A — Stage-1 admission is the main bottleneck

Signature:

```text
P(trigger|W) very low
but conditional Stage-2 treatment on triggered-W is reasonably effective
and C→W remains low
```

Next improvement should target Stage-1 admission.

### Case B — Stage-2 intervention selectivity is the main bottleneck

Signature:

```text
Stage-1 triggers enough W
but Stage-2 mostly stays FULL on W
or intervenes similarly on C and W
```

Next improvement should target Stage-2 confidence/selectivity.

### Case C — Stage-2 treatment quality is the main bottleneck

Signature:

```text
non-FULL occurs on W
but W→C remains low
and/or C→W high
```

Next improvement should target action/timing supervision.

### Case D — TextVQA-specific regression dominates

Signature:

```text
overall negative result mostly explained by TextVQA
```

Next improvement should first explain that task-specific failure before changing the global architecture.

### Case E — Mixed bottleneck

If benchmarks fail at different stages, do not force one global explanation.

The next method improvement should target the largest contribution to overall negative Net correction.

---

## 18. No New Training or Search

Do not run:

```text
Stage-2 retraining
Stage-1 retuning
P95/P98 comparison
valid-set-loss model
MCTS model
on-policy collection
new single/MCTS search
```

This phase is analysis only.

---

## 19. Required Outputs

Use:

```text
analysis/dense_failure_stage2/full_benchmark_exhaustive_audit/
```

Create:

```text
protocol.md

funnel/
    benchmark_funnel.csv
    pooled_funnel.csv
    triggered_c_vs_w_action_behavior.csv
    bottleneck_classification.csv

answer_changes/
    all_22_answer_changes.jsonl
    rescues_3.jsonl
    regressions_19.jsonl
    rescue_vs_regression_summary.csv

benchmarks/
    chartqa_audit.md
    textvqa_audit.md
    mmmu_pro_audit.md
    pope_audit.md

metrics/
    stage1_admission.csv
    stage2_intervention.csv
    treatment_quality.csv
    action_distribution.csv
    trigger_action_timing.csv
    dataset_bottleneck_summary.csv

figures/
    funnel_by_benchmark.png
    trigger_vs_nonfull_by_correctness.png
    rescue_vs_regression_actions.png
    rescue_vs_regression_timing.png
    textvqa_regression_breakdown.png
    benchmark_bottleneck_map.png

summaries/
    exhaustive_audit_summary.md
    next_improvement_recommendation.md

artifact_manifest.json
```

---

## 20. `exhaustive_audit_summary.md` Must Answer

1. For each benchmark, how many Dense-W samples reach Stage-1 trigger?
2. For triggered W, how often does Stage-2 use non-FULL?
3. For triggered C, how often does Stage-2 use non-FULL?
4. Conditional on intervention, how often does W→C occur?
5. Conditional on intervention, how often does C→W occur?
6. What exact action/timing patterns caused the 19 regressions?
7. What exact action/timing patterns caused the 3 rescues?
8. Why is TextVQA the dominant regression source?
9. Why does MMMU-Pro have 0 rescue / 4 regression?
10. Why is POPE completely inactive?
11. Is the current full-eval failure mainly admission, intervention, treatment quality, preservation, or mixed?
12. Which benchmark contributes most to the current failure mode?
13. What does the audit **not** justify concluding?

---

## 21. `next_improvement_recommendation.md`

Recommend exactly **one** next improvement based on the dominant full-eval bottleneck.

State:

```text
Why it matters
Which observed failure pattern motivates it
What experiment would test it
What a positive result would mean
What a negative result would mean
What should not be concluded from one negative result
```

Do not recommend several simultaneous changes.

---

## 22. Stop Rule

STOP after:

```text
full four-family funnel
all 22 answer-change audits
TextVQA/MMMU-Pro/POPE focused analyses
dominant bottleneck classification
one next-improvement recommendation
```

Do not change the model in this phase.

---

## 23. Core Principle

The current full benchmark result is:

```text
3 rescues
19 regressions
```

but this alone does not tell us **why**.

The next useful question is:

> **At which stage of the pipeline does the current selective-routing method fail: admission, intervention, or treatment quality?**

Because only 22 samples changed correctness, every rescue and regression can be inspected directly.

This audit should turn:

> "full benchmark routing regressed"

into a precise, actionable diagnosis.
