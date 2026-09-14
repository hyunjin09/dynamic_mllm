# Stage-1 Gate → Visual-Intervention Correctability Plan

## 1. Goal

We already have three Stage-1 failure-detection substrates:

1. **Shared Random-4 Sequential**
2. **Independent Sequential**
3. **Shared Fixed-L27 detect-and-replay baseline**

The next question is not which gate predicts `wrong` best.

It is:

> **When a gate identifies a high-risk sample, how often is that sample actually correctable by changing visual computation under the deployment regime implied by that gate?**

This phase connects:

```text
failure detection
→ treatment opportunity
```

without yet training a learned READ_OFF / WRITE_OFF / BOTH_OFF Stage-2 head.

---

## 2. Main Questions

### Q1 — Dynamic treatment potential

For a sample detected at layer `l`:

> Can changing visual computation from layer `l` onward rescue the final answer?

### Q2 — Early trigger value

> Do earlier dynamic triggers leave more intervention capacity than late triggers?

### Q3 — Detect-and-replay value

For Fixed-L27:

> If failure is detected only at layer 27, can a second pass from layer 0 recover the sample with modified visual computation?

### Q4 — Gate enrichment

> Are gate-triggered dense-wrong samples more four-action-correctable than dense-wrong samples in general?

### Q5 — Preservation

> For gate-triggered dense-correct samples, can intervention preserve correctness?

---

## 3. Frozen Gate Candidates

Do not retrain Stage-1 predictors.

### A. Shared Random-4 Sequential

Dynamic main candidate.

Trigger:

```text
l* = first layer where the frozen shared sequential gate fires
```

If no trigger:

```text
remain dense FULL
```

### B. Independent Sequential

Dynamic baseline.

Trigger:

```text
l* = first layer where the frozen independent sequential rule fires
```

### C. Shared Fixed-L27

Detect-and-replay baseline.

Run dense computation through L27.

If the gate fires:

```text
restart from layer 0
with four-action treatment
```

Do not interpret Fixed-L27 as "only modify layer 27."

---

## 4. Gate Operating Points

Do not optimize new thresholds from treatment outcomes in this phase.

For each gate use:

1. its frozen selected operating point;
2. optionally one more conservative reference point;
3. optionally one more aggressive reference point.

Suggested reference points:

```text
selected
~99% preservation
~95% preservation
```

If two policies are numerically identical, keep only unique ones.

---

## 5. Evaluation Population

Use the existing current-runtime dense split.

Primary development:

```text
validation
```

Final confirmation:

```text
test
```

For each gate and operating point, freeze the UID sets:

```text
Triggered dense-wrong
Triggered dense-correct
Non-triggered dense-wrong
Non-triggered dense-correct
```

Do not change treatment procedure by benchmark.

---

## 6. Definition: Intervention-Correctable

A triggered dense-wrong sample is **intervention-correctable** if at least one permitted four-action intervention trajectory under that gate's deployment regime produces an lmms-eval-correct final answer.

A triggered dense-correct sample is **intervention-preservable** if at least one permitted intervention trajectory remains correct.

These are oracle/treatment-feasibility quantities.

They are not learned Stage-2 performance.

---

## 7. Dynamic-Gate Treatment Contract

Applies to:

```text
Shared Random-4 Sequential
Independent Sequential
```

If a sample first triggers at layer `l*`:

```text
layers 0 ... l*-1
=
the already executed dense FULL prefix
```

Treatment may modify visual computation only from:

```text
l* ... 27
```

Do not restart from layer 0 for the primary dynamic analysis.

This directly tests whether the **current failure state** provides useful intervention timing.

---

## 8. Dynamic Treatment Search

Use the current verified four-action executor.

Actions:

```text
FULL
READ_ONLY
WRITE_ONLY
IGNORE
```

Do not exhaustively enumerate `4^(remaining layers)`.

### Step 1 — Immediate single-intervention test

At trigger layer `l*`, test:

```text
READ_ONLY at l*, then FULL afterward
WRITE_ONLY at l*, then FULL afterward
IGNORE at l*, then FULL afterward
```

`FULL` is the native control.

Measure:

> Can a single intervention at the detected failure point rescue the sample?

### Step 2 — Bounded suffix search

For triggered dense-wrong samples not rescued by Step 1, run a bounded search over:

```text
l* ... 27
```

Reuse the current validated four-action search machinery where possible.

Freeze the search budget before seeing results.

Start from the actual dense hidden state reached through layer `l*-1`.

Every discovered route must be executed under the current runtime.

Do not treat old cached correctness as authority.

---

## 9. Fixed-L27 Detect-and-Replay Contract

If Fixed-L27 fires:

```text
Pass 1:
FULL L0 ... L27

Pass 2:
restart from L0
with four-action treatment
```

Use the same general bounded search policy as the dynamic analysis, adapted to the full 0-27 treatment interval.

This compares:

```text
Dynamic:
detect during computation
→ repair remaining trajectory

Fixed-L27:
detect late
→ pay for a second treatment pass from L0
```

Report compute separately.

---

## 10. Search Fairness

Freeze and report:

```text
max evaluated routes per triggered sample
search algorithm
search seed
action ordering
early-stop rule
```

Because Fixed-L27 has a longer treatment interval, also report:

```text
route evaluations/sample
GPU time/sample
```

If useful, report two comparisons:

```text
A. equal search budget
B. best-feasible oracle upper bound
```

Do not silently give one regime much more search effort.

---

## 11. Core Metrics

For each gate / operating point report:

### 11.1 Stage-1 trigger statistics

```text
dense-correct pre-treatment preservation
dense-wrong trigger recall
trigger precision
median trigger layer
```

### 11.2 Triggered-wrong correctability

```text
Correctable@single-intervention
Correctable@bounded-search
```

Definition:

```text
triggered dense-wrong with >=1 correct treatment
/
all triggered dense-wrong
```

### 11.3 Triggered-correct preservability

```text
triggered dense-correct with >=1 correctness-preserving treatment
/
all triggered dense-correct
```

### 11.4 Population-level oracle rescue potential

Directly compute:

```text
dense-wrong samples
that are both:
    triggered
    and treatment-correctable
/
all dense-wrong samples
```

This is the most important treatment upper-bound quantity.

---

## 12. Gate Enrichment

Compare:

```text
P(correctable | triggered dense-wrong)
```

against:

```text
P(correctable | all dense-wrong)
```

under the same treatment/search contract.

Define:

```text
Correctability enrichment
=
P(correctable | triggered wrong)
/
P(correctable | all wrong)
```

Question:

> Does the failure gate preferentially select failures that visual intervention can actually fix?

This explicitly tests:

```text
failure detection
!=
treatment benefit
```

---

## 13. Trigger-Depth Correctability

For dynamic gates stratify triggered dense-wrong samples by trigger depth:

```text
Early: 0-8
Middle: 9-18
Late: 19-27
```

Report:

```text
trigger count
single-intervention rescue
bounded-search correctability
```

Main question:

> Does earlier failure detection actually provide greater treatment opportunity?

If sample counts permit, also report exact-layer trends descriptively.

---

## 14. Dataset Breakdown

Report separately for:

```text
GQA
ChartQA
TextVQA
```

Metrics:

```text
trigger recall
trigger precision
conditional treatment correctability
population-level oracle rescue potential
```

Do not tune treatment separately by dataset.

---

## 15. Primary Comparison Table

Produce:

| Gate | Pre-treatment preservation | Wrong trigger recall | Trigger precision | Triggered-wrong correctability | Population oracle rescue | Median trigger |
|---|---:|---:|---:|---:|---:|---:|
| Shared Random-4 Sequential | | | | | | |
| Independent Sequential | | | | | | |
| Shared Fixed-L27 Replay | | | | | | |

Also report compute:

```text
average dense passes
average route evaluations
relative treatment cost
```

---

## 16. Dynamic vs Fixed-L27

Compare in two dimensions.

### A. Treatment opportunity

```text
How much of the dense-wrong population is actually correctable?
```

### B. Compute

```text
Dynamic:
one dense prefix + intervention suffix

Fixed-L27:
full dense pass + second treatment pass for admitted samples
```

Do not discard Fixed-L27 only because it requires replay.

It remains an important detect-then-replay baseline.

---

## 17. Decision Cases

### Case A — Dynamic has higher treatment potential

If early/mid dynamic detection allows substantially more correction than Fixed-L27 under reasonable search:

> Dynamic failure timing is useful for treatment.

Proceed with dynamic Stage-2 development.

### Case B — Fixed-L27 replay matches/exceeds dynamic

Then:

> Early detection is not necessary for treatment quality.

Keep Fixed-L27 as a serious fallback method.

Dynamic routing would need an efficiency or representation argument.

### Case C — Gate detects wrong but does not enrich correctability

If:

```text
P(correctable | triggered wrong)
≈
P(correctable | all wrong)
```

then the failure state predicts error but not treatment responsiveness.

Do not yet train Stage 2 from `z_fail`.

### Case D — Triggered wrong samples are strongly enriched

Then the hierarchy is supported:

```text
WHEN:
failure predictor

WHAT:
visual action policy
```

Proceed to Stage 2.

---

## 18. Future Stage-2 Manifest

If dynamic treatment feasibility is positive, save for every triggered state:

```text
UID
trigger layer
current hidden feature h_l
failure probability p_wrong
failure predictor penultimate representation z_fail
successful treatment actions/routes
failed treatment actions/routes
```

This becomes future Stage-2 supervision.

Intended later architecture:

```text
h_l
+
z_fail
+
layer information
→
visual action head
→
READ_OFF / WRITE_OFF / BOTH_OFF / optional FULL
```

Do not train the action head in this phase.

---

## 19. Required Figures

Create only:

```text
gate_correctability_comparison.png
correctability_enrichment.png
correctability_by_trigger_depth.png
dataset_treatment_potential.png
oracle_rescue_vs_compute.png
```

---

## 20. Required Outputs

Use:

```text
analysis/dense_failure_stage1/treatment_correctability/
```

Create:

```text
protocol.md

gate_trigger_manifests/
    shared_random4.jsonl
    independent_sequential.jsonl
    fixed_l27.jsonl

treatment_search/
    execution_rows.jsonl
    route_cache.jsonl
    search_summary.json

metrics/
    gate_correctability.csv
    trigger_depth_correctability.csv
    dataset_breakdown.csv
    enrichment.csv
    compute_comparison.csv

stage2_future_manifest.jsonl

figures/
    gate_correctability_comparison.png
    correctability_enrichment.png
    correctability_by_trigger_depth.png
    dataset_treatment_potential.png
    oracle_rescue_vs_compute.png

decision_summary.md
```

---

## 21. Final Questions

The report must answer:

1. Among high-risk samples selected by each Stage-1 gate, how many dense-wrong cases are actually four-action-correctable?
2. Does Stage-1 gating enrich intervention-correctable failures relative to the full dense-wrong population?
3. Does earlier dynamic triggering increase treatment correctability?
4. Which dynamic gate is the better treatment substrate: Shared Random-4 or Independent Sequential?
5. How does dynamic treatment opportunity compare with Fixed-L27 detect-and-replay?
6. Are gate-triggered dense-correct samples usually preservable under intervention?
7. Is there enough evidence to proceed to a learned Stage-2 action head conditioned on the current failure state?

---

## 22. Stop Rule

STOP after treatment-correctability / oracle-feasibility analysis.

Do not yet:

```text
train the Stage-2 action head
resume broad W→C repair
run full MCTS over all 7,999 samples
perform external evaluation
```

The next method decision should be based on:

```text
failure detection quality
×
treatment correctability
×
trigger timing
×
compute cost
```

rather than failure-detection AUROC alone.
