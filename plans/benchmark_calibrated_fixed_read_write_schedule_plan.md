# Benchmark-Calibrated Fixed READ/WRITE Schedule Plan
## Can weak population-level READ/WRITE structure support a non-learned benchmark-specific intervention schedule?

## 0. Core idea

Previous phases found that READ/WRITE harm exists, but per-instance local action prediction is weak. This phase therefore tests a coarser hypothesis:

> Can a small calibration set identify benchmark-specific READ-OFF and WRITE-OFF layers that generalize to held-out samples from the same benchmark?

There is **no Stage-1 trigger** and **no learned router**.

Every sample in benchmark `b` receives exactly the same fixed schedule.

READ and WRITE are calibrated independently, then combined into the original four actions:

```text
READ ON,  WRITE ON  -> FULL
READ ON,  WRITE OFF -> READ_ONLY
READ OFF, WRITE ON  -> WRITE_ONLY
READ OFF, WRITE OFF -> IGNORE
```

---

# 1. Scientific question

Old question:

```text
Given this sample/state, which action should I take now?
```

New question:

```text
For this benchmark population,
which layer is systematically beneficial to suppress READ at?

For this benchmark population,
which layer is systematically beneficial to suppress WRITE at?
```

Primary claim under test:

> Visual-computation harm may have stable benchmark-level layer structure even when per-instance local utility is difficult to predict.

---

# 2. Stage-1 is completely disabled

Do not use:

```text
Stage-1 trigger
P90 threshold
failure score
triggered subset
post-trigger domain
```

All samples are processed from layer 0 to layer 27 under the same benchmark-specific schedule.

This prevents mixing benchmark-level calibration with sample-level failure detection.

---

# 3. No learning

Do not train:

```text
MLP router
linear probe
policy network
value network
trajectory decoder
RL policy
```

Calibration consists only of measuring single-bit layer effects on a small calibration set and selecting fixed layer indices.

No gradient-based fitting.

---

# 4. Primary benchmarks

Use the same four benchmarks as the prior external method evaluation:

```text
ChartQA
TextVQA
MMMU-Pro
POPE
```

Optionally include GQA only as an internal sanity benchmark if essentially free.

Do not use GQA to tune decisions for the four primary benchmarks.

---

# 5. Calibration / held-out split contract

For each benchmark define:

```text
CAL_b
TEST_b
```

with no UID/image-group overlap.

Preferred rule:

```text
if a labeled official train/dev split exists and matches the evaluation contract:
    draw CAL_b from that non-test split
    keep the existing evaluation split untouched as TEST_b

otherwise:
    create one deterministic group-disjoint calibration/held-out split
    from the available labeled benchmark population
    and report only the held-out portion as TEST_b
```

Never calibrate on samples used for the reported held-out result.

Freeze:

```text
split_registry.jsonl
```

before intervention outcomes are inspected.

Group by image/content identity wherever multiple questions can share the same image.

---

# 6. Calibration pool size

Primary maximum calibration pool:

```text
approximately 256 UIDs per benchmark
```

while respecting image/content groups.

Use one frozen calibration pool and nested prefixes:

```text
N = 32
N = 64
N = 128
N = 256
```

The N=256 selector is the primary schedule.

The smaller N values are calibration-efficiency/stability diagnostics and require no new branch execution once N=256 has been measured.

If a benchmark cannot supply 256 valid calibration UIDs, use the largest valid size and document it before outcome inspection.

---

# 7. Dense baseline

For every calibration and held-out sample compute/store:

```text
FULL at every layer
```

Record:

```text
generated answer
benchmark-native correctness
token-normalized gold-answer q when valid
```

All calibrated methods are compared against this exact dense baseline.

---

# 8. READ calibration sweep

For every calibration sample and every eligible layer `l`, execute:

```text
layer l:
    WRITE_ONLY = READ0 WRITE1

all other layers:
    FULL
```

This is one isolated READ-OFF intervention.

Define:

```text
DeltaAcc_R(i,l)
=
correct_WO(i,l) - correct_FULL(i)
```

with:

```text
+1 : W->C
 0 : unchanged
-1 : C->W
```

Also record:

```text
H_R(i,l)
=
q_WO(i,l) - q_FULL(i)
```

when q is valid.

---

# 9. WRITE calibration sweep

For every calibration sample and every eligible layer `l`, execute:

```text
layer l:
    READ_ONLY = READ1 WRITE0

all other layers:
    FULL
```

Define:

```text
DeltaAcc_W(i,l)
=
correct_RO(i,l) - correct_FULL(i)
```

and:

```text
H_W(i,l)
=
q_RO(i,l) - q_FULL(i)
```

when valid.

Layer 27 WRITE must follow the frozen semantics.

If it is provably zero-effect because no later layer can consume the updated visual state, retain it as a semantic control but exclude it from WRITE selection.

---

# 10. Primary calibration statistics

For each benchmark and layer:

## READ

```text
Net_R(l)
=
#(W->C) - #(C->W)

Gain_R(l)
=
Net_R(l) / N_cal
```

Secondary:

```text
MeanQ_R(l)
=
mean_i H_R(i,l)
```

## WRITE

```text
Net_W(l)
=
#(W->C) - #(C->W)

Gain_W(l)
=
Net_W(l) / N_cal
```

Secondary:

```text
MeanQ_W(l)
=
mean_i H_W(i,l)
```

Benchmark-native correctness is the primary calibration criterion.

q is only a tie-breaker / secondary diagnostic.

---

# 11. Primary layer selector

READ and WRITE are selected independently.

## READ

```text
l_R*(b)
=
argmax_l Gain_R_b(l)
```

with:

```text
if max_l Gain_R_b(l) <= 0:
    READ selection = NONE
```

Tie-break:

```text
1. larger MeanQ_R
2. fewer C->W regressions
3. lower layer index
```

## WRITE

```text
l_W*(b)
=
argmax_l Gain_W_b(l)
```

using the same NONE and tie-break rules.

Freeze these rules before held-out evaluation.

---

# 12. Fixed benchmark schedule

For every held-out sample from benchmark `b`:

```text
all layers = FULL
```

except the selected bit interventions.

### Different selected layers

Example:

```text
l_R* = 12
l_W* = 20
```

then:

```text
L12 -> WRITE_ONLY
L20 -> READ_ONLY
all others -> FULL
```

### Same selected layer

If:

```text
l_R* = l_W* = 20
```

then:

```text
L20 -> IGNORE
all others -> FULL
```

### One side NONE

Use only the selected bit intervention.

### Both NONE

Use Dense/FULL.

---

# 13. Interaction rule

READ and WRITE are calibrated independently.

Do **not** assume isolated gains add.

After selecting `l_R*` and `l_W*`, execute the actual combined schedule.

The combined route is evaluated directly.

Interaction is itself an experimental result.

Do not revise the selected layers after seeing combined held-out performance.

---

# 14. Required held-out methods

Evaluate on the exact same `TEST_b`:

## M0 — Dense

```text
FULL everywhere
```

## M1 — READ-calibrated only

```text
READ OFF at l_R*
WRITE ON everywhere
```

## M2 — WRITE-calibrated only

```text
WRITE OFF at l_W*
READ ON everywhere
```

## M3 — Combined calibrated

Combine the independently selected READ/WRITE bits into the appropriate four-action schedule.

M3 is the primary calibrated method.

M1 and M2 are required ablations.

---

# 15. Global-calibration control

Pool the primary benchmark calibration sets and select:

```text
one global READ layer
one global WRITE layer
```

using the exact same selection rule.

Apply this one global schedule to every benchmark held-out set.

Question:

> Does benchmark-specific calibration outperform one universal fixed schedule?

This control is required to support a benchmark/task-specific structure claim.

---

# 16. Random matched-sparsity control

For each benchmark, generate random schedules with the same bit sparsity as M3.

Example:

```text
M3 has one READ-OFF layer
and one WRITE-OFF layer
```

then random schedules also contain exactly:

```text
one random valid READ-OFF layer
one random valid WRITE-OFF layer
```

Use fixed seeds.

Primary target:

```text
100 random schedules
```

If full held-out execution is too expensive, use at least 20 and record that compute-driven change before looking at outcomes.

Report:

```text
random mean
random std
calibrated percentile among random schedules
```

---

# 17. Calibration-size stability

From the already measured N=256 calibration intervention matrix, independently select schedules using nested:

```text
N=32
N=64
N=128
N=256
```

For each benchmark/bit report:

```text
selected layer
Gain of selected layer
rank of the smaller-N selected layer under N=256 calibration
```

Also bootstrap the calibration set and report:

```text
selection frequency per layer
selection entropy
```

Question:

> How many calibration examples are needed before the fixed layer becomes stable?

---

# 18. Split-half structure stability

Split the N=256 calibration pool into two fixed group-disjoint halves:

```text
CAL-A
CAL-B
```

For READ and WRITE separately compute the per-layer effect vectors:

```text
Gain_R^A(l), Gain_R^B(l)
Gain_W^A(l), Gain_W^B(l)
```

Report:

```text
Spearman across layers
top-1 agreement
top-3 overlap
sign agreement
```

This tests whether the benchmark-level layer profile itself is stable rather than a winner's-curse artifact.

---

# 19. Held-out evaluation metrics

For every benchmark and method report:

```text
accuracy
W->C
C->W
Net = W->C - C->W
Delta accuracy vs Dense
```

Use paired image/content-group bootstrap 95% CIs for:

```text
M1 - Dense
M2 - Dense
M3 - Dense
benchmark-specific M3 - global schedule
```

Report each benchmark separately first.

---

# 20. Macro summary

Primary cross-benchmark summary:

```text
macro mean Delta accuracy
```

across:

```text
ChartQA
TextVQA
MMMU-Pro
POPE
```

Pooled results may be reported only as secondary because benchmark sizes differ.

---

# 21. Calibration-overfitting diagnostic

For selected layers report:

```text
calibration Gain
held-out Delta accuracy
```

and:

```text
calibration W->C / C->W
held-out W->C / C->W
```

Question:

> Does the selected effect replicate out of sample?

This is central to deciding whether coarse benchmark-level structure is real.

---

# 22. Optional held-out oracle diagnostic

This is analysis only and must never alter the method.

If cheap from existing caches or a fixed small audit subset, compute:

```text
best READ layer in hindsight
best WRITE layer in hindsight
```

Label:

```text
POST-HOC ORACLE — NOT A METHOD
```

Purpose:

```text
distinguish:
    calibration failed to find an existing fixed-layer effect
from
    no useful fixed-layer effect exists.
```

Do not use this oracle for schedule selection.

---

# 23. Primary decision categories

## CAL-A — Stable benchmark-specific fixed structure

Support if:

```text
benchmark-specific M3 improves held-out performance on multiple benchmarks,
macro Delta is positive,
and benchmark-specific calibration beats global/random controls.
```

Interpretation:

> Per-instance action utility is weak, but benchmark-level fixed READ/WRITE structure is exploitable.

## CAL-B — Bit-specific structure

Pattern:

```text
READ-only or WRITE-only calibration replicates,
but the independent combination does not consistently help.
```

Interpretation:

> One visual-computation bit has stable benchmark-level structure while combining both introduces interaction.

## CAL-C — Calibration unstable

Pattern:

```text
selected layers vary substantially with N,
split-half layer correlation is weak,
held-out gains are inconsistent.
```

Interpretation:

> The apparent population structure is too noisy for reliable exact-layer calibration.

## CAL-D — No held-out benefit

Pattern:

```text
calibration gains do not replicate,
benchmark-specific M3 does not beat Dense/global/random.
```

Interpretation:

> Weak population-level structure is insufficient for a useful benchmark-specific fixed schedule.

---

# 24. Important non-claims

Even CAL-A does not establish:

```text
instance-specific routing
OOD generalization to unseen benchmarks
zero-shot calibration
causal mechanism of harmful READ/WRITE
universal layer importance
total-system compute savings without calibration amortization
```

The method is:

```text
benchmark-adaptive
non-learned
fixed within the benchmark at inference
```

---

# 25. Optional sparse top-2 follow-up

Do **not** run automatically.

Run only if primary top-1-per-bit M3 gives convincing held-out benefit.

Then allow:

```text
up to 2 READ-OFF layers
up to 2 WRITE-OFF layers
```

Selection must remain calibration-only.

Do not simply take the top two isolated single-layer scores.

Use greedy calibration:

```text
1. choose best first layer
2. evaluate adding one second layer on CAL only
3. accept second layer only if joint calibration correctness improves
```

This optional stage is a separate follow-up.

---

# 26. Efficiency accounting

Report calibration and inference cost separately.

Calibration:

```text
N_cal samples
x all single READ-OFF layers
x all single WRITE-OFF layers
```

Held-out inference:

```text
one fixed route per sample
with only the selected layer actions changed
```

No Stage-1 and no sample-specific branching are used at test time.

Report:

```text
calibration GPU-hours
held-out throughput
relative inference FLOPs/token if measurable
```

Do not claim end-to-end efficiency without an amortization assumption.

---

# 27. Execution order

Run exactly in this order:

```text
1. freeze split registry
2. verify Dense baselines
3. run N=256 READ single-layer calibration sweep
4. run N=256 WRITE single-layer calibration sweep
5. compute per-layer READ/WRITE effect tables
6. select l_R* and l_W* independently
7. derive N=32/64/128 stability from the same intervention matrix
8. run split-half/bootstrap stability
9. freeze benchmark-specific schedules
10. freeze the global schedule
11. evaluate M0/M1/M2/M3 on held-out TEST
12. evaluate the global schedule
13. evaluate matched random schedules
14. compute macro and overfitting diagnostics
15. make CAL-A/B/C/D decision
16. only if primary result is positive, consider the optional top-2 follow-up
```

Do not inspect held-out method results before schedules are frozen.

---

# 28. Output directory

Use:

```text
analysis/benchmark_calibrated_fixed_rw_schedule/
```

Required artifacts:

```text
protocol.md
frozen_contract.json

splits/
    split_registry.jsonl
    benchmark_support.csv
    leakage_check.md

calibration/
    chartqa_read_layer_effects.csv
    chartqa_write_layer_effects.csv
    textvqa_read_layer_effects.csv
    textvqa_write_layer_effects.csv
    mmmupro_read_layer_effects.csv
    mmmupro_write_layer_effects.csv
    pope_read_layer_effects.csv
    pope_write_layer_effects.csv
    selected_layers.csv
    calibration_size_stability.csv
    split_half_stability.csv
    bootstrap_layer_selection.csv

schedules/
    benchmark_specific_schedules.json
    global_schedule.json
    random_schedule_registry.jsonl
    schedule_freeze_report.md

heldout/
    dense_results.csv
    read_only_results.csv
    write_only_results.csv
    combined_results.csv
    global_schedule_results.csv
    random_schedule_results.csv
    per_uid_results.jsonl

metrics/
    benchmark_accuracy_summary.csv
    correction_regression_summary.csv
    paired_bootstrap_ci.csv
    calibration_vs_test.csv
    benchmark_vs_global.csv
    benchmark_vs_random.csv
    macro_summary.csv

efficiency/
    calibration_cost.csv
    inference_cost.csv

figures/
    chartqa_read_write_layer_calibration.png
    textvqa_read_write_layer_calibration.png
    mmmupro_read_write_layer_calibration.png
    pope_read_write_layer_calibration.png
    calibration_size_stability.png
    split_half_layer_stability.png
    heldout_accuracy_delta.png
    correction_vs_regression.png
    benchmark_vs_global_schedule.png
    calibrated_vs_random_percentile.png

summaries/
    calibration_summary.md
    heldout_result_summary.md
    benchmark_structure_characterization.md
    next_research_direction.md

artifact_manifest.json
```

---

# 29. `calibration_summary.md` must answer

1. What split was used for calibration/test for each benchmark?
2. Was image/content-group leakage absent?
3. What READ layer was selected for each benchmark?
4. What WRITE layer was selected?
5. Was NONE selected for any bit/benchmark?
6. What are the calibration W->C/C->W/Net statistics of each selected layer?
7. How stable are selected READ layers across N=32/64/128/256?
8. How stable are selected WRITE layers?
9. What is the split-half layer-effect correlation?
10. Which layers dominate bootstrap selection?
11. What global READ/WRITE layers were selected?
12. Were schedules frozen before held-out evaluation?

---

# 30. `heldout_result_summary.md` must answer

For each benchmark:

1. Dense accuracy?
2. READ-calibrated-only accuracy and Delta?
3. WRITE-calibrated-only accuracy and Delta?
4. Combined calibrated accuracy and Delta?
5. W->C / C->W / Net for each?
6. Paired bootstrap CI versus Dense?
7. Does combined beat both individual bits?
8. Does benchmark-specific beat global calibration?
9. Where does calibrated performance fall among random matched schedules?
10. Does calibration gain replicate or collapse?
11. What is the macro Delta across benchmarks?

---

# 31. `benchmark_structure_characterization.md`

Explicitly answer:

```text
Is there stable benchmark-level READ structure?
Is there stable benchmark-level WRITE structure?
Are selected layers benchmark-specific or mostly universal?
Does independently combining READ/WRITE bits help?
How many calibration examples are needed before layer selection stabilizes?
Does coarse benchmark-level structure survive despite weak per-instance predictability?
```

End with exactly one primary category:

```text
CAL-A
CAL-B
CAL-C
CAL-D
```

---

# 32. `next_research_direction.md`

Recommend exactly one next step.

## If CAL-A

Recommend:

```text
formalize benchmark-calibrated sparse visual computation
```

and choose exactly one next test:

```text
calibration sample efficiency
cross-benchmark transfer
few-shot calibration on a new benchmark
```

## If CAL-B

Recommend:

```text
retain only the stable READ or WRITE bit
```

and test the minimal fixed schedule.

## If CAL-C

Recommend:

```text
coarser stage/region-level calibration
```

rather than exact-layer selection.

## If CAL-D

Recommend:

```text
close benchmark-level fixed calibration as insufficient
```

and return to the broader trajectory/nonlocal intervention interpretation.

---

# 33. Stop rule

Minimum required result:

```text
1. group-disjoint calibration/test splits
2. full single-layer READ calibration sweep
3. full single-layer WRITE calibration sweep
4. independent READ/WRITE layer selection
5. schedule freeze
6. held-out Dense / READ-only / WRITE-only / Combined evaluation
7. global calibration control
8. matched random schedule control
9. calibration-size and split-half stability
10. CAL-A/B/C/D decision
11. exactly one next-step recommendation
```

Do not:

```text
use Stage-1
train a router
retune on held-out test
run MCTS/beam search
launch top-2 schedules unless the primary result is positive
```

---

# 34. Core principle

The previous dynamic-routing hypothesis assumed:

```text
the correct visual-computation action is identifiable per sample.
```

Current evidence weakens that assumption.

This experiment tests a coarser alternative:

```text
benchmark
→ fixed sparse READ/WRITE layer schedule
```

If successful, the method becomes:

```text
small labeled calibration set
→ no model training
→ select READ layer
→ select WRITE layer
→ combine them into a fixed four-action schedule
→ apply unchanged to every held-out sample in that benchmark
```

This directly tests whether weak population-level READ/WRITE structure is exploitable without Stage-1 or per-instance routing.
