# POLAR-Style Post-Trigger Suffix-Program Predictor
## Full training on all available route labels + full evaluation on ChartQA, TextVQA, MMMU-Pro, and POPE

## 0. Executive decision

Keep the current two-stage handoff:

```text
Before first Stage-1 trigger:
    always FULL

At first trigger layer L*:
    Stage-2 begins

From L* through layer 27:
    Stage-2 controls the visual-computation suffix
```

The experiment changes only the Stage-2 supervision unit and predictor form.

Current Stage-2:

```text
observe current routed state
→ predict one local action
→ execute
→ repeat
```

New Stage-2:

```text
at first trigger:
    observe trigger state once
    predict the complete suffix program
        [a_L*, ..., a_27]
    execute that complete program
```

This is motivated by the finding that many post-trigger states admit multiple successful local actions, so collapsing MCTS trajectories into local valid-action targets can erase useful trajectory structure.

This is not a small diagnostic. Train on the full available training-side route corpus and then run the complete four-benchmark evaluation.

---

## 1. Research question

Does preserving MCTS supervision as complete post-trigger suffix programs, rather than reducing it to local action labels / local valid-action sets, produce a Stage-2 policy with better end-to-end rescue-regression behavior?

Primary criterion:

```text
W→C > C→W
```

Stronger criterion:

```text
W→C > C→W
AND
routed accuracy >= Dense accuracy
```

---

## 2. Why this experiment matters

Previous experiments established:

1. Single-intervention routes can rescue some Dense-W samples.
2. MCTS finds many additional multi-layer corrective trajectories.
3. Local imitation of MCTS actions generalizes poorly.
4. Observed-valid-set loss improves oracle local-action agreement but not free-run benchmark behavior.
5. The large label-completeness audit showed severe local action ambiguity:
   - old KEEP invalidation: 99.2%;
   - old INTERVENE invalidation: 54.2%;
   - 822 / 1,200 audited states had all four first actions connected to some successful bounded suffix.

Hypothesis:

> The MCTS trajectory may contain useful joint structure even when its individual per-layer actions are not uniquely identifiable in isolation.

This phase tests that directly.

---

## 3. Fixed components

Keep fixed:

```text
Qwen2.5-VL-7B-Instruct
28 decoder layers
robust ALL-source Stage-1
P90 Stage-1 threshold
four action semantics
same-layer READ semantics
LMMS-Eval correctness
four external benchmark families
```

Actions:

```text
FULL       = READ1 WRITE1
READ_ONLY  = READ1 WRITE0
WRITE_ONLY = READ0 WRITE1
IGNORE     = READ0 WRITE0
```

Before trigger:

```text
a_l = FULL for l < L*
```

Stage-2 never controls pre-trigger layers.

---

## 4. Primary Stage-1 operating point

Use the frozen robust ALL-source Stage-1 at P90.

Do not tune Stage-1 in this phase.

Reason: the experiment should isolate whether trajectory-level Stage-2 supervision is better than the previous local Stage-2 formulation.

---

## 5. Full training corpus

Do not train on a small subset.

Construct one full P90-aligned post-trigger program corpus from every available training-side replay-valid source:

```text
1. preservation trajectories
2. single-intervention successful trajectories
3. original MCTS successful trajectories
4. robust-gate corrective-search trajectories
5. successful trajectories recovered by the
   large treatment-label completeness audit
```

Use all eligible programs after:

```text
P90 alignment
exact replay validation
deduplication
train-side provenance checks
```

Do not cap programs per UID. Control route-rich UIDs by weighting.

---

## 6. P90-align every program

For each UID:

```text
L* = first robust Stage-1 P90 trigger
```

Canonical target:

```text
tau_i = [a_L*, a_L*+1, ..., a_27]
```

If a route was discovered under P98/P95 or from an intermediate routed state:

1. reconstruct the exact action prefix from P90 trigger to that state;
2. concatenate the stored route suffix;
3. replay the complete P90-aligned suffix from L*;
4. include only if exact replay remains correct.

Do not assume cross-threshold route compatibility.

---

## 7. Convert completeness-audit discoveries into complete programs

Each completeness-audit success has:

```text
P90 trigger
→ exact routed prefix
→ forced newly successful action
→ successful suffix
```

Concatenate these into a complete P90 suffix program and replay from the P90 trigger.

Only exact-replay-valid complete programs enter training.

Do not convert them back into local action labels.

---

## 8. Dense-C preservation supervision

For any training-side sample that:

```text
P90 Stage-1 triggers
AND
Dense FULL answer is correct
```

use the canonical target:

```text
[FULL, FULL, ..., FULL]
```

from L* to 27.

Do not promote alternative non-FULL programs for Dense-C to equal primary targets in this first experiment.

Reason:

> Dense-C is already correct under all-FULL, so preserving FULL is the safest deployment behavior and directly targets prior C→W regressions.

---

## 9. Dense-W corrective supervision

For any training-side sample that:

```text
P90 Stage-1 triggers
AND
Dense FULL answer is wrong
```

retain all unique replay-valid successful corrective suffix programs:

```text
T_i = {tau_i1, tau_i2, ..., tau_iK}
```

Do not collapse them into:

```text
per-layer majority labels
local valid-action sets
one arbitrary route
```

The complete trajectory is the supervision unit.

---

## 10. Program deduplication

Canonicalize every route as its complete P90 suffix action string.

Two routes for the same UID are duplicates if the complete action strings are identical.

Keep one canonical copy and preserve provenance:

```text
single
MCTS
robust search
completeness audit
multiple provenance sources
```

Report unique program count per UID and overlap by provenance.

---

## 11. Training weights

### UID-level balance

Each UID contributes equal total weight within its dataset/source/outcome cell.

### Dataset/source/outcome balance

Balance optimization contribution across available:

```text
dataset × source-regime × dense-outcome
```

cells where support is sufficient.

Do not feed dataset/source IDs into the model.

### Route-level balance inside a W UID

If W UID `i` has `K_i` valid programs:

```text
each unique program receives 1 / K_i
```

of that UID's route weight.

For the primary experiment:

```text
no shortest-route preference
no explicit non-FULL penalty
no hand-designed route utility
```

First test the supervision-unit change itself.

---

## 12. Predictor input

At the first trigger layer L*, use the actual trigger state.

Reuse the current Stage-2 representation family:

```text
z_R(L*)
z_W(L*)
```

Construct:

```text
c = MLP([z_R(L*); z_W(L*)])
```

as the program context.

This keeps the representation family close to the current Stage-2 so the main change is trajectory-level supervision.

---

## 13. Program decoder architecture

Use a small autoregressive Transformer action decoder.

For layer position `l >= L*`, inputs include:

```text
absolute layer embedding
relative-to-trigger position embedding
previous-action embedding
trigger context c
```

The first position uses a learned BOS action token.

Recommended minimal architecture:

```text
2 Transformer decoder blocks
small router hidden width
4-8 heads as dimension permits
causal mask over action tokens
conditioning/cross-attention on trigger context c
```

Keep it lightweight relative to the frozen MLLM.

Do not add:

```text
dataset ID
source ID
Stage-1 probability
```

to the primary model.

---

## 14. Why autoregressive program decoding

A simultaneous independent per-layer head can still average incompatible routes.

Example:

```text
Route A: WO   → FULL → IGNORE
Route B: FULL → RO   → FULL
```

Independent marginals can produce a hybrid program that was never a successful trajectory.

Autoregressive decoding models:

```text
P(tau | trigger state)
=
Π_l P(a_l | trigger state, previous program actions)
```

so later actions depend on the chosen program prefix.

---

## 15. Training objective

For one replay-valid program:

```text
tau = [a_L*, ..., a_27]
```

use teacher-forced sequence NLL:

```text
L_program(tau)
=
- Σ_{l=L*}^{27}
    log P_theta(
        a_l
        | c, L*, a_L*, ..., a_{l-1}
    )
```

Apply the hierarchical weights above.

For multiple valid W programs, train on every unique complete program with normalized per-UID route weights.

---

## 16. Initialization

Initialize compatible parts of:

```text
READ branch
WRITE branch
trigger-context projection
```

from the existing Stage-2-A checkpoint.

Initialize the new program decoder randomly.

Train:

```text
READ branch
WRITE branch
context projection
program decoder
```

jointly.

Freeze:

```text
MLLM backbone
Stage-1
```

---

## 17. Internal train/dev protocol

The scientific result must use the full training corpus.

Use a two-stage optimization protocol:

### Step A — group-disjoint internal development

Reserve a UID/image-group-disjoint internal dev split only to determine:

```text
stable optimizer settings
epoch count / early-stop point
```

Do not perform architecture sweeps.

Do not tune on the four external benchmarks.

### Step B — full-corpus refit

After the epoch count is frozen:

```text
reinitialize from the same starting point
train on 100% of eligible training-side UIDs/programs
for the frozen epoch count
```

Only the full-refit checkpoint is used for final external evaluation.

---

## 18. Small smoke is implementation-only

A tiny smoke run is allowed only to validate:

```text
program corpus loading
P90 alignment
padding/masking
teacher forcing
decoder output
program execution
same-layer READ semantics
checkpoint save/load
```

Do not use smoke accuracy as scientific evidence.

Do not stop after a 50/100/500-sample result unless there is an implementation/runtime failure.

---

## 19. Inference contract

For a test sample:

```text
run FULL normally
evaluate frozen Stage-1 sequentially

if no trigger:
    remain FULL
    Stage-2 never runs

if first trigger occurs at L*:
    compute z_R(L*), z_W(L*)
    build trigger context c
    decode one complete suffix program
    execute that program from L* through layer 27
```

The full suffix program is chosen before routed suffix execution.

Stage-2 does not re-decide after each executed layer in this experiment.

---

## 20. Inference decoding

Primary decoding:

```text
beam search width = 8
```

over the 4-action autoregressive decoder.

Suffix length is fixed by L*, so score with:

```text
sum of action log probabilities
```

Select the highest-scoring complete program.

No:

```text
correctness oracle
MCTS at inference
external-benchmark reranking
hand-designed intervention penalty
```

Also log greedy decoding for diagnostics, but beam-8 top-1 is the pre-specified primary method.

---

## 21. Preserve the Stage-1 handoff semantics

The contract remains:

```text
l < L*:
    FULL

l >= L*:
    predicted suffix program
```

Interpretation:

```text
Stage-1:
    when should dense computation stop being trusted?

Stage-2:
    once handed off, what READ/WRITE program should govern the remaining suffix?
```

---

## 22. Mandatory full external evaluation

After full-corpus training, run all four complete benchmark families:

```text
1. ChartQA
2. TextVQA
3. MMMU-Pro
4. POPE
```

Use the same benchmark variants, prompts, answer normalization, manifests, and evaluator contracts as the previous full evaluation.

Expected established counts:

```text
ChartQA   ~ 2,500
TextVQA   ~ 5,000
MMMU-Pro  ~ 3,460
POPE      ~ 9,000
Total     ~ 19,960
```

Do not substitute a smaller subset.

---

## 23. Baselines on the same manifest

Compare:

```text
B0. Dense FULL
B1. robust Stage-1 P90 + current sequential Stage-2-A
B2. robust Stage-1 P90 + new suffix-program predictor
```

Prefer rerunning all three in the same environment.

If old B0/B1 outputs are reused, require exact:

```text
model
manifest
prompt
decoder
evaluator
```

parity and record hashes.

---

## 24. Primary metrics

For every benchmark and pooled overall report:

```text
Dense accuracy
Sequential-A accuracy
Program-router accuracy

Program - Dense delta
Program - Sequential-A delta
```

Against Dense, report:

```text
W→C
C→W
C→C
W→W
Net = W→C - C→W
```

Primary success condition:

```text
pooled Net > 0
```

Stronger success condition:

```text
pooled Net > 0
AND
Program accuracy >= Dense
```

---

## 25. Paired uncertainty

Use sample/UID-level paired bootstrap for:

```text
Program accuracy - Dense accuracy
```

Report 95% CI overall and benchmark-wise where support permits.

Do not overinterpret a tiny positive point estimate with an inconclusive interval.

---

## 26. Stage-1 / Stage-2 funnel

For each benchmark report:

```text
N
Dense C / W
Stage-1 triggered C / W
P(trigger | C)
P(trigger | W)

triggered samples with all-FULL predicted program
triggered samples with any non-FULL
triggered-W with any non-FULL
triggered-C with any non-FULL

W→C
C→W
```

This separates Stage-1 admission from Stage-2 treatment quality.

---

## 27. Program-level diagnostics

For each triggered test sample log:

```text
trigger layer L*
predicted suffix program
beam top-1 score
beam top-k scores
# non-FULL actions
first non-FULL layer
action histogram
# contiguous non-FULL segments
```

Aggregate:

```text
mean/median # non-FULL
all-FULL program rate
first-intervention delay
RO / WO / IGNORE usage
```

---

## 28. Program coherence diagnostics

On training-side internal dev only, measure:

```text
exact match to any known valid program
prefix match
edit distance to nearest known valid program
```

These are diagnostics, not primary success criteria.

A newly generated program may be valid even if it was not in the MCTS corpus.

End-to-end answer correctness takes precedence.

---

## 29. Multiple-valid-program diagnostics

For W UIDs with multiple valid programs report:

```text
# known programs per UID
pairwise Hamming distance
action entropy by relative suffix position
```

Check whether the predictor outputs one coherent trajectory rather than a hybrid of unrelated local marginals.

---

## 30. Dense-C preservation analysis

Among Stage-1-triggered Dense-C report:

```text
% predicted all-FULL program
% predicted any non-FULL
C→W rate conditional on any non-FULL
C→C preservation
```

The intended conservative behavior is:

```text
triggered Dense-C → all-FULL suffix
```

unless the model has strong learned evidence otherwise.

---

## 31. Triggered-W treatment analysis

Among Stage-1-triggered Dense-W report:

```text
% predicted all-FULL
% predicted any non-FULL
W→C conditional on any non-FULL
W→C by first non-FULL action
W→C by # non-FULL actions
W→C by trigger-layer bin
```

This tests treatment quality rather than intervention frequency alone.

---

## 32. Benchmark-specific inspection

### ChartQA / TextVQA

Previous Stage-2 had both rescues and regressions, especially TextVQA.

Question:

```text
Does program-level supervision reduce C→W while retaining/increasing W→C?
```

### MMMU-Pro

Previous sequential Stage-2 intervened on many W samples but produced zero rescues.

Question:

```text
Does coherent suffix-program prediction produce any real rescue?
```

### POPE

Previous P90 Stage-1 was effectively inactive.

Question:

```text
Does the new Stage-2 remain inactive when Stage-1 does not hand off?
```

Do not change Stage-1 to force POPE activity.

---

## 33. Efficiency statistics

Secondary only.

Report:

```text
average READ-off layers after trigger
average WRITE-off layers after trigger
average IGNORE layers
fraction of full 28-layer READs preserved
fraction of full 28-layer WRITEs preserved
```

If reliable profiling already exists, report approximate visual-computation savings.

Do not introduce a new efficiency-accuracy trade-off objective in this phase.

---

## 34. Failure interpretation

### Outcome A — positive pooled Net and accuracy recovery/improvement

Interpretation:

> Full post-trigger trajectory supervision is better aligned with the routing problem than local action imitation.

### Outcome B — fewer regressions but still few rescues

Interpretation:

> Program prediction improves conservatism/coherence, but corrective-program generalization or Stage-1 admission remains limiting.

Do not immediately abandon the trajectory formulation.

### Outcome C — many interventions but poor W→C

Interpretation:

> Full-program supervision alone is insufficient; the trigger-state representation may not predict which program transfers to unseen samples.

### Outcome D — almost always all-FULL

Interpretation:

> Preservation supervision dominates the program model.

Inspect W-program likelihood and decoding before redesigning the architecture.

### Outcome E — internal route imitation is strong but external benchmark remains negative

Interpretation:

> Fitting known programs is not sufficient for OOD treatment generalization.

The next bottleneck is likely representation/distribution generalization rather than local label ambiguity.

---

## 35. What a positive result can establish

A positive result supports:

```text
trajectory/program-level supervision
>
local action imitation
```

for this post-trigger READ/WRITE routing problem under the current Stage-1 and route corpus.

It can also show that MCTS labels are useful when retained as coherent programs.

---

## 36. What this cannot establish

Do not claim:

```text
the learned program is globally optimal
MCTS found every valid route
the trigger point is optimal
a unique correct route exists
this method is identical to POLAR
```

This is POLAR-inspired program supervision adapted to the current READ/WRITE controller.

---

## 37. No small-subset scientific gate

Explicitly do not:

```text
train on 50/100/500 samples
evaluate on tiny benchmark subsets
stop the direction because of a tiny result
```

Allowed only:

```text
implementation smoke
internal training-stability check
```

After parity passes, proceed to:

```text
full route-corpus training
+
full four-benchmark evaluation
```

---

## 38. Required artifacts

Use:

```text
analysis/dense_failure_stage2/polar_suffix_program/
```

Create:

```text
protocol.md

corpus/
    p90_program_corpus_manifest.jsonl
    uid_program_index.json
    program_provenance.jsonl
    corpus_summary.csv
    source_outcome_counts.csv
    program_cardinality_per_uid.csv
    replay_validation.jsonl
    completeness_audit_program_import.jsonl

splits/
    internal_dev_groups.json
    full_refit_manifest.json

model/
    model_config.json
    initialization_report.md
    parameter_count.json

training/
    internal_dev_train_log.jsonl
    internal_dev_summary.md
    frozen_epoch_selection.json
    full_refit_train_log.jsonl
    full_refit_checkpoint_manifest.json

decoding/
    beam8_config.json
    decoding_contract.md

evaluation/
    chartqa/
    textvqa/
    mmmu_pro/
    pope/

metrics/
    full_benchmark_summary.csv
    transition_counts.csv
    paired_bootstrap.csv
    stage1_stage2_funnel.csv
    action_usage.csv
    program_statistics.csv
    trigger_layer_breakdown.csv
    dense_c_preservation.csv
    triggered_w_treatment.csv
    benchmark_breakdown.csv

diagnostics/
    internal_program_match.csv
    multi_route_coherence.csv
    nearest_valid_program_distance.csv
    beam_statistics.csv

figures/
    benchmark_accuracy_comparison.png
    rescue_regression_comparison.png
    stage1_stage2_funnel.png
    nonfull_program_length.png
    trigger_to_first_intervention.png
    action_usage_by_benchmark.png
    program_coherence.png

summaries/
    polar_suffix_program_full_eval_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 39. `polar_suffix_program_full_eval_summary.md` must answer

1. How many P90-triggered training UIDs were used?
2. How many unique complete suffix programs were used?
3. How many came from preservation / single / MCTS / robust search / completeness audit?
4. What was the valid-program count distribution per W UID?
5. Did full-corpus training complete?
6. What were Dense / Sequential-A / Program accuracies on all four benchmarks?
7. What were W→C and C→W per benchmark and pooled?
8. Is pooled Net positive?
9. Is Program accuracy >= Dense?
10. Did C preservation improve over Sequential-A?
11. Did treated-W success improve?
12. Did MMMU-Pro obtain rescues?
13. Did POPE remain inactive under unchanged Stage-1?
14. How often did the predictor output all-FULL after trigger?
15. How many non-FULL actions were typically predicted?
16. Did predicted programs resemble coherent known valid trajectories?
17. Is the result evidence for or against the trajectory-level Stage-2 formulation?
18. What bottleneck remains?

---

## 40. `next_stage2_recommendation.md`

Recommend exactly one next step.

If pooled Net becomes positive:

```text
refine the program predictor / preference objective
without changing Stage-1 yet
```

If regressions fall but rescue remains weak:

```text
diagnose corrective-program generalization on triggered-W
```

If it fails similarly to sequential Stage-2:

```text
shift attention from local-label ambiguity
to trigger-state representation / distribution generalization
```

State:

```text
why the next experiment is discriminating
what it can establish
what it cannot establish
```

---

## 41. Stop rule

This phase ends only after:

```text
1. full P90-aligned program-corpus construction
2. exact replay validation
3. full program-predictor training
4. full-corpus refit
5. complete ChartQA evaluation
6. complete TextVQA evaluation
7. complete MMMU-Pro evaluation
8. complete POPE evaluation
9. pooled rescue/regression analysis
10. one evidence-based next-step recommendation
```

Do not stop after a small pilot unless there is an implementation/runtime failure.

---

## 42. Core principle

The Stage-1 handoff remains:

```text
when should dense computation stop being trusted?
```

The new Stage-2 question is:

```text
given that handoff, what complete READ/WRITE routing program should govern the remaining suffix?
```

MCTS was introduced because correction is a joint multi-layer trajectory problem.

This experiment trains Stage-2 using that trajectory as the supervision unit instead of collapsing it into ambiguous local action labels.
