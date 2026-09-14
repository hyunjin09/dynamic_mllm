# Closed-Loop Stage-2 + Trajectory-Set Objective
## Full training on all available P90-aligned routes + full evaluation on ChartQA, TextVQA, MMMU-Pro, and POPE

## 1. Goal

Keep the original intended Stage-2 inference:

```text
Before first Stage-1 trigger:
    FULL only

At first trigger layer L*:
    Stage-2 begins

For every layer l >= L*:
    observe the actual current routed hidden state s_l
    predict {FULL, READ_ONLY, WRITE_ONLY, IGNORE}
    execute the action
    observe the resulting routed state s_{l+1}
    repeat
```

The new experiment changes the **training objective**, not the Stage-1 handoff or the basic Stage-2 representation.

The central hypothesis is:

> Multiple successful MCTS/search trajectories should be supervised as a trajectory set. The router should place high probability on at least one coherent successful trajectory, rather than being forced to imitate every discovered route or collapsing the routes into local valid-action sets.

This phase must run:

```text
full route-corpus training
→ full-corpus refit
→ full ChartQA
→ full TextVQA
→ full MMMU-Pro
→ full POPE
```

Tiny subsets are allowed only for implementation smoke tests.

---

## 2. Why this is the next discriminating experiment

Earlier results:

```text
Sequential-A:
    W→C = 3
    C→W = 19
    Net = -16

Open-loop suffix-program predictor:
    W→C = 3
    C→W = 8
    Net = -5
```

The label-completeness audit also showed that local action validity is highly ambiguous:

```text
KEEP invalidation      = 99.2%
INTERVENE invalidation = 54.2%
822 / 1,200 audited states:
    all four first actions had some successful bounded suffix
```

The program predictor preserved whole trajectories, but predicted the whole suffix from the trigger snapshot:

```text
s_L* → [a_L*, a_L*+1, ..., a_27]
```

The beam audit showed poor corrective candidate generation:

```text
Triggered W      = 496
W→C@1            = 3
W→C@8            = 33
Ranking failures = 30
Generation fail  = 463
```

However, that result applies to an **open-loop** predictor. Our actual intended router is **closed-loop**:

```text
s_L* → a_L*
      ↓ execute
s_L*+1 → a_L*+1
        ↓ execute
...
```

This experiment tests whether actual per-layer state feedback plus trajectory-level supervision is sufficient.

---

## 3. Freeze everything else

Freeze:

```text
Qwen2.5-VL-7B-Instruct
28 decoder layers
robust ALL-source Stage-1
P90 Stage-1 threshold
same trigger semantics
same four actions
same READ/WRITE execution semantics
same LMMS-Eval correctness
same external benchmark manifests/contracts
```

Actions:

```text
FULL       = READ1 WRITE1
READ_ONLY  = READ1 WRITE0
WRITE_ONLY = READ0 WRITE1
IGNORE     = READ0 WRITE0
```

Do not change Stage-1 in this phase.

---

## 4. Stage-2 representation

Use the existing closed-loop Stage-2 representation family.

At routed layer `l`:

```text
current text/query hidden state + current routed visual tokens
    → READ branch
    → z_R(s_l)

current routed visual tokens
    → WRITE branch
    → z_W(s_l)

[z_R ; z_W]
    → shared MLP/action head
    → 4 logits
```

Primary experiment must not add:

```text
layer embeddings
dataset/source IDs
Stage-1 score
pre-trigger history
program decoder
```

The purpose is to isolate the objective/formulation change.

---

## 5. Full training population

Use the full P90-aligned route corpus already built:

```text
569 UIDs
4,948 unique replay-valid complete suffix programs
```

Use all eligible provenance:

```text
preservation
single-intervention
original MCTS
robust corrective search
completeness-audit recovered programs
```

Do not train on a reduced scientific subset.

---

## 6. Dense-C target

For a training UID where:

```text
P90 triggers
AND
Dense FULL answer is correct
```

use one canonical preservation trajectory:

```text
FULL, FULL, ..., FULL
```

from trigger layer `L*` through layer 27.

Thus:

```text
T_i = {tau_FULL}
```

Do not add alternative Dense-C non-FULL routes in the primary run.

Reason: a false-positive Stage-1 trigger should conservatively preserve the already-correct Dense computation.

---

## 7. Dense-W target

For a training UID where:

```text
P90 triggers
AND
Dense FULL answer is wrong
```

keep every unique replay-valid successful corrective trajectory:

```text
T_i = {tau_i1, tau_i2, ..., tau_iK}
```

Each trajectory must contain the actual sequence:

```text
(s_L*, a_L*),
(s_L*+1, a_L*+1),
...
(s_27, a_27)
```

where every `s_l` is produced by executing the previous actions of that same route.

Do not collapse these into:

```text
local majority labels
local valid-action sets
one arbitrary "best" route
```

---

## 8. Exact routed-state reconstruction

Replay every training program from the P90 trigger.

For each layer cache enough frozen-backbone state to reconstruct the trainable Stage-2 input exactly.

Store:

```text
UID
image/content group
dataset/source regime
Dense C/W outcome
trigger layer
trajectory ID
trajectory provenance
layer
action
exact prefix-action hash
text/control hidden input
routed visual-token hidden input
state hash/reference
```

The cached state is the state **before choosing the current layer action**.

Because z_R/z_W remain trainable, cache their input hidden states rather than only frozen z_R/z_W outputs.

---

## 9. Replay parity

Every retained route must pass exact replay:

```text
same P90 trigger
same action prefix
same routed states
same final answer/correctness
```

Quarantine any route with:

```text
prefix mismatch
state mismatch
missing cached feature
final correctness mismatch
runtime error
```

Report retained/quarantined counts.

---

## 10. Do not reuse the old objectives

### Old independent-route CE

Do not optimize:

```text
(1/K_i) Σ_k [-Σ_l log π(a_l^ik | s_l^ik)]
```

because this asks one policy to imitate all discovered successful trajectories simultaneously.

### Local valid-set loss

Do not optimize:

```text
-log Σ_{a in A_valid(s)} π(a|s)
```

as the main objective.

The completeness audit showed that `A_valid(s)` can become nearly all four actions, erasing preference/trajectory information.

---

## 11. Primary trajectory-set marginal objective

For successful trajectory `tau_ik`:

```text
route_logp_ik
=
Σ_{l=L*}^{27}
log π_theta(a_l^ik | s_l^ik)
```

For UID `i` with `K_i` successful trajectories:

```text
L_i
=
-(1/T_i)
[
    logsumexp(route_logp_i1, ..., route_logp_iK)
    - log K_i
]
```

where:

```text
T_i = 28 - L*
```

is suffix length.

Interpretation:

> For this sample, the router only needs to assign high probability mass to at least one coherent successful trajectory.

The `1/T_i` term prevents earlier-triggered samples from receiving larger loss simply because their suffix is longer.

---

## 12. Soft responsibility over routes

The log-sum-exp induces soft trajectory responsibilities:

```text
r_ik = softmax_k(route_logp_ik)
```

This allows the model to specialize toward one or a few successful strategies for a UID instead of averaging incompatible routes.

Log:

```text
responsibility entropy
top-responsibility route share
```

during training.

Do not use hard route selection.

---

## 13. No extra preference heuristics yet

Primary run must not add:

```text
shortest-route preference
non-FULL penalty
action-specific reward
search-order weight
MCTS visit-count weight
```

All replay-valid Dense-W trajectories enter the set symmetrically.

This keeps the experiment focused on the supervision-unit/objective change.

---

## 14. UID weighting

One UID is one primary loss unit.

The number of routes `K_i` must not linearly increase that UID's weight.

Use the existing balancing policy across:

```text
dataset × source regime × Dense outcome
```

where support permits.

These identities are optimization metadata only, not model inputs.

---

## 15. Exact objective implementation

Batch by UID.

Each UID contributes all its successful routes to one marginal loss.

If a UID has too many routes for one device batch:

```text
microbatch routes
but compute the exact gradient-preserving logsumexp over all K_i
```

Do not replace the exact set objective with random one-route sampling in the primary experiment.

Before training, validate:

```text
manual small-case logsumexp equality
gradient parity against a brute-force implementation
numerical stability for long suffixes
```

---

## 16. Initialization and trainable parameters

Initialize from the existing closed-loop Sequential-A Stage-2 configuration/checkpoint where compatible.

Train:

```text
READ branch
WRITE branch
shared action head
```

Freeze:

```text
MLLM backbone
Stage-1
```

Do not use the open-loop program decoder.

---

## 17. Training protocol

### A. Implementation smoke

Use only enough UIDs to verify:

```text
route grouping
cached-state loading
trajectory log-likelihood
set marginal loss
gradient flow
closed-loop execution
checkpoint save/load
```

No scientific conclusion from smoke performance.

### B. Internal group-disjoint development

Use UID/image-group-disjoint development only to freeze:

```text
optimizer stability
epoch count / early-stop point
learning rate only if necessary
```

Avoid broad hyperparameter sweeps.

Do not tune on the four external benchmarks.

### C. Full-corpus refit

After the schedule is frozen:

```text
reinitialize identically
train on all eligible 569 UIDs
and all replay-valid trajectories
for the frozen epoch count
```

Only this full-refit checkpoint goes to external evaluation.

---

## 18. Closed-loop inference

For each evaluation sample:

```text
run FULL normally before trigger

if Stage-1 never triggers:
    remain FULL

if first trigger occurs at L*:
    for l = L* ... 27:
        read actual current routed state s_l
        compute z_R(s_l), z_W(s_l)
        choose argmax action
        execute it
        continue from the actual resulting state
```

Primary inference is deterministic greedy per-layer routing.

No:

```text
beam search
MCTS
oracle lookahead
program reranker
stochastic action sampling
```

---

## 19. Mandatory full evaluation

Run the complete established four benchmark families:

```text
ChartQA   ~ 2,500
TextVQA   ~ 5,000
MMMU-Pro  ~ 3,460
POPE      ~ 9,000
Total     ~ 19,960
```

Use the exact existing:

```text
sample manifests
prompt contracts
generation settings
answer normalization
correctness evaluator
```

Do not substitute a smaller subset.

---

## 20. Required baselines

Compare on the exact same manifest:

```text
B0 Dense FULL
B1 P90 + Sequential-A
B2 P90 + Open-loop Program predictor
B3 P90 + Closed-loop Trajectory-Set policy
```

Known prior transition counts:

```text
Sequential-A:
    W→C = 3
    C→W = 19
    Net = -16

Open-loop Program:
    W→C = 3
    C→W = 8
    Net = -5
```

Reuse prior outputs only if exact parity is verified.

---

## 21. Primary metrics

For every benchmark and pooled overall report:

```text
accuracy
W→C
C→W
C→C
W→W
Net = W→C - C→W
```

Main table:

| Method | Accuracy | W→C | C→W | Net |
|---|---:|---:|---:|---:|
| Dense | | | | |
| Sequential-A | | 3 | 19 | -16 |
| Open-loop Program | | 3 | 8 | -5 |
| Closed-loop Trajectory-Set | | | | |

Primary success:

```text
pooled Net > 0
```

Stronger success:

```text
pooled Net > 0
AND
accuracy >= Dense
```

---

## 22. Paired uncertainty

Use paired sample-level bootstrap for:

```text
Closed-loop - Dense accuracy
Closed-loop - Open-loop Program accuracy
```

Report point estimates and 95% CIs overall and per benchmark where appropriate.

---

## 23. Stage-1 / Stage-2 funnel

Per benchmark report:

```text
N
Dense C / W
triggered C / W
P(trigger | C)
P(trigger | W)

triggered-C with any non-FULL
triggered-W with any non-FULL

W→C
C→W
```

Stage-1 is unchanged, so this directly isolates treatment behavior.

---

## 24. Closed-loop routing diagnostics

For every triggered sample log:

```text
trigger layer
action at every routed layer
# non-FULL actions
first non-FULL layer
trigger-to-first-intervention delay
RO / WO / IGNORE counts
action switches
```

Aggregate separately for:

```text
Dense-C
Dense-W
W→C
C→W
C→C
W→W
```

---

## 25. State-feedback sanity check

Verify that later Stage-2 decisions are computed from the actual state created by earlier actions.

For routed examples log:

```text
previous action
next-layer state reference/hash
next-layer logits
```

This is a contract/sanity check, not a causal analysis.

---

## 26. Benchmark-specific questions

### ChartQA

Does closed-loop feedback increase rescue without returning to the high regression rate of Sequential-A?

### TextVQA

Open-loop Program had:

```text
W→C = 0
C→W = 0
```

Does actual per-layer state feedback recover any corrective behavior?

### MMMU-Pro

Open-loop Program had:

```text
W→C = 2
C→W = 2
```

Does closed-loop feedback increase the number of real rescues?

### POPE

Stage-1 was effectively inactive.

Do not alter Stage-1 to create activity.

---

## 27. Outcome interpretation

### Outcome A — W rescue rises and Net becomes positive

Interpretation:

> Closed-loop state feedback + trajectory-set supervision is better aligned with the routing problem than both local imitation and open-loop program prediction.

### Outcome B — W rescue rises but C→W also rises

Interpretation:

> Correction became learnable, but preservation/selectivity remains insufficient.

Do not abandon closed-loop routing.

### Outcome C — C preservation remains good but W→C stays near 3

Interpretation:

> Multiple-route objective and closed-loop feedback are not enough with the current z_R/z_W representation.

Representation/generalization becomes the next bottleneck.

### Outcome D — internal trajectory-set fit is strong but external free-run rescue remains weak

Interpretation:

> Offline successful-route states may not match the policy's on-policy state distribution.

A later on-policy diagnostic/relabeling experiment becomes justified.

### Outcome E — even internal trajectory-set fit is poor

Interpretation:

> Check optimization/representation capacity before blaming exposure.

---

## 28. What a positive result supports

A positive full-benchmark result supports:

```text
closed-loop per-layer state feedback
+
trajectory-set supervision
```

over:

```text
local single-label CE
local valid-action-set loss
open-loop whole-suffix prediction
```

under the same Stage-1 handoff.

---

## 29. What a negative result does not prove

A negative result does not establish that:

```text
dynamic READ/WRITE routing is impossible
MCTS is useless
Stage-1 is wrong
richer state representations cannot work
on-policy relabeling cannot help
```

It only tests this exact formulation.

---

## 30. No small-subset scientific gate

Do not:

```text
train on a few dozen/few hundred samples
run a tiny benchmark subset
stop because a pilot score is weak
```

Small runs are allowed only for implementation correctness.

Once implementation parity passes:

```text
FULL TRAIN
→ FULL REFIT
→ FULL 4-BENCHMARK EVAL
```

must be completed.

---

## 31. Required artifacts

Use:

```text
analysis/dense_failure_stage2/closed_loop_trajectory_set/
```

Create:

```text
protocol.md

corpus/
    trajectory_set_manifest.jsonl
    uid_to_trajectories.json
    trajectory_provenance.jsonl
    replay_validation.jsonl
    corpus_summary.csv
    trajectory_count_per_uid.csv
    routed_state_cache_manifest.jsonl
    state_feature_schema.md

objective/
    objective_implementation.md
    numerical_stability_tests.md
    gradient_parity_tests.md

splits/
    internal_dev_groups.json
    full_refit_manifest.json

training/
    initialization_report.md
    model_config.json
    parameter_count.json
    internal_dev_train_log.jsonl
    internal_dev_summary.md
    frozen_epoch_selection.json
    full_refit_train_log.jsonl
    responsibility_statistics.csv
    full_refit_checkpoint_manifest.json

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
    intervention_statistics.csv
    trigger_layer_breakdown.csv
    dense_c_preservation.csv
    triggered_w_treatment.csv
    benchmark_breakdown.csv

diagnostics/
    internal_trajectory_likelihood.csv
    trajectory_responsibility.csv
    state_feedback_analysis.csv

figures/
    benchmark_accuracy_comparison.png
    rescue_regression_comparison.png
    stage1_stage2_funnel.png
    action_usage_by_benchmark.png
    trigger_to_first_intervention.png
    trajectory_responsibility_entropy.png
    closed_loop_vs_open_loop.png

summaries/
    closed_loop_trajectory_set_full_eval_summary.md
    next_stage2_recommendation.md

artifact_manifest.json
```

---

## 32. `closed_loop_trajectory_set_full_eval_summary.md` must answer

1. How many training UIDs and replay-valid trajectories were used?
2. How many were Dense-C preservation versus Dense-W corrective?
3. What was the provenance distribution?
4. Did routed-state replay parity pass?
5. Did the exact marginal objective pass numerical/gradient tests?
6. Did full-corpus refit complete?
7. What are Dense / Sequential-A / Open-loop Program / Closed-loop accuracies?
8. What are W→C/C→W/Net per benchmark and pooled?
9. Is pooled Net positive?
10. Is routed accuracy >= Dense?
11. Did W rescue improve over both prior Stage-2 methods?
12. Did C preservation remain improved over Sequential-A?
13. What fraction of triggered-W receives any non-FULL?
14. What fraction of triggered-C receives any non-FULL?
15. What is trigger-to-first-intervention delay?
16. Did TextVQA obtain rescues?
17. Did MMMU-Pro improve beyond two rescues?
18. Did POPE remain inactive when Stage-1 did not trigger?
19. Did route responsibilities specialize onto a subset of successful trajectories?
20. Does the evidence support the closed-loop trajectory-set formulation?
21. If it still fails, is the next bottleneck more consistent with optimization, representation/generalization, or on-policy state-distribution shift?
22. What does the experiment not justify concluding?

---

## 33. `next_stage2_recommendation.md`

Recommend exactly one next experiment.

If Net becomes positive:

```text
keep this formulation and refine only the smallest remaining bottleneck
```

If rescue rises but regression is excessive:

```text
training-side conservative treatment/selectivity mechanism
```

If internal trajectory fit is good but external rescue remains weak:

```text
one on-policy state-distribution diagnostic/relabeling experiment
```

If internal fit is weak:

```text
one minimal Stage-2 representation enrichment experiment
```

State why the next experiment is discriminating and what positive/negative outcomes mean.

---

## 34. Stop rule

This phase ends only after:

```text
1. all 4,948 programs reconstructed as closed-loop routed trajectories
2. exact routed-state cache/replay validation
3. objective numerical + gradient validation
4. internal group-disjoint schedule selection
5. full 569-UID refit
6. full ChartQA
7. full TextVQA
8. full MMMU-Pro
9. full POPE
10. pooled W→C / C→W / Net analysis
11. one evidence-based next-step recommendation
```

Do not stop after a small pilot unless there is an implementation/runtime failure.

---

## 35. Core principle

The intended Stage-2 is:

```text
trigger
→ observe current routed hidden state
→ choose current action
→ execute
→ observe the actual next routed hidden state
→ choose again
```

MCTS/search provides multiple successful **closed-loop trajectories**.

Therefore the training target should not ask:

```text
"What is the unique correct local action?"
```

and should not ask:

```text
"Can one trigger snapshot predict the whole future program?"
```

Instead:

> **Can the closed-loop router assign high probability to at least one coherent successful trajectory for each triggered failure case, while preserving all-FULL behavior for false-positive triggers?**

This experiment tests that question with full training and the complete four-benchmark evaluation.
