# READ Counterfactual Stage-1 Branch-Critic Diagnostic
## Can the frozen Stage-1 failure predictor choose between READ ON and READ OFF after one hypothetical layer?

## 0. Immediate instruction: stop the previous bounded-planning phase

The previous `READ-only bounded planning / search` phase is no longer the active priority.

Before running this experiment:

```text
STOP submitting any new bounded-planning/search jobs.
STOP the currently running Phase-85 / read_only_bounded_planning jobs.
```

Do this safely:

1. identify only jobs/processes that belong to the READ-only bounded-planning phase;
2. snapshot the current progress/status and currently written artifacts;
3. record active job IDs / PIDs / SLURM IDs and the reason for stopping;
4. terminate those jobs gracefully;
5. verify that no bounded-planning workers remain;
6. do not delete or overwrite any completed/intermediate results;
7. do not kill unrelated experiments or infrastructure.

If a running process cannot be confidently attributed to the old bounded-planning phase, do **not** kill it blindly. Record it and stop for clarification.

Preserve the existing interim report and caches as historical evidence. They may be useful later, but they are not the active experiment.

Create:

```text
analysis/read_counterfactual_stage1_branch_critic/
old_phase_stop_report.md
```

containing:

```text
what was running
what was stopped
job/process identifiers
last known progress
which artifacts were preserved
whether any old job remains
```

---

# 1. Why this experiment is needed

Previous READ experiments asked variants of:

```text
current state
→ predict READ utility directly
```

or:

```text
READ-ON/OFF state difference
→ predict q_WRITE_ONLY - q_FULL
```

Those were weak.

However, Stage-1 solves a different task:

```text
hidden state
→ probability that the trajectory will eventually be wrong
```

Stage-1 has already shown substantial in-domain failure-prediction signal.

Therefore we have not yet tested:

> Execute READ ON and READ OFF for one layer, score both resulting states with Stage-1, and choose the branch predicted to be safer.

This is not direct READ-utility regression.

It is a **counterfactual branch-outcome critic**.

---

# 2. Core question

At the same exact pre-action state `s_l`:

```text
READ ON:
    FULL = READ1 WRITE1
    → resulting state s_ON

READ OFF:
    WRITE_ONLY = READ0 WRITE1
    → resulting state s_OFF
```

Apply the same frozen Stage-1 failure predictor:

```text
p_ON  = Stage1(s_ON)
p_OFF = Stage1(s_OFF)
```

where larger score means higher predicted probability/risk of eventual failure.

Then ask:

```text
Does p_ON - p_OFF predict which branch actually has the better downstream outcome?
```

If:

```text
p_ON > p_OFF
```

the critic prefers READ OFF.

If:

```text
p_ON < p_OFF
```

the critic prefers READ ON.

---

# 3. Frozen population

Use the complete existing dense READ counterfactual population:

```text
1,413 P90-triggered UIDs
1,385 image groups
15,185 exact dense post-trigger states
```

Use the exact same state identities as the previous READ phases.

Do not create a small scientific subset.

If cached one-step branch states already exist, reuse them.

No new full-suffix Qwen execution is required for the primary diagnostic.

---

# 4. Frozen branch semantics

For every exact state `s_l`:

## ON branch

```text
layer l:
    FULL = READ1 WRITE1

later layers for the frozen ground-truth outcome:
    FULL
```

## OFF branch

```text
layer l:
    WRITE_ONLY = READ0 WRITE1

later layers for the frozen ground-truth outcome:
    FULL
```

The two branches differ only in READ at the current layer.

Reuse the existing frozen Step-A branch outcomes:

```text
q_ON
q_OFF
correct_ON
correct_OFF
```

where:

```text
q_ON  = q_FULL
q_OFF = q_WRITE_ONLY
```

and:

```text
H_R = q_OFF - q_ON
```

No new MCTS/search labels.

---

# 5. Critical Stage-1 indexing rule

This must be audited before any metric is computed.

After action at layer `l`, the resulting branch state must be scored at the **corresponding next valid Stage-1 evaluation position according to the original Stage-1 representation contract**.

Conceptually:

```text
s_l
→ execute action at l
→ resulting post-action state
→ Stage-1 score at the corresponding next-state position
```

Do not simply reuse the layer-`l` head/index if that is not how Stage-1 was trained.

Do not invent a synthetic layer 28.

For layer 27:

```text
if the frozen Stage-1 contract has no valid post-action evaluation point:
    exclude layer-27 states from the primary branch-critic metric
    report them separately
```

If the contract does define a valid final-state Stage-1 scoring representation, use it exactly and document it.

Before the full run, write:

```text
stage1_post_action_indexing_contract.md
```

with the exact tensor location and layer mapping.

---

# 6. Frozen Stage-1 model

Use the exact selected Stage-1 predictor from the existing pipeline:

```text
same checkpoint
same architecture
same feature extraction
same calibration
same P90 threshold tau
```

Do not retrain.

Do not recalibrate on the counterfactual states.

Do not choose a new threshold after inspecting results.

This experiment first tests **zero-shot reuse of the existing Stage-1 critic**.

---

# 7. Required branch scores

For every eligible state compute:

```text
p_ON
p_OFF
delta_p = p_ON - p_OFF
```

Interpretation:

```text
delta_p > 0:
    Stage-1 predicts ON is riskier
    → prefer READ OFF

delta_p < 0:
    Stage-1 predicts OFF is riskier
    → prefer READ ON
```

Store raw logits as well as calibrated probabilities if Stage-1 has both.

Primary comparisons should use the same score type that was used for the original Stage-1 AUROC.

---

# 8. Branch-wise failure prediction

Evaluate Stage-1 separately on the two branch populations.

## ON branch

Target:

```text
y_ON = 1 if ON/FULL branch eventually answers wrong
       0 if correct
```

Predictor:

```text
p_ON
```

Report:

```text
AUROC
AUPRC
Brier score
ECE / calibration curve
failure prevalence
```

## OFF branch

Target:

```text
y_OFF = 1 if OFF/WRITE_ONLY branch eventually answers wrong
        0 if correct
```

Predictor:

```text
p_OFF
```

Report the same metrics.

This comparison answers:

> Does the frozen Dense-trained Stage-1 generalize to a READ-OFF counterfactual state?

---

# 9. Branch-distribution shift diagnostic

The ON branch is essentially the normal dense continuation.

The OFF branch is counterfactual.

Compare:

```text
Stage-1 score distribution ON vs OFF
feature/representation norms where available
layer-wise score distribution
dataset/source score distribution
```

Do not interpret representation distance alone as failure.

The important result is whether `p_OFF` still predicts `y_OFF`.

---

# 10. Paired preference diagnostic

The key score is:

```text
delta_p = p_ON - p_OFF
```

Ground-truth branch preference should be measured in two ways.

## Continuous preference

```text
H_R = q_OFF - q_ON
```

Positive `H_R` means OFF is better.

Report:

```text
Spearman(delta_p, H_R)
Pearson(delta_p, H_R)
sign agreement for H_R > 0 / H_R < 0
```

Do not overinterpret tiny q differences.

---

# 11. Strong behavioral flip preference

This is the cleanest test.

Existing cohorts:

```text
READ-harmful flip:
    ON/FULL = Wrong
    OFF/WO  = Correct

READ-beneficial flip:
    ON/FULL = Correct
    OFF/WO  = Wrong
```

Expected approximate counts from the frozen READ census:

```text
harmful flips:    625
beneficial flips:  41
```

Verify exact eligible counts after indexing/layer-27 handling.

Desired critic behavior:

```text
harmful flip:
    p_ON > p_OFF

beneficial flip:
    p_ON < p_OFF
```

Report separately:

```text
harmful-flip preference accuracy
beneficial-flip preference accuracy
balanced preference accuracy
discordant-pair AUROC using delta_p
```

Do not report only raw accuracy because the flip classes are highly imbalanced.

Use image-group bootstrap 95% CIs.

---

# 12. Frozen threshold-crossing quadrants

Use the existing P90 Stage-1 threshold:

```text
tau
```

Classify every state into:

```text
Q1: ON safe,    OFF safe
    p_ON < tau, p_OFF < tau

Q2: ON safe,    OFF trigger
    p_ON < tau, p_OFF >= tau

Q3: ON trigger, OFF safe
    p_ON >= tau, p_OFF < tau

Q4: ON trigger, OFF trigger
    p_ON >= tau, p_OFF >= tau
```

For each quadrant report:

```text
state count
UID count
mean/median H_R
harmful prevalence H_R>0
ON correctness
OFF correctness
harmful-flip prevalence
beneficial-flip prevalence
```

Q3 is especially important.

Hypothesis:

> If READ OFF genuinely resolves the failure state, `ON trigger / OFF safe` should be enriched for ON-wrong/OFF-correct outcomes.

---

# 13. First-trigger-only offline policy

Before implementing any closed-loop controller, evaluate one conservative policy at each UID's **first Stage-1 trigger only**.

At first trigger `L*`, compute:

```text
p_ON
p_OFF
```

Use this frozen rule:

```text
if p_ON < tau:
    choose ON

else if p_OFF < tau:
    choose OFF

else:
    choose the branch with lower Stage-1 failure score
```

Equivalent table:

| ON next-state | OFF next-state | Choice |
|---|---|---|
| safe | safe | ON |
| safe | trigger | ON |
| trigger | safe | OFF |
| trigger | trigger | lower failure score |

Reason:

```text
if ordinary FULL already moves into a Stage-1-safe state,
do not suppress READ unnecessarily.
```

For exact ties:

```text
choose ON
```

---

# 14. Offline first-trigger outcome evaluation

Because this policy chooses only one action at the first trigger and then assumes FULL continuation, its outcome already exists in the frozen branch corpus.

For all 1,413 triggered UIDs report:

```text
Dense-W:
    # choose ON
    # choose OFF
    W→C
    W remains W

Dense-C:
    # choose ON
    # choose OFF
    C→W
    C remains C
```

Primary policy numbers:

```text
W→C
C→W
Net = W→C - C→W
```

Also report:

```text
treated-W success rate
treated-C regression rate
```

This is an offline one-decision diagnostic only.

It is **not** the final closed-loop controller result.

---

# 15. Decision categories

Do not decide from a single metric.

## BC-A — Strong zero-shot branch critic

Evidence pattern:

```text
OFF-branch failure AUROC remains clearly above chance;
delta_p separates harmful vs beneficial flips;
Q3 (ON-trigger/OFF-safe) is enriched for actual OFF-beneficial outcomes;
first-trigger-only policy has positive Net without excessive C→W.
```

Next step:

```text
full closed-loop Stage-1 branch-critic READ controller
```

## BC-B — Dense critic works, OFF branch suffers shift

Pattern:

```text
ON AUROC strong
OFF AUROC weak / near chance
paired preference weak
```

Interpretation:

> Existing Stage-1 does not generalize to READ-OFF routed states.

Next step:

```text
refit branch-failure critic on Dense + READ-OFF states
```

Do not conclude that branch-outcome prediction itself is impossible.

## BC-C — Both branches predictable but relative choice fails

Pattern:

```text
ON AUROC strong
OFF AUROC strong
but delta_p poorly predicts branch preference
```

Interpretation:

> Absolute failure prediction is learnable, but branch scores are not directly comparable/calibrated enough for action selection.

Next step:

```text
paired branch-risk calibration/ranking experiment
```

## BC-D — Branch-outcome prediction itself is weak

Pattern:

```text
ON/OFF branch AUROC weak
paired preference weak
threshold quadrants uninformative
```

Do not proceed to the closed-loop controller.

---

# 16. No arbitrary hard AUROC threshold

Do not decide from a post-hoc rule such as `AUROC > 0.7` alone.

Use together:

```text
branch AUROC
bootstrap uncertainty
flip preference
threshold-quadrant enrichment
first-trigger policy Net
```

The purpose is to determine whether the branch critic is actionable.

---

# 17. Layer / regime analysis

Report branch AUROC and paired preference by:

```text
absolute layer
Early / Middle / Late
trigger-relative depth
Dense-W vs Dense-C
dataset/source cell
```

These are descriptive subgroup checks.

Do not select one favorable subgroup after inspecting the results and call the method solved.

---

# 18. Cache and parity requirements

Before scoring verify:

```text
exact UID/layer alignment
same pre-state for ON/OFF
ON state equals canonical FULL post-state
OFF state equals canonical WRITE_ONLY post-state
frozen branch correctness/q labels
```

Run a fresh stratified parity smoke on at least:

```text
32 UIDs
```

covering:

```text
datasets
sources
early/middle/late layers
```

No scientific result until parity passes.

---

# 19. Compute policy

This experiment should be cheap.

Primary work should be:

```text
cached-state loading
Stage-1 head inference
metric computation
```

Do **not** launch:

```text
MCTS
beam search
B=256/512 search
new full-suffix branch generation
```

If required post-action states are missing, first report the missing fraction.

Only regenerate missing one-layer ON/OFF states if necessary.

---

# 20. Explicit stop: no closed-loop free-run yet

This phase stops after the offline branch-critic diagnostic.

Do **not** yet run the full controller:

```text
trigger
→ ON/OFF hypothetical execution
→ Stage-1 compare
→ commit
→ repeat at next layer
```

That controller is the next phase only if the diagnostic is promising.

---

# 21. Future controller semantics if BC-A is supported

For reference only:

```text
for each layer l:

    score the current actual routed state with Stage-1

    if current score < tau:
        execute FULL
        continue

    else:
        hypothetically execute:
            ON  = FULL
            OFF = WRITE_ONLY

        score resulting states:
            p_ON
            p_OFF

        if p_ON < tau:
            commit ON

        elif p_OFF < tau:
            commit OFF

        else:
            commit the lower-risk branch

    continue from the committed actual routed state
```

Stage-1 remains active later.

If the route becomes safe, use FULL.

If it triggers again, run a fresh ON/OFF comparison.

Do not implement this inside the current phase.

---

# 22. Output directory

Use:

```text
analysis/read_counterfactual_stage1_branch_critic/
```

Required artifacts:

```text
old_phase_stop_report.md
protocol.md
frozen_contract.json
stage1_post_action_indexing_contract.md

population/
    eligible_state_manifest.jsonl
    excluded_state_manifest.jsonl
    first_trigger_uid_manifest.jsonl
    cohort_counts.csv

parity/
    state_alignment.csv
    on_full_parity.csv
    off_wo_parity.csv
    smoke_report.md

scores/
    branch_scores.jsonl
    on_branch_metrics.csv
    off_branch_metrics.csv
    calibration_metrics.csv

paired/
    continuous_preference_metrics.csv
    strong_flip_preference.csv
    threshold_quadrants.csv
    layer_breakdown.csv
    trigger_relative_breakdown.csv
    dataset_source_breakdown.csv

policy/
    first_trigger_policy_decisions.jsonl
    first_trigger_policy_summary.csv
    first_trigger_dense_w.csv
    first_trigger_dense_c.csv

statistics/
    group_bootstrap_ci.csv

figures/
    on_vs_off_failure_roc.png
    on_vs_off_calibration.png
    delta_p_vs_read_harm.png
    strong_flip_delta_p.png
    threshold_quadrant_outcomes.png
    first_trigger_policy_outcomes.png

summaries/
    branch_critic_summary.md
    next_step_recommendation.md

artifact_manifest.json
```

---

# 23. `branch_critic_summary.md` must answer

1. Were the old bounded-planning jobs safely stopped?
2. How many READ states were eligible for Stage-1 post-action scoring?
3. How was post-action layer indexing mapped into Stage-1?
4. Were any layer-27 states excluded, and why?
5. What is ON-branch failure AUROC/AUPRC?
6. What is OFF-branch failure AUROC/AUPRC?
7. How well calibrated is Stage-1 on ON vs OFF states?
8. Does OFF show a large generalization collapse?
9. What is Spearman between `delta_p` and `H_R`?
10. On harmful flips, how often is `p_ON > p_OFF`?
11. On beneficial flips, how often is `p_ON < p_OFF`?
12. What is balanced flip-preference accuracy / AUROC?
13. How many states fall into each threshold quadrant?
14. Is `ON trigger / OFF safe` enriched for ON-wrong/OFF-correct?
15. Is `ON safe / OFF trigger` enriched for the opposite?
16. What does the first-trigger-only policy do on Dense-W?
17. What is its W→C count/rate?
18. What is its C→W count/rate on Dense-C?
19. What is Net?
20. Does the result support BC-A/B/C/D?
21. What does this experiment not establish?

---

# 24. `next_step_recommendation.md`

Recommend exactly one next step.

```text
BC-A:
    full closed-loop Stage-1 branch-critic READ controller

BC-B:
    counterfactual-state branch-failure critic refit

BC-C:
    paired branch-risk calibration/ranking experiment

BC-D:
    stop this branch-critic direction
```

---

# 25. Stop rule

STOP after:

```text
1. safely stopping old bounded-planning jobs
2. exact Stage-1 indexing audit
3. cached ON/OFF parity
4. frozen Stage-1 scoring of all eligible states
5. ON/OFF branch failure metrics
6. paired delta_p preference analysis
7. strong-flip analysis
8. threshold-quadrant analysis
9. first-trigger-only offline policy evaluation
10. one BC-A/B/C/D decision
11. exactly one next-step recommendation
```

Do not restart bounded search.

Do not retrain Stage-1.

Do not run the full closed-loop controller inside this phase.

---

# 26. Core principle

Previous READ experiments asked:

```text
Can I infer READ utility directly from the current state?
```

This experiment asks a different question:

> **If I actually execute the two candidate actions for one layer, can the already-successful Stage-1 failure predictor tell which resulting branch is safer?**

If yes, Stage-2 may not need to learn READ utility directly.

It may instead be:

```text
candidate action
→ resulting hidden state
→ failure critic
→ branch choice
```

That is the hypothesis tested here.
