# Stage-1 READ branch-critic diagnostic — BC-C

**Decision: BC-C.** The closest supported category is BC-C, qualified: both branches retain moderate above-chance absolute failure signal, but relative branch choice is unsupported. Balanced flip accuracy is 47.77%, discordant AUROC is 0.4684, and delta-p Spearman is -0.0187. Q3 rescue enrichment is uncertain. The offline policy has Net +24 (30 rescues, 6 regressions), but this does not establish useful action ranking or controller readiness. The apparent ON-to-OFF AUROC drop is not evidence of an OFF-score collapse: on fixed ON labels, ON/OFF scores give 0.6728/0.6710 AUROC; on fixed OFF labels they give 0.5948/0.5949. Changing the labels accounts for the observed gap even with the score held fixed. This supports a relative-choice failure description, not a claim that simple recalibration will solve it. The cause of poor paired preference remains unknown.

The complete requested offline diagnostic is finished. All results below use **15,185 states, 1,413 P90-triggered UIDs and 1,385 image groups**. They are conditional on this selected population and frozen five-head reuse; they are not held-out population-level Stage-1 training performance.

**Execution and indexing**

Phase85 was safely stopped: supervisor PID 2138508 and workers 2138512–2138515 were terminated with SIGTERM after progress/artifact snapshots. No live bounded-planning worker remains. Completed and intermediate search results, including the PRELIMINARY report, were preserved. See [old_phase_stop_report.md](../old_phase_stop_report.md).

All **15,185** states are eligible, including **1,413 layer-27 states**; excluded states: **0**. Original native Stage1 features are decoder-output features indexed by the layer that produced them. Post-action layer l therefore maps to **Stage1 index l**, including valid index27. No synthetic layer28 and no final decoder normalization were introduced. Exact feature order is final user token, mean user text, mean visual tokens, followed by the original normalization and predictor.

The old compact branch caches contain the last control token, not both Stage1 user-text summaries. OFF Stage1 features were missing for 100% of states. Only the required one-layer representations were reconstructed from dense baselines; all native ON features were reused and matched exactly. No branch suffix, answer generation, new search labels, retraining or calibration fit occurred.

Fresh stratified parity passed on **32 UIDs / 308 states** before the full run. Full extraction verifies compact prestate identity, shared branch inputs, action bits, native ON pooled features, canonical ON output and exact Phase82 ON/OFF compact post-state hashes at every state. Frozen q/correctness labels align with all 30,370 original branch records. Original ensemble ON scores also reproduce within the recorded numerical tolerance. See [indexing contract](../stage1_post_action_indexing_contract.md) and [full verification](../parity/full_scoring_verification.json).

The original Stage-1 trigger at layer l already uses the dense output of that layer; the frozen Step-A action is at its pre-input. Consequently this first-trigger comparison is retrospective and would require rollback to realize. The exact frozen labels and population were retained. A forward-only controller with a newly defined next-layer action is not evaluated here.

**Branch-wise prediction and calibration**

| Branch | AUROC [95% CI] | AUPRC [95% CI] | Brier | ECE | Failure prevalence |
|---|---|---|---:|---:|---:|
| ON | 0.6728 [0.6200, 0.7247] | 0.9671 [0.9548, 0.9773] | 0.0580 | 0.0110 | 93.70% |
| OFF | 0.5949 [0.5557, 0.6334] | 0.9235 [0.9085, 0.9377] | 0.0911 | 0.0284 | 89.85% |

Paired OFF-minus-ON AUROC: **-0.0779 [-0.1119, -0.0480]**. AUPRC is average precision and must be compared with each branch's high failure prevalence. ECE uses 10 fixed equal-width bins. The frozen score is the mean of five sigmoid probabilities with no additional fitted calibration; per-head raw logits and probabilities are retained. This selected P90 cohort and its training exposure limit any claim of broad Dense-to-counterfactual generalization. The [fixed-target cross-score diagnostic](../scores/fixed_target_cross_score_diagnostic.md) separates the observed change in target difficulty from changing the predictor scores.

[ROC figure](../figures/on_vs_off_failure_roc.png) · [Calibration figure](../figures/on_vs_off_calibration.png)

Distribution diagnostics (distance alone is not a failure criterion):

| branch   | variable                |      mean |       std |       p05 |    median |       p95 |
|:---------|:------------------------|----------:|----------:|----------:|----------:|----------:|
| ON       | p                       |   0.92593 |   0.04335 |   0.85031 |   0.92833 |   0.98588 |
| ON       | feature_norm            | 380.261   | 254.501   | 116.525   | 273.492   | 845.541   |
| ON       | normalized_feature_norm | 122.241   |  86.101   |  48.7035  |  83.6726  | 320.463   |
| OFF      | p                       |   0.92516 |   0.04409 |   0.84813 |   0.9282  |   0.98566 |
| OFF      | feature_norm            | 380.67    | 255.293   | 116.579   | 273.838   | 847.777   |
| OFF      | normalized_feature_norm | 122.605   |  86.5919  |  48.7727  |  83.8497  | 321.772   |

Layer-wise and dataset/source score distributions are in [branch_distribution_shift.csv](../scores/branch_distribution_shift.csv).

**Paired preference and strong behavioral flips**

- Spearman(delta_p, H_R): **-0.0187 [-0.0347, -0.0025]**.
- Pearson(delta_p, H_R): **-0.0185 [-0.0345, -0.0015]**.
- Strict sign agreement on nonzero H_R: **49.21%**. Fixed tiny-q sensitivity thresholds are separately reported; tiny q changes do not override behavioral flips.
- Harmful-flip preference accuracy: **54.08% [50.09, 58.16]**.
- Beneficial-flip preference accuracy: **41.46% [22.73, 60.00]**.
- Balanced flip-preference accuracy: **47.77% [38.29, 57.11]**.
- Discordant-pair AUROC (harmful versus beneficial, scored by delta_p): **0.4684 [0.3574, 0.5799]**.

| flip       |   states |   uids |   preferred_correctly |   score_ties |   preference_accuracy |   median_delta_p |
|:-----------|---------:|-------:|----------------------:|-------------:|----------------------:|-----------------:|
| harmful    |      625 |    223 |                   338 |            0 |              0.5408   |      0.000243873 |
| beneficial |       41 |     21 |                    17 |            0 |              0.414634 |      0.000808097 |

These exact eligible flip counts include layer27. Score ties count as neither strict ON nor strict OFF preference; policy ties retain ON. Confidence intervals use **5,000 paired image-group bootstrap draws**. Undefined missing-class replicates are counted in [group_bootstrap_ci.csv](../statistics/group_bootstrap_ci.csv), not replaced by chance scores.

[Continuous preference figure](../figures/delta_p_vs_read_harm.png) · [Strong-flip figure](../figures/strong_flip_delta_p.png)

**Frozen threshold quadrants**

P90 tau is **0.9061332901863008**. Historical admission remains strict >tau. This plan's quadrants use >=tau for trigger and <tau for safe. Exact threshold ties: **0**.

| quadrant   |   states |   uids |   mean_H_R |   median_H_R |   harmful_prevalence |   ON_correctness |   OFF_correctness |   harmful_flips |   harmful_flip_prevalence |   beneficial_flips |   beneficial_flip_prevalence |
|:-----------|---------:|-------:|-----------:|-------------:|---------------------:|-----------------:|------------------:|----------------:|--------------------------:|-------------------:|-----------------------------:|
| Q1         |     3353 |    754 |   -0.0049  |     -0.00069 |              0.48882 |          0.11452 |           0.14852 |             132 |                   0.03937 |                 18 |                      0.00537 |
| Q2         |      167 |    142 |   -0.02358 |     -0.00341 |              0.45509 |          0.11377 |           0.15569 |               7 |                   0.04192 |                  0 |                      0       |
| Q3         |      417 |    348 |   -0.01651 |     -0.00527 |              0.44844 |          0.08153 |           0.14149 |              27 |                   0.06475 |                  2 |                      0.0048  |
| Q4         |    11248 |   1353 |   -0.01202 |     -0.00249 |              0.47857 |          0.04623 |           0.08517 |             459 |                   0.04081 |                 21 |                      0.00187 |

Q3 (ON-trigger/OFF-safe) harmful-flip prevalence is **6.47% [3.96, 9.16]**; enrichment difference versus all states is **2.36 [-0.06, 4.99] percentage points**. Q2 (ON-safe/OFF-trigger) beneficial-flip prevalence is **0.00% [0.00, 0.00]**; enrichment difference is **-0.27 [-0.41, -0.15] percentage points**. Quadrant enrichment is descriptive and does not substitute for net policy outcomes. The zero-width empirical bootstrap interval for Q2 reflects zero observed beneficial flips; it does not bound the probability of an unseen event.

[Quadrant outcomes figure](../figures/threshold_quadrant_outcomes.png)

**First-trigger-only offline policy**

Use the exact frozen rule: if ON is safe, choose ON; otherwise choose OFF only when its score is lower than ON. This includes the OFF-safe case. Exact score ties choose ON. There are exactly **1,413** decisions, each evaluated using its already measured one-action/FULL-continuation outcome.

| dense_class   |   uids |   choose_ON |   choose_OFF |   changed |   unchanged |   change_rate |   treated_change_rate |
|:--------------|-------:|------------:|-------------:|----------:|------------:|--------------:|----------------------:|
| W             |   1307 |         716 |          591 |        30 |        1277 |     0.0229533 |             0.0507614 |
| C             |    106 |          49 |           57 |         6 |         100 |     0.0566038 |             0.105263  |

- Dense-W: choose ON **716**, choose OFF **591**; **W→C=30**, W remains W=1277. W→C rate: **2.30%**; treated-W success: **5.08%**.
- Dense-C: choose ON **49**, choose OFF **57**; **C→W=6**, C remains C=100. C→W rate: **5.66%**; treated-C regression: **10.53%**.
- **Net=+24 UIDs**, bootstrap interval **24.0000 [12.0000, 36.0000]**; net rate **1.70% [0.85, 2.54]**.

First-trigger quadrant counts: `{'Q3': 152, 'Q4': 1261}`. ON-safe first-trigger cases are constrained by the historical post-layer trigger definition, not evidence that ordinary FULL can never resolve later risk.

[First-trigger outcomes figure](../figures/first_trigger_policy_outcomes.png)

**Regimes, interpretation and limits**

Descriptive complete-population breakdowns are saved by [absolute layer](../paired/layer_breakdown.csv), [Early/Middle/Late](../paired/depth_regime_breakdown.csv), [trigger-relative depth](../paired/trigger_relative_breakdown.csv), [Dense W/C](../paired/dense_class_breakdown.csv) and [dataset/source cell](../paired/dataset_source_breakdown.csv). Single-class AUROCs are undefined, not evidence of chance performance. Per-UID metric summaries are also preserved. No favorable subgroup is selected as a new primary population.

The closest supported category is BC-C, qualified: both branches retain moderate above-chance absolute failure signal, but relative branch choice is unsupported. Balanced flip accuracy is 47.77%, discordant AUROC is 0.4684, and delta-p Spearman is -0.0187. Q3 rescue enrichment is uncertain. The offline policy has Net +24 (30 rescues, 6 regressions), but this does not establish useful action ranking or controller readiness. The apparent ON-to-OFF AUROC drop is not evidence of an OFF-score collapse: on fixed ON labels, ON/OFF scores give 0.6728/0.6710 AUROC; on fixed OFF labels they give 0.5948/0.5949. Changing the labels accounts for the observed gap even with the score held fixed. This supports a relative-choice failure description, not a claim that simple recalibration will solve it. The cause of poor paired preference remains unknown.

This establishes only the frozen critic's offline behavior on the exact selected corpus. It does not establish held-out failure-prediction learnability, causally realizable forward-only triggering, closed-loop trajectory safety, repeated-action correctness, deployment benefit or performance on other populations. Gold q is used only as the frozen evaluation target, never as the critic's input or branch-selection rule. No controller or follow-on experiment was run.

Independent interpretation review returned **stable**, with **medium confidence** in the qualified BC-C category. The strongest objection is retained: OFF absolute prediction is moderate and this is a selected internal corpus. Independent direct reconstruction of all branch scores, primary AUROCs, Spearman, policy counts, cache hashes and bootstrap quantiles passed; see [result verification](../parity/independent_result_verification.json).

**Exactly one next-step recommendation:** Paired branch-risk calibration/ranking experiment, as a separately authorized phase with held-out image-group evaluation.
