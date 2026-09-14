"""Render verified branch-critic results after the bounded research interpretation."""
import argparse,json
import pandas as pd
from experiments.run_read_counterfactual_stage1_branch_critic import OUT,read_json,TAU


def render(category,reason):
    s=read_json(OUT/'summaries/metric_summary.json');ci=pd.read_csv(OUT/'statistics/group_bootstrap_ci.csv').set_index('metric')
    def interval(k,percent=False,points=False):
        r=ci.loc[k];f=100 if percent else 1;fmt='.2f' if percent else '.4f';suffix='%' if percent and not points else ''
        return f"{r.estimate*f:{fmt}}{suffix} [{r.ci_low*f:{fmt}}, {r.ci_high*f:{fmt}}]"
    branch=pd.DataFrame(s['branch']);q=pd.DataFrame(s['quadrants']);policy=pd.DataFrame(s['policy']);flips=pd.DataFrame(s['flips']);shift=pd.read_csv(OUT/'scores/branch_distribution_shift.csv');shift=shift[shift.grouping=='ALL']
    q=q[['quadrant','states','uids','mean_H_R','median_H_R','harmful_prevalence','ON_correctness','OFF_correctness','harmful_flips','harmful_flip_prevalence','beneficial_flips','beneficial_flip_prevalence']]
    policy=policy[['dense_class','uids','choose_ON','choose_OFF','changed','unchanged','change_rate','treated_change_rate']]
    flips=flips[['flip','states','uids','preferred_correctly','score_ties','preference_accuracy','median_delta_p']]
    pair=s['paired'];wc=policy.iloc[0];cw=policy.iloc[1];net=int(wc.changed-cw.changed)
    timing='The original Stage-1 trigger at layer l already uses the dense output of that layer; the frozen Step-A action is at its pre-input. Consequently this first-trigger comparison is retrospective and would require rollback to realize. The exact frozen labels and population were retained. A forward-only controller with a newly defined next-layer action is not evaluated here.'
    nexts={'BC-A':'Full closed-loop Stage-1 branch-critic READ controller, as a separately authorized phase with explicit rollback/indexing semantics and safety evaluation.','BC-B':'Counterfactual-state branch-failure critic refit on Dense plus READ-OFF representations, as a separately authorized phase with held-out image-group evaluation.','BC-C':'Paired branch-risk calibration/ranking experiment, as a separately authorized phase with held-out image-group evaluation.','BC-D':'Stop this branch-critic direction; do not proceed to the closed-loop controller.'}
    report=f'''# Stage-1 READ branch-critic diagnostic — {category}

**Decision: {category}.** {reason}

The complete requested offline diagnostic is finished. All results below use **15,185 states, 1,413 P90-triggered UIDs and 1,385 image groups**. They are conditional on this selected population and frozen five-head reuse; they are not held-out population-level Stage-1 training performance.

**Execution and indexing**

Phase85 was safely stopped: supervisor PID 2138508 and workers 2138512–2138515 were terminated with SIGTERM after progress/artifact snapshots. No live bounded-planning worker remains. Completed and intermediate search results, including the PRELIMINARY report, were preserved. See [old_phase_stop_report.md](../old_phase_stop_report.md).

All **15,185** states are eligible, including **1,413 layer-27 states**; excluded states: **0**. Original native Stage1 features are decoder-output features indexed by the layer that produced them. Post-action layer l therefore maps to **Stage1 index l**, including valid index27. No synthetic layer28 and no final decoder normalization were introduced. Exact feature order is final user token, mean user text, mean visual tokens, followed by the original normalization and predictor.

The old compact branch caches contain the last control token, not both Stage1 user-text summaries. OFF Stage1 features were missing for 100% of states. Only the required one-layer representations were reconstructed from dense baselines; all native ON features were reused and matched exactly. No branch suffix, answer generation, new search labels, retraining or calibration fit occurred.

Fresh stratified parity passed on **32 UIDs / 308 states** before the full run. Full extraction verifies compact prestate identity, shared branch inputs, action bits, native ON pooled features, canonical ON output and exact Phase82 ON/OFF compact post-state hashes at every state. Frozen q/correctness labels align with all 30,370 original branch records. Original ensemble ON scores also reproduce within the recorded numerical tolerance. See [indexing contract](../stage1_post_action_indexing_contract.md) and [full verification](../parity/full_scoring_verification.json).

{timing}

**Branch-wise prediction and calibration**

| Branch | AUROC [95% CI] | AUPRC [95% CI] | Brier | ECE | Failure prevalence |
|---|---|---|---:|---:|---:|
| ON | {interval('ON_AUROC')} | {interval('ON_AUPRC')} | {s['branch'][0]['Brier']:.4f} | {s['branch'][0]['ECE']:.4f} | {s['branch'][0]['failure_prevalence']:.2%} |
| OFF | {interval('OFF_AUROC')} | {interval('OFF_AUPRC')} | {s['branch'][1]['Brier']:.4f} | {s['branch'][1]['ECE']:.4f} | {s['branch'][1]['failure_prevalence']:.2%} |

Paired OFF-minus-ON AUROC: **{interval('OFF_minus_ON_AUROC')}**. AUPRC is average precision and must be compared with each branch's high failure prevalence. ECE uses 10 fixed equal-width bins. The frozen score is the mean of five sigmoid probabilities with no additional fitted calibration; per-head raw logits and probabilities are retained. This selected P90 cohort and its training exposure limit any claim of broad Dense-to-counterfactual generalization. The [fixed-target cross-score diagnostic](../scores/fixed_target_cross_score_diagnostic.md) separates the observed change in target difficulty from changing the predictor scores.

[ROC figure](../figures/on_vs_off_failure_roc.png) · [Calibration figure](../figures/on_vs_off_calibration.png)

Distribution diagnostics (distance alone is not a failure criterion):

{shift[['branch','variable','mean','std','p05','median','p95']].round(5).to_markdown(index=False)}

Layer-wise and dataset/source score distributions are in [branch_distribution_shift.csv](../scores/branch_distribution_shift.csv).

**Paired preference and strong behavioral flips**

- Spearman(delta_p, H_R): **{interval('Spearman')}**.
- Pearson(delta_p, H_R): **{interval('Pearson')}**.
- Strict sign agreement on nonzero H_R: **{pair['sign_agreement']:.2%}**. Fixed tiny-q sensitivity thresholds are separately reported; tiny q changes do not override behavioral flips.
- Harmful-flip preference accuracy: **{interval('harmful_flip_accuracy',True)}**.
- Beneficial-flip preference accuracy: **{interval('beneficial_flip_accuracy',True)}**.
- Balanced flip-preference accuracy: **{interval('balanced_flip_accuracy',True)}**.
- Discordant-pair AUROC (harmful versus beneficial, scored by delta_p): **{interval('discordant_AUROC')}**.

{flips.to_markdown(index=False)}

These exact eligible flip counts include layer27. Score ties count as neither strict ON nor strict OFF preference; policy ties retain ON. Confidence intervals use **5,000 paired image-group bootstrap draws**. Undefined missing-class replicates are counted in [group_bootstrap_ci.csv](../statistics/group_bootstrap_ci.csv), not replaced by chance scores.

[Continuous preference figure](../figures/delta_p_vs_read_harm.png) · [Strong-flip figure](../figures/strong_flip_delta_p.png)

**Frozen threshold quadrants**

P90 tau is **{TAU}**. Historical admission remains strict >tau. This plan's quadrants use >=tau for trigger and <tau for safe. Exact threshold ties: **{s['threshold_ties']}**.

{q.round(5).to_markdown(index=False)}

Q3 (ON-trigger/OFF-safe) harmful-flip prevalence is **{interval('Q3_harmful_flip_prevalence',True)}**; enrichment difference versus all states is **{interval('Q3_harmful_enrichment_difference',True,True)} percentage points**. Q2 (ON-safe/OFF-trigger) beneficial-flip prevalence is **{interval('Q2_beneficial_flip_prevalence',True)}**; enrichment difference is **{interval('Q2_beneficial_enrichment_difference',True,True)} percentage points**. Quadrant enrichment is descriptive and does not substitute for net policy outcomes. The zero-width empirical bootstrap interval for Q2 reflects zero observed beneficial flips; it does not bound the probability of an unseen event.

[Quadrant outcomes figure](../figures/threshold_quadrant_outcomes.png)

**First-trigger-only offline policy**

Use the exact frozen rule: if ON is safe, choose ON; otherwise choose OFF only when its score is lower than ON. This includes the OFF-safe case. Exact score ties choose ON. There are exactly **1,413** decisions, each evaluated using its already measured one-action/FULL-continuation outcome.

{policy.to_markdown(index=False)}

- Dense-W: choose ON **{int(wc.choose_ON)}**, choose OFF **{int(wc.choose_OFF)}**; **W→C={int(wc.changed)}**, W remains W={int(wc.unchanged)}. W→C rate: **{wc.change_rate:.2%}**; treated-W success: **{wc.treated_change_rate:.2%}**.
- Dense-C: choose ON **{int(cw.choose_ON)}**, choose OFF **{int(cw.choose_OFF)}**; **C→W={int(cw.changed)}**, C remains C={int(cw.unchanged)}. C→W rate: **{cw.change_rate:.2%}**; treated-C regression: **{cw.treated_change_rate:.2%}**.
- **Net={net:+d} UIDs**, bootstrap interval **{interval('policy_Net')}**; net rate **{interval('policy_Net_rate',True)}**.

First-trigger quadrant counts: `{s['first_trigger_quadrants']}`. ON-safe first-trigger cases are constrained by the historical post-layer trigger definition, not evidence that ordinary FULL can never resolve later risk.

[First-trigger outcomes figure](../figures/first_trigger_policy_outcomes.png)

**Regimes, interpretation and limits**

Descriptive complete-population breakdowns are saved by [absolute layer](../paired/layer_breakdown.csv), [Early/Middle/Late](../paired/depth_regime_breakdown.csv), [trigger-relative depth](../paired/trigger_relative_breakdown.csv), [Dense W/C](../paired/dense_class_breakdown.csv) and [dataset/source cell](../paired/dataset_source_breakdown.csv). Single-class AUROCs are undefined, not evidence of chance performance. Per-UID metric summaries are also preserved. No favorable subgroup is selected as a new primary population.

{reason}

This establishes only the frozen critic's offline behavior on the exact selected corpus. It does not establish held-out failure-prediction learnability, causally realizable forward-only triggering, closed-loop trajectory safety, repeated-action correctness, deployment benefit or performance on other populations. Gold q is used only as the frozen evaluation target, never as the critic's input or branch-selection rule. No controller or follow-on experiment was run.

Independent interpretation review returned **stable**, with **medium confidence** in the qualified BC-C category. The strongest objection is retained: OFF absolute prediction is moderate and this is a selected internal corpus. Independent direct reconstruction of all branch scores, primary AUROCs, Spearman, policy counts, cache hashes and bootstrap quantiles passed; see [result verification](../parity/independent_result_verification.json).

**Exactly one next-step recommendation:** {nexts[category]}
'''
    (OUT/'summaries/branch_critic_summary.md').write_text(report)
    (OUT/'summaries/next_step_recommendation.md').write_text(f'# Next-step recommendation — {category}\n\n{nexts[category]}\n\nReason: {reason}\n\nThis recommendation is unexecuted. The authorized offline diagnostic stops here.\n')
    print(OUT/'summaries/branch_critic_summary.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--category',required=True,choices=['BC-A','BC-B','BC-C','BC-D']);p.add_argument('--reason',required=True);a=p.parse_args();render(a.category,a.reason)
