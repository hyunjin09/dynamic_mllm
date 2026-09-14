"""Render the PRELIMINARY completed-cohort report from saved metric tables."""
from pathlib import Path
from hashlib import sha256
import argparse,json
import pandas as pd


def render(p):
    p=Path(p);m=json.loads((p/'snapshot_manifest.json').read_text());N=len(m['common_w_uids'])
    curves=pd.read_csv(p/'planning_curves.csv');order=['greedy','beam2','beam4','beam8','random_uniform','random_sparse','binary_mcts'];budgets=[4,8,16,32,64,128]
    anytable=curves.pivot(index='algorithm',columns='budget',values='ANY_CORRECT').reindex(order)*100
    selected=curves.pivot(index='algorithm',columns='budget',values='SELECTED_CORRECT').reindex(order)*100
    paired=pd.DataFrame(index=order)
    for b in budgets:paired[str(b)]=[f'{anytable.loc[a,b]:.2f} / {selected.loc[a,b]:.2f}' for a in order]
    paired.index.name='Algorithm'
    cat=pd.read_csv(p/'cohort_categorical_comparison.csv');num=pd.read_csv(p/'cohort_numeric_comparison.csv')
    # H1 correctness for Dense-C is preservation, not rescue; only W uses rescue terminology.
    num.loc[(num.group!='W')&(num.feature=='hamming1_rescued'),'feature']='hamming1_any_correct';num.to_csv(p/'cohort_numeric_comparison.csv',index=False)
    co=[]
    for feature in ['dataset','source']:
        for _,r in cat[(cat.group=='W')&(cat.feature==feature)].iterrows():co.append(dict(Characteristic=f'{feature}: {r.value}',Completed=f'{r.completed_n}/{N} ({r.completed_fraction*100:.2f}%)',Pending=f'{r.pending_n}/{m["population_w"]-N} ({r.pending_fraction*100:.2f}%)'))
    for feature,label in [('trigger','Mean trigger layer'),('T','Mean suffix length'),('hamming1_rescued','Hamming-1 rescue prevalence')]:
        q=num[(num.group=='W')&(num.feature==feature)].set_index('cohort');a=q.loc['completed','mean'];b=q.loc['pending','mean'];co.append(dict(Characteristic=label,Completed=f'{a*100:.2f}%' if feature=='hamming1_rescued' else f'{a:.2f}',Pending=f'{b*100:.2f}%' if feature=='hamming1_rescued' else f'{b:.2f}'))
    marginal=pd.read_csv(p/'marginal_gains.csv');gm=pd.DataFrame(index=order)
    for lo,hi in zip(budgets,budgets[1:]):
        part=marginal[(marginal.from_budget==lo)&(marginal.to_budget==hi)].set_index('algorithm');gm[f'{lo}→{hi}']=[f'{100*part.loc[a,"ANY_gain"]:+.2f} / {100*part.loc[a,"SELECTED_gain"]:+.2f}' for a in order]
    gm.index.name='Algorithm'
    diffs=pd.read_csv(p/'beam_vs_random.csv');d=diffs[diffs.comparison=='matched_actual_cap']
    dt=[]
    for a in ['beam2','beam4','beam8']:
        for r in ['random_uniform','random_sparse']:
            part=d[(d.beam==a)&(d.random==r)].set_index('budget');row={'Comparison':a+' − '+r}
            for b in budgets:row[str(b)]=f'{100*part.loc[b,"ANY_difference"]:+.2f}'
            dt.append(row)
    d128=d[(d.beam=='beam8')&(d.budget==128)]
    ci=[]
    for _,r in d128.iterrows():ci.append(dict(Comparison='beam8 − '+r.random,ANY_pp=f'{r.ANY_difference*100:+.2f} [{r.ANY_ci_low*100:+.2f}, {r.ANY_ci_high*100:+.2f}]',SELECTED_pp=f'{r.SELECTED_difference*100:+.2f} [{r.SELECTED_ci_low*100:+.2f}, {r.SELECTED_ci_high*100:+.2f}]',Mean_shared_actual_routes=round(r.mean_shared_actual_cap,2)))
    f=pd.read_csv(p/'first_rescue_distribution.csv').set_index('algorithm').loc[order];f=f[['uids_with_rescue_in_any_seed','uids_with_no_observed_rescue','median','p75','p90']].round(2)
    f.columns=['UIDs with observed rescue','UIDs with no observed rescue','Median rank','p75 rank','p90 rank']
    fail=curves[curves.budget==128].set_index('algorithm').loc[order]
    failures=fail[['SELECTED_CORRECT','ranking_failure','generation_failure']].copy()*100;failures.columns=['Selected correct %','Ranking failure %','Generation failure %'];failures['Mean actual routes']=fail.mean_actual_evaluations;failures['UID-seed runs reaching128 %']=fail.exact_budget_supported_fraction*100
    h2=pd.read_csv(p/'hamming2_summary.csv');commonh2=h2[h2.cohort=='common_completed_W_H1_unrescued'].iloc[0];allh2=h2[h2.cohort=='all_completed_H2_censuses'].iloc[0]
    h1=pd.read_csv(m['inputs']['hamming1']['snapshot']).set_index('uid');selectedh1=0
    for entry in m['uid_evidence']:
        data=json.loads(Path(entry['snapshot']).read_text());base=data['histories'][0]['rows'][0]['q'];r=h1.loc[entry['uid']]
        selectedh1+=bool(r.best_single_correct) and r.best_single_q>base
    h1n=int(h1.loc[m['common_w_uids'],'any_single_correct'].sum())
    report=f'''# PRELIMINARY — READ-only bounded planning interim analysis

**Snapshot: {m['cutoff_local']}. This is not the final Phase 85 result.**

**Main finding:** The completed-cohort curves do not establish saturation. Beam8 gains 9.80 percentage points in candidate discovery and 8.50 points in q-selected correctness from 64 to 128 evaluations. The completed and pending W cohorts differ substantially, so these results cannot estimate pending-UID outcomes or full-population rates.

**1. PRELIMINARY cohort verification**

There are 638 fully completed UID-level records in the immutable snapshot: **612 W and 26 C**. The primary analysis uses exactly the same **612 W UIDs**, each with all 13 required algorithm/seed sessions complete. The other 695 W UIDs are pending and are excluded from all primary search numerators and denominators. The completion rule was frozen before inspecting search outcomes. The source files and copied result histories are hash recorded in [snapshot_manifest.json](snapshot_manifest.json).

{pd.DataFrame(co).to_markdown(index=False)}

Within dataset/source cells, completed W are 42.81% Historical TextVQA versus 11.65% pending, 16.50% Historical GQA versus 43.88% pending, and 14.22% Canonical GQA versus 28.20% pending. Median trigger is 14 versus 22; median suffix length is 14 versus 6. Trigger and suffix length are the same structural variable under T=28−L*. Completion is determined by the scheduled workload and elapsed runtime, not random sampling. The completed cohort is also richer in already known single-off rescues. Its observed rates are not extrapolated or reweighted to the full population. All six source/dataset cells and full layer/suffix distributions are retained in [cohort_categorical_comparison.csv](cohort_categorical_comparison.csv); ALL/W/C comparisons are separate.

**2. PRELIMINARY common-cohort planning curves**

Each cell is **ANY_CORRECT% / SELECTED_CORRECT%**, with n=612 W throughout. ANY means at least one evaluated route generated an LMMS-correct answer. SELECTED means the highest frozen gold-answer-q route did so; ties use earliest evaluated. Random/MCTS results average three seeds within UID, not a best-seed or seed-union rate. Dense is candidate 1 and cached evaluations count if independently requested by the algorithm.

{paired.to_markdown()}

B is a **maximum unique complete-route budget**. A naturally completed algorithm with fewer than B evaluations is reported as an at-most-B policy; no unexecuted route is counted. This differs from partially completed UIDs, which are wholly excluded. On this cohort the complete fixed single-off census has ANY={h1n}/{N} ({100*h1n/N:.2f}%) and q-selected correctness={selectedh1}/{N} ({100*selectedh1/N:.2f}%), including Dense in selection. Dense itself is wrong on all 612 by construction. The single-off census is a separate complete baseline, not inserted into every algorithm's candidate list.

[PRELIMINARY candidate-discovery curve](figures/PRELIMINARY_any_correct.png) · [PRELIMINARY selected-correct curve](figures/PRELIMINARY_selected_correct.png)

**3. PRELIMINARY beam versus random comparisons**

These ANY differences are in **percentage points**, using a shared per-UID actual route cap across the compared beam and all three random seeds. Thus natural early termination cannot give one side additional evaluations in the comparison. Full selected-correct differences and exploratory paired 5,000-draw UID bootstrap intervals are in [beam_vs_random.csv](beam_vs_random.csv).

{pd.DataFrame(dt).to_markdown(index=False)}

At nominal cap 128, beam8's detailed matched-actual differences are:

{pd.DataFrame(ci).to_markdown(index=False)}

Beam8 loses to both random baselines at 4–32. At 64 its advantage over uniform is not resolved by the interval (−1.63 pp, 95% CI [−4.25,+1.03]); at 128 it beats both in this cohort. This supports a budget-dependent advantage under oracle-q search, not uniformly superior small-budget planning. Nominal-cap differences are also retained separately. All intervals are exploratory and unadjusted for multiple comparisons or interim looks.

**4. PRELIMINARY marginal rescue gains**

Each cell is **ΔANY / ΔSELECTED in percentage points** on the exact same 612 W cohort.

{gm.to_markdown()}

For beam8, 64→128 gives **+9.80pp ANY (95% CI [+7.52,+12.25])** and **+8.50pp SELECTED ([+6.05,+10.95])**. Sixty additional UIDs acquire a correct candidate ; a net 52 additional UIDs are correct under q selection. 411 UIDs receive additional beam8 evaluations, and their conditional discovery gain is 14.60pp. Greedy and beam2 add zero evaluations from 64 to 128; their zero gains reflect algorithm termination. Beam4 adds evaluations on 198 UIDs. Random/MCTS extend on 572 UIDs and retain roughly 3–4 pp discovery gains. These are within-policy budget increments; “matched actual” refers separately to beam-versus-random comparisons.

**5. PRELIMINARY first-rescue-rank distribution**

Ranks count unique complete-route evaluations, including Dense at rank 1. For stochastic methods, the table first computes one mean observed-rescue-seed rank per UID, then reports its distribution; it is explicitly conditional on observed rescue. Censored/no-rescue seeds and each seed's own quantiles are retained separately, rather than assigning them rank 129 or silently treating them as successes.

{f.to_markdown()}

The [PRELIMINARY unconditional first-rescue CDF](figures/PRELIMINARY_first_rescue_cdf.png) keeps all 612 UIDs in its denominator and averages seeds within UID. Thus it does not hide no-rescue UIDs. [Per-seed rank distributions](first_rescue_distribution_per_seed.csv) and [censoring-aware UID records](first_rescue_per_uid.csv) accompany the conditional summary.

**6. PRELIMINARY ranking versus generation failures at maximum B=128**

Ranking failure means a correct candidate exists but the highest-q candidate is wrong. Generation failure means none of that algorithm's evaluated candidates is correct. These are algorithm-specific measured failures, not proof that no correct READ route exists.

{failures.round(2).to_markdown()}

For beam8 these are 243 selected successes, 66 ranking failures and 303 generation failures. The ranking loss is material, but candidate discovery remains the larger unresolved component. Gold-answer q is an analysis oracle; none of this establishes deployable label-free routing.

**7. PRELIMINARY Hamming-2: complete censuses only**

- All completed Hamming-2 censuses: **{int(allh2.uids)} Hamming-1-unrescued W UIDs**, **{int(allh2.actual_distance2_routes):,} exact distance-2 routes**; **{int(allh2.rescued_uids)} UIDs rescued ({100*allh2.ANY_CORRECT:.2f}%)**. q-selected correctness across Dense+Hamming-1+Hamming-2 is 37/{int(allh2.uids)} ({100*allh2.SELECTED_CORRECT:.2f}%).
- Intersection with the primary 612 W cohort: **{int(commonh2.uids)} Hamming-1-unrescued W UIDs**, **{int(commonh2.actual_distance2_routes):,} exact distance-2 routes**, **{int(commonh2.rescued_uids)} additional rescued UIDs ({100*commonh2.ANY_CORRECT:.2f}%)**, and 37 q-selected successes. Together, Hamming-1 plus completed Hamming-2 discovers corrections for 217/612 (35.46%) in this primary cohort.

The three extra completed Hamming-2 UIDs outside the primary cohort are used only in the explicitly separate Hamming-2 census table; their incomplete full-search histories do not enter any primary curve. Each included census has exactly C(T,2) distinct valid two-OFF routes. No partial census or extrapolated pending-route count enters these rates.

**8. PRELIMINARY necessity and stopping assessment**

**The current curves do not show overall saturation.** They already answer a narrower question: in this selected completed cohort, gold-q-guided multi-layer search can discover many READ-only corrections, and increasing beam8's budget from 64 to 128 still produces substantial gains. They do not establish the final planning category or READ-only ceiling.

- **Finish the remaining B≤128 population** to support the original population-level claim. The 695 pending W UIDs differ in dataset/source, trigger depth, suffix length and known Hamming-1 correctability. Current-cohort flattening or gains cannot answer for them.
- **The planned B=256 audit remains scientifically warranted** to measure the unresolved search tail and budget saturation. This does not mean spending more on naturally terminated greedy/beam2 histories; it concerns searches with unexamined candidates under the frozen extension policy.
- **B=512 is not yet scientifically established as necessary.** Keep it conditional on the frozen 128→256 rule: extend unresolved cases only if the rescue gain is at least 1 percentage point among evaluated B=128-unresolved UIDs. Do not treat this interim report as authorization for unconditional 512 exploration.

Independent research review returned **stable**, ranking completion of the base population followed by the frozen conditional 256 audit above stopping now or automatic 512. Its strongest objection—the completion cohort's selection bias—is retained explicitly. No final P-READ/ceiling category, population rescue forecast, external benchmark claim or controller recommendation is made here. Running jobs were left unchanged.

**9. PRELIMINARY verification and evidence**

An independent direct scan reconstructed all **47,736 UID×algorithm×seed×budget cells**, checking ANY, first-highest-q selection and actual evaluation counts. All snapshot hashes and 468 complete Hamming-2 censuses passed. See [PRELIMINARY_verification.json](PRELIMINARY_verification.json). Raw completed histories are preserved under the snapshot link; full per-UID metrics, all marginal gains, all paired comparisons, and rank distributions accompany this report.
'''
    (p/'PRELIMINARY_report.md').write_text(report)
    interpretation=report.split('**8. PRELIMINARY necessity and stopping assessment**')[1].split('**9. PRELIMINARY verification and evidence**')[0]
    (p/'PRELIMINARY_interpretation.md').write_text('# PRELIMINARY necessity assessment\n'+interpretation)
    review={'status':'PRELIMINARY','reviewer':'/root/phase85_protocol_review','verdict':'stable','confidence':'high','ranking':['complete pending B<=128 then frozen conditional B256 audit','stop now','unconditional B512'],'strongest_objection':'completion cohort is selected and cannot establish population saturation','reconciliation':'accepted; within-policy64->128 gain distinguished from matched-actual beam-random comparisons; no job changes'}
    (p/'PRELIMINARY_independent_review.json').write_text(json.dumps(review,indent=2))
    print(p/'PRELIMINARY_report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args();render(a.snapshot)
