"""Apply the pre-fit secondary-control claim safeguards without changing primary gates."""
import numpy as np
import pandas as pd
from experiments.run_write_harm_structure_learnability import *
from dense_failure_stage2.write_bootstrap import bootstrap_comparison


def main():
    verify();contract=read_json(OUT/'propagation/secondary_interpretation_contract.json');assert contract['frozen_before_W3_fitting'];assert file_sha256(Path(__file__))==contract['code_sha256'];original=read_json(OUT/'propagation/propagation_decision.json');d=pd.read_csv(OUT/'propagation/horizon_metrics.csv');d=d[d.support=='common'];flags=[]
    for pop in ['full','dense_w']:
        baseline=d[(d.population==pop)&(d.condition=='delta')&(d.horizon==0)].iloc[0]
        for _,r in d[(d.population==pop)&(d.condition!='delta')].iterrows():
            rho=r.spearman-baseline.spearman;auc=r.harmful_auroc-baseline.harmful_auroc;high=r['recall_at_precision_0.9']>=.05 and r['precision_at_0.1']>=r.harmful_prevalence+.1
            if rho>=.1 or auc>=.08 or high:flags.append(dict(population=pop,condition=r.condition,horizon=int(r.horizon),spearman_gain_over_delta_H0=rho,auc_gain_over_delta_H0=auc,useful_high_precision=bool(high)))
    category=original['category'];differences=[]
    if category=='W-PROP-C' and flags:category='W-PROP-INCONCLUSIVE'
    if category=='W-PROP-B':
        h=original['smallest_successful_horizon'];pred=read_jsonl(OUT/'propagation/oof_predictions.jsonl');rows={r['state_id']:r for r in read_jsonl(OUT/'population/dense_write_state_manifest.jsonl')}
        for pop in ['full','dense_w']:
            by={condition:{r['state_id']:r for r in pred if r['support']=='common' and r['population']==pop and r['horizon']==h and r['condition']==condition} for condition in ['pair','on','off']};ids=sorted(by['pair']);rr=[rows[s] for s in ids];y=np.array([rows[s]['h_w'] for s in ids]);left=np.array([by['pair'][s]['prediction'] for s in ids])
            for single in ['on','off']:
                assert set(by[single])==set(ids);right=np.array([by[single][s]['prediction'] for s in ids]);comp=bootstrap_comparison(rr,y,left,right);bounded=comp['spearman_ci_high']<.05 and comp['harmful_auc_ci_high']<.04;differences.append(dict(population=pop,horizon=h,left='pair',right=single,bounded_small=bounded,**comp))
                if not bounded:category='W-PROP-INCONCLUSIVE'
    out={**original,'primary_frozen_category':original['category'],'category':category,'secondary_claim_flags':flags,'ordered_pair_B_checks':differences,'generalization_gate':category in ['W-PROP-A','W-PROP-B'],'secondary_interpretation_contract_sha256':file_sha256(OUT/'propagation/secondary_interpretation_contract.json'),'no_secondary_model_promoted':True}
    atomic_json(OUT/'propagation/interpretation_decision.json',out);pd.DataFrame(flags).to_csv(OUT/'statistics/secondary_claim_flags.csv',index=False)
    if differences:pd.DataFrame(differences).to_csv(OUT/'statistics/ordered_pair_B_checks.csv',index=False)
    (OUT/'propagation/reconciled_interpretation.md').write_text('# '+category+'\n\nThe original prospective primary gate remains '+original['category']+'. Pre-fit secondary-control safeguards prevent an overbroad negative or single-branch-sufficiency claim; they never promote a secondary model after outcomes.\n\n'+str(flags)+'\n\n'+str(differences)+'\n')
    print(category,'secondary flags',len(flags),flush=True)

if __name__=='__main__':main()
