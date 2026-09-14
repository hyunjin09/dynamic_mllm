"""Independent direct metric reconstruction and artifact verification."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd

out=Path('analysis/read_counterfactual_stage1_branch_critic')
# Preserve JSON IEEE doubles: pandas' default JSON parser perturbs near-ties.
d=pd.DataFrame([json.loads(line) for line in (out/'scores/branch_scores.jsonl').open()])
summary=json.loads((out/'summaries/metric_summary.json').read_text())
assert len(d)==15185 and d.state_id.nunique()==15185 and d.uid.nunique()==1413 and d.image_group_id.nunique()==1385
for b in ['ON','OFF']:
    logits=np.stack(d['logits_'+b]);p=(1/(1+np.exp(-logits))).mean(axis=1)
    assert np.allclose(p,d['p_'+b],rtol=0,atol=1e-14)
assert np.allclose(d.delta_p,d.p_ON-d.p_OFF,rtol=0,atol=1e-14)
assert np.allclose(d.H_R,d.q_OFF-d.q_ON,rtol=0,atol=1e-14)

def direct_auc(y,p):
    pos=np.asarray(p)[np.asarray(y,bool)];neg=np.asarray(p)[~np.asarray(y,bool)];total=0.
    for start in range(0,len(pos),256):
        block=pos[start:start+256,None];total+=(block>neg).sum()+.5*(block==neg).sum()
    return total/(len(pos)*len(neg))

checks={}
for b in ['ON','OFF']:
    a=direct_auc(~d['correct_'+b],d['p_'+b]);expected=next(r for r in summary['branch'] if r['branch']==b)['AUROC']
    assert abs(a-expected)<1e-12;checks[b+'_AUROC_direct_pairs']=a
flip=d[d.correct_ON!=d.correct_OFF];auc=direct_auc(~flip.correct_ON,flip.delta_p)
assert abs(auc-summary['paired']['discordant_AUROC'])<1e-12;checks['discordant_AUROC_direct_pairs']=auc
rho=d.delta_p.rank(method='average').corr(d.H_R.rank(method='average'));assert abs(rho-summary['paired']['Spearman'])<1e-12
pol=d[d.layer==d.trigger_layer];off=(pol.p_ON>=.9061332901863008)&(pol.p_OFF<pol.p_ON);chosen=np.where(off,pol.correct_OFF,pol.correct_ON)
assert len(pol)==1413 and int((pol.dense_wrong&chosen).sum())==30 and int((~pol.dense_wrong&~chosen).sum())==6
hashes=[]
for line in (out/'work/feature_cache_manifest.jsonl').open():
    r=json.loads(line);assert sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];hashes.append(r)
assert len(hashes)==1413
boot=pd.read_csv(out/'statistics/group_bootstrap_draws.csv');assert len(boot)==5000
for _,r in pd.read_csv(out/'statistics/group_bootstrap_ci.csv').iterrows():
    assert abs(boot[r.metric].quantile(.025)-r.ci_low)<1e-12 and abs(boot[r.metric].quantile(.975)-r.ci_high)<1e-12
verification={'passed':True,'states':15185,'uids':1413,'groups':1385,'raw_feature_cache_hashes_checked':len(hashes),'five_head_score_reconstruction':'PASS','q_delta_and_probability_delta':'PASS','direct_auc_checks':checks,'independent_Spearman':rho,'independent_first_trigger_policy':{'W_to_C':30,'C_to_W':6,'Net':24},'bootstrap_quantile_reconstruction':'PASS 5000 image-group draws','verification_parser_note':'Initial verifier used pandas default JSON parsing and altered near-ties by <1e-15, shifting rho by 1.82e-8; lossless json.loads reproduces the frozen analysis exactly. Analysis itself already used json.loads; no metric or experiment was changed.'}
(out/'parity/independent_result_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
