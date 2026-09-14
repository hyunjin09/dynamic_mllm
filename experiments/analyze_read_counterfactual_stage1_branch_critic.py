"""Offline frozen branch-critic metrics with paired image-group bootstrap."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.run_read_counterfactual_stage1_branch_critic import OUT,TAU,verify,file_sha256,atomic_json,atomic_jsonl,read_json,read_jsonl


def weighted_mean(x,w):
    return float(np.dot(x,w)/w.sum()) if w.sum()>0 else np.nan


def rank_values(x,w):
    _,inverse=np.unique(x,return_inverse=True);mass=np.bincount(inverse,weights=w)
    return (np.cumsum(mass)-mass/2)[inverse]


def correlation(x,y,w,rank=False):
    if rank:x=rank_values(x,w);y=rank_values(y,w)
    xc=x-weighted_mean(x,w);yc=y-weighted_mean(y,w)
    denom=np.sqrt(np.dot(xc*xc,w)*np.dot(yc*yc,w))
    return float(np.dot(xc*yc,w)/denom) if denom>0 else np.nan


class BinaryMetric:
    """Exact weighted tied-score AUROC and stepwise average precision."""
    def __init__(self,y,p):
        self.y=np.asarray(y,float);self.p=np.asarray(p,float)
        unique,self.inverse=np.unique(self.p,return_inverse=True);self.n=len(unique)
        self.bins=np.clip((self.p*10).astype(int),0,9)
    def __call__(self,w):
        pos=np.bincount(self.inverse,weights=w*self.y,minlength=self.n);neg=np.bincount(self.inverse,weights=w*(1-self.y),minlength=self.n)
        P=pos.sum();N=neg.sum()
        auc=float(np.dot(pos,np.cumsum(neg)-neg/2)/(P*N)) if P>0 and N>0 else np.nan
        tp=np.cumsum(pos[::-1]);total=np.cumsum((pos+neg)[::-1]);precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        ap=float(np.dot(precision,pos[::-1])/P) if P>0 else np.nan
        count=np.bincount(self.bins,weights=w,minlength=10);err=np.bincount(self.bins,weights=w*(self.y-self.p),minlength=10)
        return dict(AUROC=auc,AUPRC=ap,Brier=weighted_mean((self.p-self.y)**2,w),ECE=float(np.abs(err).sum()/w.sum()) if w.sum() else np.nan,failure_prevalence=weighted_mean(self.y,w))


def policy_off(on,off,tau=TAU):
    return (np.asarray(on)>=tau)&(np.asarray(off)<np.asarray(on))


def quadrants(on,off,tau=TAU):
    return np.where(np.asarray(on)<tau,np.where(np.asarray(off)<tau,'Q1','Q2'),np.where(np.asarray(off)<tau,'Q3','Q4'))


def prep_frame():
    d=pd.DataFrame(read_jsonl(OUT/'scores/branch_scores.jsonl'))
    d['y_ON']=~d.correct_ON;d['y_OFF']=~d.correct_OFF
    d['harmful_flip']=~d.correct_ON & d.correct_OFF;d['beneficial_flip']=d.correct_ON & ~d.correct_OFF
    d['quadrant']=quadrants(d.p_ON,d.p_OFF)
    d['depth_bin']=pd.cut(d.layer,[-1,8,18,27],labels=['Early','Middle','Late']).astype(str)
    d['choose_OFF']=policy_off(d.p_ON,d.p_OFF);d['policy_correct']=np.where(d.choose_OFF,d.correct_OFF,d.correct_ON)
    return d


def paired_metrics(d,w=None):
    if w is None:w=np.ones(len(d))
    dp=d.delta_p.to_numpy();hr=d.H_R.to_numpy();harm=d.harmful_flip.to_numpy();benef=d.beneficial_flip.to_numpy();flip=harm|benef
    ha=weighted_mean((dp>0).astype(float),w*harm);ba=weighted_mean((dp<0).astype(float),w*benef)
    return dict(Spearman=correlation(dp,hr,w,True),Pearson=correlation(dp,hr,w),sign_agreement=weighted_mean((np.sign(dp)==np.sign(hr)).astype(float),w*(hr!=0)),harmful_flip_accuracy=ha,beneficial_flip_accuracy=ba,balanced_flip_accuracy=(ha+ba)/2,discordant_AUROC=BinaryMetric(harm[flip],dp[flip])(w[flip])['AUROC'] if flip.any() else np.nan)


def subgroup(d,cols):
    output=[]
    for key,g in d.groupby(cols,observed=True,sort=True):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(cols,key));row.update(states=len(g),uids=g.uid.nunique(),harmful_flips=int(g.harmful_flip.sum()),beneficial_flips=int(g.beneficial_flip.sum()))
        for branch in ['ON','OFF']:
            row.update({branch+'_'+k:v for k,v in BinaryMetric(g['y_'+branch],g['p_'+branch])(np.ones(len(g))).items()})
        row.update(paired_metrics(g));output.append(row)
    return pd.DataFrame(output)


def summary_vector(d,pol):
    on=BinaryMetric(d.y_ON,d.p_ON);off=BinaryMetric(d.y_OFF,d.p_OFF)
    dp=d.delta_p.to_numpy();hr=d.H_R.to_numpy();harm=d.harmful_flip.to_numpy();benef=d.beneficial_flip.to_numpy();flip=harm|benef
    # AUROC need not receive probabilities, so the separate helper's calibration bins are irrelevant.
    disc=BinaryMetric(harm[flip],np.clip(dp[flip]/2+.5,0,1))
    q3=(d.quadrant=='Q3').to_numpy();q2=(d.quadrant=='Q2').to_numpy()
    pw=pol.dense_wrong.to_numpy();pc=~pw;po=pol.choose_OFF.to_numpy();correct=pol.policy_correct.to_numpy()
    def evaluate(w,wp):
        a=on(w);b=off(w);o={**{'ON_'+k:v for k,v in a.items()},**{'OFF_'+k:v for k,v in b.items()}}
        o['OFF_minus_ON_AUROC']=b['AUROC']-a['AUROC'];o['Spearman']=correlation(dp,hr,w,True);o['Pearson']=correlation(dp,hr,w)
        o['harmful_flip_accuracy']=weighted_mean((dp>0).astype(float),w*harm);o['beneficial_flip_accuracy']=weighted_mean((dp<0).astype(float),w*benef)
        o['balanced_flip_accuracy']=(o['harmful_flip_accuracy']+o['beneficial_flip_accuracy'])/2;o['discordant_AUROC']=disc(w[flip])['AUROC']
        o['Q3_harmful_flip_prevalence']=weighted_mean(harm.astype(float),w*q3);o['Q3_harmful_enrichment_difference']=o['Q3_harmful_flip_prevalence']-weighted_mean(harm.astype(float),w)
        o['Q2_beneficial_flip_prevalence']=weighted_mean(benef.astype(float),w*q2);o['Q2_beneficial_enrichment_difference']=o['Q2_beneficial_flip_prevalence']-weighted_mean(benef.astype(float),w)
        o['policy_W_to_C']=float(np.dot(wp,pw&correct));o['policy_C_to_W']=float(np.dot(wp,pc&~correct));o['policy_Net']=o['policy_W_to_C']-o['policy_C_to_W']
        o['policy_W_to_C_rate']=weighted_mean(correct.astype(float),wp*pw);o['policy_C_to_W_rate']=weighted_mean((~correct).astype(float),wp*pc)
        o['treated_W_success']=weighted_mean(correct.astype(float),wp*pw*po);o['treated_C_regression']=weighted_mean((~correct).astype(float),wp*pc*po)
        o['policy_Net_rate']=o['policy_Net']/wp.sum()
        return o
    return evaluate


def analyze():
    verify();assert read_json(OUT/'parity/full_scoring_verification.json')['passed']
    ac=read_json(OUT/'analysis_contract.json');assert file_sha256(__file__)==ac['analysis_code_sha256']
    d=prep_frame();assert len(d)==15185 and d.uid.nunique()==1413
    pol=d[d.layer==d.trigger_layer].copy();assert len(pol)==1413
    tables=[];cal=[]
    for b in ['ON','OFF']:
        met=BinaryMetric(d['y_'+b],d['p_'+b])(np.ones(len(d)));row=dict(branch=b,states=len(d),uids=d.uid.nunique(),**met)
        pd.DataFrame([row]).to_csv(OUT/f'scores/{b.lower()}_branch_metrics.csv',index=False);tables.append(row)
        bins=np.minimum((d['p_'+b].to_numpy()*10).astype(int),9)
        for i in range(10):
            g=d[bins==i];cal.append(dict(branch=b,bin=i,lower=i/10,upper=(i+1)/10,states=len(g),mean_probability=g['p_'+b].mean(),failure_prevalence=g['y_'+b].mean()))
    pd.DataFrame(tables).to_csv(OUT/'scores/calibration_metrics.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'scores/calibration_bins.csv',index=False)
    pair=paired_metrics(d);pd.DataFrame([dict(states=len(d),nonzero_H_R=int((d.H_R!=0).sum()),**pair)]).to_csv(OUT/'paired/continuous_preference_metrics.csv',index=False)
    sensitivity=[]
    for eps in [0,1e-4,1e-3,1e-2]:
        g=d[d.H_R.abs()>eps];sensitivity.append(dict(min_abs_H_R=eps,states=len(g),**paired_metrics(g)))
    pd.DataFrame(sensitivity).to_csv(OUT/'paired/tiny_q_sensitivity.csv',index=False)
    flips=[]
    for name,mask in [('harmful',d.harmful_flip),('beneficial',d.beneficial_flip)]:
        g=d[mask];correct=g.delta_p>0 if name=='harmful' else g.delta_p<0
        flips.append(dict(flip=name,states=len(g),uids=g.uid.nunique(),preferred_correctly=int(correct.sum()),score_ties=int((g.delta_p==0).sum()),preference_accuracy=correct.mean(),median_delta_p=g.delta_p.median()))
    pd.DataFrame(flips).to_csv(OUT/'paired/strong_flip_preference.csv',index=False)
    qrows=[]
    for name in ['Q1','Q2','Q3','Q4']:
        g=d[d.quadrant==name];qrows.append(dict(quadrant=name,states=len(g),uids=g.uid.nunique(),mean_H_R=g.H_R.mean(),median_H_R=g.H_R.median(),harmful_prevalence=(g.H_R>0).mean(),ON_correctness=g.correct_ON.mean(),OFF_correctness=g.correct_OFF.mean(),harmful_flips=int(g.harmful_flip.sum()),harmful_flip_prevalence=g.harmful_flip.mean(),beneficial_flips=int(g.beneficial_flip.sum()),beneficial_flip_prevalence=g.beneficial_flip.mean()))
    pd.DataFrame(qrows).to_csv(OUT/'paired/threshold_quadrants.csv',index=False)
    for cols,name in [(['layer'],'layer_breakdown'),(['depth_bin'],'depth_regime_breakdown'),(['trigger_relative_depth'],'trigger_relative_breakdown'),(['dataset','source_regime'],'dataset_source_breakdown'),(['dense_wrong'],'dense_class_breakdown')]:subgroup(d,cols).to_csv(OUT/f'paired/{name}.csv',index=False)
    dist=[]
    for cols in [[],['layer'],['dataset','source_regime']]:
        groups=[('ALL',d)] if not cols else d.groupby(cols,observed=True)
        for key,g in groups:
          for b in ['ON','OFF']:
            for col in ['p_','feature_norm_','normalized_feature_norm_']:
                v=g[col+b];dist.append(dict(grouping='+'.join(cols) or 'ALL',group=str(key),branch=b,variable=col.rstrip('_'),states=len(g),mean=v.mean(),std=v.std(),p05=v.quantile(.05),median=v.median(),p95=v.quantile(.95)))
    pd.DataFrame(dist).to_csv(OUT/'scores/branch_distribution_shift.csv',index=False)
    macro=subgroup(d,['uid']);macro.to_csv(OUT/'paired/uid_macro_metrics.csv',index=False)
    keep=['state_id','uid','image_group_id','dataset','source_regime','layer','dense_wrong','correct_ON','correct_OFF','p_ON','p_OFF','delta_p','quadrant','choose_OFF','policy_correct']
    atomic_jsonl(OUT/'policy/first_trigger_policy_decisions.jsonl',pol[keep].to_dict('records'))
    prows=[]
    for name,wrong in [('dense_w',True),('dense_c',False)]:
        g=pol[pol.dense_wrong==wrong];g[keep].to_csv(OUT/f'policy/first_trigger_{name}.csv',index=False)
        choose=int(g.choose_OFF.sum());changed=int(g.policy_correct.sum()) if wrong else int((~g.policy_correct).sum())
        prows.append(dict(dense_class='W' if wrong else 'C',uids=len(g),choose_ON=len(g)-choose,choose_OFF=choose,changed=changed,unchanged=len(g)-changed,change_rate=changed/len(g),treated_change_rate=changed/choose if choose else np.nan))
    pd.DataFrame(prows).to_csv(OUT/'policy/first_trigger_policy_summary.csv',index=False)
    evaluation=summary_vector(d,pol);estimate=evaluation(np.ones(len(d)),np.ones(len(pol)))
    groups=sorted(d.image_group_id.unique());index={g:i for i,g in enumerate(groups)};gi=d.image_group_id.map(index).to_numpy();pi=pol.image_group_id.map(index).to_numpy();rng=np.random.default_rng(2026091301);draws=[]
    for i in range(5000):
        mass=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));draws.append(evaluation(mass[gi],mass[pi]))
        if (i+1)%500==0:print('bootstrap',i+1,flush=True)
    boot=pd.DataFrame(draws);ci=[]
    for k,v in estimate.items():
        a=boot[k].dropna();ci.append(dict(metric=k,estimate=v,ci_low=a.quantile(.025),ci_high=a.quantile(.975),valid_draws=len(a),invalid_draws=5000-len(a),bootstrap_unit='image_group_id',draws=5000))
    pd.DataFrame(ci).to_csv(OUT/'statistics/group_bootstrap_ci.csv',index=False);boot.to_csv(OUT/'statistics/group_bootstrap_draws.csv',index=False)
    atomic_json(OUT/'summaries/metric_summary.json',dict(population=dict(states=len(d),uids=d.uid.nunique(),groups=len(groups),layer27=int((d.layer==27).sum())),branch=tables,paired=pair,flips=flips,quadrants=qrows,policy=prows,estimates=estimate,first_trigger_quadrants=pol.quadrant.value_counts().to_dict(),score_ties=int((d.delta_p==0).sum()),threshold_ties=int(((d.p_ON==TAU)|(d.p_OFF==TAU)).sum())))
    figures(d,pol,pd.DataFrame(cal),pd.DataFrame(qrows),prows)
    print('PASS metrics, bootstrap and figures',flush=True)


def roc(y,p):
    o=np.argsort(-np.asarray(p),kind='stable');y=np.asarray(y)[o];v=np.asarray(p)[o];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];tp=np.cumsum(y)[ends];fp=np.cumsum(1-y)[ends]
    return np.r_[0,fp/(1-y).sum()],np.r_[0,tp/y.sum()]


def figures(d,pol,cal,q,policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):plt.tight_layout();plt.savefig(OUT/f'figures/{name}.png');plt.close()
    plt.figure(figsize=(6,5))
    for b in ['ON','OFF']:
        x,y=roc(d['y_'+b],d['p_'+b]);plt.plot(x,y,label=b)
    plt.plot([0,1],[0,1],'k--',alpha=.3);plt.legend();plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title('Frozen Stage-1 branch failure prediction');save('on_vs_off_failure_roc')
    plt.figure(figsize=(6,5))
    for b in ['ON','OFF']:
        g=cal[(cal.branch==b)&(cal.states>0)];plt.plot(g.mean_probability,g.failure_prevalence,'o-',label=b)
    plt.plot([0,1],[0,1],'k--',alpha=.3);plt.legend();plt.xlabel('Mean predicted failure');plt.ylabel('Observed failure');plt.title('Frozen branch calibration (10 fixed bins)');save('on_vs_off_calibration')
    plt.figure(figsize=(6,5));plt.hexbin(d.delta_p,d.H_R,gridsize=50,bins='log',mincnt=1);plt.colorbar(label='State count (log scale)');plt.axhline(0,color='grey',lw=.5);plt.axvline(0,color='grey',lw=.5);plt.xlabel('p_ON − p_OFF');plt.ylabel('q_OFF − q_ON');plt.title('Paired probability preference versus READ harm');save('delta_p_vs_read_harm')
    plt.figure(figsize=(6,5));plt.boxplot([d.loc[d.harmful_flip,'delta_p'],d.loc[d.beneficial_flip,'delta_p']],tick_labels=['ON wrong / OFF correct','ON correct / OFF wrong'],showfliers=True);plt.axhline(0,color='grey',lw=.8);plt.ylabel('p_ON − p_OFF');plt.title('Behavioral flip preference');save('strong_flip_delta_p')
    plt.figure(figsize=(7,5));x=np.arange(4);plt.bar(x-.18,q.harmful_flip_prevalence,width=.36,label='ON wrong / OFF correct');plt.bar(x+.18,q.beneficial_flip_prevalence,width=.36,label='ON correct / OFF wrong');plt.xticks(x,q.quadrant);plt.ylabel('Flip prevalence');plt.legend();plt.title('Frozen threshold quadrant outcomes');save('threshold_quadrant_outcomes')
    plt.figure(figsize=(6,5));wc=policy[0]['changed'];cw=policy[1]['changed'];plt.bar(['W→C','C→W','Net'],[wc,cw,wc-cw]);plt.ylabel('UID count');plt.title('Offline first-trigger single-decision policy');save('first_trigger_policy_outcomes')

if __name__=='__main__':analyze()
