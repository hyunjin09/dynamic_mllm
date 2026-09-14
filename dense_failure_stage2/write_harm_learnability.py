"""WRITE-specific visual-update features and bounded horizon contracts."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

HORIZONS=(0,1,2,4,8)
SEEDS=(2026091101,2026091102,2026091103)

def horizons_for_layer(layer):
    if not 0<=layer<28:raise ValueError('invalid decoder layer')
    return tuple(h for h in HORIZONS if layer+h<28)

def harm_metrics(truth,prediction,flip=None):
    from experiments.analyze_read_counterfactual_stage1_branch_critic import BinaryMetric,correlation
    y=np.asarray(truth,float);p=np.asarray(prediction,float);w=np.ones(len(y));labels=(y>0).astype(int)
    if len(y)==0:return {}
    a=BinaryMetric(labels,p)(w);o=dict(spearman=correlation(y,p,w,True),pearson=correlation(y,p,w),mae=float(np.abs(y-p).mean()),rmse=float(np.sqrt(np.square(y-p).mean())),harmful_auroc=a['AUROC'],harmful_auprc=a['AUPRC'],harmful_prevalence=float(labels.mean()),states=len(y))
    order=np.argsort(-p,kind='stable');z=labels[order]
    for c in [.05,.1,.2]:o[f'precision_at_{c:g}']=float(z[:max(1,math.ceil(c*len(z)))].mean())
    # Threshold-qualified prefixes include all ties. No optimistic tie ordering for recall.
    end=np.r_[np.flatnonzero(p[order][1:]!=p[order][:-1]),len(p)-1]
    precision=np.cumsum(z)[end]/(end+1);recall=np.cumsum(z)[end]/max(1,z.sum())
    for t in [.9,.95]:o[f'recall_at_precision_{t:g}']=float(recall[precision>=t].max()) if (precision>=t).any() else 0.
    if flip is not None:
        b=BinaryMetric(np.asarray(flip,int),p)(w);o.update(harmful_flip_auroc=b['AUROC'],harmful_flip_auprc=b['AUPRC'])
    return o

def _stats(x):
    return {'mean':x.mean(),'std':x.std(unbiased=False),'min':x.min(),'max':x.max(),'q10':torch.quantile(x,.1),'median':torch.quantile(x,.5),'q90':torch.quantile(x,.9)}

def diversity(v):
    n=len(v);centered=v-v.mean(0);unit=F.normalize(v,dim=1,eps=1e-12)
    pair=(unit.sum(0).square().sum()-unit.square().sum())/max(1,n*(n-1))
    gram=centered@centered.T if n<=v.shape[1] else centered.T@centered
    eig=torch.linalg.eigvalsh(gram/max(1,n-1)).clamp_min(0)
    mass=eig/eig.sum().clamp_min(1e-12);entropy=-(mass*mass.clamp_min(1e-30).log()).sum()
    return {'pair_cosine':pair,'token_variance':centered.square().mean(),'effective_rank':entropy.exp() if eig.sum()>0 else entropy.new_tensor(0.),'spectral_entropy':entropy,'centroid_distance':centered.norm(dim=1).mean()}

def write_features(pre,full,off,query,coords):
    """Exact FP32 geometry, with covariance's full nonzero spectrum (no projection)."""
    pre=pre.float();full=full.float();off=off.float();query=query.float();delta=full-off
    n=len(pre);mag=delta.norm(dim=1);den=mag.sum().clamp_min(1e-12);mass=mag/den
    f={};put=lambda group,name,value:f.__setitem__(f'{group}_{name}',float(value))
    for name,v in {'mean_token_norm':mag.mean(),'max_token_norm':mag.max(),'frobenius':delta.norm(),'over_pre':delta.norm()/pre.norm().clamp_min(1e-12),'over_off':delta.norm()/off.norm().clamp_min(1e-12)}.items():put('f1',name,v)
    for name in ['mean_token_norm','max_token_norm','frobenius']:put('f1','log1p_'+name,math.log1p(f['f1_'+name]))
    for name,v in [('pre',pre),('off',off),('full',full)]:
        for stat,x in _stats(F.cosine_similarity(delta,v,dim=1,eps=1e-12)).items():put('f2',f'cos_{name}_{stat}',x)
        put('f2',f'pooled_cos_{name}',F.cosine_similarity(delta.mean(0),v.mean(0),dim=0,eps=1e-12))
    descending=mag.sort(descending=True).values;ascending=mag.sort().values
    put('f3','top1_share',descending[0]/den);put('f3','top5_share',descending[:min(5,n)].sum()/den);put('f3','max_over_mean',mag.max()/mag.mean().clamp_min(1e-12));put('f3','entropy',-(mass*mass.clamp_min(1e-30).log()).sum())
    put('f3','gini',(2*(torch.arange(1,n+1,device=mag.device)*ascending).sum()/(n*den)-(n+1)/n) if mag.sum()>0 else 0.)
    geom={name:diversity(v) for name,v in [('pre',pre),('full',full)]}
    geom['off']=geom['pre'] if torch.equal(pre,off) else diversity(off)
    for name,values in geom.items():
        for stat,x in values.items():put('f4',f'{name}_{stat}',x)
    for ref in ['pre','off']:
        for stat in geom['full']:put('f4',f'full_minus_{ref}_{stat}',geom['full'][stat]-geom[ref][stat])
    align={}
    for name,v in [('pre',pre),('full',full),('off',off)]:
        a=F.cosine_similarity(v,query[None,:],dim=1,eps=1e-12);align[name]={'mean':a.mean(),'max':a.max(),'top5':a.topk(min(5,n)).values.mean()}
        for stat,x in align[name].items():put('f5',f'{name}_{stat}',x)
    for ref in ['pre','off']:
        for stat in align['full']:put('f5',f'full_minus_{ref}_{stat}',align['full'][stat]-align[ref][stat])
    for frac in [.5,.8]:put('f7',f'tokens_for_{int(frac*100)}pct',int(torch.searchsorted(torch.cumsum(descending/den,0),torch.tensor(frac,device=mag.device)))+1 if mag.sum()>0 else 0)
    idx=torch.linspace(0,1,n,device=mag.device);center=(idx*mass).sum();put('f7','index_center',center);put('f7','index_spread',torch.sqrt(((idx-center).square()*mass).sum()))
    if coords is None:raise ValueError('frozen F7 requires validated visual grid coordinates')
    xy=coords.to(device=pre.device,dtype=torch.float32);xy=(xy-xy.min(0).values)/(xy.max(0).values-xy.min(0).values).clamp_min(1)
    centroid=(xy*mass[:,None]).sum(0);spread=((xy-centroid).square().sum(1)*mass).sum().sqrt()
    put('f7','center_h',centroid[0]);put('f7','center_w',centroid[1]);put('f7','spatial_spread',spread);put('f7','spatial_concentration',1/(1+spread))
    if not all(math.isfinite(v) for v in f.values()):raise ValueError('nonfinite WRITE feature')
    return f

def feature_schema():
    v=torch.tensor([[1.,0.,2.],[0.,2.,1.],[2.,1.,0.]])
    f=write_features(v,v+.1,v,torch.ones(3),torch.tensor([[0,0],[0,1],[1,0]]))
    groups={k:[x for x in f if x.startswith(k.lower()+'_')] for k in ['F1','F2','F3','F4','F5','F7']}
    return {'groups':groups,'F_ALL':list(f),'omitted':{'F6':'optional; visual-row attention/contribution reconstruction requires unavailable full pretext caches and additional operation extraction'},'numeric_dtype':'FP32','covariance':'centered, exact full spectrum of smaller Gram; nonnegative eigenvalue clipping for roundoff','top_k':5}
