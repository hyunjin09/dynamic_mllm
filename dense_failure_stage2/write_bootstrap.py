"""Exact tied-rank bootstrap with ordering cached before resampling."""
import numpy as np
from dense_failure_stage2.write_harm_learnability import harm_metrics


class PairedRankAUC:
    def __init__(self,truth,left,right):
        self.arrays=[np.asarray(x,float) for x in [truth,left,right]]
        self.inverse=[np.unique(x,return_inverse=True)[1] for x in self.arrays]
        self.sizes=[int(i.max())+1 for i in self.inverse]
        self.positive=self.arrays[0]>0
    def __call__(self,w):
        w=np.asarray(w,float);ranks=[]
        for inv,n in zip(self.inverse,self.sizes):
            mass=np.bincount(inv,weights=w,minlength=n);r=(mass.cumsum()-mass/2)[inv];ranks.append(r-(r*w).sum()/w.sum())
        variance=[(r*r*w).sum() for r in ranks]
        rho=[(ranks[0]*ranks[j]*w).sum()/np.sqrt(variance[0]*variance[j]) if variance[0]*variance[j]>0 else np.nan for j in [1,2]]
        auc=[]
        for inv,n in zip(self.inverse[1:],self.sizes[1:]):
            pos=np.bincount(inv,weights=w*self.positive,minlength=n);neg=np.bincount(inv,weights=w*(~self.positive),minlength=n);den=pos.sum()*neg.sum();auc.append((pos*(neg.cumsum()-neg/2)).sum()/den if den>0 else np.nan)
        return [rho[0]-rho[1],auc[0]-auc[1]]


def bootstrap_comparison(rows,truth,left,right,seed=2026091499):
    groups=sorted({r['image_group_id'] for r in rows});index={g:i for i,g in enumerate(groups)};gi=np.array([index[r['image_group_id']] for r in rows]);rng=np.random.default_rng(seed);metric=PairedRankAUC(truth,left,right)
    draws=np.array([metric(np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))[gi]) for _ in range(5000)])
    point=metric(np.ones(len(rows)))
    return dict(spearman_gain=float(point[0]),spearman_ci_low=float(np.nanquantile(draws[:,0],.025)),spearman_ci_high=float(np.nanquantile(draws[:,0],.975)),harmful_auc_gain=float(point[1]),harmful_auc_ci_low=float(np.nanquantile(draws[:,1],.025)),harmful_auc_ci_high=float(np.nanquantile(draws[:,1],.975)),draws=5000,bootstrap_unit='image_group_id')
