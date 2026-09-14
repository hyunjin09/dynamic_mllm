import numpy as np
import torch
from dense_failure_stage2.write_harm_learnability import horizons_for_layer,diversity,write_features,harm_metrics


def test_write_horizon_counts_layers_after_intervention():
    assert horizons_for_layer(27)==(0,)
    assert horizons_for_layer(19)==(0,1,2,4,8)
    assert horizons_for_layer(20)==(0,1,2,4)


def test_exact_covariance_rank_and_zero_variance():
    d=diversity(torch.eye(3))
    assert torch.isclose(d['effective_rank'],torch.tensor(2.),atol=1e-5)
    assert diversity(torch.ones(3,4))['effective_rank']==0


def test_write_geometry_zero_update_and_uniform_update():
    v=torch.eye(3);q=torch.ones(3);coords=torch.tensor([[0,0],[0,1],[1,0]])
    zero=write_features(v,v,v,q,coords)
    assert zero['f1_frobenius']==0 and zero['f3_gini']==0 and zero['f7_tokens_for_50pct']==0
    assert zero['f4_full_minus_off_effective_rank']==0
    uniform=write_features(v,v+.25,v,q,coords)
    assert np.isclose(uniform['f3_top1_share'],1/3) and np.isclose(uniform['f3_gini'],0,atol=1e-6)
    assert np.isclose(uniform['f1_frobenius'],.75)


def test_baseline_sign_transform_and_zero_harm_denominator():
    old=np.array([-.3,0,.2,.4]);prediction=np.array([-.2,.01,.1,.5])
    a=harm_metrics(-old,-prediction);b=harm_metrics(old,prediction)
    assert a['spearman']==b['spearman'] and a['mae']==b['mae']
    assert a['harmful_prevalence']==.25 and b['harmful_prevalence']==.5


def test_precision_recall_does_not_split_score_ties():
    m=harm_metrics(np.array([1.,1.,0.,-1.]),np.ones(4))
    assert m['recall_at_precision_0.9']==0
