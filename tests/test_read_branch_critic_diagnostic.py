import numpy as np
import torch
from types import SimpleNamespace
from experiments.analyze_read_counterfactual_stage1_branch_critic import BinaryMetric,correlation,policy_off,quadrants
from experiments.run_read_counterfactual_stage1_branch_critic import stage1_pool


def test_weighted_auc_and_ap_match_explicit_replication():
    y=np.array([0,1,0,1]);p=np.array([-.2,.1,.1,.9]);w=np.array([2,3,1,4])
    a=BinaryMetric(y,p)(w);b=BinaryMetric(np.repeat(y,w),np.repeat(p,w))(np.ones(w.sum()))
    assert np.isclose(a['AUROC'],b['AUROC']) and np.isclose(a['AUPRC'],b['AUPRC'])
    assert np.isclose(correlation(p,y,w,True),correlation(np.repeat(p,w),np.repeat(y,w),np.ones(w.sum()),True))


def test_tied_scores_and_perfect_ranking():
    y=np.array([0,0,1,1]);a=BinaryMetric(y,np.ones(4)*.5)(np.ones(4))
    assert a['AUROC']==.5 and a['AUPRC']==.5 and a['Brier']==.25 and a['ECE']==0
    perfect=BinaryMetric(y,y)(np.ones(4));assert perfect['AUROC']==1 and perfect['AUPRC']==1
    assert np.isnan(BinaryMetric([1,1],[.2,.3])(np.ones(2))['AUROC'])


def test_frozen_policy_quadrants_and_ties():
    on=np.array([.2,.2,.8,.8,.5,.8]);off=np.array([.1,.8,.2,.7,.5,.8])
    assert quadrants(on,off,.5).tolist()==['Q1','Q2','Q3','Q4','Q4','Q4']
    assert policy_off(on,off,.5).tolist()==[False,False,True,True,False,False]


def test_stage1_uses_user_text_not_last_control_token():
    text=torch.tensor([[[1.,2.],[3.,4.],[100.,200.]]],dtype=torch.bfloat16)
    visual=torch.tensor([[[5.,6.],[7.,8.]]],dtype=torch.bfloat16)
    prepared=SimpleNamespace(text_indices=torch.tensor([[0,3,4]]),visual_valid_mask=torch.tensor([[True,True]]))
    pos=SimpleNamespace(user_text=(0,3),final_user_token=3)
    assert torch.equal(stage1_pool(text,visual,prepared,pos),torch.tensor([3.,4.,2.,3.,6.,7.],dtype=torch.bfloat16))
