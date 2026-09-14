from types import SimpleNamespace
import torch
from experiments.run_write_harm_propagation import rollout


def test_write_rollout_isolates_visual_change_then_propagates_into_text(monkeypatch):
    import binary_policy.executor.four_action as actions
    import binary_policy.executor.inputs as inputs
    decoder=SimpleNamespace(layers=list(range(28)))
    monkeypatch.setattr(inputs,'resolve_decoder',lambda wrapped:decoder)
    calls=[]
    def layer(wrapped,module,text,visual,prepared,**kwargs):
        action=kwargs['action'];calls.append((kwargs['layer_index'],action,id(kwargs['cache'])))
        return text+visual.mean(1,keepdim=True),visual+1 if action=='FULL' else visual,SimpleNamespace(read_on=True,write_on=action=='FULL')
    monkeypatch.setattr(actions,'four_action_layer',layer)
    baseline=SimpleNamespace(pre_layer_states=[(torch.zeros(1,2,4),torch.ones(1,3,4)) for _ in range(28)],native_causal=True)
    prepared=SimpleNamespace(text_valid_mask=torch.ones(1,2,dtype=torch.bool),visual_valid_mask=torch.ones(1,3,dtype=torch.bool))
    on=rollout(None,baseline,prepared,19,'FULL',(0,1,2,4,8));off=rollout(None,baseline,prepared,19,'READ_ONLY',(0,1,2,4,8))
    assert torch.equal(on[0]['text'],off[0]['text'])
    assert not torch.equal(on[0]['visual'],off[0]['visual'])
    assert not torch.equal(on[1]['text'],off[1]['text'])
    assert off[8]['action_trace']==['READ_ONLY']+['FULL']*8
    assert calls[8][:2]==(27,'FULL') and calls[9][:2]==(19,'READ_ONLY')
    assert torch.equal(baseline.pre_layer_states[19][0],torch.zeros(1,2,4))
    calls.clear();last=rollout(None,baseline,prepared,27,'READ_ONLY',(0,))
    assert len(calls)==1 and list(last)==[0]
