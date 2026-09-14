from dense_failure_stage2.benchmark_fixed_schedule import content_groups, group_prefix, schedule, select_layer


def test_transitive_groups_cover_cross_variant_and_multiimage():
    rows=[{'uid':'a','group_keys':['rgb:1','native:mmmu:x']}, {'uid':'b','group_keys':['native:mmmu:x','rgb:2']}, {'uid':'c','group_keys':['rgb:2','rgb:3']}, {'uid':'d','group_keys':['rgb:4']}]
    groups=content_groups(rows)
    assert sorted(map(len,groups))==[1,3]
    assert rows[0]['image_group_id']==rows[2]['image_group_id']
    assert rows[0]['image_group_id']!=rows[3]['image_group_id']


def test_prefix_never_splits_an_image_group():
    groups=[[1,2],[3,4,5],[6]]
    assert group_prefix(groups,4)==[1,2]
    assert group_prefix(groups,5)==[1,2,3,4,5]


def test_selector_none_terminal_exclusion_and_tie_order():
    effects=[dict(layer=i,gain=0,mean_q=0,c_to_w=0,selectable=i<27) for i in range(28)]
    assert select_layer(effects) is None
    effects[27].update(gain=100)
    assert select_layer(effects) is None
    effects[1].update(gain=.1,mean_q=.2,c_to_w=2)
    effects[2].update(gain=.1,mean_q=.3,c_to_w=3)
    assert select_layer(effects)==2
    effects[1]['mean_q']=.3
    assert select_layer(effects)==1
    effects[2]['c_to_w']=2
    assert select_layer(effects)==1


def test_schedule_combines_bits_and_none():
    assert schedule(7,7)[7]=='IGNORE'
    actions=schedule(4,9)
    assert actions[4]=='WRITE_ONLY' and actions[9]=='READ_ONLY'
    assert actions.count('FULL')==26
    assert schedule()==['FULL']*28


def test_missing_q_does_not_discard_native_correctness(monkeypatch):
    from types import SimpleNamespace
    import torch
    from binary_policy.executor.four_action import FourActionLayerExecution
    from experiments import run_benchmark_fixed_schedule as runner
    monkeypatch.setattr(runner.reference,'_generate',lambda *args:dict(correct=False,score=0.,generated_answer='unknown',generated_token_ids=[1]))
    runtime=runner.Runtime.__new__(runner.Runtime)
    output=SimpleNamespace(layer_actions=['FULL']*28,layer_stats=[FourActionLayerExecution(i,'FULL',True,True,3,3,1) for i in range(28)],inputs=SimpleNamespace(visual_states=torch.zeros(1,2,3)))
    result=runtime.measure(output,{'input_ids':torch.ones(1,4,dtype=torch.long)},{'benchmark':'textvqa','answer':'','all_answer_norms':['']},True)
    assert result['correct'] is False and result['score']==0
    assert result['q'] is None and result['q_invalid_reason']=='TextVQA q references are empty'
    assert result['actions']==['FULL']*28


def test_native_punctuation_is_retained_but_score_mismatch_stops(monkeypatch):
    from experiments import run_benchmark_fixed_schedule as runner
    import pytest
    native=dict(generated_token_ids=[33,13,151645],generated_answer='B.',correct=True,score=1.)
    unified=dict(generated_token_ids=[33,151645],generated_answer='B',correct=True,score=1.,q=-.4)
    monkeypatch.setattr(runner.reference,'_native_dense_generation',lambda *args,**kwargs:native)
    result=runner.authoritative_dense(None,{'uid':'sentinel'},unified,native,'test')
    assert result['generated_answer']=='B.' and result['unified_generated_answer']=='B'
    assert result['q']==-.4 and result['unified_token_mismatch']
    with pytest.raises(AssertionError,match='scorer mismatch'):
        runner.authoritative_dense(None,{'uid':'sentinel'},dict(unified,correct=False,score=0.),native,'test')


def test_native_decode_continues_last_position_not_prefix_maximum():
    from types import SimpleNamespace
    import torch
    import pytest
    from experiments.run_benchmark_fixed_schedule import align_native_decode_positions
    meta=SimpleNamespace(full_attention_mask=torch.ones(1,4),full_position_ids=torch.tensor([[[0,8,2,3]]]*3),full_prompt_len=torch.tensor([4]),rope_deltas=torch.tensor([[5]]))
    before=meta.full_position_ids.clone()
    align_native_decode_positions(meta)
    assert (meta.full_prompt_len+meta.rope_deltas[:,0]).item()==4
    assert torch.equal(meta.full_position_ids,before)
    meta.full_position_ids[1,0,-1]=9
    with pytest.raises(AssertionError,match='rotary planes'):
        align_native_decode_positions(meta)
