from dense_failure_stage2.read_only_planning import SearchSession, actions, curve, enumeration, route_hash
import pytest


def oracle(route):
    # Two OFF edits are necessary; single edits lure greedy elsewhere.
    return {'q': 5. if route.startswith('00') else float(route.count('1')), 'correct': route.startswith('00')}


@pytest.mark.parametrize('algorithm',['greedy','beam2','beam4','beam8','random_uniform','random_sparse','binary_mcts'])
def test_resume_unique_budget_and_determinism(algorithm):
    a=SearchSession(algorithm,4,'uid',9)
    a.advance(4,oracle)
    a.advance(16,oracle)
    b=SearchSession(algorithm,4,'uid',9)
    b.advance(16,oracle)
    assert a.rows==b.rows
    assert len({r['suffix'] for r in a.rows})==len(a.rows)
    assert [r['evaluation_rank'] for r in a.rows]==list(range(1,len(a.rows)+1))
    assert all(set(r['suffix'])<=set('01') for r in a.rows)
    assert a.rows[0]['suffix']=='1111'


def test_width_exposes_coordinated_correction():
    a=SearchSession('beam4',4,'uid'); a.advance(16,oracle)
    assert curve(a.rows,16)['any_correct']


def test_ranking_failure_distinct_from_discovery():
    rows=[dict(q=2.,correct=False,suffix='11',evaluation_rank=1),dict(q=1.,correct=True,suffix='00',evaluation_rank=2)]
    assert curve(rows,2)['outcome']=='RANKING_FAILURE'
    assert curve(rows,1)['outcome']=='GENERATION_FAILURE'


def test_finite_support_and_enumeration():
    for alg in ('binary_mcts','random_sparse','random_uniform'):
        s=SearchSession(alg,2,'u'); s.advance(128,oracle)
        assert s.exhausted and len(s.rows)==4
    assert len(list(enumeration(28)))==407


def test_route_contract():
    assert actions(26,'01')==('FULL',)*26+('WRITE_ONLY','FULL')
    assert route_hash('a',26,'01')!=route_hash('b',26,'01')
    with pytest.raises(ValueError): actions(26,'02')
