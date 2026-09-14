"""Synthetic rubric checks; these fixtures are not experimental measurements."""
import json
import pandas as pd
import pytest

from experiments import summarize_benchmark_fixed_schedule as report
from dense_failure_stage2.benchmark_fixed_schedule import FAMILIES


def fixture_tables(root,case):
    for folder in ['metrics','calibration','heldout']:(root/folder).mkdir()
    (root/'metrics/complete.json').write_text(json.dumps({'passed':True}))
    selected=[];half=[];boot=[];nested=[];cis=[];random=[];raw=[];combined=[]
    none=case=='none'
    for b in FAMILIES:
        for bit,pick in [('R',3),('W',4)]:
            selected.append(dict(benchmark=b,bit=bit,selected_layer=None if none else pick))
            half.append(dict(benchmark=b,bit=bit,spearman=None if none else .1 if case=='unstable' else .8))
            boot.append(dict(benchmark=b,bit=bit,layer='NONE' if none else str(pick),frequency=1 if none else .2 if case=='unstable' else .8))
            for i in range(4):nested.append(dict(benchmark=b,bit=bit,selected_layer=None if none else i+1 if case=='unstable' else pick))
        for method in ['M1-Dense','M2-Dense','M3-Dense','GLOBAL-Dense','M3-GLOBAL']:
            positive=(method=='M3-Dense' and case in ['positive','mixed']) or (method=='M1-Dense' and case=='bit')
            cis.append(dict(scope='family',benchmark=b,comparison=method,ci_low=.01 if positive else -.01))
        random.append(dict(scope='family',benchmark=b,calibrated_accuracy=.9 if case=='positive' else .8))
        for uid in range(10):combined.append(dict(benchmark=b,benchmark_family=b,correct=uid<(9 if case=='positive' else 8)))
        for index in range(20):
            for uid in range(10):raw.append(dict(benchmark=b,benchmark_family=b,method=f'RANDOM_{index:03d}',correct=uid<8))
    macro=[]
    for method in ['M1-Dense','M2-Dense','M3-Dense','GLOBAL-Dense','M3-GLOBAL']:
        positive=(case=='positive' and method in ['M3-Dense','M3-GLOBAL']) or (case=='mixed' and method=='M3-Dense') or (case=='bit' and method=='M1-Dense')
        macro.append(dict(comparison=method,macro_delta=.02 if positive else 0,ci_low=.01 if positive else -.01))
    for name,data in [('calibration/selected_layers.csv',selected),('calibration/split_half_stability.csv',half),('calibration/bootstrap_layer_selection.csv',boot),('calibration/calibration_size_stability.csv',nested),('metrics/paired_bootstrap_ci.csv',cis),('metrics/macro_summary.csv',macro),('metrics/benchmark_vs_random.csv',random),('heldout/random_schedule_results.csv',raw),('heldout/combined_results.csv',combined)]:pd.DataFrame(data).to_csv(root/name,index=False)


@pytest.mark.parametrize('case,expected',[('positive','CAL-A'),('bit','CAL-B'),('unstable','CAL-C'),('mixed','CAL-D'),('none','CAL-D')])
def test_prespecified_patterns_and_mixed_evidence(tmp_path,monkeypatch,case,expected):
    fixture_tables(tmp_path,case)
    monkeypatch.setattr(report,'OUT',tmp_path)
    result=report.decision()
    assert result['category']==expected
    if case=='mixed':assert result['mixed_positive_dense_evidence']
    if case=='none':assert result['unstable_benchmarks']==[]
    random=pd.read_csv(tmp_path/'metrics/benchmark_vs_random.csv')
    if case=='positive':assert random.monte_carlo_p.tolist()==pytest.approx([1/21]*len(random))
    else:assert (random.monte_carlo_p==1).all()


def test_random_ties_use_integer_counts_despite_csv_rounding(tmp_path,monkeypatch):
    import numpy as np
    fixture_tables(tmp_path,'mixed')
    path=tmp_path/'metrics/benchmark_vs_random.csv'
    frame=pd.read_csv(path);frame['calibrated_accuracy']=np.nextafter(.8,1.)
    frame.to_csv(path,index=False)
    monkeypatch.setattr(report,'OUT',tmp_path)
    report.decision()
    result=pd.read_csv(path)
    assert (result.monte_carlo_p==1).all()
    assert (result.calibrated_percentile==50).all()


def test_calibration_precluded_category_d_keeps_positive_efficacy(tmp_path,monkeypatch):
    fixture_tables(tmp_path,'positive')
    half=pd.read_csv(tmp_path/'calibration/split_half_stability.csv')
    boot=pd.read_csv(tmp_path/'calibration/bootstrap_layer_selection.csv')
    profiles={('chartqa','R'):(.5006,.286),('chartqa','W'):(.677,.298),('textvqa','R'):(.0916,.5865),('textvqa','W'):(.3341,.556),('mmmupro','R'):(.3503,.2925),('mmmupro','W'):(.2528,.393),('pope','R'):(.3834,.561),('pope','W'):(0,.2225)}
    for i,r in half.iterrows():half.loc[i,'spearman']=profiles[r.benchmark,r.bit][0]
    for i,r in boot.iterrows():boot.loc[i,'frequency']=profiles[r.benchmark,r.bit][1]
    half.to_csv(tmp_path/'calibration/split_half_stability.csv',index=False)
    boot.to_csv(tmp_path/'calibration/bootstrap_layer_selection.csv',index=False)
    monkeypatch.setattr(report,'OUT',tmp_path)
    result=report.decision()
    assert result['category']=='CAL-D' and result['category_determined_by_calibration']
    assert result['m3_replicates'] and result['mixed_positive_dense_evidence']
    assert len(result['positive_m3_families'])==4
