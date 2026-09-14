import numpy as np
from dense_failure_stage2.write_bootstrap import PairedRankAUC
from experiments.analyze_read_counterfactual_stage1_branch_critic import correlation,BinaryMetric


def test_cached_rank_bootstrap_matches_original_with_ties_and_zero_weight_groups():
    rng=np.random.default_rng(10);truth=rng.integers(-2,3,97).astype(float);a=rng.integers(-3,4,97).astype(float);b=rng.normal(size=97);metric=PairedRankAUC(truth,a,b)
    for _ in range(50):
        w=rng.integers(0,4,97);expected=[correlation(truth,a,w,True)-correlation(truth,b,w,True),BinaryMetric(truth>0,a)(w)['AUROC']-BinaryMetric(truth>0,b)(w)['AUROC']]
        np.testing.assert_allclose(metric(w),expected,rtol=1e-12,atol=1e-12)
    assert np.isnan(PairedRankAUC(truth,np.ones(97),b)(np.ones(97))[0])
