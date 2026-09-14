from experiments.finalize_read_only_bounded_planning import matched_any


def history(n, rescue):
    return [dict(q=float(i),correct=i==rescue,suffix=str(i),evaluation_rank=i+1) for i in range(n)]


def test_exhausted_search_requires_equal_actual_comparator_cap():
    # Random's correction at rank4 is unavailable at the structured policy's2 evaluations.
    cap,a,b=matched_any([history(2,None)],[history(8,3)],8)
    assert (cap,a,b)==(2,0.,0.)


def test_seed_average_is_inside_uid_comparison():
    cap,a,b=matched_any([history(8,2)],[history(8,2),history(8,None),history(8,None)],8)
    assert cap==8 and a==1. and b==1/3
