from experiments.summarize_binary_pareto_training_fit import first_best_epoch


def test_first_best_epoch_uses_earliest_tie():
    rows = [{"epoch": 1, "value": 0.2}, {"epoch": 2, "value": 0.4}, {"epoch": 3, "value": 0.4}]

    assert first_best_epoch(rows, lambda row: row["value"], maximize=True) == 2
    assert first_best_epoch(rows, lambda row: row["value"], maximize=False) == 1
