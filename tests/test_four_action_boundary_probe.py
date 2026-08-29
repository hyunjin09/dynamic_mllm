from __future__ import annotations

from collections import Counter

from experiments.prepare_four_action_boundary_probe import (
    build_matched_boundary_probe_records,
)
from four_action_online_router.boundary_probe import (
    BoundaryProbe,
    binary_classification_metrics,
    paired_uid_bootstrap_auc_difference,
)


def test_boundary_probe_pairs_are_balanced_within_split_dataset_and_layer() -> None:
    boundaries = []
    for split in ("train", "validation"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            for layer, count in ((0, 2), (1, 3), (2, 2)):
                for index in range(count):
                    boundaries.append(
                        {
                            "uid": f"{split}:{dataset}:{layer}:{index}",
                            "split": split,
                            "dataset": dataset,
                            "boundary_layer": layer,
                        }
                    )

    records, audit = build_matched_boundary_probe_records(boundaries, seed=19)

    counts = Counter(
        (row["split"], row["dataset"], row["target_layer"], row["label"])
        for row in records
    )
    cells = {(split, dataset, layer) for split, dataset, layer, _ in counts}
    assert all(
        counts[(split, dataset, layer, 0)]
        == counts[(split, dataset, layer, 1)]
        for split, dataset, layer in cells
    )
    assert all(row["target_layer"] < row["source_boundary_layer"] for row in records if row["label"] == 0)
    assert all(row["target_layer"] == row["source_boundary_layer"] for row in records if row["label"] == 1)
    assert audit["balanced"] is True
    assert audit["max_target_layer"] == 1
    assert audit["excluded_positive_boundary_records"] > 0


def test_boundary_probe_metrics_and_paired_bootstrap_are_deterministic() -> None:
    labels = [0, 0, 1, 1]
    upfront = [0.4, 0.1, 0.6, 0.9]
    online = [0.1, 0.2, 0.8, 0.9]
    groups = ["a", "b", "c", "d"]

    metrics = binary_classification_metrics(labels, online)
    assert metrics == {"auroc": 1.0, "accuracy": 1.0, "f1": 1.0}
    first = paired_uid_bootstrap_auc_difference(
        labels, upfront, online, groups, draws=100, seed=7
    )
    second = paired_uid_bootstrap_auc_difference(
        labels, upfront, online, groups, draws=100, seed=7
    )
    assert first == second
    assert first["point_estimate"] == 0.0


def test_boundary_probe_has_the_same_capacity_for_both_representations() -> None:
    upfront = BoundaryProbe(
        hidden_width=16,
        num_layers=28,
        branch_width=8,
        layer_embedding_width=4,
        classifier_hidden_width=8,
        dropout=0.1,
    )
    online = BoundaryProbe(
        hidden_width=16,
        num_layers=28,
        branch_width=8,
        layer_embedding_width=4,
        classifier_hidden_width=8,
        dropout=0.1,
    )
    assert sum(parameter.numel() for parameter in upfront.parameters()) == sum(
        parameter.numel() for parameter in online.parameters()
    )
