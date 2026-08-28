import pytest

from experiments.run_four_action_answer_alignment import select_rows
from tools.research_analysis.four_action.parallelism import (
    artifact_names,
    partition_gpu_rows,
    worker_layout,
)


def test_two_replicas_map_to_each_of_all_eight_gpus():
    layouts = [worker_layout(rank, world_size=16, gpu_count=8) for rank in range(16)]

    assert {layout.gpu_index for layout in layouts} == set(range(8))
    assert {layout.replica_index for layout in layouts} == {0, 1}
    assert all(layout.replicas_per_gpu == 2 for layout in layouts)
    assert layouts[0].gpu_index == layouts[8].gpu_index == 0
    assert layouts[0].replica_index == 0
    assert layouts[8].replica_index == 1


def test_worker_layout_rejects_nonuniform_or_out_of_range_workers():
    with pytest.raises(ValueError, match="multiple"):
        worker_layout(rank=0, world_size=15, gpu_count=8)
    with pytest.raises(ValueError, match="rank"):
        worker_layout(rank=16, world_size=16, gpu_count=8)


def test_replica_partitions_are_disjoint_complete_and_balanced():
    rows = [{"uid": f"u{index}"} for index in range(11)]

    parts = [partition_gpu_rows(rows, replica, replicas_per_gpu=3) for replica in range(3)]

    assert {row["uid"] for part in parts for row in part} == {row["uid"] for row in rows}
    assert sum(len(part) for part in parts) == len(rows)
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_replica_artifacts_do_not_share_append_files():
    assert artifact_names(replicas_per_gpu=1, replica_index=0) == {
        "results": "results.jsonl",
        "failures": "failures.jsonl",
        "runtime": "runtime.json",
    }
    assert artifact_names(replicas_per_gpu=2, replica_index=0)["results"] == "results_replica_00.jsonl"
    assert artifact_names(replicas_per_gpu=2, replica_index=1)["results"] == "results_replica_01.jsonl"


def test_production_selection_splits_one_gpu_shard_between_replicas():
    rows = [
        {"uid": f"u{index}", "cohort": "primary_a_plus", "shard": 0}
        for index in range(5)
    ] + [{"uid": "other_gpu", "cohort": "primary_a_plus", "shard": 1}]
    eligibility = {
        row["uid"]: {"uid": row["uid"], "eligible": True}
        for row in rows
    }

    replica_zero = select_rows(rows, {}, "primary", 0, 16, eligibility)
    replica_one = select_rows(rows, {}, "primary", 8, 16, eligibility)

    assert [row["uid"] for row in replica_zero] == ["u0", "u2", "u4"]
    assert [row["uid"] for row in replica_one] == ["u1", "u3"]


def test_no_correction_selection_excludes_unscorable_textvqa_targets():
    base = {
        "dataset": "textvqa",
        "cohort": "control_no_correction_found",
        "shard": 0,
        "answer": "cat",
        "metric_name": "textvqa_evalai_consensus",
        "correctness_threshold": 0.5,
    }
    rows = [
        {**base, "uid": "valid", "all_answer_norms": ["cat"] * 3 + ["dog"] * 7},
        {
            **base,
            "uid": "unscorable",
            "all_answer_norms": [f"unique {index}" for index in range(10)],
        },
    ]
    eligibility = {row["uid"]: {"uid": row["uid"], "eligible": True} for row in rows}

    selected = select_rows(rows, {}, "control_no_correction", 0, 8, eligibility)

    assert [row["uid"] for row in selected] == ["valid"]
