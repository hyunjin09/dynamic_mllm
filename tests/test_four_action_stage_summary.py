from experiments.summarize_four_action_stage import (
    artifact_paths,
    drift_distribution,
    expected_count,
    sample_gate_semantically_passes,
    worker_contract_passes,
)


def summary():
    return {
        "smoke_ids": list(range(8)),
        "pilot_ids": list(range(56)),
        "primary_rows": 1912,
        "taxonomy": {
            "gqa": {
                "control_no_correction_found": 614,
                "control_full_correct_all_off_wrong": 1145,
            },
            "textvqa": {
                "control_no_correction_found": 254,
                "control_full_correct_all_off_wrong": 965,
            },
        },
    }


def test_expected_stage_counts():
    state = summary()
    assert expected_count("preflight", state) == 8
    assert expected_count("smoke", state) == 8
    assert expected_count("pilot", state) == 56
    assert expected_count("primary", state) == 1912
    assert expected_count("control_no_correction", state) == 868
    assert expected_count("control_vision_required", state) == 2110
    eligibility = {
        "eligible_counts": {
            "primary_a_plus": 1900,
            "control_no_correction_found": 860,
            "control_full_correct_all_off_wrong": 2100,
        }
    }
    assert expected_count("primary", state, eligibility) == 1900
    assert expected_count("control_no_correction", state, eligibility) == 860
    assert expected_count(
        "control_no_correction", state, eligibility, target_unscorable_count=11
    ) == 849
    assert expected_count("control_vision_required", state, eligibility) == 2100


def test_drift_distribution_reports_requested_statistics():
    result = drift_distribution([-2.0, 0.0, 2.0])
    assert result["count"] == 3
    assert result["mean"] == 0.0
    assert result["median"] == 0.0
    assert result["minimum"] == -2.0
    assert result["maximum"] == 2.0
    assert {"std", "p90", "p95", "p99"} <= set(result)


def test_historical_anchor_token_drift_is_not_a_semantic_gate():
    row = {
        "sample_gate": {
            "passed": False,
            "checks": {
                "baseline_cached_ids_match": False,
                "unified_full_native_semantic_parity": True,
                "unified_ignore_binary_semantic_parity": True,
            },
        }
    }
    assert sample_gate_semantically_passes(row)
    row["sample_gate"]["checks"]["unified_full_native_semantic_parity"] = False
    assert not sample_gate_semantically_passes(row)


def test_trajectory_identity_gate_can_be_rechecked_at_current_bf16_tolerance():
    row = {
        "sample_gate": {
            "passed": False,
            "checks": {"trajectory_final_margin_identity": False},
        },
        "unified_full_answer_trajectory": {
            "final_margin_vs_factorial_baseline_abs_diff": 9.3e-5
        },
    }

    assert not sample_gate_semantically_passes(row, trajectory_atol=1e-5)
    assert sample_gate_semantically_passes(row, trajectory_atol=1e-4)


def test_stage_merger_discovers_base_and_replica_artifacts(tmp_path):
    for name in (
        "results.jsonl",
        "results_replica_00.jsonl",
        "results_replica_01.jsonl",
        "runtime.json",
        "runtime_replica_00.json",
        "runtime_replica_01.json",
        "failures_replica_01.jsonl",
    ):
        (tmp_path / name).write_text("")

    assert [path.name for path in artifact_paths(tmp_path, "results", ".jsonl")] == [
        "results.jsonl",
        "results_replica_00.jsonl",
        "results_replica_01.jsonl",
    ]
    assert [path.name for path in artifact_paths(tmp_path, "runtime", ".json")] == [
        "runtime.json",
        "runtime_replica_00.json",
        "runtime_replica_01.json",
    ]
    assert [path.name for path in artifact_paths(tmp_path, "failures", ".jsonl")] == [
        "failures_replica_01.jsonl"
    ]


def test_worker_contract_accepts_complete_two_replica_resume_with_legacy_metadata():
    legacy = [{"rank": rank, "world_size": 8} for rank in range(8)]
    multiplexed = [
        {
            "rank": replica * 8 + gpu,
            "world_size": 16,
            "gpu_index": gpu,
            "replica_index": replica,
            "replicas_per_gpu": 2,
        }
        for replica in range(2)
        for gpu in range(8)
    ]

    assert worker_contract_passes(legacy)
    assert worker_contract_passes(legacy + multiplexed)
    assert not worker_contract_passes(legacy + multiplexed[:-1])
