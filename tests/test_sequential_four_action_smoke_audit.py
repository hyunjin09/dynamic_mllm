from __future__ import annotations

from experiments.audit_sequential_four_action_smoke import (
    conversion_semantics_valid,
    build_smoke_audit,
)


def _evaluation(correct=True):
    return {
        "correct": correct,
        "generated_ids": [1],
        "generated_answer": "yes",
        "correct_target_scores": [{"text": "yes"}],
    }


def test_conversion_truth_table_and_early_to_late_context_are_audited():
    conversion = {
        "status": "converted",
        "source_binary_route": [0, 0],
        "label_semantics": "corrective_w2c",
        "steps": [
            {
                "layer": 0,
                "incoming_branch_count": 1,
                "full_restored_count": 0,
                "read_only_only_count": 0,
                "write_only_only_count": 0,
                "both_partial_correct_count": 1,
                "ignore_fallback_count": 0,
                "outgoing_branch_count": 2,
            },
            {
                "layer": 1,
                "incoming_branch_count": 2,
                "full_restored_count": 1,
                "read_only_only_count": 0,
                "write_only_only_count": 0,
                "both_partial_correct_count": 0,
                "ignore_fallback_count": 1,
                "outgoing_branch_count": 2,
            },
        ],
        "final_branches": [
            {
                "route": ["READ_ONLY", "FULL"],
                "evaluation": _evaluation(),
                "decisions": [
                    {
                        "layer": 0,
                        "action": "READ_ONLY",
                        "full_correct": False,
                        "read_only_correct": True,
                        "write_only_correct": True,
                    },
                    {
                        "layer": 1,
                        "action": "FULL",
                        "full_correct": True,
                        "read_only_correct": None,
                        "write_only_correct": None,
                    },
                ],
            },
            {
                "route": ["WRITE_ONLY", "IGNORE"],
                "evaluation": _evaluation(),
                "decisions": [
                    {
                        "layer": 0,
                        "action": "WRITE_ONLY",
                        "full_correct": False,
                        "read_only_correct": True,
                        "write_only_correct": True,
                    },
                    {
                        "layer": 1,
                        "action": "IGNORE",
                        "full_correct": False,
                        "read_only_correct": False,
                        "write_only_correct": False,
                    },
                ],
            },
        ],
    }

    assert conversion_semantics_valid(conversion, route_type="W2C")
    conversion["steps"][1]["layer"] = 0
    assert not conversion_semantics_valid(conversion, route_type="W2C")


def test_replay_failure_is_valid_only_when_it_is_explicit_and_unrefined():
    failure = {
        "status": "source_route_replay_failure",
        "failure_reason": "current unified executor did not reproduce correctness",
    }

    assert conversion_semantics_valid(failure, route_type="W2C")
    failure["final_branches"] = [{"route": ["IGNORE"]}]
    assert not conversion_semantics_valid(failure, route_type="W2C")


def test_smoke_audit_requires_all_workers_and_semantic_parity():
    datasets = ["gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro"]
    records = []
    for rank in range(8):
        is_w2c = rank == 0
        source_route = [0, 1] if is_w2c else [1, 1]
        evaluation = _evaluation()
        if is_w2c:
            conversion = {
                "status": "converted",
                "source_binary_route": source_route,
                "source_route_evaluation": evaluation,
                "label_semantics": "corrective_w2c",
                "steps": [
                    {
                        "layer": 0,
                        "incoming_branch_count": 1,
                        "full_restored_count": 0,
                        "read_only_only_count": 1,
                        "write_only_only_count": 0,
                        "both_partial_correct_count": 0,
                        "ignore_fallback_count": 0,
                        "outgoing_branch_count": 1,
                    }
                ],
                "final_branches": [
                    {
                        "route": ["READ_ONLY", "FULL"],
                        "evaluation": evaluation,
                        "decisions": [
                            {
                                "layer": 0,
                                "action": "READ_ONLY",
                                "full_correct": False,
                                "read_only_correct": True,
                                "write_only_correct": False,
                            }
                        ],
                    }
                ],
            }
            unique_route = ["READ_ONLY", "FULL"]
        else:
            conversion = {
                "status": "converted",
                "source_binary_route": source_route,
                "source_route_evaluation": evaluation,
                "label_semantics": "preserving_c2c",
                "steps": [],
                "final_branches": [
                    {"route": ["FULL", "FULL"], "evaluation": evaluation, "decisions": []}
                ],
            }
            unique_route = ["FULL", "FULL"]
        records.append(
            {
                "uid": f"u{rank}",
                "dataset": datasets[rank % 5],
                "passed": True,
                "route_type": "W2C" if is_w2c else "C2C",
                "current_unified_full": {
                    **evaluation,
                    "correct": not is_w2c,
                },
                "current_unified_all_off": evaluation,
                "source_positive_route_count": 1,
                "source_route_replay_valid_count": 1,
                "source_route_replay_failure_count": 0,
                "raw_conversions": [conversion],
                "unique_valid_four_action_routes": [
                    {"four_action_route": unique_route, "evaluation": evaluation}
                ],
                "pilot_old_binary_semantic_checks": [
                    {
                        "generated_ids_match": True,
                        "generated_answer_match": True,
                        "correctness_match": True,
                    }
                ],
                "route_evaluation_cache": {"cache_hits": 1},
                "execution_contract": {"contract_sha256": "contract"},
            }
        )
    progress = []
    for rank in range(8):
        progress.extend(
            [
                {"event": "worker_start", "rank": rank, "gpu_index": rank, "replica_index": 0},
                {"event": "worker_complete", "rank": rank},
            ]
        )

    report = build_smoke_audit(
        records,
        expected_uids={f"u{x}" for x in range(8)},
        failure_rows=[],
        progress=progress,
        checksum_errors=[],
        resume_verified=True,
        slurm_jobs=[{"state": "COMPLETED", "exit_code": "0:0"}],
    )

    assert report["passed"]
    assert all(report["checks"].values())
