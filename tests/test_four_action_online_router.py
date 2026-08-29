from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

import pytest
import torch

from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.data import (
    boundary_teacher_route,
    choose_smoke_indices,
    mandatory_boundary_record,
    manifest_route_tensor,
    select_boundary_pilot,
)
from four_action_online_router.metrics import (
    mandatory_boundary_metrics,
    mandatory_boundary_pilot_gate,
    execution_checkpoint_key,
    summarize_execution_rows,
    summarize_node_predictions,
)
from four_action_online_router.runtime import select_last_text_state
from four_action_online_router.supervision import (
    PrefixTrie,
    balanced_epoch_indices,
    set_valued_action_loss,
)


def _sha256(path: Path) -> str:
    from experiments.train_binary_polar import file_sha256

    return file_sha256(path)


def test_prefix_trie_preserves_every_valid_next_action() -> None:
    routes = [
        (3, 1, 0),
        (3, 2, 3),
        (0, 3, 3),
    ]
    trie = PrefixTrie(routes)

    assert trie.valid_actions(()) == frozenset({0, 3})
    assert trie.valid_actions((3,)) == frozenset({1, 2})
    assert trie.valid_actions((3, 1)) == frozenset({0})
    masks = trie.valid_action_masks_for_route(routes[0])
    assert masks.tolist() == [
        [True, False, False, True],
        [False, True, True, False],
        [True, False, False, False],
    ]


def test_set_valued_loss_rewards_total_valid_probability_mass() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0, -1.0]], requires_grad=True)
    valid = torch.tensor([[False, True, True, False]])

    loss = set_valued_action_loss(logits, valid)
    expected = -torch.logsumexp(logits[:, 1:3], dim=-1) + torch.logsumexp(
        logits, dim=-1
    )

    assert torch.allclose(loss, expected.mean())
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 2] < 0


def test_balanced_epoch_sampler_is_type_dataset_balanced_and_reproducible() -> None:
    rows = []
    for route_type in ("W2C", "C2C"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            for index in range(3):
                rows.append(
                    {
                        "uid": f"{route_type}:{dataset}:{index}",
                        "route_type": route_type,
                        "dataset": dataset,
                    }
                )

    first = balanced_epoch_indices(
        rows, samples_per_epoch=24, seed=17, epoch=2, world_size=4
    )
    repeated = balanced_epoch_indices(
        rows, samples_per_epoch=24, seed=17, epoch=2, world_size=4
    )
    changed = balanced_epoch_indices(
        rows, samples_per_epoch=24, seed=17, epoch=3, world_size=4
    )

    assert first == repeated
    assert first != changed
    assert len(first) == 24
    counts = Counter((rows[index]["route_type"], rows[index]["dataset"]) for index in first)
    assert set(counts.values()) == {4}
    assert all(len(first[rank::4]) == 6 for rank in range(4))


def test_router_emits_structured_joint_logits_and_separate_visual_queries() -> None:
    torch.manual_seed(4)
    router = OnlineFourActionRouter(
        hidden_size=12,
        num_layers=3,
        d_router=8,
        num_heads=2,
        mlp_hidden_size=16,
        dropout=0.0,
        interaction_scale=0.1,
    )
    text = torch.randn(2, 12)
    visual = torch.randn(2, 5, 12)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    layers = torch.tensor([0, 2])

    features = router.forward_features(text, visual, mask, layers)
    changed = router.forward_features(text + 0.5, visual, mask, layers)
    logits = router(text, visual, mask, layers)

    assert logits.shape == (2, 4)
    assert features.read_visual.shape == (2, 8)
    assert features.write_visual.shape == (2, 8)
    assert not torch.equal(features.read_visual, changed.read_visual)
    assert torch.equal(features.write_visual, changed.write_visual)

    logits.sum().backward()
    assert router.read_layer_queries.weight.grad is not None
    assert router.write_layer_queries.weight.grad is not None
    assert router.read_layer_queries.weight.grad[0].abs().sum() > 0
    assert router.write_layer_queries.weight.grad[2].abs().sum() > 0


def test_external_loader_requires_preselected_checksum_bound_router(tmp_path: Path) -> None:
    from experiments.evaluate_four_action_online_router_external import load_router

    config = {
        "router": {
            "hidden_size": 12,
            "num_layers": 3,
            "d_router": 8,
            "num_heads": 2,
            "mlp_hidden_size": 16,
            "dropout": 0.0,
            "interaction_scale": 0.1,
        }
    }
    router = OnlineFourActionRouter(**config["router"])
    checkpoint = tmp_path / "router.pt"
    torch.save({"router": router.state_dict(), "config_sha256": "frozen"}, checkpoint)
    selection = tmp_path / "best.json"
    selection.write_text(
        json.dumps(
            {
                "selected_before_external_evaluation": True,
                "config_sha256": "frozen",
                "best_checkpoint": str(checkpoint),
                "best_checkpoint_sha256": _sha256(checkpoint),
                "best_epoch": 4,
            }
        ),
        encoding="utf-8",
    )

    loaded, metadata, resolved = load_router(
        config, selection, torch.device("cpu"), config_sha256="frozen"
    )

    assert resolved == checkpoint
    assert metadata["best_epoch"] == 4
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(router.state_dict().values(), loaded.state_dict().values())
    )

    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_before_external_evaluation"] = False
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not selected"):
        load_router(config, selection, torch.device("cpu"), config_sha256="frozen")


def test_smoke_compute_calibration_accounts_for_all_requested_stages() -> None:
    from experiments.record_four_action_online_router_compute import calibrated_estimate

    estimate = calibrated_estimate(
        {
            "passed": True,
            "measured_smoke_body_seconds": 60.0,
            "qwen_route_equivalents_per_rank": 3,
        }
    )

    assert estimate["calibration"]["seconds_per_route_equivalent_per_gpu"] == 20.0
    assert estimate["training_teacher_replay"]["route_equivalents"] == 61_440
    assert estimate["epoch_validation_teacher_plus_routed"]["route_equivalents"] == 17_320
    assert estimate["external_routed_plus_unified_full"]["route_equivalents"] == 29_920
    assert estimate["combined"]["route_equivalents"] == 108_680


def test_epoch_artifacts_commit_checkpoint_and_validation_together(tmp_path: Path) -> None:
    from experiments.train_four_action_online_router import save_epoch_artifacts

    metadata = save_epoch_artifacts(
        tmp_path,
        2,
        {"router": {"weight": torch.tensor([1.0])}, "metrics": {"epoch": 2}},
        [{"uid": "sample", "correct": True}],
    )

    epoch_dir = tmp_path / "epoch_02"
    assert epoch_dir.is_dir()
    assert Path(metadata["checkpoint"]) == epoch_dir / "router_checkpoint.pt"
    assert (epoch_dir / "metadata.json").is_file()
    assert json.loads((epoch_dir / "validation_outputs.jsonl").read_text()) == {
        "uid": "sample",
        "correct": True,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        save_epoch_artifacts(tmp_path, 2, {}, [])


def test_smoke_output_directory_is_created_once_by_rank_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.train_four_action_online_router import prepare_smoke_output_dir

    barriers = []
    monkeypatch.setattr(
        "experiments.train_four_action_online_router.dist.barrier",
        lambda: barriers.append("barrier"),
    )
    output_dir = tmp_path / "smoke"

    prepare_smoke_output_dir(output_dir, rank=0)
    prepare_smoke_output_dir(output_dir, rank=1)

    assert output_dir.is_dir()
    assert barriers == ["barrier", "barrier"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_smoke_output_dir(output_dir, rank=0)


def test_smoke_and_training_share_the_warmup_scheduler_contract() -> None:
    from experiments.train_four_action_online_router import (
        build_training_optimizer_and_scheduler,
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer, scheduler = build_training_optimizer_and_scheduler(
        [parameter],
        {
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "total_optimizer_steps": 480,
        },
    )

    learning_rates = []
    for _ in range(4):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    assert learning_rates == pytest.approx([0.0, 5e-5, 1e-4, 1.5e-4])


def test_structured_logits_use_ignore_read_write_full_action_order() -> None:
    read = torch.tensor([2.0])
    write = torch.tensor([3.0])
    residual = torch.zeros(1, 4)

    logits = OnlineFourActionRouter.structured_logits(
        read, write, residual, interaction_scale=0.1
    )

    assert logits.tolist() == [[-5.0, -1.0, 1.0, 5.0]]


def test_uniform_logits_have_log_four_set_loss_for_one_valid_action() -> None:
    logits = torch.zeros(7, 4)
    valid = torch.zeros(7, 4, dtype=torch.bool)
    valid[:, 3] = True

    assert set_valued_action_loss(logits, valid).item() == torch.tensor(
        math.log(4.0)
    ).item()


def test_last_text_state_uses_each_samples_final_valid_control_row() -> None:
    states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False]], dtype=torch.bool
    )

    selected = select_last_text_state(states, valid)

    assert torch.equal(selected[0], states[0, 1])
    assert torch.equal(selected[1], states[1, 2])


def test_label_audit_counts_prefix_nodes_and_multi_valid_states() -> None:
    from experiments.audit_four_action_online_router import build_label_trie_audit

    rows = [
        {
            "uid": "gqa:a",
            "dataset": "gqa",
            "split": "train",
            "route_type": "W2C",
            "executor_contract_sha256": "contract",
            "valid_routes": [
                {"actions": ["FULL", "READ_ONLY"]},
                {"actions": ["FULL", "WRITE_ONLY"]},
            ],
        },
        {
            "uid": "chartqa:b",
            "dataset": "chartqa",
            "split": "validation",
            "route_type": "C2C",
            "executor_contract_sha256": "contract",
            "valid_routes": [{"actions": ["IGNORE", "FULL"]}],
        },
    ]

    audit = build_label_trie_audit(rows, expected_contract="contract", num_layers=2)

    assert audit["samples"] == 2
    assert audit["routes"] == 3
    assert audit["trie_nodes"]["total"] == 4
    assert audit["valid_action_multiplicity"]["two"] == 1
    assert audit["valid_action_multiplicity"]["one"] == 3


def test_node_metrics_treat_any_valid_next_action_as_correct() -> None:
    logits = torch.tensor(
        [[0.0, 3.0, 2.0, 1.0], [0.0, 0.0, 1.0, 4.0], [5.0, 0.0, 0.0, 0.0]]
    )
    valid = torch.tensor(
        [
            [False, True, True, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )

    metrics = summarize_node_predictions(logits, valid)

    assert metrics["states"] == 3
    assert metrics["valid_action_at_1"] == pytest.approx(2 / 3)
    assert metrics["predicted_action_counts"] == {
        "IGNORE": 1,
        "READ_ONLY": 1,
        "WRITE_ONLY": 0,
        "FULL": 1,
    }


def test_execution_metrics_and_checkpoint_key_prioritize_balanced_behavior() -> None:
    rows = [
        {"uid": "w1", "dataset": "gqa", "route_type": "W2C", "correct": True, "actions": ["FULL", "IGNORE"]},
        {"uid": "w2", "dataset": "chartqa", "route_type": "W2C", "correct": False, "actions": ["FULL", "FULL"]},
        {"uid": "c1", "dataset": "gqa", "route_type": "C2C", "correct": True, "actions": ["READ_ONLY", "FULL"]},
        {"uid": "c2", "dataset": "textvqa", "route_type": "C2C", "correct": True, "actions": ["WRITE_ONLY", "FULL"]},
    ]
    summary = summarize_execution_rows(rows)

    assert summary["w2c_rescue_rate"] == 0.5
    assert summary["c2c_preservation_rate"] == 1.0
    assert summary["balanced_execution_score"] == 0.75
    assert summary["overall_routed_accuracy"] == 0.75
    assert summary["c2c_regressions"] == 0

    better_balanced = {"epoch": 2, "execution": {**summary, "balanced_execution_score": 0.8}}
    earlier = {"epoch": 1, "execution": summary}
    assert execution_checkpoint_key(better_balanced) > execution_checkpoint_key(earlier)


def test_online_data_helpers_encode_routes_and_choose_balanced_smoke() -> None:
    rows = []
    for route_type in ("W2C", "C2C"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            for index in range(2):
                rows.append(
                    {
                        "uid": f"{route_type}:{dataset}:{index}",
                        "dataset": dataset,
                        "route_type": route_type,
                        "valid_routes": [
                            {"actions": ["FULL", "IGNORE"]},
                            {"actions": ["FULL", "READ_ONLY"]},
                        ],
                    }
                )
    encoded = manifest_route_tensor(rows[0], num_layers=2)
    selected = choose_smoke_indices(rows, records=8, seed=5)

    assert encoded.tolist() == [[3, 0], [3, 1]]
    assert len(selected) == 8
    chosen = [rows[index] for index in selected]
    assert {row["dataset"] for row in chosen} == {"gqa", "chartqa", "textvqa"}
    assert {row["route_type"] for row in chosen} == {"W2C", "C2C"}


def test_mandatory_boundary_uses_latest_reachable_all_full_prefix() -> None:
    row = {
        "uid": "gqa:boundary",
        "dataset": "gqa",
        "split": "train",
        "route_type": "W2C",
        "valid_routes": [
            {
                "route_key": "short",
                "source_binary_route_ids": ["binary:short"],
                "actions": ["FULL", "READ_ONLY", "FULL", "FULL"],
            },
            {
                "route_key": "latest-read",
                "source_binary_route_ids": ["binary:read"],
                "actions": ["FULL", "FULL", "READ_ONLY", "IGNORE"],
            },
            {
                "route_key": "latest-write",
                "source_binary_route_ids": ["binary:write"],
                "actions": ["FULL", "FULL", "WRITE_ONLY", "FULL"],
            },
        ],
    }

    boundary = mandatory_boundary_record(row, num_layers=4)

    assert boundary == {
        "uid": "gqa:boundary",
        "dataset": "gqa",
        "boundary_layer": 2,
        "all_full_prefix_length": 2,
        "all_full_prefix": ["FULL", "FULL"],
        "valid_nonfull_actions": ["READ_ONLY", "WRITE_ONLY"],
        "boundary_route_indices": [1, 2],
        "boundary_route_keys": ["latest-read", "latest-write"],
        "source_binary_route_ids": ["binary:read", "binary:write"],
        "singleton": False,
    }


def test_mandatory_boundary_rejects_an_all_full_w2c_route() -> None:
    row = {
        "uid": "gqa:invalid",
        "dataset": "gqa",
        "split": "train",
        "route_type": "W2C",
        "valid_routes": [
            {"route_key": "all-full", "actions": ["FULL", "FULL"]},
        ],
    }

    with pytest.raises(ValueError, match="all-FULL route"):
        mandatory_boundary_record(row, num_layers=2)


def test_boundary_teacher_route_reaches_the_exact_frozen_boundary() -> None:
    row = {
        "uid": "gqa:teacher",
        "route_type": "W2C",
        "valid_routes": [
            {"actions": ["FULL", "IGNORE", "FULL"]},
            {"actions": ["FULL", "FULL", "WRITE_ONLY"]},
        ],
    }
    boundary = {
        "uid": "gqa:teacher",
        "boundary_layer": 2,
        "valid_nonfull_actions": ["WRITE_ONLY"],
        "boundary_route_indices": [1],
        "teacher_route_index": 1,
    }

    route = boundary_teacher_route(row, boundary, num_layers=3)

    assert route.tolist() == [3, 3, 2]


def test_boundary_pilot_selection_is_fixed_balanced_and_route_compatible() -> None:
    boundary_rows = []
    manifest_rows = []
    actions = ("IGNORE", "READ_ONLY", "WRITE_ONLY")
    for dataset in ("gqa", "chartqa", "textvqa"):
        for index in range(12):
            uid = f"{dataset}:w{index:02d}"
            action = actions[index % len(actions)]
            boundary_rows.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "boundary_layer": index % 8,
                    "valid_nonfull_actions": [action],
                    "singleton": True,
                }
            )
            manifest_rows.append(
                {"uid": uid, "dataset": dataset, "route_type": "W2C"}
            )
        for index in range(5):
            uid = f"{dataset}:c{index:02d}"
            manifest_rows.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "route_type": "C2C",
                    "valid_routes": [
                        {"actions": ["FULL", "FULL"]},
                        {"actions": ["FULL", "IGNORE"]},
                    ],
                }
            )

    selected = select_boundary_pilot(
        manifest_rows,
        boundary_rows,
        w2c_per_dataset=6,
        c2c_per_dataset=3,
        seed=17,
        num_layers=2,
    )
    repeated = select_boundary_pilot(
        manifest_rows,
        boundary_rows,
        w2c_per_dataset=6,
        c2c_per_dataset=3,
        seed=17,
        num_layers=2,
    )

    assert selected == repeated
    assert len(selected["w2c_uids"]) == 18
    assert len(selected["c2c_uids"]) == 9
    assert len(set(selected["w2c_uids"] + selected["c2c_uids"])) == 27
    for dataset in ("gqa", "chartqa", "textvqa"):
        assert selected["counts_by_dataset"][dataset] == {"W2C": 6, "C2C": 3}
        chosen = [
            row for row in boundary_rows if row["uid"] in selected["w2c_uids"]
            and row["dataset"] == dataset
        ]
        assert {row["valid_nonfull_actions"][0] for row in chosen} == set(actions)


def test_mandatory_boundary_metrics_include_timing_and_singleton_recall() -> None:
    rows = [
        {
            "uid": "gqa:a",
            "dataset": "gqa",
            "boundary_layer": 2,
            "valid_nonfull_actions": ["READ_ONLY"],
            "singleton": True,
            "predicted_boundary_action": "READ_ONLY",
            "actions": ["FULL", "FULL", "READ_ONLY", "FULL"],
            "correct": True,
        },
        {
            "uid": "chartqa:b",
            "dataset": "chartqa",
            "boundary_layer": 1,
            "valid_nonfull_actions": ["WRITE_ONLY"],
            "singleton": True,
            "predicted_boundary_action": "FULL",
            "actions": ["FULL", "FULL", "FULL", "FULL"],
            "correct": False,
        },
        {
            "uid": "textvqa:c",
            "dataset": "textvqa",
            "boundary_layer": 3,
            "valid_nonfull_actions": ["IGNORE", "WRITE_ONLY"],
            "singleton": False,
            "predicted_boundary_action": "IGNORE",
            "actions": ["FULL", "FULL", "IGNORE", "FULL"],
            "correct": False,
        },
    ]

    metrics = mandatory_boundary_metrics(rows, num_layers=4)

    assert metrics["records"] == 3
    assert metrics["valid_action_at_1"] == pytest.approx(2 / 3)
    assert metrics["nonfull_recall"] == pytest.approx(2 / 3)
    assert metrics["singleton"]["valid_action_at_1"] == pytest.approx(0.5)
    assert metrics["singleton_action_recall"]["READ_ONLY"] == pytest.approx(1.0)
    assert metrics["singleton_action_recall"]["WRITE_ONLY"] == pytest.approx(0.0)
    assert metrics["free_rollout"]["left_all_full_fraction"] == pytest.approx(2 / 3)
    assert metrics["free_rollout"]["exact_boundary_fraction"] == pytest.approx(1 / 3)
    assert metrics["free_rollout"]["within_2_fraction"] == pytest.approx(2 / 3)
    assert metrics["free_rollout"]["early_fraction"] == pytest.approx(1 / 3)
    assert metrics["free_rollout"]["late_or_no_deviation_fraction"] == pytest.approx(1 / 3)


def test_mandatory_boundary_pilot_gate_is_prospective_and_action_aware() -> None:
    metrics = {
        "boundary": {
            "valid_action_at_1": 0.96,
            "nonfull_recall": 0.97,
            "singleton_action_records": {"IGNORE": 8, "READ_ONLY": 8, "WRITE_ONLY": 8},
            "singleton_action_recall": {"IGNORE": 0.875, "READ_ONLY": 1.0, "WRITE_ONLY": 0.75},
            "free_rollout": {"left_all_full_fraction": 0.95},
        },
        "execution": {"w2c_rescue_rate": 0.30, "c2c_preservation_rate": 0.95},
    }
    gates = {
        "boundary_valid_action_at_1": 0.95,
        "boundary_nonfull_recall": 0.95,
        "singleton_action_recall": 0.80,
        "singleton_min_records": 5,
        "free_rollout_leave_full": 0.90,
        "w2c_rescue_rate": 0.25,
        "c2c_preservation_rate": 0.90,
    }

    result = mandatory_boundary_pilot_gate(metrics, gates)

    assert result["passed"] is False
    assert result["checks"]["singleton_WRITE_ONLY_recall"] is False
    metrics["boundary"]["singleton_action_recall"]["WRITE_ONLY"] = 0.875
    assert mandatory_boundary_pilot_gate(metrics, gates)["passed"] is True


def test_boundary_pilot_parent_contract_rejects_a_second_scientific_change() -> None:
    from experiments.train_four_action_boundary_pilot import verify_parent_contract

    parent = {
        "base_model": {"path": "model", "revision": "rev", "frozen": True},
        "executor": {"implementation": "unified", "contract_sha256": "contract"},
        "data": {
            "manifest": "manifest", "manifest_sha256": "manifest-sha",
            "source_manifest": "source", "source_manifest_sha256": "source-sha",
            "supervision": "set-valued", "route_sampling": "deterministic",
        },
        "router": {"architecture": "same", "dropout": 0.1},
        "training": {
            "optimizer": "AdamW", "learning_rate": 5e-4, "weight_decay": 0.01,
            "scheduler": "cosine", "warmup_steps": 10,
            "gradient_clip_norm": 1.0, "precision": "bfloat16",
            "deterministic_algorithms": True,
        },
    }
    pilot = json.loads(json.dumps(parent))
    pilot["data"]["c2c_route_sampling"] = "deterministic"

    verify_parent_contract(pilot, parent)
    pilot["router"]["dropout"] = 0.0

    with pytest.raises(RuntimeError, match="matched parent field"):
        verify_parent_contract(pilot, parent)


def test_boundary_pilot_resume_verifies_checksum_and_preserves_passed_gate(
    tmp_path: Path,
) -> None:
    from experiments.train_four_action_boundary_pilot import load_resume_payload

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config_sha256": "config",
            "epoch": 5,
            "global_step": 30,
            "metrics": {"validation": {"gate": {"passed": True}}},
            "rng_states": [
                {
                    "python": random.getstate(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.get_rng_state(),
                }
            ],
        },
        checkpoint,
    )
    epoch_dir = tmp_path / "epoch_05"
    epoch_dir.mkdir()
    metadata = epoch_dir / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    payload = load_resume_payload(metadata, config_sha256="config", world_size=1)

    assert payload["metrics"]["validation"]["gate"]["passed"] is True
    with checkpoint.open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_resume_payload(metadata, config_sha256="config", world_size=1)


def test_boundary_pilot_requires_slurm_and_canonical_output(tmp_path: Path) -> None:
    from experiments.train_four_action_boundary_pilot import (
        acquire_run_lock,
        latest_resume_metadata,
        require_slurm_environment,
        release_run_lock,
        verify_output_contract,
    )

    with pytest.raises(RuntimeError, match="Slurm allocation"):
        require_slurm_environment({})
    require_slurm_environment({"SLURM_JOB_ID": "17"})

    configured = tmp_path / "pilot"
    verify_output_contract(configured, configured)
    with pytest.raises(RuntimeError, match="canonical output"):
        verify_output_contract(tmp_path / "other", configured)

    first = acquire_run_lock(configured, resume=False)
    with pytest.raises(RuntimeError, match="already active"):
        acquire_run_lock(configured, resume=True)
    release_run_lock(first)
    resumed = acquire_run_lock(configured, resume=True)
    release_run_lock(resumed)
    assert latest_resume_metadata(configured) is None


def test_boundary_artifact_freeze_is_idempotent_and_exclusive(tmp_path: Path) -> None:
    from experiments.prepare_four_action_collapse import write_frozen

    artifact = tmp_path / "boundary.jsonl"
    write_frozen(artifact, "first\n")
    write_frozen(artifact, "first\n")
    assert artifact.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(FileExistsError, match="different artifact"):
        write_frozen(artifact, "second\n")

    locked = tmp_path / "pilot.json"
    locked.with_suffix(".json.lock").touch()
    with pytest.raises(FileExistsError, match="lock already exists"):
        write_frozen(locked, "{}\n")
