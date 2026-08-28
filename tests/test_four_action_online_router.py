from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pytest
import torch

from four_action_online_router.model import OnlineFourActionRouter
from four_action_online_router.data import choose_smoke_indices, manifest_route_tensor
from four_action_online_router.metrics import (
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
