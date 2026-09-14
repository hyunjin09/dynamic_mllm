from __future__ import annotations

import pytest
import torch

from dense_failure_stage1.ood_signal import (
    PreDecoderInputCollector,
    audit_cross_dataset_groups,
    choose_representative_layer,
    validate_ood_membership,
)


class _ToyDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **_kwargs):
        return self.linear(hidden_states)


class _ToyModel(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_predecoder_collector_captures_native_input_and_stops_all_layers() -> None:
    layers = [_ToyDecoder(4) for _ in range(3)]
    model = _ToyModel(layers)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    collector = PreDecoderInputCollector(
        layers,
        visual_positions=[0, 1],
        user_text_positions=[2, 3, 4],
        final_user_token_position=4,
    )

    capture = collector.consume(model, hidden)

    expected = torch.cat((hidden[0, 4], hidden[0, 2:5].mean(0), hidden[0, :2].mean(0)))
    assert torch.equal(capture.feature, expected)
    assert capture.sequence_length == 6
    assert capture.hidden_size == 4
    assert capture.capture_count == 1
    assert capture.decoder_forward_count == 0
    assert collector.active_handles == 0


def test_predecoder_collector_fails_if_boundary_is_not_reached() -> None:
    layer = _ToyDecoder(2)
    collector = PreDecoderInputCollector(
        [layer], visual_positions=[0], user_text_positions=[1], final_user_token_position=1
    )

    with pytest.raises(RuntimeError, match="without reaching"):
        collector.consume(lambda: None)
    assert collector.active_handles == 0


def _rows():
    rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        for split in ("train", "val", "test"):
            for wrong in (False, True):
                rows.append(
                    {
                        "uid": f"{dataset}:{split}:{wrong}",
                        "dataset": dataset,
                        "split": split,
                        "current_dense_wrong": wrong,
                        "image_group_id": f"{dataset}:{split}:{wrong}",
                    }
                )
    return rows


def test_ood_membership_never_uses_target_for_source_decisions() -> None:
    rows = _rows()
    membership = validate_ood_membership(
        rows, source_datasets=["gqa", "chartqa"], target_dataset="textvqa"
    )

    assert all(rows[index]["dataset"] != "textvqa" for index in membership["source_train"])
    assert all(rows[index]["dataset"] != "textvqa" for index in membership["source_validation"])
    assert all(rows[index]["dataset"] == "textvqa" for index in membership["target_all"])
    assert len(membership["target_phase48_test"]) == 2


def test_cross_dataset_group_audit_rejects_identical_content_across_tasks() -> None:
    clean = _rows()
    assert audit_cross_dataset_groups(clean)["passed"] is True
    contaminated = list(clean)
    contaminated[0] = dict(contaminated[0], image_group_id=contaminated[-1]["image_group_id"])
    audit = audit_cross_dataset_groups(contaminated)
    assert audit["passed"] is False
    assert audit["cross_dataset_image_groups"] == 1


def test_representative_layer_uses_source_validation_and_lower_tie_break() -> None:
    rows = [{"layer": layer, "validation_auroc": 0.6 + layer / 1000} for layer in range(28)]
    rows[7]["validation_auroc"] = 0.95
    rows[21]["validation_auroc"] = 0.95

    assert choose_representative_layer(rows) == 7

    with pytest.raises(ValueError, match="exactly layers"):
        choose_representative_layer(rows[:-1])
