"""Helpers for the leave-one-dataset-out dense-failure diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


INPUT_FEATURE_NAMES = ("text_final", "text_mean", "visual_mean")
OOD_RUN_TARGETS = {
    "leave_textvqa_out": "textvqa",
    "leave_chartqa_out": "chartqa",
    "leave_gqa_out": "gqa",
}


class _CaptureComplete(BaseException):
    """Private non-error control flow used to stop before decoder layer 0."""

    def __init__(self, owner: object):
        super().__init__("native pre-decoder representation captured")
        self.owner = owner


@dataclass(frozen=True)
class PreDecoderCapture:
    feature: torch.Tensor
    sequence_length: int
    hidden_size: int
    capture_count: int
    decoder_forward_count: int


class PreDecoderInputCollector:
    """Capture the native merged prompt at layer-0 input and abort immediately.

    This boundary is after token embedding, visual encoding, and multimodal token
    insertion, but before any language-decoder layer executes.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        visual_positions: Sequence[int],
        user_text_positions: Sequence[int],
        final_user_token_position: int,
    ):
        if not layers:
            raise ValueError("at least one decoder layer is required")
        self.layers = list(layers)
        self.visual_positions = tuple(int(value) for value in visual_positions)
        self.user_text_positions = tuple(int(value) for value in user_text_positions)
        self.final_user_token_position = int(final_user_token_position)
        if not self.visual_positions or not self.user_text_positions:
            raise ValueError("visual and user-text positions must be nonempty")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._feature: torch.Tensor | None = None
        self._sequence_length: int | None = None
        self.capture_count = 0
        self.decoder_forward_count = 0

    def _pre_hook(self, _module, args, kwargs):
        if self.capture_count:
            raise RuntimeError("decoder layer-0 pre-hook fired more than once")
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("layer-0 pre-hook did not receive rank-3 hidden states")
        if hidden.shape[0] != 1:
            raise RuntimeError("pre-decoder extraction requires batch size one")
        positions = (
            *self.visual_positions,
            *self.user_text_positions,
            self.final_user_token_position,
        )
        if min(positions) < 0 or max(positions) >= hidden.shape[1]:
            raise RuntimeError("frozen token positions lie outside pre-decoder sequence")
        final = hidden[0, self.final_user_token_position]
        text = hidden[0, list(self.user_text_positions)].mean(dim=0)
        visual = hidden[0, list(self.visual_positions)].mean(dim=0)
        feature = torch.cat((final, text, visual), dim=-1).detach().cpu()
        if feature.ndim != 1 or not torch.isfinite(feature.float()).all():
            raise RuntimeError("pre-decoder pooled feature is invalid")
        self._feature = feature
        self._sequence_length = int(hidden.shape[1])
        self.capture_count += 1
        raise _CaptureComplete(self)

    def _forward_hook(self, _module, _args, _output):
        self.decoder_forward_count += 1

    def __enter__(self):
        if self._handles:
            raise RuntimeError("pre-decoder collector is already active")
        self._handles.append(
            self.layers[0].register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        )
        self._handles.extend(
            layer.register_forward_hook(self._forward_hook) for layer in self.layers
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def consume(self, function, /, *args, **kwargs) -> PreDecoderCapture:
        """Run ``function`` and accept only this collector's private sentinel."""

        with self:
            try:
                function(*args, **kwargs)
            except _CaptureComplete as exc:
                if exc.owner is not self:
                    raise
            else:
                raise RuntimeError("model completed without reaching decoder layer 0")
        if self._feature is None or self._sequence_length is None:
            raise RuntimeError("pre-decoder hook stopped without a captured feature")
        if self.capture_count != 1 or self.decoder_forward_count != 0:
            raise RuntimeError(
                "capture boundary violated: expected one pre-hook and zero decoder forwards"
            )
        return PreDecoderCapture(
            feature=self._feature,
            sequence_length=self._sequence_length,
            hidden_size=int(self._feature.numel() // len(INPUT_FEATURE_NAMES)),
            capture_count=self.capture_count,
            decoder_forward_count=self.decoder_forward_count,
        )

    @property
    def active_handles(self) -> int:
        return len(self._handles)


def audit_cross_dataset_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that identical image content never crosses dataset identity."""

    datasets_by_group: dict[str, set[str]] = defaultdict(set)
    records_by_group: Counter[str] = Counter()
    for row in rows:
        group = str(row["image_group_id"])
        datasets_by_group[group].add(str(row["dataset"]))
        records_by_group[group] += 1
    crossing = {
        group: sorted(datasets)
        for group, datasets in datasets_by_group.items()
        if len(datasets) > 1
    }
    return {
        "records": len(rows),
        "image_groups": len(datasets_by_group),
        "cross_dataset_image_groups": len(crossing),
        "cross_dataset_records": sum(records_by_group[group] for group in crossing),
        "examples": dict(list(sorted(crossing.items()))[:20]),
        "passed": not crossing,
    }


def validate_ood_membership(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_datasets: Sequence[str],
    target_dataset: str,
) -> dict[str, list[int]]:
    """Return frozen source-train/source-val/full-target row indices."""

    source = set(source_datasets)
    target = str(target_dataset)
    if target in source or len(source) != 2:
        raise ValueError("OOD run requires two source datasets and one distinct target")
    train = [
        index
        for index, row in enumerate(rows)
        if str(row["dataset"]) in source and str(row["split"]) == "train"
    ]
    validation = [
        index
        for index, row in enumerate(rows)
        if str(row["dataset"]) in source and str(row["split"]) == "val"
    ]
    target_all = [
        index for index, row in enumerate(rows) if str(row["dataset"]) == target
    ]
    target_test = [
        index
        for index, row in enumerate(rows)
        if str(row["dataset"]) == target and str(row["split"]) == "test"
    ]
    if not train or not validation or not target_all or not target_test:
        raise ValueError("OOD membership produced an empty required partition")
    memberships = {
        "source_train": train,
        "source_validation": validation,
        "target_all": target_all,
        "target_phase48_test": target_test,
    }
    for name, indices in memberships.items():
        labels = {int(bool(rows[index]["current_dense_wrong"])) for index in indices}
        if labels != {0, 1}:
            raise ValueError(f"{name} does not contain both current-label classes")
    used = set(train) | set(validation)
    if used & set(target_all):
        raise ValueError("target dataset leaked into source train/validation")
    return memberships


def choose_representative_layer(
    validation_rows: Sequence[Mapping[str, Any]],
) -> int:
    """Select by source-validation AUROC only, with a lower-layer tie break."""

    by_layer = {int(row["layer"]): float(row["validation_auroc"]) for row in validation_rows}
    if set(by_layer) != set(range(28)) or len(by_layer) != len(validation_rows):
        raise ValueError("representative-layer selection requires exactly layers 0-27")
    if not all(np.isfinite(value) for value in by_layer.values()):
        raise ValueError("source-validation AUROCs must be finite")
    return min(range(28), key=lambda layer: (-by_layer[layer], layer))
