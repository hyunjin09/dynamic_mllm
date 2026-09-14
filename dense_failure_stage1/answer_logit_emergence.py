"""Corrected answer-position helpers for dense logit-emergence analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch

from dense_failure_stage1.logit_emergence import first_persistent_layer


DATASETS = ("gqa", "chartqa", "textvqa")


def answer_prediction_position(attention_mask: torch.Tensor) -> int:
    """Return the final attended prompt token for a batch of exactly one."""

    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError("answer-position extraction requires batch size one")
    attended = attention_mask[0].to(torch.bool).detach().cpu().tolist()
    if not any(attended):
        raise ValueError("attention mask is empty")
    count = sum(attended)
    if attended != [True] * count + [False] * (len(attended) - count):
        raise ValueError("attention mask must be contiguous from the first token")
    return count - 1


def append_teacher_prefix(
    inputs: Mapping[str, object], prefix_ids: Sequence[int]
) -> dict[str, object]:
    """Append text tokens to the multimodal prompt without changing image inputs."""

    result = dict(inputs)
    prefix = [int(value) for value in prefix_ids]
    if not prefix:
        return result
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        raise TypeError("input_ids and attention_mask must be tensors")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("teacher forcing requires batch size one")
    token_tensor = torch.tensor(
        [prefix], device=input_ids.device, dtype=input_ids.dtype
    )
    result["input_ids"] = torch.cat((input_ids, token_tensor), dim=1)
    result["attention_mask"] = torch.cat(
        (
            attention_mask,
            torch.ones(
                (1, len(prefix)),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            ),
        ),
        dim=1,
    )
    mm_types = inputs.get("mm_token_type_ids")
    if isinstance(mm_types, torch.Tensor):
        result["mm_token_type_ids"] = torch.cat(
            (
                mm_types,
                torch.zeros(
                    (1, len(prefix)),
                    device=mm_types.device,
                    dtype=mm_types.dtype,
                ),
            ),
            dim=1,
        )
    return result


def stable_reference_order(answers: Sequence[str]) -> list[str]:
    """Order references by frequency, breaking ties by first occurrence."""

    values = [str(answer) for answer in answers if str(answer)]
    counts = Counter(values)
    first = {value: values.index(value) for value in counts}
    return sorted(counts, key=lambda value: (-counts[value], first[value]))


def _terminated(ids: Sequence[int], im_end_token_id: int) -> tuple[int, ...]:
    values = [int(value) for value in ids]
    try:
        end = values.index(int(im_end_token_id))
    except ValueError:
        values.append(int(im_end_token_id))
    else:
        values = values[: end + 1]
    return tuple(values)


@dataclass(frozen=True)
class DivergenceSpec:
    usable: bool
    reason: str | None
    gt_source: str
    gt_sequence_ids: tuple[int, ...]
    generated_sequence_ids: tuple[int, ...]
    common_prefix_ids: tuple[int, ...]
    divergence_index: int | None
    gt_token_id: int | None
    generated_token_id: int | None


def build_divergence_spec(
    *,
    canonical_gt_ids: Sequence[int],
    generated_ids: Sequence[int],
    im_end_token_id: int,
    fallback_gt_sequences: Sequence[tuple[str, Sequence[int]]] = (),
) -> DivergenceSpec:
    """Freeze the first answer-token divergence without consulting logits."""

    generated = _terminated(generated_ids, im_end_token_id)
    candidates = [("canonical", canonical_gt_ids), *fallback_gt_sequences]
    seen: set[tuple[int, ...]] = set()
    for source, raw_gt in candidates:
        gt = _terminated(raw_gt, im_end_token_id)
        if gt in seen:
            continue
        seen.add(gt)
        if gt == generated:
            continue
        divergence = next(
            index
            for index, (gt_token, generated_token) in enumerate(zip(gt, generated))
            if gt_token != generated_token
        )
        return DivergenceSpec(
            usable=True,
            reason=None,
            gt_source=str(source),
            gt_sequence_ids=gt,
            generated_sequence_ids=generated,
            common_prefix_ids=gt[:divergence],
            divergence_index=divergence,
            gt_token_id=gt[divergence],
            generated_token_id=generated[divergence],
        )
    return DivergenceSpec(
        usable=False,
        reason="no_gt_sequence_diverges_from_generation",
        gt_source="none",
        gt_sequence_ids=(),
        generated_sequence_ids=generated,
        common_prefix_ids=(),
        divergence_index=None,
        gt_token_id=None,
        generated_token_id=None,
    )


def cached_replay_matches(
    prefix_ids: Sequence[int], next_token_id: int, replay_ids: Sequence[int]
) -> bool:
    """Check the processed greedy replay through one exact continuation token."""

    expected = tuple(int(value) for value in prefix_ids) + (int(next_token_id),)
    return tuple(int(value) for value in replay_ids) == expected


def token_metrics(logits: torch.Tensor, *, token_id: int) -> dict[str, torch.Tensor]:
    """Return raw logits, minimum ranks, top-1, and strongest non-target logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [layers,vocabulary]")
    target = int(token_id)
    if target < 0 or target >= logits.shape[1]:
        raise ValueError(f"target token ID is outside vocabulary: {target}")
    token_logits = logits[:, target]
    token_ranks = (logits > token_logits[:, None]).sum(dim=1) + 1
    top_values, top_ids = logits.topk(k=2, dim=1)
    competitor = torch.where(
        top_ids[:, 0] == target, top_values[:, 1], top_values[:, 0]
    )
    return {
        "token_logits": token_logits,
        "token_ranks": token_ranks,
        "top1_token_ids": top_ids[:, 0],
        "top1_logits": top_values[:, 0],
        "strongest_non_target_logits": competitor,
    }


def summarize_layerwise_trajectories(
    *,
    gt_logits: np.ndarray,
    other_logits: np.ndarray,
    gt_ranks: np.ndarray,
    other_ranks: np.ndarray,
    top1_logits: np.ndarray,
    margins: np.ndarray,
    gt_is_top1: np.ndarray,
    other_is_top1: np.ndarray,
) -> list[dict[str, float | int]]:
    """Summarize aligned sample-by-layer trajectory matrices."""

    arrays = {
        "gt_logits": np.asarray(gt_logits),
        "other_logits": np.asarray(other_logits),
        "gt_ranks": np.asarray(gt_ranks),
        "other_ranks": np.asarray(other_ranks),
        "top1_logits": np.asarray(top1_logits),
        "margins": np.asarray(margins),
        "gt_is_top1": np.asarray(gt_is_top1),
        "other_is_top1": np.asarray(other_is_top1),
    }
    shape = arrays["gt_logits"].shape
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError("trajectory arrays must be nonempty [samples,layers]")
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"trajectory shape differs for {name}: {array.shape}")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"trajectory values are non-finite for {name}")

    def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"mean_{prefix}": float(values.mean()),
            f"q25_{prefix}": float(np.quantile(values, 0.25)),
            f"median_{prefix}": float(np.median(values)),
            f"q75_{prefix}": float(np.quantile(values, 0.75)),
        }

    rows = []
    for layer in range(shape[1]):
        row: dict[str, float | int] = {"layer": layer, "n": shape[0]}
        for key in (
            "gt_logits",
            "other_logits",
            "gt_ranks",
            "other_ranks",
            "top1_logits",
            "margins",
        ):
            singular = key[:-1] if key.endswith("s") else key
            row.update(stats(arrays[key][:, layer], singular))
        row["gt_top1_fraction"] = float(arrays["gt_is_top1"][:, layer].mean())
        row["other_top1_fraction"] = float(
            arrays["other_is_top1"][:, layer].mean()
        )
        row["positive_margin_fraction"] = float(
            (arrays["margins"][:, layer] > 0).mean()
        )
        row["negative_margin_fraction"] = float(
            (arrays["margins"][:, layer] < 0).mean()
        )
        row["zero_margin_fraction"] = float(
            (arrays["margins"][:, layer] == 0).mean()
        )
        rows.append(row)
    return rows


class LayerPositionCollector:
    """Collect one post-layer hidden vector at an exact sequence position."""

    def __init__(self, layers: Sequence[torch.nn.Module], *, position: int):
        self.layers = list(layers)
        self.position = int(position)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._states: list[torch.Tensor | None] = [None] * len(self.layers)

    def _hook(self, layer_index: int):
        def capture(_module, _inputs, output):
            if self._states[layer_index] is not None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("decoder layer output must have shape [1,sequence,hidden]")
            if self.position < 0 or self.position >= hidden.shape[1]:
                raise ValueError(
                    f"requested position {self.position} is outside layer output {hidden.shape}"
                )
            self._states[layer_index] = hidden[0, self.position].detach()
            return None

        return capture

    def __enter__(self):
        if self._handles:
            raise RuntimeError("layer-position collector is already active")
        self._handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def stacked(self) -> torch.Tensor:
        missing = [index for index, state in enumerate(self._states) if state is None]
        if missing:
            raise RuntimeError(f"decoder layers were not observed: {missing}")
        return torch.stack([state for state in self._states if state is not None])


class GenerationCallCollector:
    """Collect the final token state from one exact cached-generation call."""

    def __init__(
        self, layers: Sequence[torch.nn.Module], *, target_call_index: int
    ) -> None:
        if int(target_call_index) < 0:
            raise ValueError("target generation call index must be nonnegative")
        self.layers = list(layers)
        self.target_call_index = int(target_call_index)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._call_counts = [0] * len(self.layers)
        self._states: list[torch.Tensor | None] = [None] * len(self.layers)

    def _hook(self, layer_index: int):
        def capture(_module, _inputs, output):
            call_index = self._call_counts[layer_index]
            self._call_counts[layer_index] += 1
            if call_index != self.target_call_index:
                return None
            if self._states[layer_index] is not None:
                raise RuntimeError("target generation call was observed more than once")
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("decoder layer output must have shape [1,sequence,hidden]")
            self._states[layer_index] = hidden[0, -1].detach()
            return None

        return capture

    def __enter__(self):
        if self._handles:
            raise RuntimeError("generation-call collector is already active")
        self._handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def stacked(self) -> torch.Tensor:
        missing = [index for index, state in enumerate(self._states) if state is None]
        if missing:
            raise RuntimeError(
                f"target generation call was not observed by layers: {missing}"
            )
        expected_calls = self.target_call_index + 1
        if any(count < expected_calls for count in self._call_counts):
            raise RuntimeError("generation ended before the requested cached step")
        return torch.stack([state for state in self._states if state is not None])


@dataclass(frozen=True)
class LayerPositionReadout:
    position: int
    layer_logits: torch.Tensor
    model_final_logits: torch.Tensor
    final_top1_matches_model: bool
    reconstructed_final_top1_matches_model: bool
    final_max_abs_logit_difference: float


@dataclass(frozen=True)
class GeneratedPrefixReadout:
    layer_logits: torch.Tensor
    generated_ids: tuple[int, ...]
    final_top1_matches_raw_generation_logits: bool
    reconstructed_final_top1_matches_raw_generation_logits: bool
    final_max_abs_logit_difference: float


@torch.inference_mode()
def read_layer_logits_at_position(
    model: torch.nn.Module,
    inputs: Mapping[str, object],
    *,
    position: int,
) -> LayerPositionReadout:
    """Run one dense prefill and read every layer through the exact final head."""

    decoder = model.model.language_model
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    with LayerPositionCollector(decoder.layers, position=position) as collector:
        output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            logits_to_keep=1,
            return_dict=True,
        )
    states = collector.stacked()
    layer_logits = model.lm_head(decoder.norm(states)).float()
    model_final_logits = output.logits[0, -1].float()
    difference = float((layer_logits[-1] - model_final_logits).abs().max().item())
    reconstructed_match = int(layer_logits[-1].argmax().item()) == int(
        model_final_logits.argmax().item()
    )
    layer_logits[-1].copy_(model_final_logits)
    return LayerPositionReadout(
        position=int(position),
        layer_logits=layer_logits.detach().cpu(),
        model_final_logits=model_final_logits.detach().cpu(),
        final_top1_matches_model=True,
        reconstructed_final_top1_matches_model=reconstructed_match,
        final_max_abs_logit_difference=difference,
    )


@torch.inference_mode()
def read_layer_logits_after_generated_prefix(
    model: torch.nn.Module,
    inputs: Mapping[str, object],
    *,
    prefix_ids: Sequence[int],
) -> GeneratedPrefixReadout:
    """Replay exact cached greedy decoding and read after the common prefix."""

    prefix = tuple(int(value) for value in prefix_ids)
    if not prefix:
        raise ValueError("cached prefix readout requires at least one token")
    decoder = model.model.language_model
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    with GenerationCallCollector(
        decoder.layers, target_call_index=len(prefix)
    ) as collector:
        output = model.generate(
            **inputs,
            max_new_tokens=len(prefix) + 1,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
        )
    states = collector.stacked()
    layer_logits = model.lm_head(decoder.norm(states)).float()
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = tuple(
        int(value) for value in output.sequences[0, prompt_length:].detach().cpu()
    )
    raw_generation_logits = output.logits[-1][0].float()
    difference = float(
        (layer_logits[-1] - raw_generation_logits).abs().max().item()
    )
    reconstructed_match = int(layer_logits[-1].argmax().item()) == int(
        raw_generation_logits.argmax().item()
    )
    layer_logits[-1].copy_(raw_generation_logits)
    return GeneratedPrefixReadout(
        layer_logits=layer_logits.detach().cpu(),
        generated_ids=generated,
        final_top1_matches_raw_generation_logits=True,
        reconstructed_final_top1_matches_raw_generation_logits=(
            reconstructed_match
        ),
        final_max_abs_logit_difference=difference,
    )


def classify_wrong_trajectory(
    margins: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.0,
    consecutive: int = 3,
    early_cutoff: int = 4,
) -> str:
    """Apply the prospectively frozen wrong-trajectory taxonomy."""

    wrong_layer = first_persistent_layer(
        margins,
        threshold=-float(threshold),
        direction="less",
        consecutive=consecutive,
    )
    gt_layer = first_persistent_layer(
        margins,
        threshold=float(threshold),
        direction="greater",
        consecutive=consecutive,
    )
    if wrong_layer is not None and gt_layer is not None and gt_layer < wrong_layer:
        return "answer_erosion"
    if wrong_layer is None:
        return "ambiguous"
    if wrong_layer <= int(early_cutoff):
        return "early_wrong"
    return "progressive_wrong"


def _stable_key(uid: str, seed: int) -> tuple[str, str]:
    return sha256(f"{seed}\0{uid}".encode("utf-8")).hexdigest(), uid


def select_position_sanity(
    rows: Sequence[Mapping],
    *,
    seed: int,
    per_cell: int,
    teacher_forced_per_wrong_dataset: int = 0,
) -> list[dict]:
    """Select outcome strata and guaranteed shared-prefix wrong comparisons."""

    required_teacher_forced = int(teacher_forced_per_wrong_dataset)
    if required_teacher_forced < 0 or required_teacher_forced > int(per_cell):
        raise ValueError("invalid teacher-forced sanity quota")

    chosen: list[dict] = []
    used_groups: set[str] = set()
    for dataset in DATASETS:
        for correct in (False, True):
            candidates = sorted(
                (
                    dict(row)
                    for row in rows
                    if str(row["dataset"]) == dataset
                    and bool(row["current_dense_correct"]) is correct
                ),
                key=lambda row: _stable_key(str(row["uid"]), seed),
            )
            cell = []
            if not correct and required_teacher_forced:
                prefix_candidates = [
                    row
                    for row in candidates
                    if bool(row.get("comparison_usable", True))
                    and bool(row.get("comparison_common_prefix_ids"))
                ]
                for row in prefix_candidates:
                    group = str(row["image_group_id"])
                    if group in used_groups:
                        continue
                    cell.append(row)
                    used_groups.add(group)
                    if len(cell) == required_teacher_forced:
                        break
                if len(cell) != required_teacher_forced:
                    raise ValueError(
                        f"insufficient shared-prefix sanity records for {dataset}: "
                        f"{len(cell)} < {required_teacher_forced}"
                    )
            for row in candidates:
                if any(existing["uid"] == row["uid"] for existing in cell):
                    continue
                group = str(row["image_group_id"])
                if group in used_groups:
                    continue
                cell.append(row)
                used_groups.add(group)
                if len(cell) == per_cell:
                    break
            if len(cell) != per_cell:
                raise ValueError(
                    f"insufficient sanity records for {dataset}/{correct}: "
                    f"{len(cell)} < {per_cell}"
                )
            chosen.extend(cell)
    return chosen
