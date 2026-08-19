from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import torch


def deterministic_rank(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def choose_one_record_per_image(
    rows: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    ranked = sorted(
        rows,
        key=lambda row: (
            deterministic_rank(str(row["id"]), seed),
            str(row["id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        image_id = str(row["image_id"])
        if image_id in seen:
            continue
        selected.append(row)
        seen.add(image_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique-image records; {count} required")
    return selected


def reserve_multi_question_groups(
    rows: Iterable[dict[str, Any]],
    excluded_image_ids: set[str],
    group_count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in excluded_image_ids:
            groups[image_id].append(row)
    eligible = [
        (image_id, sorted(group, key=lambda row: str(row["id"])))
        for image_id, group in groups.items()
        if len(group) >= 2
    ]
    eligible.sort(key=lambda item: (deterministic_rank(item[0], seed), item[0]))
    if len(eligible) < group_count:
        raise ValueError(f"Only {len(eligible)} multi-question groups; {group_count} required")
    return dict(eligible[:group_count])


def image_group_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["image_id"]) for row in rows)
    histogram = Counter(counts.values())
    values = sorted(counts.values())
    return {
        "record_count": sum(values),
        "image_count": len(values),
        "multi_question_image_count": sum(value >= 2 for value in values),
        "records_in_multi_question_images": sum(value for value in values if value >= 2),
        "questions_per_image_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
    }


def search_cells(layers: Sequence[int], actions: Sequence[str]) -> list[dict[str, Any]]:
    if not layers or not actions:
        raise ValueError("layers and actions must both be nonempty")
    if len(set(layers)) != len(layers) or len(set(actions)) != len(actions):
        raise ValueError("layers and actions must not contain duplicates")
    return [{"layer": int(layer), "action": action} for layer in layers for action in actions]


def normal_mde(sd: float, clusters: int, alpha: float, power: float) -> float:
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("sd must be finite and positive")
    if clusters < 2:
        raise ValueError("clusters must be at least two")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie in (0, 1)")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * sd / math.sqrt(clusters)


def right_pad_prompt_inputs(
    inputs: dict[str, Any], target_length: int, pad_token_id: int
) -> dict[str, Any]:
    current = int(inputs["input_ids"].shape[1])
    if target_length < current:
        raise ValueError("target_length cannot shorten the prompt")
    amount = target_length - current
    result = dict(inputs)
    if amount:
        result["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, amount), value=int(pad_token_id)
        )
        result["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, amount), value=0
        )
    return result
