"""Convert strict local counterfactual edges into trajectory BT pairs."""

from __future__ import annotations

import hashlib
from typing import Any


DATASET_VERSION = "correctness_first_preference_gt_v31"
PAIR_POLICY = "hamming1_continuation_aware_trajectory_bt_v1"


def _changed_layers(left: str, right: str) -> list[int]:
    if len(left) != len(right):
        return []
    return [index for index, (a, b) in enumerate(zip(left, right)) if a != b]


def _preference_pair_id(source_pair_id: str) -> str:
    digest = hashlib.sha256(f"h1bt:{source_pair_id}".encode("utf-8")).hexdigest()[:24]
    return f"h1bt:{digest}"


def validate_hamming1_route_preference_pair(pair: dict[str, Any], *, num_layers: int) -> None:
    chosen = str(pair["chosen_mask_key"])
    rejected = str(pair["rejected_mask_key"])
    if len(chosen) != num_layers or len(rejected) != num_layers:
        raise ValueError("trajectory masks do not match num_layers")
    changed = _changed_layers(chosen, rejected)
    layer = int(pair["changed_layer_index"])
    if changed != [layer]:
        raise ValueError(f"pair is not an exact Hamming-1 intervention at layer {layer}: {changed}")
    if chosen[:layer] != rejected[:layer] or chosen[layer + 1 :] != rejected[layer + 1 :]:
        raise ValueError("Hamming-1 pair must share both pre-action prefix and future suffix")
    if not bool(pair["chosen_correct"]) or bool(pair["rejected_correct"]):
        raise ValueError("trajectory preference must orient correct > incorrect")
    if int(pair["chosen_budget"]) != chosen.count("1"):
        raise ValueError("chosen budget does not match mask")
    if int(pair["rejected_budget"]) != rejected.count("1"):
        raise ValueError("rejected budget does not match mask")
    if int(pair["budget_delta_rejected_minus_chosen"]) != rejected.count("1") - chosen.count("1"):
        raise ValueError("budget delta does not match masks")
    subtype = str(pair["pair_subtype"])
    if subtype == "h1_off_win" and not (chosen[layer] == "0" and rejected[layer] == "1"):
        raise ValueError("h1_off_win must prefer OFF at the changed layer")
    if subtype == "h1_on_win" and not (chosen[layer] == "1" and rejected[layer] == "0"):
        raise ValueError("h1_on_win must prefer ON at the changed layer")
    if subtype not in {"h1_off_win", "h1_on_win"}:
        raise ValueError(f"unknown Hamming-1 pair subtype: {subtype!r}")


def hamming1_route_preference_pair(row: dict[str, Any], *, num_layers: int = 28) -> dict[str, Any]:
    """Orient one strict decisive edge as a complete-route BT preference.

    The routes share their prefix and future suffix. The preferred route is the
    benchmark-correct route, so route-level BT keeps downstream trajectory
    effects while assigning the only explicit action difference to one layer.
    """
    off_mask = str(row["off_mask_key"])
    on_mask = str(row["on_mask_key"])
    layer = int(row["layer_index"])
    if _changed_layers(off_mask, on_mask) != [layer]:
        raise ValueError(f"source row is not an exact Hamming-1 edge at layer {layer}")
    if off_mask[layer] != "0" or on_mask[layer] != "1":
        raise ValueError("source edge must be oriented OFF to ON")
    prefix = str(row["common_prefix_mask"])
    if len(prefix) != layer or off_mask[:layer] != prefix or on_mask[:layer] != prefix:
        raise ValueError("source common prefix does not match intervention masks")

    action = str(row["label_action"])
    off_correct = bool(row["off_correct"])
    on_correct = bool(row["on_correct"])
    if action == "off":
        if not off_correct or on_correct:
            raise ValueError("label_action=off does not match edge correctness")
        chosen_side, rejected_side = "off", "on"
    elif action == "on":
        if not on_correct or off_correct:
            raise ValueError("label_action=on does not match edge correctness")
        chosen_side, rejected_side = "on", "off"
    else:
        raise ValueError(f"unsupported label_action: {action!r}")

    def side(name: str, field: str) -> Any:
        return row[f"{name}_{field}"]

    chosen_mask = str(side(chosen_side, "mask_key"))
    rejected_mask = str(side(rejected_side, "mask_key"))
    pair = {
        "uid": str(row["uid"]),
        "sample_id": str(row["sample_id"]),
        "benchmark": str(row["benchmark"]),
        "split": str(row["split"]),
        "source_bucket": str(row["source_bucket"]),
        "dataset_version": DATASET_VERSION,
        "model_runtime_id": str(row["model_runtime_id"]),
        "num_layers": int(num_layers),
        "pair_id": _preference_pair_id(str(row["pair_id"])),
        "source_local_pair_id": str(row["pair_id"]),
        "pair_policy": PAIR_POLICY,
        "pair_type": "correctness",
        "pair_subtype": f"h1_{row['outcome']}",
        "preference_rule": "correct_over_incorrect_exact_hamming1_fixed_suffix",
        "objective_level": 1,
        "recommended_weight": 1.0,
        "changed_layer_index": layer,
        "changed_layer_one_based": layer + 1,
        "common_prefix_mask": prefix,
        "common_suffix_mask": off_mask[layer + 1 :],
        "hamming_distance": 1,
        "is_local_pair": True,
        "chosen_mask_key": chosen_mask,
        "rejected_mask_key": rejected_mask,
        "chosen_budget": chosen_mask.count("1"),
        "rejected_budget": rejected_mask.count("1"),
        "budget_delta_rejected_minus_chosen": rejected_mask.count("1") - chosen_mask.count("1"),
        "chosen_correct": True,
        "rejected_correct": False,
        "chosen_score": float(side(chosen_side, "score")),
        "rejected_score": float(side(rejected_side, "score")),
        "chosen_route_id": str(side(chosen_side, "route_id")),
        "rejected_route_id": str(side(rejected_side, "route_id")),
        "chosen_families": list(side(chosen_side, "families")),
        "rejected_families": list(side(rejected_side, "families")),
        "chosen_action_at_changed_layer": chosen_mask[layer],
        "rejected_action_at_changed_layer": rejected_mask[layer],
        "source_weight": float(row.get("recommended_primary_weight", 1.0)),
    }
    validate_hamming1_route_preference_pair(pair, num_layers=num_layers)
    return pair
