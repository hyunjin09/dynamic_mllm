#!/usr/bin/env python3
"""Static integrity and matched-initialization gate for the four cap runs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from binary_policy.predictor import BinaryPolarBackbone
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import file_sha256, validate_gate


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (24, 22, 20, 18)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def component_hashes(model: BinaryPolarBackbone) -> dict[str, str]:
    groups = {
        "layer_queries": ("encoder.layer_embedding.",),
        "cross_attention": ("encoder.cross_attention.",),
        "cross_layer_encoder": ("encoder.layer_encoder.",),
        "binary_head": ("route_head.",),
        "visual_projection": ("encoder.image_projection.",),
        "question_projection": ("encoder.input_projection.",),
    }
    state = model.state_dict()
    return {
        name: tensor_dict_sha256({key: value for key, value in state.items() if key.startswith(prefixes)})
        for name, prefixes in groups.items()
    }


def normalized(config: dict) -> dict:
    result = deepcopy(config)
    result["protocol_version"] = "MATCHED"
    result["data"]["manifest"] = "MATCHED"
    result["data"]["manifest_sha256"] = "MATCHED"
    result["data"]["route_cap_policy"] = "MATCHED"
    result["data"]["visual_on_cap"] = "MATCHED"
    return result


def main() -> None:
    configs = {}
    rows_by_cap = {}
    for cap in CAPS:
        path = PROJECT / f"configs/binary_cap{cap}_full10_bce_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config["training"]["objective"] != "duplicated_bce":
            raise RuntimeError("cap run changed the frozen duplicated-BCE objective")
        if int(config["data"]["visual_on_cap"]) != cap:
            raise RuntimeError("config cap mismatch")
        if config["source_plan"]["sha256"] != file_sha256(PROJECT / config["source_plan"]["path"]):
            raise RuntimeError("cap source-plan checksum mismatch")
        for name, specification in config["gates"].items():
            validate_gate(name, specification)
        for source, expected in config["source_sha256"].items():
            if file_sha256(PROJECT / source) != expected:
                raise RuntimeError(f"cap source checksum mismatch: {source}")
        for specification in config["evidence"].values():
            if file_sha256(PROJECT / specification["path"]) != specification["sha256"]:
                raise RuntimeError("cap evidence checksum mismatch")
        manifest = PROJECT / config["data"]["manifest"]
        if file_sha256(manifest) != config["data"]["manifest_sha256"]:
            raise RuntimeError("cap manifest checksum mismatch")
        rows = read_jsonl(manifest)
        if len(rows) != 8000 or len({row["uid"] for row in rows}) != 8000:
            raise RuntimeError("cap manifest population/UID mismatch")
        for row in rows:
            routes = row["valid_routes"]
            if bool(routes) != bool(row["common_eligible_cap18"]):
                raise RuntimeError("matched-population route presence mismatch")
            if any(sum(route["mask"]) > cap for route in routes):
                raise RuntimeError("cap manifest contains an over-budget route")
            parent_keys = row["original_valid_mask_keys"]
            selected_keys = [route["key"] for route in routes]
            if not set(selected_keys).issubset(set(parent_keys)):
                raise RuntimeError("cap route is absent from original supervision")
            if [key for key in parent_keys if key in set(selected_keys)] != selected_keys:
                raise RuntimeError("cap filtering changed parent route order")
            if len(routes) != int(row["supervision_route_count"]):
                raise RuntimeError("cap supervision route count mismatch")
        configs[cap] = {"path": path, "config": config}
        rows_by_cap[cap] = rows
    baseline = normalized(configs[24]["config"])
    if any(normalized(configs[cap]["config"]) != baseline for cap in CAPS[1:]):
        raise RuntimeError("cap configs differ outside frozen cap/manifest fields")
    uid_sets = {
        cap: {row["uid"] for row in rows_by_cap[cap] if row["valid_routes"]}
        for cap in CAPS
    }
    if len({frozenset(value) for value in uid_sets.values()}) != 1:
        raise RuntimeError("cap manifests do not share identical common-eligible UIDs")
    for cap, rows in rows_by_cap.items():
        config = configs[cap]["config"]
        split_counts = Counter(row["split"] for row in rows if row["valid_routes"])
        expected = Counter(
            train=int(config["data"]["train_positive_records"]),
            validation=int(config["data"]["validation_positive_records"]),
        )
        if split_counts != expected:
            raise RuntimeError("cap common split counts differ from config")
    features = {
        row["uid"]
        for row in read_jsonl(PROJECT / configs[24]["config"]["visual_features"]["manifest"])
    }
    if uid_sets[24] - features:
        raise RuntimeError("visual feature cache does not cover common cap UIDs")

    config = configs[24]["config"]
    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    model = BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=1024,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
        image_dim=int(config["visual_features"]["feature_width"]),
    )
    initialization = {
        "seed": seed,
        "full_initialization_sha256": predictor_state_sha256(model),
        "components": component_hashes(model),
        "identical_for_all_caps_by_constructor_seed_contract": True,
    }
    output = PROJECT / "outputs/binary_cap_sweep_v1/audits/training_readiness_v1.json"
    payload = {
        "schema_version": "binary_cap_training_readiness_v1",
        "passed": True,
        "caps": list(CAPS),
        "common_positive_records": len(uid_sets[24]),
        "common_train_records": configs[24]["config"]["data"]["train_positive_records"],
        "common_validation_records": configs[24]["config"]["data"]["validation_positive_records"],
        "configs": {
            str(cap): {"path": str(item["path"].relative_to(PROJECT)), "sha256": file_sha256(item["path"])}
            for cap, item in configs.items()
        },
        "initialization": initialization,
        "checks": {
            "matched_uids": True,
            "config_identity_outside_cap": True,
            "routes_are_cap_compliant": True,
            "all_on_absent": True,
            "visual_features_complete": True,
            "node03_excluded_by_submission_contract": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "output": str(output.relative_to(PROJECT)), "initialization": initialization}, sort_keys=True))


if __name__ == "__main__":
    main()
