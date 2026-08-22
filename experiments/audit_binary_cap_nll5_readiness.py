#!/usr/bin/env python3
"""Static readiness audit for the two CAP-NLL5 pipelines."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.train_binary_polar import file_sha256


CAPS = (26, 24)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    output = Path("outputs/binary_cap_nll5_v1/audits/training_readiness_v1.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readiness artifact: {output}")
    configs = {}
    populations = {}
    shared_uids = None
    shared_initialization_contract = None
    for cap in CAPS:
        path = Path(f"configs/binary_cap{cap}_nll5_execval_v1.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config["training"]["objective"] != "exact_set_nll" or int(config["training"]["epochs"]) != 5:
            raise RuntimeError(f"CAP{cap} objective/epoch contract mismatch")
        if config["evaluation"]["internal_primary"] != "executed_validation_accuracy":
            raise RuntimeError(f"CAP{cap} does not use executed validation accuracy")
        if int(config["data"]["visual_on_cap"]) != cap or int(config["data"]["common_eligibility_cap"]) != 24:
            raise RuntimeError(f"CAP{cap} cap metadata mismatch")
        for source, digest in config["source_sha256"].items():
            if file_sha256(Path(source)) != digest:
                raise RuntimeError(f"CAP{cap} source hash mismatch: {source}")
        manifest = Path(config["data"]["manifest"])
        if file_sha256(manifest) != config["data"]["manifest_sha256"]:
            raise RuntimeError(f"CAP{cap} manifest hash mismatch")
        rows = [row for row in read_jsonl(manifest) if row.get("valid_routes")]
        counts = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation")
        }
        if counts != {"train": 6007, "validation": 872}:
            raise RuntimeError(f"CAP{cap} population mismatch: {counts}")
        uids = {row["uid"] for row in rows}
        if shared_uids is None:
            shared_uids = uids
        elif uids != shared_uids:
            raise RuntimeError("CAP26 and CAP24 use different matched UIDs")
        initialization_contract = {
            key: config["training"][key]
            for key in (
                "seed", "physical_batch_size", "gradient_accumulation_steps",
                "effective_batch_size", "learning_rate", "weight_decay",
                "scheduler", "warmup_steps",
            )
        }
        if shared_initialization_contract is None:
            shared_initialization_contract = initialization_contract
        elif initialization_contract != shared_initialization_contract:
            raise RuntimeError("CAP26 and CAP24 optimization contracts differ")
        configs[str(cap)] = {"path": str(path), "sha256": file_sha256(path)}
        populations[str(cap)] = counts
    payload = {
        "schema_version": "binary_cap_nll5_training_readiness_v1",
        "passed": True,
        "integrity_status": "PASS",
        "configs": configs,
        "populations": populations,
        "common_uid_count": len(shared_uids or ()),
        "same_common_uids": True,
        "same_initialization_and_optimization": True,
        "only_intended_differences": ["visual_on_cap", "cap_filtered_valid_route_sets"],
        "checkpoint_selection": (
            "max_executed_accuracy_then_min_mean_visual_on_then_"
            "min_validation_set_nll_then_earlier_epoch"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
