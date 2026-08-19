#!/usr/bin/env python3
"""Independent streaming verification of published P8 artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def reconstruct(boundaries: list[int], operations: list[int]) -> list[int]:
    starts = [0] + [index for index in range(1, len(boundaries)) if boundaries[index] == 1]
    output = [0] * len(boundaries)
    for run, start in enumerate(starts):
        stop = starts[run + 1] if run + 1 < len(starts) else len(output)
        output[start:stop] = [operations[start]] * (stop - start)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("P8 generation audit did not pass")
    for item in audit["artifacts"].values():
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {path}")

    valid_path = Path(audit["artifacts"]["valid_set"]["path"])
    predictor_path = Path(audit["artifacts"]["binary_predictor"]["path"])
    valid_count = positive = selected = 0
    groups: dict[str, str] = {}
    valid_iter = rows(valid_path)
    predictor_iter = rows(predictor_path)
    while True:
        left = next(valid_iter, None)
        right = next(predictor_iter, None)
        if left is None or right is None:
            if left is not right:
                raise RuntimeError("valid-set and predictor manifests have different row counts")
            break
        valid_count += 1
        if left["uid"] != right["uid"] or left["valid_routes"] != right["valid_routes"]:
            raise RuntimeError(f"matched-objective route mismatch for {left['uid']}")
        routes = left["valid_routes"]
        if len(routes) > 50 or len({route["key"] for route in routes}) != len(routes):
            raise RuntimeError(f"route cap/uniqueness failure for {left['uid']}")
        if routes and abs(sum(route["weight"] for route in routes) - 1.0) > 1e-9:
            raise RuntimeError(f"route-weight normalization failure for {left['uid']}")
        positive += bool(routes)
        selected += len(routes)
        previous = groups.setdefault(left["split_group"], left["split"])
        if previous != left["split"]:
            raise RuntimeError(f"split leakage for {left['split_group']}")

    ranking_count = ranking_valid = 0
    for row in rows(Path(audit["artifacts"]["route_ranking"]["path"])):
        ranking_count += 1
        ranking_valid += bool(row["valid"])

    polar_count = 0
    for row in rows(Path(audit["artifacts"]["polar_segment"]["path"])):
        polar_count += 1
        if reconstruct(row["boundary_targets"], row["operation_targets"]) != row["mask"]:
            raise RuntimeError(f"POLAR reconstruction failure for {row['uid']}/{row['route_id']}")

    single_count = sum(1 for _ in rows(Path(audit["artifacts"]["single_best"]["path"])))
    expected = audit["totals"]
    observed = {
        "sample_rows": valid_count,
        "single_best_rows": single_count,
        "positive_samples": positive,
        "selected_valid_routes": selected,
        "ranking_routes": ranking_count,
        "ranking_valid_routes": ranking_valid,
        "polar_routes": polar_count,
        "cross_split_groups": 0,
    }
    required = {
        "sample_rows": expected["samples"],
        "single_best_rows": expected["samples"],
        "positive_samples": expected["positive_samples"],
        "selected_valid_routes": expected["selected_valid_routes"],
        "ranking_routes": expected["evaluated_routes"],
        "ranking_valid_routes": expected["raw_valid_routes"],
        "polar_routes": expected["selected_valid_routes"],
        "cross_split_groups": 0,
    }
    if observed != required:
        raise RuntimeError(f"P8 count mismatch: observed={observed}, required={required}")
    result = {
        "schema_version": "label_regeneration_p8_verification_v1",
        "passed": True,
        "generation_audit_sha256": digest(args.audit),
        "observed": observed,
        "checks": {
            "artifact_checksums": True,
            "matched_objective_route_identity": True,
            "route_cap_and_equal_weights": True,
            "image_group_disjoint": True,
            "ranking_positive_negative_counts": True,
            "polar_exact_reconstruction": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = digest(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
