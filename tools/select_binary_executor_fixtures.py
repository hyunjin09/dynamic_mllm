#!/usr/bin/env python3
"""Freeze two outcome-blind executor fixtures per benchmark/difficulty cell."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_policy.labels import iter_source_json


def digest_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.per_cell < 1:
        raise ValueError("per-cell must be positive")

    candidates: dict[str, list[dict]] = {}
    invalid = []
    for path in iter_source_json(args.source):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            sample = record["sample"]
            benchmark = str(sample["benchmark"])
            difficulty = str(sample["mcts_difficulty"])
            uid = str(sample["uid"])
            num_layers = int(record["runtime"]["num_layers"])
            candidate_masks = {
                tuple(row["visual_on_mask"])
                for row in record.get("candidate_executions", [])
                if isinstance(row.get("visual_on_mask"), list)
            }
            required_masks = {(1,) * num_layers, (0,) * num_layers}
            best = record.get("mcts", {}).get("best_mask")
            if isinstance(best, list) and len(best) == num_layers:
                required_masks.add(tuple(best))
            if not required_masks.issubset(candidate_masks):
                raise ValueError("missing cached candidate IDs for a required technical route")
            image_path = Path(sample["local_image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"image missing: {image_path}")
            cell = f"{benchmark}/{difficulty}"
            candidates.setdefault(cell, []).append(
                {
                    "rank_key": digest_text(f"{args.seed}:{uid}"),
                    "uid": uid,
                    "benchmark": benchmark,
                    "difficulty": difficulty,
                    "image_sha256": sample.get("image_content_sha256"),
                    "record_file": str(path.resolve()),
                    "record_sha256": file_sha256(path),
                }
            )
        except Exception as exc:
            invalid.append({"record_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    selected = []
    for cell, rows in sorted(candidates.items()):
        rows.sort(key=lambda row: (row["rank_key"], row["uid"]))
        selected.extend(rows[: args.per_cell])
    expected_cells = {
        f"{benchmark}/{difficulty}"
        for benchmark in ("chartqa", "docvqa", "gqa", "textvqa")
        for difficulty in ("easy", "hard")
    }
    selected_counts = {
        cell: sum(f"{row['benchmark']}/{row['difficulty']}" == cell for row in selected)
        for cell in sorted(expected_cells)
    }
    report = {
        "schema_version": "binary_executor_fixtures_v1",
        "selection_uses_outcomes": False,
        "selection_rule": "smallest sha256(seed:uid) within benchmark/difficulty cell",
        "seed": args.seed,
        "per_cell": args.per_cell,
        "source_record_count": sum(len(rows) for rows in candidates.values()) + len(invalid),
        "technical_invalid_records": invalid,
        "selected_counts": selected_counts,
        "records": selected,
        "passed": (
            not invalid
            and set(candidates) == expected_cells
            and all(value == args.per_cell for value in selected_counts.values())
            and len({row["image_sha256"] for row in selected}) == len(selected)
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError("executor fixture freeze failed")


if __name__ == "__main__":
    main()
