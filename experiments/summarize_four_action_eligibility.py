#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from tools.research_analysis.four_action.eligibility import summarize_eligibility


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    config = yaml.safe_load(Path("configs/four_action_answer_alignment.yaml").read_text())
    root = Path(config["eligibility_root"])
    rows = []
    failures = []
    runtimes = []
    missing = []
    for shard in range(8):
        directory = root / f"shard_{shard:02d}"
        if not (directory / "runtime.json").is_file() or not (directory / "results.jsonl").is_file():
            missing.append(shard)
            continue
        runtimes.append(json.loads((directory / "runtime.json").read_text()))
        rows.extend(read_jsonl(directory / "results.jsonl"))
        failures.extend(read_jsonl(directory / "failures.jsonl"))
    manifest = read_jsonl(Path(config["cohort_manifest"]))
    candidate_counts = dict(Counter(row["cohort"] for row in manifest))
    summary = summarize_eligibility(rows, candidate_counts) if not missing else {}
    checks = {
        "all_eight_shards_present": not missing and len(runtimes) == 8,
        "eight_worker_contract": len(runtimes) == 8
        and {row["rank"] for row in runtimes} == set(range(8))
        and all(row["world_size"] == 8 for row in runtimes),
        "no_failures": not failures,
        "all_candidates_present": not missing and len(rows) == len(manifest),
    }
    summary.update(
        {
            "checks": checks,
            "passed": all(checks.values()),
            "missing_shards": missing,
            "failures": failures,
            "dataset_candidate_counts": dict(Counter(row["dataset"] for row in rows)),
        }
    )
    rows.sort(key=lambda row: (row["dataset"], row["uid"]))
    write_once(
        root / "merged_results.jsonl",
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows),
    )
    write_once(root / "summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
