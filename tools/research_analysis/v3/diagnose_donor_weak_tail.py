from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


COVERAGE = Path("outputs/v3_preflight/donor_coverage_audit_v1.json")
GEOMETRY = Path("artifacts/v3_null_calibration/read_write_geometry_v1/geometry.jsonl")
DONORS = Path(
    "artifacts/v3_null_calibration/paired_donor_index_v1/paired_donor_index.jsonl"
)
OUTPUT = Path("outputs/v3_preflight/donor_weak_tail_diagnostic_v1.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return math.inf
    return max(first / second, second / first)


def fields(row: dict) -> dict[str, float]:
    return {
        "read_norm": float(row["read"]["frobenius_norm"]),
        "write_norm": float(row["write"]["frobenius_norm"]),
        "read_rows": float(row["read_shape"][0]),
        "write_rows": float(row["write_shape"][0]),
        "image_tokens": float(row["image_tokens"]),
        "prompt_tokens": float(row["prompt_tokens"]),
        "read_scale_ratio": float(row["read_rmsnorm_scale_ratio"]),
        "write_scale_ratio": float(row["write_rmsnorm_scale_ratio"]),
        "read_row_cv": float(row["read"]["row_norm_cv"]),
        "write_row_cv": float(row["write"]["row_norm_cv"]),
    }


def main() -> None:
    coverage = json.loads(COVERAGE.read_text())
    geometry = {
        (row["sample_id"], int(row["layer"])): row
        for row in (json.loads(line) for line in GEOMETRY.read_text().splitlines())
    }
    donors = {
        (row["sample_id"], int(row["layer"])): row
        for row in (json.loads(line) for line in DONORS.read_text().splitlines())
    }
    rows = []
    dominant = Counter()
    for weak in coverage["weak_targets"]:
        key = (weak["sample_id"], int(weak["layer"]))
        donor_id = donors[key]["donors"][7]["sample_id"]
        target_values = fields(geometry[key])
        donor_values = fields(geometry[(donor_id, key[1])])
        components = {
            name: ratio(target_values[name], donor_values[name]) for name in target_values
        }
        maximum = max(components.values())
        names = sorted(name for name, value in components.items() if abs(value - maximum) < 1e-9)
        dominant.update(names)
        rows.append(
            {
                **weak,
                "eighth_donor_id": donor_id,
                "component_ratios": components,
                "dominant_components": names,
            }
        )
    payload = {
        "schema_version": "v3_donor_weak_tail_diagnostic_v1",
        "outcome_blind": True,
        "terminal_answer_or_action_outcomes_used": False,
        "question": "Which frozen geometry component determines each >1.5 eighth-donor distance?",
        "weak_target_count": len(rows),
        "dominant_component_counts": dict(dominant),
        "rows": rows,
        "sources": {str(path): sha256(path) for path in (COVERAGE, GEOMETRY, DONORS)},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "dominant": dict(dominant)}))


if __name__ == "__main__":
    main()
