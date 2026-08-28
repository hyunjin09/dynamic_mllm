#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tools.research_analysis.four_action.route_conditioned import classify_route_conditioned_cell


ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_action_cells(frame: pd.DataFrame) -> dict[str, Any]:
    keys = ["uid", "target_layer"]
    groups = frame.groupby(keys, sort=False)
    action_sets_exact = True
    target_identity_exact = True
    action_semantics_exact = True
    target_is_anchor_off = True
    m00_correct = True
    taxonomy_exact = True
    maximum_factorial_error = 0.0
    maximum_margin_error = float(
        np.max(np.abs(frame.margin - (frame.S_correct - frame.S_original_full_wrong)))
    )
    for (_, layer), group in groups:
        indexed = group.set_index("action")
        action_sets_exact &= len(group) == 4 and set(indexed.index) == set(ACTIONS)
        if not action_sets_exact:
            continue
        target_identity_exact &= group.fixed_correct_target_text.nunique() == 1
        target_identity_exact &= group.fixed_wrong_target_text.nunique() == 1
        mask = list(group.anchor_route_mask.iloc[0])
        target_is_anchor_off &= int(mask[int(layer)]) == 0
        target_is_anchor_off &= int(group.anchor_off_count.iloc[0]) == sum(
            int(value) == 0 for value in mask
        )
        for action in ACTIONS:
            row = indexed.loc[action]
            action_semantics_exact &= bool(row.read_on) == (action in {"READ_ONLY", "FULL"})
            action_semantics_exact &= bool(row.write_on) == (action in {"WRITE_ONLY", "FULL"})
            action_semantics_exact &= bool(row.new_evaluation) == (action != "IGNORE")
        m00_correct &= bool(indexed.loc["IGNORE", "correct"])
        correctness = {action: bool(indexed.loc[action, "correct"]) for action in ACTIONS}
        expected_taxonomy = classify_route_conditioned_cell(correctness)
        taxonomy_exact &= group.taxonomy.nunique() == 1
        taxonomy_exact &= str(group.taxonomy.iloc[0]) == expected_taxonomy
        margins = {action: float(indexed.loc[action, "margin"]) for action in ACTIONS}
        expected_effects = {
            "read_w0": margins["READ_ONLY"] - margins["IGNORE"],
            "write_r0": margins["WRITE_ONLY"] - margins["IGNORE"],
            "read_w1": margins["FULL"] - margins["WRITE_ONLY"],
            "write_r1": margins["FULL"] - margins["READ_ONLY"],
            "interaction": margins["FULL"]
            - margins["READ_ONLY"]
            - margins["WRITE_ONLY"]
            + margins["IGNORE"],
        }
        for name, expected in expected_effects.items():
            maximum_factorial_error = max(
                maximum_factorial_error,
                float(np.max(np.abs(group[name].to_numpy(dtype=float) - expected))),
            )
    checks = {
        "four_actions_exact_per_sample_layer": bool(action_sets_exact),
        "fixed_target_identity_exact": bool(target_identity_exact),
        "action_read_write_semantics_exact": bool(action_semantics_exact),
        "target_layer_is_anchor_off_and_off_count_exact": bool(target_is_anchor_off),
        "every_anchor_m00_correct": bool(m00_correct),
        "taxonomy_recomputation_exact": bool(taxonomy_exact),
        "margin_formula_exact": maximum_margin_error <= 1e-12,
        "factorial_formulas_exact": maximum_factorial_error <= 1e-12,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "action_row_count": len(frame),
        "cell_count": groups.ngroups,
        "sample_count": frame.uid.nunique(),
        "maximum_margin_formula_abs_error": maximum_margin_error,
        "maximum_factorial_formula_abs_error": maximum_factorial_error,
    }


def _checksum_audit(root: Path) -> dict[str, Any]:
    failures = []
    sidecars = sorted(root.rglob("*.sha256"))
    for sidecar in sidecars:
        parts = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        if len(parts) != 2:
            failures.append({"sidecar": str(sidecar), "reason": "malformed"})
            continue
        expected, name = parts
        target = sidecar.parent / name.strip()
        if not target.exists():
            failures.append({"sidecar": str(sidecar), "reason": "target_missing"})
        elif _sha256_file(target) != expected:
            failures.append({"sidecar": str(sidecar), "reason": "digest_mismatch"})
    return {
        "passed": not failures,
        "sidecar_count": len(sidecars),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed route-conditioned outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/four_action_route_conditioned.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    anchor = json.loads((root / "anchor_route_summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((root / "pilot_benchmark_summary.json").read_text(encoding="utf-8"))
    full = json.loads((Path(config["full_root"]) / "full" / "stage_summary.json").read_text(encoding="utf-8"))
    aggregate = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    cells = pd.read_parquet(root / "route_conditioned_cells.parquet")
    cell_audit = audit_action_cells(cells)
    required = [
        root / "implementation_audit.md",
        root / "pilot_report.md",
        root / "anchor_route_manifest.jsonl",
        root / "anchor_route_manifest.parquet",
        root / "route_conditioned_cells.jsonl",
        root / "route_conditioned_cells.parquet",
        root / "aggregate_summary.json",
        root / "route_conditioned_decomposition_report.md",
    ]
    figures = sorted((root / "figures").glob("*.png"))
    cross_checks = {
        "all_stage_summaries_pass": bool(anchor["passed"] and pilot["passed"] and full["passed"] and aggregate["passed"]),
        "sample_count_consistent": int(anchor["validated_anchor_count"])
        == int(full["sample_count"])
        == int(aggregate["validated_anchor_sample_count"])
        == int(cell_audit["sample_count"]),
        "off_position_count_consistent": int(full["anchor_off_position_count"])
        == int(aggregate["anchor_off_position_count"])
        == int(cell_audit["cell_count"]),
        "action_row_count_consistent": int(full["flat_action_row_count"])
        == int(aggregate["flat_action_row_count"])
        == int(cell_audit["action_row_count"]),
        "new_cell_count_is_exactly_3k": int(full["new_intervention_cell_count"])
        == 3 * int(full["anchor_off_position_count"]),
        "zero_disqualifying_failures": int(full["disqualifying_failure_count"]) == 0,
        "all_eight_gpu_worker_contract": bool(full["all_eight_gpu_worker_contract"])
        and int(full["worker_count"]) == 16,
        "required_artifacts_present": all(path.exists() for path in required),
        "required_figures_present": len(figures) >= 6,
    }
    checksum_audit = _checksum_audit(root)
    output = {
        "schema_version": "route_conditioned_final_integrity_audit_v1",
        "passed": bool(
            cell_audit["passed"]
            and checksum_audit["passed"]
            and all(cross_checks.values())
        ),
        "cell_audit": cell_audit,
        "cross_artifact_checks": cross_checks,
        "checksum_audit_before_this_file": checksum_audit,
        "required_artifacts": [str(path) for path in required],
        "figure_count": len(figures),
        "figure_paths": [str(path) for path in figures],
    }
    path = root / "final_integrity_audit.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
