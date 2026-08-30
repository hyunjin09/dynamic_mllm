#!/usr/bin/env python3
"""Validate the W2C repair smoke and summarize the completed repair."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from four_action_policy.when_repair import (
    build_known_full_candidates,
    local_suffix_search_plan,
    maximal_full_boundary,
    repair_w2c_sample,
)


Route = tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _route_key(actions: Sequence[str]) -> str:
    return "|".join(str(action) for action in actions)


def _route(actions: Sequence[str]) -> Route:
    return tuple(str(action) for action in actions)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _write_frozen(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    path.write_bytes(encoded)


def _write_jsonl_frozen(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = b"".join(_json_bytes(dict(row)) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    path.write_bytes(encoded)


def _write_text_frozen(path: Path, value: str) -> None:
    encoded = value.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    path.write_bytes(encoded)


def record_file_sha256s(record_root: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(record_root.glob("*.json"))
        if path.is_file()
    }


def compare_resume_snapshot(
    baseline: Mapping[str, str], current: Mapping[str, str]
) -> dict[str, Any]:
    baseline_keys = set(baseline)
    current_keys = set(current)
    changed = sorted(
        name
        for name in baseline_keys & current_keys
        if str(baseline[name]) != str(current[name])
    )
    missing = sorted(baseline_keys - current_keys)
    added = sorted(current_keys - baseline_keys)
    return {
        "passed": not (changed or missing or added),
        "baseline_records": len(baseline),
        "current_records": len(current),
        "changed": changed,
        "missing": missing,
        "added": added,
    }


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def audit_smoke_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_by_uid: Mapping[str, Mapping[str, Any]],
    manifest_by_uid: Mapping[str, Mapping[str, Any]],
    search_budget: int,
    seed: int,
) -> dict[str, Any]:
    """Reconstruct every smoke repair and apply plan checks 1--8 and 10."""

    issues: dict[str, list[str]] = {
        "exact_old_route_replay": [],
        "full_insertion_at_candidate_boundary": [],
        "compatible_suffix_enumeration": [],
        "deduplication": [],
        "correct_cache_update": [],
        "candidate_boundary_moves_after_rescue": [],
        "iterative_re_evaluation": [],
        "bounded_search_after_known_exhaustion": [],
        "output_determinism": [],
    }
    observed_rescue = False
    observed_iteration = False
    reconstructed = 0

    for record in records:
        uid = str(record["uid"])
        source = source_by_uid[uid]
        manifest = manifest_by_uid[uid]
        repair = record.get("repair") or {}
        if record.get("status") != "completed":
            for values in issues.values():
                values.append(f"{uid}: record is not completed")
            continue

        original_routes = [_route(row["actions"]) for row in source["valid_routes"]]
        original_keys = [_route_key(route) for route in original_routes]
        replays = list(record.get("old_route_replays", []))
        replay_keys = [str(row.get("route_key")) for row in replays]
        if (
            len(replays) != int(manifest["valid_route_count"])
            or replay_keys != original_keys
            or any(row.get("correct") is not True for row in replays)
        ):
            issues["exact_old_route_replay"].append(uid)

        execution_rows = list(repair.get("route_execution_cache", []))
        execution_keys = [str(row["route_key"]) for row in execution_rows]
        repaired_rows = list(repair.get("repaired_routes", []))
        repaired_keys = [str(row["route_key"]) for row in repaired_rows]
        if (
            len(execution_keys) != len(set(execution_keys))
            or len(repaired_keys) != len(set(repaired_keys))
        ):
            issues["deduplication"].append(uid)

        execution_by_route = {
            _route(row["actions"]): dict(row) for row in execution_rows
        }
        for execution in execution_rows:
            boundary = int(execution["candidate_boundary"])
            actions = list(execution["actions"])
            if (
                execution["stage"] == "known_suffix_repair"
                and (
                    boundary >= len(actions)
                    or actions[boundary] != "FULL"
                    or any(action != "FULL" for action in actions[:boundary])
                )
            ):
                issues["full_insertion_at_candidate_boundary"].append(
                    f"{uid}:round={execution['round']}"
                )

        correct_routes: dict[Route, str] = {
            route: "original_cache" for route in original_routes
        }
        execution_seen: set[Route] = set()
        history = list(repair.get("history", []))
        observed_iteration = observed_iteration or len(history) > 1
        trace_valid = True
        for round_index, history_row in enumerate(history):
            ordered_routes = sorted(correct_routes)
            boundary, _ = maximal_full_boundary(ordered_routes)
            known = build_known_full_candidates(ordered_routes, boundary=boundary)
            known_keys = {str(row["route_key"]) for row in known}
            known_routes = {_route(row["actions"]) for row in known}
            known_rows = [
                execution_by_route[route]
                for route in sorted(known_routes)
                if route in execution_by_route
            ]
            executed_known_keys = {str(row["route_key"]) for row in known_rows}
            if (
                int(history_row["round"]) != round_index
                or int(history_row["boundary"]) != boundary
                or int(history_row["known_candidates"]) != len(known)
                or known_keys != executed_known_keys
            ):
                issues["compatible_suffix_enumeration"].append(
                    f"{uid}:round={round_index}"
                )
                trace_valid = False

            for row in known_rows:
                execution_seen.add(_route(row["actions"]))
            known_correct = [row for row in known_rows if bool(row["correct"])]
            bounded_rows: list[dict[str, Any]] = []
            if not known_correct:
                bounded_plan = local_suffix_search_plan(
                    known,
                    boundary=boundary,
                    uid=uid,
                    seed=seed,
                    budget=search_budget,
                    excluded_routes=set(correct_routes) | execution_seen,
                )
                expected_bounded = {
                    str(row["route_key"]) for row in bounded_plan["candidates"]
                }
                bounded_routes = {
                    _route(row["actions"]) for row in bounded_plan["candidates"]
                }
                bounded_rows = [
                    execution_by_route[route]
                    for route in sorted(bounded_routes)
                    if route in execution_by_route
                ]
                actual_bounded = {str(row["route_key"]) for row in bounded_rows}
                if (
                    int(history_row["bounded_available"])
                    != int(bounded_plan["available_candidates"])
                    or int(history_row["bounded_selected"]) != len(bounded_rows)
                    or expected_bounded != actual_bounded
                ):
                    issues["compatible_suffix_enumeration"].append(
                        f"{uid}:bounded-round={round_index}"
                    )
                    trace_valid = False
            elif int(history_row["bounded_selected"]) != 0:
                issues["bounded_search_after_known_exhaustion"].append(
                    f"{uid}:history-round={round_index}"
                )
                trace_valid = False

            for row in bounded_rows:
                execution_seen.add(_route(row["actions"]))
            successful = known_correct or [row for row in bounded_rows if bool(row["correct"])]
            if successful:
                before = boundary
                observed_rescue = True
                for row in successful:
                    correct_routes.setdefault(_route(row["actions"]), str(row["stage"]))
                after, _ = maximal_full_boundary(sorted(correct_routes))
                if after <= before:
                    issues["candidate_boundary_moves_after_rescue"].append(
                        f"{uid}:round={round_index}"
                    )
                    trace_valid = False

        if set(repaired_keys) != {_route_key(route) for route in correct_routes}:
            issues["correct_cache_update"].append(uid)
            trace_valid = False
        new_repaired = {
            str(row["route_key"])
            for row in repaired_rows
            if row.get("source_of_discovery") != "original_cache"
        }
        successful_executions = {
            str(row["route_key"]) for row in execution_rows if bool(row["correct"])
        }
        if new_repaired != successful_executions:
            issues["correct_cache_update"].append(f"{uid}:successful-set")
            trace_valid = False

        deterministic_results = {
            _route(row["actions"]): {
                name: row[name]
                for name in (
                    "prediction",
                    "score",
                    "correct",
                    "generated_ids",
                    "execution_source",
                    "prompt_sha256",
                )
                if name in row
            }
            for row in execution_rows
        }

        def cached_evaluate(actions: Route) -> Mapping[str, Any]:
            if actions not in deterministic_results:
                raise KeyError(_route_key(actions))
            return deterministic_results[actions]

        try:
            repeated = repair_w2c_sample(
                source,
                cached_evaluate,
                search_budget=search_budget,
                seed=seed,
            )
            if _json_bytes(repeated) != _json_bytes(repair):
                issues["output_determinism"].append(uid)
                trace_valid = False
        except Exception as error:  # pragma: no cover - retained in audit payload
            issues["output_determinism"].append(
                f"{uid}:{type(error).__name__}:{error}"
            )
            trace_valid = False
        reconstructed += int(trace_valid)

    if not observed_rescue:
        issues["candidate_boundary_moves_after_rescue"].append(
            "smoke cohort did not exercise a rescue"
        )
    if not observed_iteration:
        issues["iterative_re_evaluation"].append(
            "smoke cohort did not exercise a second repair round"
        )

    checks = {
        name: _check(
            not values,
            issues=sorted(set(values)),
            **(
                {"observed": observed_rescue}
                if name == "candidate_boundary_moves_after_rescue"
                else {"observed": observed_iteration}
                if name == "iterative_re_evaluation"
                else {}
            ),
        )
        for name, values in issues.items()
    }
    return {
        "records": len(records),
        "completed": sum(record.get("status") == "completed" for record in records),
        "quarantined": sum(record.get("status") != "completed" for record in records),
        "reconstructed_records": reconstructed,
        "checks": checks,
        "all_passed": all(row["passed"] for row in checks.values()),
    }


def _load_records(record_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(record_root.glob("*.json"))
    ]


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config, file_sha256(path)


def _load_json(path: str | Path, expected_sha256: str | None = None) -> Any:
    resolved = Path(path)
    if expected_sha256 is not None and file_sha256(resolved) != expected_sha256:
        raise RuntimeError(f"checksum mismatch: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path, expected_sha256: str | None = None) -> list[Any]:
    resolved = Path(path)
    if expected_sha256 is not None and file_sha256(resolved) != expected_sha256:
        raise RuntimeError(f"checksum mismatch: {resolved}")
    return [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_smoke(config_path: Path) -> dict[str, Any]:
    config, config_sha = _load_config(config_path)
    raw_root = Path(config["execution"]["raw_output_root"])
    hashes = record_file_sha256s(raw_root / "smoke" / "records")
    expected = int(config["smoke"]["records"])
    if len(hashes) != expected:
        raise RuntimeError(f"smoke snapshot has {len(hashes)} records, expected {expected}")
    payload = {
        "schema_version": "w2c_when_repair_resume_snapshot_v1",
        "config_sha256": config_sha,
        "record_sha256": hashes,
        "records": len(hashes),
    }
    output = Path(config["reporting"]["analysis_dir"]) / "smoke" / "resume_baseline.json"
    _write_frozen(output, payload)
    return payload


def finalize_smoke(config_path: Path) -> dict[str, Any]:
    config, config_sha = _load_config(config_path)
    analysis_root = Path(config["reporting"]["analysis_dir"])
    raw_root = Path(config["execution"]["raw_output_root"])
    record_root = raw_root / "smoke" / "records"
    records = _load_records(record_root)
    baseline = _load_json(analysis_root / "smoke" / "resume_baseline.json")
    if baseline["config_sha256"] != config_sha:
        raise RuntimeError("resume baseline uses a different repair config")
    resume = compare_resume_snapshot(
        baseline["record_sha256"], record_file_sha256s(record_root)
    )

    source_rows = _load_jsonl(
        config["data"]["source_manifest"],
        config["data"]["source_manifest_sha256"],
    )
    source_by_uid = {str(row["uid"]): row for row in source_rows}
    smoke_manifest = _load_json(
        config["data"]["smoke_manifest"], config["data"]["smoke_manifest_sha256"]
    )
    manifest_by_uid = {str(row["uid"]): row for row in smoke_manifest}
    if set(manifest_by_uid) != {str(row["uid"]) for row in records}:
        raise RuntimeError("raw smoke records do not exactly match the frozen manifest")

    audit = audit_smoke_records(
        records,
        source_by_uid=source_by_uid,
        manifest_by_uid=manifest_by_uid,
        search_budget=int(config["search"]["per_state_variant_budget"]),
        seed=int(config["search"]["seed"]),
    )
    audit["checks"]["resume_restart_consistency"] = _check(
        bool(resume["passed"]),
        **{name: value for name, value in resume.items() if name != "passed"},
    )
    audit["checks"]["coverage_and_zero_quarantine"] = _check(
        len(records) == int(config["smoke"]["records"])
        and audit["quarantined"] == 0,
        expected_records=int(config["smoke"]["records"]),
        observed_records=len(records),
        quarantined=audit["quarantined"],
    )
    audit["all_passed"] = all(
        row["passed"] for row in audit["checks"].values()
    )
    audit.update(
        {
            "schema_version": "w2c_when_repair_smoke_gate_v1",
            "config_sha256": config_sha,
            "route_executions": sum(
                len(row["old_route_replays"])
                + len(row["repair"]["route_execution_cache"])
                for row in records
            ),
            "old_route_replays": sum(len(row["old_route_replays"]) for row in records),
            "new_correct_routes": sum(
                int(row["repair"]["new_correct_route_count"]) for row in records
            ),
            "rescued_samples": sum(
                int(row["repair"]["new_correct_route_count"]) > 0 for row in records
            ),
            "iterative_samples": sum(
                len(row["repair"]["history"]) > 1 for row in records
            ),
            "boundary_shift_counts": dict(
                sorted(
                    Counter(
                        str(row["repair"]["boundary_shift"]) for row in records
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            ),
        }
    )
    _write_jsonl_frozen(
        analysis_root / "smoke" / "smoke_executions.jsonl",
        sorted(records, key=lambda row: str(row["uid"])),
    )
    _write_frozen(analysis_root / "smoke" / "smoke_gate.json", audit)
    checks = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if row['passed'] else 'FAIL'} |"
        for name, row in audit["checks"].items()
    )
    report = f"""# W2C WHEN Repair Smoke Report

## Decision

**{'PASS' if audit['all_passed'] else 'FAIL'}**. The frozen 12-sample smoke
completed {audit['route_executions']:,} route executions, including
{audit['old_route_replays']:,} exact original-route replays, with
{audit['quarantined']} quarantined samples. The resume replay left all raw
record bytes unchanged.

## Repair behavior

- Samples with at least one newly verified correct route: {audit['rescued_samples']}/12
- Newly verified correct routes: {audit['new_correct_routes']}
- Samples exercising more than one repair round: {audit['iterative_samples']}/12
- Final boundary-shift counts: `{json.dumps(audit['boundary_shift_counts'], sort_keys=True)}`

## Gate checks

| Check | Result |
|---|---|
{checks}

`FULL_UNRESCUED_UNDER_BUDGET` retains bounded one-edit semantics and is not a
claim that FULL is globally invalid. A pass admits only the frozen 640-sample
repair; it does not admit gate/router training or external evaluation.
"""
    _write_text_frozen(analysis_root / "smoke" / "smoke_report.md", report)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/w2c_when_repair/repair_config.yaml"
    )
    parser.add_argument(
        "--stage", choices=("snapshot-smoke", "finalize-smoke"), required=True
    )
    args = parser.parse_args()
    if args.stage == "snapshot-smoke":
        result = snapshot_smoke(Path(args.config))
    else:
        result = finalize_smoke(Path(args.config))
    print(json.dumps({"event": args.stage, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
