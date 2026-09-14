#!/usr/bin/env python3
"""Select one frozen Stage-1 gate and threshold from existing score trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.gate_winner import (  # noqa: E402
    CANDIDATES,
    apply_control,
    independent_alpha_sweep,
    metrics_with_utility,
    paired_bootstrap,
    reference_operating_point,
    select_operating_point,
    select_winner,
    shared_threshold_sweep,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage1_gate_winner_threshold_selection_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
BOUND_CODE_PATHS = (
    "configs/stage1_gate_winner_threshold_selection_v1.json",
    "dense_failure_stage1/gate_winner.py",
    "dense_failure_stage1/sequential_gate.py",
    "dense_failure_stage1/shared_global_gate.py",
    "experiments/select_stage1_gate_winner.py",
)
SWEEP_FIELDS = (
    "candidate",
    "control_type",
    "control_value",
    "threshold",
    "layer_thresholds_json",
    "records",
    "correct",
    "wrong",
    "triggered",
    "no_trigger",
    "correct_false_triggers",
    "wrong_detected",
    "correct_preservation",
    "wrong_detection_recall",
    "failure_precision",
    "trigger_rate",
    "no_trigger_fraction",
    "median_first_trigger_layer",
    "mean_first_trigger_layer",
    "utility",
    "utility_rate",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project root: {value}")
    return path


def load_static_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage1_gate_winner_threshold_selection_config_v1":
        raise ValueError("unsupported winner-selection config")
    if tuple(item["name"] for item in config["candidates"]) != CANDIDATES:
        raise ValueError("candidate order differs from the frozen contract")
    if float(config["selection"]["minimum_failure_precision"]) != 0.9:
        raise ValueError("failure-precision floor differs from the plan")
    if int(config["population"]["validation"]) != 800:
        raise ValueError("validation population differs from the plan")
    return config


def _verify_source(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"source hash mismatch for {path}: {actual} != {expected}")


def _score_matrix(rows: Sequence[Mapping[str, Any]], *, expected_variant: str | None = None) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    normalized = []
    seen = set()
    for row in rows:
        uid = str(row["uid"])
        if uid in seen:
            raise ValueError(f"duplicate score UID: {uid}")
        seen.add(uid)
        if expected_variant is not None and str(row.get("variant")) != expected_variant:
            raise ValueError(f"unexpected score variant for {uid}")
        dataset = str(row["dataset"]).lower()
        if dataset not in DATASETS:
            raise ValueError(f"unexpected dataset for {uid}: {dataset}")
        scores = [float(row[f"p_{layer}"]) for layer in range(28)]
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite score for {uid}")
        normalized.append(
            {
                "uid": uid,
                "dataset": dataset,
                "image_group_id": str(row["image_group_id"]),
                "label": int(bool(row["current_dense_wrong"])),
                "scores": scores,
            }
        )
    normalized.sort(key=lambda item: item["uid"])
    labels = np.asarray([item["label"] for item in normalized], dtype=np.int64)
    scores = np.asarray([item["scores"] for item in normalized], dtype=np.float64)
    if len(normalized) != 800 or Counter(labels.tolist()) != Counter({0: 400, 1: 400}):
        raise ValueError("score population is not the frozen balanced 800-record split")
    return normalized, scores, labels


def _assert_aligned(reference: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> None:
    left = [(row["uid"], row["dataset"], row["image_group_id"], row["label"]) for row in reference]
    right = [(row["uid"], row["dataset"], row["image_group_id"], row["label"]) for row in other]
    if left != right:
        raise RuntimeError("candidate score trajectories do not align by UID/label/dataset/image group")


def prepare(config_path: Path) -> None:
    config = load_static_config(config_path)
    sources = config["sources"]
    phase50_manifest = read_json(resolve_path(sources["phase50_manifest"]))
    phase51_manifest = read_json(resolve_path(sources["phase51_manifest"]))
    complete_all28 = read_json(resolve_path(sources["phase51_all28_training_complete"]))
    complete_random4 = read_json(resolve_path(sources["phase51_random4_training_complete"]))
    if not phase50_manifest.get("passed") or not phase51_manifest.get("passed"):
        raise RuntimeError("an upstream gate artifact manifest did not pass")
    for complete, variant in (
        (complete_all28, "state_layer_all28"),
        (complete_random4, "state_layer_random4"),
    ):
        if not complete.get("passed") or complete.get("variant") != variant:
            raise RuntimeError(f"invalid Phase-51 validation completion: {variant}")
        if complete.get("contract_sha256") != phase51_manifest.get("contract_sha256"):
            raise RuntimeError(f"Phase-51 contract mismatch: {variant}")

    validation_hashes = {
        "phase50_validation_scores": phase50_manifest["required_files"]["validation_scores.jsonl"],
        "phase51_all28_validation_scores": complete_all28["validation_scores_sha256"],
        "phase51_random4_validation_scores": complete_random4["validation_scores_sha256"],
    }
    for name, expected in validation_hashes.items():
        _verify_source(resolve_path(sources[name]), expected)
    independent, _, _ = _score_matrix(read_jsonl(resolve_path(sources["phase50_validation_scores"])))
    all28, _, _ = _score_matrix(
        read_jsonl(resolve_path(sources["phase51_all28_validation_scores"])),
        expected_variant="state_layer_all28",
    )
    random4, _, _ = _score_matrix(
        read_jsonl(resolve_path(sources["phase51_random4_validation_scores"])),
        expected_variant="state_layer_random4",
    )
    _assert_aligned(independent, all28)
    _assert_aligned(independent, random4)

    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE_PATHS}
    source_hashes = {
        key: file_sha256(resolve_path(value))
        for key, value in sources.items()
        if key not in {"phase50_test_scores", "phase51_test_scores"}
    }
    contract: dict[str, Any] = {
        "schema_version": "stage1_gate_winner_threshold_selection_contract_v1",
        "run_id": config["run_id"],
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "static_config": config,
        "bound_code_sha256": bound_hashes,
        "source_sha256": source_hashes,
        "upstream_contract_sha256": {
            "phase50": phase50_manifest["contract_sha256"],
            "phase51": phase51_manifest["contract_sha256"],
        },
        "deferred_test_sources": {
            "phase50_test_scores": {
                "path": sources["phase50_test_scores"],
                "expected_sha256": phase50_manifest["required_files"]["test_scores.jsonl"],
            },
            "phase51_test_scores": {
                "path": sources["phase51_test_scores"],
                "expected_sha256": phase51_manifest["required_files"]["evaluation/test_scores.jsonl"],
            },
        },
        "selection_contract": {
            "utility": "wrong_triggered_minus_correct_triggered",
            "precision_floor": 0.9,
            "winner_ties": config["selection"]["tie_breakers"],
            "trigger_comparison": "strict_greater_than",
            "independent_control": "shared alpha mapped to 28 higher-quantile validation-correct thresholds",
            "shared_control": "one raw score threshold over trajectory maxima",
            "fixed_control": "Random-4 layer-27 score only",
        },
        "test_access_qualification": {
            "historically_unopened": False,
            "phase52_selection_held_out": True,
            "pre_freeze_deviation": "Five leading rows of the Phase-51 combined test score file were printed during source-format inspection; no aggregate, comparison, or selection statistic was computed.",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_once_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Stage-1 gate winner selection protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Phase-50 source contract: `{phase50_manifest['contract_sha256']}`
- Phase-51 source contract: `{phase51_manifest['contract_sha256']}`
- Population: 800 validation records (400 current-dense correct, 400 current-dense wrong) and a separate 800-record test split.
- Target: current-runtime dense failure only.
- Candidate gates: independent sequential, shared All-28 sequential, shared Random-4 sequential, and shared Random-4 fixed layer 27.
- Utility: wrong samples triggered minus correct samples triggered, divided by all samples for utility rate.
- Validation eligibility: failure precision at least 90%.
- Winner: maximum validation utility rate; ties use preservation, recall, earlier median trigger, simplicity, then frozen candidate order.
- Independent sweep: every empirical shared-alpha breakpoint, converted into 28 per-layer `higher` quantile thresholds.
- Shared sweep: every attainable validation trajectory-maximum threshold plus an all-trigger terminal point.
- Test scores are not used by `select-validation`; the selected winner and all candidate controls are frozen first.

## Test-access qualification

The Phase-50/51 test trajectories already existed and had been evaluated in their source phases, so this is a Phase-52 selection-held-out confirmation, not a historically unopened test. During implementation, five leading Phase-51 test rows were printed before freeze to inspect format. No aggregate or candidate-selection statistic was computed from them. This deviation is retained in the frozen contract and limits any claim of strictly unopened test access.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "validation_records": len(independent),
            "validation_correct": 400,
            "validation_wrong": 400,
            "candidate_alignment": True,
            "source_hashes_verified": validation_hashes,
            "test_files_hashed_or_read_by_prepare": False,
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract(config_path: Path, *, verify_test: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_static_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("frozen contract hash is invalid")
    if contract.get("static_config") != config:
        raise RuntimeError("static config differs from the frozen contract")
    for relative, expected in contract["bound_code_sha256"].items():
        _verify_source(resolve_path(relative), expected)
    for name, expected in contract["source_sha256"].items():
        _verify_source(resolve_path(config["sources"][name]), expected)
    if verify_test:
        for item in contract["deferred_test_sources"].values():
            _verify_source(resolve_path(item["path"]), item["expected_sha256"])
    return contract, output_root


def load_validation(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    sources = config["sources"]
    independent, independent_scores, labels = _score_matrix(
        read_jsonl(resolve_path(sources["phase50_validation_scores"]))
    )
    all28, all28_scores, _ = _score_matrix(
        read_jsonl(resolve_path(sources["phase51_all28_validation_scores"])),
        expected_variant="state_layer_all28",
    )
    random4, random4_scores, _ = _score_matrix(
        read_jsonl(resolve_path(sources["phase51_random4_validation_scores"])),
        expected_variant="state_layer_random4",
    )
    _assert_aligned(independent, all28)
    _assert_aligned(independent, random4)
    return independent, {
        "independent_sequential": independent_scores,
        "shared_all28": all28_scores,
        "shared_random4": random4_scores,
        "shared_fixed_l27": random4_scores,
    }, labels


def _annotate_sweep(candidate: str, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{field: ({"candidate": candidate, **row}).get(field, "") for field in SWEEP_FIELDS} for row in rows]


def _control_from_point(candidate: str, point: Mapping[str, Any]) -> dict[str, Any]:
    if candidate == "independent_sequential":
        return {
            "control_type": str(point["control_type"]),
            "alpha": point["control_value"],
            "layer_thresholds": json.loads(str(point["layer_thresholds_json"])),
        }
    return {
        "control_type": str(point["control_type"]),
        "threshold": float(point["threshold"]),
        "layers": [27] if candidate == "shared_fixed_l27" else list(range(28)),
    }


def _summary_row(candidate: str, point: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    candidate_config = next(item for item in config["candidates"] if item["name"] == candidate)
    return {
        "candidate": candidate,
        "display_name": candidate_config["display_name"],
        "control_type": point["control_type"],
        "control_value": point["control_value"],
        "correct_preservation": point["correct_preservation"],
        "wrong_detection_recall": point["wrong_detection_recall"],
        "failure_precision": point["failure_precision"],
        "triggered": point["triggered"],
        "correct_false_triggers": point["correct_false_triggers"],
        "wrong_detected": point["wrong_detected"],
        "utility": point["utility"],
        "utility_rate": point["utility_rate"],
        "median_first_trigger_layer": point["median_first_trigger_layer"],
        "mean_first_trigger_layer": point["mean_first_trigger_layer"],
        "simplicity_rank": candidate_config["simplicity_rank"],
        "candidate_order": CANDIDATES.index(candidate),
    }


def _dataset_rows(
    records: Sequence[Mapping[str, Any]], labels: np.ndarray, first_by_candidate: Mapping[str, np.ndarray], *, split: str
) -> list[dict[str, Any]]:
    output = []
    dataset_values = np.asarray([row["dataset"] for row in records])
    for candidate, first in first_by_candidate.items():
        for dataset in DATASETS:
            mask = dataset_values == dataset
            output.append({"split": split, "candidate": candidate, "dataset": dataset, **metrics_with_utility(labels[mask], first[mask])})
    return output


def select_validation(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_test=False)
    selected_path = output_root / "selected_winner.json"
    if selected_path.exists():
        raise RuntimeError("validation winner is already frozen")
    config = contract["static_config"]
    records, scores, labels = load_validation(config)
    sweeps = {
        "independent_sequential": independent_alpha_sweep(scores["independent_sequential"], labels),
        "shared_all28": shared_threshold_sweep(scores["shared_all28"], labels, layers=range(28)),
        "shared_random4": shared_threshold_sweep(scores["shared_random4"], labels, layers=range(28)),
        "shared_fixed_l27": shared_threshold_sweep(scores["shared_fixed_l27"], labels, layers=[27]),
    }
    sweep_root = output_root / "validation_threshold_sweeps"
    summaries = []
    points: dict[str, dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}
    first_by_candidate = {}
    for candidate in CANDIDATES:
        annotated = _annotate_sweep(candidate, sweeps[candidate])
        atomic_csv(sweep_root / f"{candidate}.csv", annotated)
        point = select_operating_point(
            sweeps[candidate],
            minimum_precision=float(config["selection"]["minimum_failure_precision"]),
        )
        points[candidate] = point
        controls[candidate] = _control_from_point(candidate, point)
        summaries.append(_summary_row(candidate, point, config))
        first_by_candidate[candidate] = apply_control(candidate, scores[candidate], controls[candidate])

    winner, ranking = select_winner(summaries)
    atomic_csv(output_root / "validation_summary.csv", ranking)
    reference_rows = []
    for candidate in CANDIDATES:
        for target in config["selection"]["reference_preservation_targets"]:
            point = reference_operating_point(sweeps[candidate], target_preservation=float(target))
            reference_rows.append(
                {
                    "candidate": candidate,
                    "target_preservation": target,
                    "absolute_preservation_distance": abs(float(point["correct_preservation"]) - float(target)),
                    **{key: point[key] for key in SWEEP_FIELDS if key != "candidate"},
                }
            )
    atomic_csv(output_root / "reference_operating_points.csv", reference_rows)
    atomic_csv(
        output_root / "dataset_breakdown.validation.csv",
        _dataset_rows(records, labels, first_by_candidate, split="validation"),
    )
    selection = {
        "schema_version": "stage1_gate_winner_selection_v1",
        "contract_sha256": contract["contract_sha256"],
        "test_evaluated": False,
        "selection_split": "validation",
        "selection_rule": config["selection"],
        "winner": winner,
        "runner_up": ranking[1],
        "ranking": ranking,
        "candidate_controls": controls,
        "candidate_validation_points": points,
        "validation_source_sha256": {
            name: contract["source_sha256"][name]
            for name in (
                "phase50_validation_scores",
                "phase51_all28_validation_scores",
                "phase51_random4_validation_scores",
            )
        },
        "test_source_files_read_by_selection_command": False,
        "limited_pre_freeze_source_exposure": contract["test_access_qualification"]["pre_freeze_deviation"],
    }
    write_once_json(selected_path, selection)
    print(
        json.dumps(
            {
                "passed": True,
                "winner": winner["candidate"],
                "winner_utility_rate": winner["utility_rate"],
                "runner_up": ranking[1]["candidate"],
                "runner_up_utility_rate": ranking[1]["utility_rate"],
                "utility_margin": float(winner["utility_rate"]) - float(ranking[1]["utility_rate"]),
            }
        )
    )


def load_test(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    sources = config["sources"]
    independent, independent_scores, labels = _score_matrix(
        read_jsonl(resolve_path(sources["phase50_test_scores"]))
    )
    combined = read_jsonl(resolve_path(sources["phase51_test_scores"]))
    by_variant = {
        variant: [row for row in combined if row.get("variant") == variant]
        for variant in ("state_layer_all28", "state_layer_random4")
    }
    if len(combined) != 1600 or any(len(rows) != 800 for rows in by_variant.values()):
        raise ValueError("Phase-51 combined test scores are incomplete or contain extra variants")
    all28, all28_scores, _ = _score_matrix(by_variant["state_layer_all28"], expected_variant="state_layer_all28")
    random4, random4_scores, _ = _score_matrix(by_variant["state_layer_random4"], expected_variant="state_layer_random4")
    _assert_aligned(independent, all28)
    _assert_aligned(independent, random4)
    return independent, {
        "independent_sequential": independent_scores,
        "shared_all28": all28_scores,
        "shared_random4": random4_scores,
        "shared_fixed_l27": random4_scores,
    }, labels


def _plot_results(
    output_root: Path,
    selection: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    test_first: Mapping[str, np.ndarray],
) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    sweeps = {candidate: read_csv(output_root / f"validation_threshold_sweeps/{candidate}.csv") for candidate in CANDIDATES}
    labels_by_candidate = {row["candidate"]: row["display_name"] for row in selection["ranking"]}

    fig, ax = plt.subplots(figsize=(7, 5))
    for candidate, rows in sweeps.items():
        ax.plot([float(row["correct_preservation"]) for row in rows], [float(row["wrong_detection_recall"]) for row in rows], label=labels_by_candidate[candidate], alpha=0.8)
    ax.set(xlabel="Correct preservation", ylabel="Wrong detection recall", title="Validation gate Pareto curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "gate_pareto_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for candidate, rows in sweeps.items():
        ax.plot([float(row["wrong_detection_recall"]) for row in rows], [float(row["failure_precision"]) for row in rows], label=labels_by_candidate[candidate], alpha=0.8)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Wrong detection recall", ylabel="Failure precision", title="Validation gate precision-recall")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "gate_precision_vs_recall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for candidate, rows in sweeps.items():
        ax.plot([float(row["correct_preservation"]) for row in rows], [float(row["utility_rate"]) for row in rows], label=labels_by_candidate[candidate], alpha=0.8)
        point = selection["candidate_validation_points"][candidate]
        ax.scatter(float(point["correct_preservation"]), float(point["utility_rate"]), s=28)
    ax.set(xlabel="Correct preservation", ylabel="Utility rate", title="Validation utility vs preservation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "gate_utility_vs_preservation.png", dpi=180)
    plt.close(fig)

    winner = selection["winner"]["candidate"]
    first = test_first[winner]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 28.5, 1)
    ax.hist(first[np.logical_and(labels == 0, first >= 0)], bins=bins, alpha=0.65, label="Correct false triggers")
    ax.hist(first[np.logical_and(labels == 1, first >= 0)], bins=bins, alpha=0.65, label="Wrong detected")
    ax.set(xlabel="First trigger layer", ylabel="Samples", title=f"Winner test trigger layers: {labels_by_candidate[winner]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "winner_trigger_layer_distribution.png", dpi=180)
    plt.close(fig)

    dataset_values = np.asarray([row["dataset"] for row in records])
    metrics = []
    for dataset in DATASETS:
        mask = dataset_values == dataset
        metrics.append(metrics_with_utility(labels[mask], first[mask]))
    x = np.arange(len(DATASETS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width, [row["correct_preservation"] for row in metrics], width, label="Preservation")
    ax.bar(x, [row["wrong_detection_recall"] for row in metrics], width, label="Wrong recall")
    ax.bar(x + width, [row["failure_precision"] for row in metrics], width, label="Precision")
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 1.05)
    ax.set(title="Winner test performance by dataset", ylabel="Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "dataset_winner_breakdown.png", dpi=180)
    plt.close(fig)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.4f}"


def evaluate_test(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_test=True)
    selection = read_json(output_root / "selected_winner.json")
    if selection.get("contract_sha256") != contract["contract_sha256"] or selection.get("test_evaluated") is not False:
        raise RuntimeError("validation selection is missing or incompatible")
    if (output_root / "test_completion.json").exists():
        raise RuntimeError("test comparison is already finalized")
    records, scores, labels = load_test(contract["static_config"])
    first_by_candidate = {
        candidate: apply_control(candidate, scores[candidate], selection["candidate_controls"][candidate])
        for candidate in CANDIDATES
    }
    test_rows = []
    for candidate in CANDIDATES:
        metrics = metrics_with_utility(labels, first_by_candidate[candidate])
        test_rows.append({"candidate": candidate, "selected_on_validation": candidate == selection["winner"]["candidate"], **metrics})
    atomic_csv(output_root / "test_comparison.csv", test_rows)
    dataset_rows = read_csv(output_root / "dataset_breakdown.validation.csv")
    dataset_rows.extend(_dataset_rows(records, labels, first_by_candidate, split="test"))
    atomic_csv(output_root / "dataset_breakdown.csv", dataset_rows)

    winner = selection["winner"]["candidate"]
    alternative = selection["runner_up"]["candidate"]
    bootstrap = {
        "schema_version": "stage1_gate_winner_paired_bootstrap_v1",
        "contract_sha256": contract["contract_sha256"],
        "split": "test",
        "winner": winner,
        "alternative": alternative,
        **paired_bootstrap(
            labels,
            first_by_candidate[winner],
            first_by_candidate[alternative],
            replicates=int(contract["static_config"]["bootstrap"]["replicates"]),
            seed=int(contract["static_config"]["bootstrap"]["seed"]),
            confidence=float(contract["static_config"]["bootstrap"]["confidence"]),
        ),
    }
    atomic_json(output_root / "bootstrap_comparison.json", bootstrap)
    _plot_results(output_root, selection, records, labels, first_by_candidate)

    test_by_name = {row["candidate"]: row for row in test_rows}
    validation_table = "\n".join(
        f"| {row['display_name']} | {_fmt(row['correct_preservation'])} | {_fmt(row['wrong_detection_recall'])} | {_fmt(row['failure_precision'])} | {_fmt(row['utility_rate'])} | {_fmt(row['median_first_trigger_layer'])} |"
        for row in selection["ranking"]
    )
    test_table = "\n".join(
        f"| {row['candidate']} | {_fmt(row['correct_preservation'])} | {_fmt(row['wrong_detection_recall'])} | {_fmt(row['failure_precision'])} | {_fmt(row['utility_rate'])} | {_fmt(row['median_first_trigger_layer'])} |"
        for row in test_rows
    )
    winner_test = test_by_name[winner]
    winner_control = selection["candidate_controls"][winner]
    winner_datasets = [row for row in dataset_rows if row["split"] == "test" and row["candidate"] == winner]
    dataset_text = "\n".join(
        f"| {row['dataset']} | {_fmt(row['correct_preservation'])} | {_fmt(row['wrong_detection_recall'])} | {_fmt(row['failure_precision'])} | {_fmt(row['utility_rate'])} |"
        for row in winner_datasets
    )
    ci_u = bootstrap["winner_minus_alternative_utility_rate"]
    ci_r = bootstrap["winner_minus_alternative_wrong_recall"]
    fixed = test_by_name["shared_fixed_l27"]
    summary = f"""# Stage-1 gate winner and threshold selection

Contract: `{contract['contract_sha256']}`

## Validation selection

| Candidate | Correct preservation | Wrong recall | Failure precision | Utility rate | Median trigger |
|---|---:|---:|---:|---:|---:|
{validation_table}

The frozen validation winner is **{winner}**, using `{json.dumps(winner_control, sort_keys=True)}`. It maximized validation utility rate subject to failure precision >= 90%. The frozen runner-up is **{alternative}**.

## Selection-held-out test confirmation

| Candidate | Correct preservation | Wrong recall | Failure precision | Utility rate | Median trigger |
|---|---:|---:|---:|---:|---:|
{test_table}

The winner's test preservation was {_fmt(winner_test['correct_preservation'])}, wrong recall {_fmt(winner_test['wrong_detection_recall'])}, precision {_fmt(winner_test['failure_precision'])}, and utility rate {_fmt(winner_test['utility_rate'])}. Its validation-to-test utility drift was {float(winner_test['utility_rate']) - float(selection['winner']['utility_rate']):+.4f}.

Against the validation runner-up, the paired 95% bootstrap interval for test utility-rate difference was [{ci_u['lower']:.4f}, {ci_u['upper']:.4f}] (observed {ci_u['observed']:.4f}); for wrong-recall difference it was [{ci_r['lower']:.4f}, {ci_r['upper']:.4f}] (observed {ci_r['observed']:.4f}).

The fixed-L27 control had test utility rate {_fmt(fixed['utility_rate'])} versus {_fmt(winner_test['utility_rate'])} for the winner; this is the direct evidence on whether sequential depth adds value under the selected operating points.

## Winner by dataset on test

| Dataset | Correct preservation | Wrong recall | Failure precision | Utility rate |
|---|---:|---:|---:|---:|
{dataset_text}

## Final questions

1. **Winner and threshold:** `{winner}` with the frozen control above.
2. **Test confirmation:** preservation {_fmt(winner_test['correct_preservation'])}, wrong recall {_fmt(winner_test['wrong_detection_recall'])}, precision {_fmt(winner_test['failure_precision'])}, utility {_fmt(winner_test['utility_rate'])}.
3. **Calibration stability:** validation-to-test utility drift was {float(winner_test['utility_rate']) - float(selection['winner']['utility_rate']):+.4f}; the complete comparison is in `test_comparison.csv`.
4. **Sequential versus fixed:** winner utility {_fmt(winner_test['utility_rate'])}; fixed-L27 utility {_fmt(fixed['utility_rate'])}.
5. **Dataset dominance:** the per-dataset table above shows whether the aggregate is concentrated in one task.
6. **Bootstrap uncertainty:** utility and recall intervals are reported above and frozen in `bootstrap_comparison.json`.
7. **Carry-forward:** only `{winner}` and its frozen validation-selected control should be used in the next separately authorized treatment experiment. No treatment experiment was run here.

## Qualification

This is selection-held-out within Phase 52, not historically unopened: Phase-50/51 test trajectories already existed. Additionally, five leading Phase-51 test rows were printed during format inspection before the winner freeze, without computing any aggregate or using them for selection. This limits the strictness of the holdout claim but does not alter the deterministic validation-only winner computation.
"""
    _atomic_bytes(output_root / "decision_summary.md", summary.encode())
    completion = {
        "schema_version": "stage1_gate_winner_test_completion_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "winner_frozen_before_test_command": True,
        "winner": winner,
        "alternative": alternative,
        "test_records": len(records),
        "test_source_sha256": {
            name: item["expected_sha256"] for name, item in contract["deferred_test_sources"].items()
        },
        "selection_unchanged_after_test": True,
    }
    atomic_json(output_root / "test_completion.json", completion)
    required = [
        "protocol.md",
        "frozen_protocol.json",
        "preparation_audit.json",
        *[f"validation_threshold_sweeps/{candidate}.csv" for candidate in CANDIDATES],
        "validation_summary.csv",
        "reference_operating_points.csv",
        "selected_winner.json",
        "test_comparison.csv",
        "dataset_breakdown.csv",
        "bootstrap_comparison.json",
        "figures/gate_pareto_curves.png",
        "figures/gate_precision_vs_recall.png",
        "figures/gate_utility_vs_preservation.png",
        "figures/winner_trigger_layer_distribution.png",
        "figures/dataset_winner_breakdown.png",
        "decision_summary.md",
        "test_completion.json",
    ]
    atomic_json(
        output_root / "artifact_manifest.json",
        {
            "schema_version": "stage1_gate_winner_selection_artifact_manifest_v1",
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "winner": winner,
            "required_file_count": len(required),
            "required_files": {relative: file_sha256(output_root / relative) for relative in required},
            "test_selection_held_out": True,
            "strictly_unopened_test": False,
        },
    )
    print(json.dumps({"passed": True, "winner": winner, "test_utility_rate": winner_test["utility_rate"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "select-validation", "evaluate-test"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "select-validation":
        select_validation(config_path)
    else:
        evaluate_test(config_path)


if __name__ == "__main__":
    main()
