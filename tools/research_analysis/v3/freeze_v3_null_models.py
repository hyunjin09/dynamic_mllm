from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from nulls.joint_four_action import (
    JointPathCovariance,
    PairedGeometryMetadata,
    _decode_path,
    _path_scores,
    fit_joint_path_covariance,
    fit_paired_donor_calipers,
    generate_joint_path_null,
    generate_paired_isotropic_null,
    generate_paired_real_null,
    paired_geometry_distance,
    search_budget_cells,
    select_paired_donors,
)
from nulls.structured_read import FixedGridCovariance, map_rows


LAYERS = [0, 4, 8, 12, 16, 20, 24]
ACTIONS = ["IGNORE", "READ_ONLY", "WRITE_ONLY"]
GRID_ROWS = 32
VARIANCE_CANDIDATES = [0.85, 0.90, 0.95]
SHRINKAGE_CANDIDATES = [0.05, 0.10, 0.20]
CV_FOLDS = 5
DONOR_COUNT = 8
TIGHT_DONOR_CALIPER = 1.5
LOCAL_REPAIR_MAX_CALIPER = 1.6
LOCAL_REPAIR_MAX_TAIL_FRACTION = 0.01
DONOR_SEED = 2026080607
COVARIANCE_SEED = 2026080608
ISOTROPIC_SEED = 2026080612
PRECISION_BOOTSTRAP_SEED = 2026080609
MAX_PRECISION_DRAWS = 128
DRAW_CANDIDATES = [4, 8, 16, 32, 64]
JOINT_SCORE_FIDELITY_DRAWS = 32768
JOINT_SCORE_COVARIANCE_RELATIVE_TOLERANCE = 0.15
FINAL_NATIVE_SUBSPACE_RELATIVE_TOLERANCE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze v3 null models from Stage B geometry.")
    parser.add_argument(
        "--geometry-root", default="artifacts/v3_null_calibration/read_write_geometry_v1"
    )
    parser.add_argument(
        "--covariance-root", default="artifacts/v3_null_calibration/joint_covariance_model_v1"
    )
    parser.add_argument(
        "--donor-root", default="artifacts/v3_null_calibration/paired_donor_index_v1"
    )
    parser.add_argument(
        "--coverage-output", default="outputs/v3_preflight/donor_coverage_audit_v1.json"
    )
    parser.add_argument(
        "--precision-output", default="outputs/v3_preflight/null_draw_precision_v1.json"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(base: int, *parts: Any) -> int:
    material = ":".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2**63 - 1)


def stable_fold(sample_id: str) -> int:
    return derived_seed(2026080610, sample_id) % CV_FOLDS


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_geometry(root: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["sample_count"] != 400 or manifest["sample_layer_count"] != 2800:
        raise RuntimeError("Geometry manifest is incomplete")
    summaries = {
        (row["sample_id"], int(row["layer"])): row
        for row in (json.loads(line) for line in (root / "geometry.jsonl").read_text().splitlines())
    }
    tensors = {}
    for row in json.loads((root / "tensor_index.json").read_text()):
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Tensor checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensors[payload["sample_id"]] = payload
    if len(tensors) != 400 or len(summaries) != 2800:
        raise RuntimeError("Loaded geometry cardinality mismatch")
    return tensors, summaries


def path_payload(path: FixedGridCovariance) -> dict[str, Any]:
    return {
        "mean": path.mean.detach().cpu(),
        "basis": path.basis.detach().cpu(),
        "eigenvalues": path.eigenvalues.detach().cpu(),
        "rank": path.rank,
        "explained_variance": path.explained_variance,
        "grid_rows": path.grid_rows,
        "hidden_size": path.hidden_size,
        "calibration_samples": path.calibration_samples,
        "variance_target": path.variance_target,
        "eigen_shrinkage": path.eigen_shrinkage,
    }


def fit_payload(fit: JointPathCovariance) -> dict[str, Any]:
    return {
        "read": path_payload(fit.read),
        "write": path_payload(fit.write),
        "score_mean": fit.score_mean.detach().cpu(),
        "score_covariance": fit.score_covariance.detach().cpu(),
        "score_factor": fit.score_factor.detach().cpu(),
        "joint_shrinkage": fit.joint_shrinkage,
    }


def path_from_payload(payload: dict[str, Any], device: torch.device) -> FixedGridCovariance:
    return FixedGridCovariance(
        mean=payload["mean"].to(device),
        basis=payload["basis"].to(device),
        eigenvalues=payload["eigenvalues"].to(device),
        rank=int(payload["rank"]),
        explained_variance=float(payload["explained_variance"]),
        grid_rows=int(payload["grid_rows"]),
        hidden_size=int(payload["hidden_size"]),
        calibration_samples=int(payload["calibration_samples"]),
        variance_target=float(payload["variance_target"]),
        eigen_shrinkage=float(payload["eigen_shrinkage"]),
    )


def fit_from_payload(payload: dict[str, Any], device: torch.device) -> JointPathCovariance:
    return JointPathCovariance(
        read=path_from_payload(payload["read"], device),
        write=path_from_payload(payload["write"], device),
        score_mean=payload["score_mean"].to(device),
        score_covariance=payload["score_covariance"].to(device),
        score_factor=payload["score_factor"].to(device),
        joint_shrinkage=float(payload["joint_shrinkage"]),
    )


def reconstruction_error(residuals: list[torch.Tensor], fit: FixedGridCovariance) -> float:
    scores = _path_scores(residuals, fit)
    decoded = torch.stack([_decode_path(fit, score) for score in scores])
    actual = torch.stack([map_rows(item.float(), fit.grid_rows) for item in residuals])
    errors = (decoded - actual).flatten(1).norm(dim=1) / actual.flatten(1).norm(dim=1).clamp_min(1e-12)
    return float(errors.mean().item())


def joint_score_covariance_error(
    fit: JointPathCovariance, seed: int, draws: int
) -> float:
    """Monte Carlo fidelity of the joint standardized READ/WRITE score law."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        draws,
        fit.score_factor.shape[1],
        generator=generator,
        dtype=torch.float32,
    ).to(fit.score_factor.device)
    scores = fit.score_mean.float().unsqueeze(0) + noise @ fit.score_factor.float().T
    centered = scores - scores.mean(dim=0, keepdim=True)
    empirical = centered.T @ centered / (draws - 1)
    return float(
        ((empirical - fit.score_covariance.float()).norm()
         / fit.score_covariance.float().norm().clamp_min(1e-12)).item()
    )


def cv_candidate(
    sample_ids: list[str],
    read: list[torch.Tensor],
    write: list[torch.Tensor],
    variance_target: float,
    shrinkage: float,
) -> dict[str, Any]:
    fold_rows = []
    for fold in range(CV_FOLDS):
        train = [index for index, sample_id in enumerate(sample_ids) if stable_fold(sample_id) != fold]
        test = [index for index, sample_id in enumerate(sample_ids) if stable_fold(sample_id) == fold]
        if len(train) < 3 or not test:
            raise RuntimeError("Invalid deterministic calibration fold")
        fit = fit_joint_path_covariance(
            [read[index] for index in train],
            [write[index] for index in train],
            GRID_ROWS,
            variance_target,
            shrinkage,
            marginal_eigen_shrinkage=shrinkage,
        )
        test_read = [read[index] for index in test]
        test_write = [write[index] for index in test]
        read_scores = _path_scores(test_read, fit.read)
        write_scores = _path_scores(test_write, fit.write)
        scores = torch.cat([read_scores, write_scores], dim=1)
        centered = scores - fit.score_mean
        solved = torch.linalg.solve(fit.score_covariance, centered.transpose(0, 1)).transpose(0, 1)
        mahalanobis_per_dimension = float(
            (centered * solved).sum(dim=1).mean().item() / scores.shape[1]
        )
        condition = float(torch.linalg.cond(fit.score_covariance).item())
        fold_rows.append(
            {
                "fold": fold,
                "train": len(train),
                "test": len(test),
                "read_rank": fit.read.rank,
                "write_rank": fit.write.rank,
                "read_reconstruction_error": reconstruction_error(test_read, fit.read),
                "write_reconstruction_error": reconstruction_error(test_write, fit.write),
                "mahalanobis_per_dimension": mahalanobis_per_dimension,
                "condition_number": condition,
            }
        )
    recon = sum(
        row["read_reconstruction_error"] + row["write_reconstruction_error"]
        for row in fold_rows
    ) / (2 * len(fold_rows))
    mahal = sum(abs(row["mahalanobis_per_dimension"] - 1.0) for row in fold_rows) / len(fold_rows)
    condition_log = sum(math.log10(max(row["condition_number"], 1.0)) for row in fold_rows) / len(fold_rows)
    rank_fraction = sum(
        (row["read_rank"] + row["write_rank"]) / (2 * (row["train"] - 1))
        for row in fold_rows
    ) / len(fold_rows)
    score = recon + 0.05 * mahal + 0.01 * rank_fraction + 0.001 * condition_log
    return {
        "variance_target": variance_target,
        "eigenvalue_and_joint_shrinkage": shrinkage,
        "selection_score": score,
        "mean_reconstruction_error": recon,
        "mean_absolute_mahalanobis_calibration_error": mahal,
        "mean_log10_condition": condition_log,
        "mean_rank_fraction": rank_fraction,
        "folds": fold_rows,
    }


def metadata_from_summary(row: dict[str, Any]) -> PairedGeometryMetadata:
    return PairedGeometryMetadata(
        sample_id=row["sample_id"],
        image_id=row["image_id"],
        dataset=row["dataset"],
        layer=int(row["layer"]),
        read_norm=float(row["read"]["frobenius_norm"]),
        write_norm=float(row["write"]["frobenius_norm"]),
        read_rows=int(row["read_shape"][0]),
        write_rows=int(row["write_shape"][0]),
        image_tokens=int(row["image_tokens"]),
        prompt_tokens=int(row["prompt_tokens"]),
        read_scale_ratio=float(row["read_rmsnorm_scale_ratio"]),
        write_scale_ratio=float(row["write_rmsnorm_scale_ratio"]),
        read_row_cv=float(row["read"]["row_norm_cv"]),
        write_row_cv=float(row["write"]["row_norm_cv"]),
    )


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        str(q): float(torch.quantile(tensor, q).item())
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def bootstrap_ci(values: torch.Tensor, draws: int, seed: int) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(draws):
        indices = torch.randint(values.numel(), (values.numel(),), generator=generator)
        means.append(values[indices].mean())
    tensor = torch.stack(means)
    probabilities = torch.tensor(
        [0.025, 0.975], dtype=tensor.dtype, device=tensor.device
    )
    return tuple(float(item) for item in torch.quantile(tensor, probabilities))


def geometry_proxy(
    actual_read: torch.Tensor,
    actual_write: torch.Tensor,
    null_read: torch.Tensor,
    null_write: torch.Tensor,
) -> float:
    read_distance = float(
        (null_read - actual_read).norm().item() / max(float(actual_read.norm().item()), 1e-12)
    )
    write_distance = float(
        (null_write - actual_write).norm().item() / max(float(actual_write.norm().item()), 1e-12)
    )
    return max(read_distance, write_distance, math.sqrt((read_distance**2 + write_distance**2) / 2))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    geometry_root = Path(args.geometry_root)
    covariance_root = Path(args.covariance_root)
    donor_root = Path(args.donor_root)
    covariance_root.mkdir(parents=True, exist_ok=True)
    donor_root.mkdir(parents=True, exist_ok=True)
    tensors, summaries = load_geometry(geometry_root)
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for sample_id, layer in summaries:
        grouped[(summaries[(sample_id, layer)]["dataset"], layer)].append(sample_id)

    fitted: dict[tuple[str, int], JointPathCovariance] = {}
    model_paths = []
    cv_rows = []
    fidelity_rows = []
    for (dataset, layer), sample_ids in sorted(grouped.items()):
        sample_ids = sorted(sample_ids)
        read = [tensors[sample_id]["layers"][layer]["read"].float().to(device) for sample_id in sample_ids]
        write = [tensors[sample_id]["layers"][layer]["write"].float().to(device) for sample_id in sample_ids]
        candidates = []
        for variance_target in VARIANCE_CANDIDATES:
            for shrinkage in SHRINKAGE_CANDIDATES:
                candidates.append(
                    cv_candidate(sample_ids, read, write, variance_target, shrinkage)
                )
        candidates.sort(key=lambda row: (row["selection_score"], row["variance_target"], row["eigenvalue_and_joint_shrinkage"]))
        selected = candidates[0]
        fit = fit_joint_path_covariance(
            read,
            write,
            GRID_ROWS,
            selected["variance_target"],
            selected["eigenvalue_and_joint_shrinkage"],
            marginal_eigen_shrinkage=selected["eigenvalue_and_joint_shrinkage"],
        )
        fitted[(dataset, layer)] = fit
        model_path = covariance_root / f"{dataset}_layer_{layer:02d}.pt"
        torch.save(
            {
                "schema_version": "v3_joint_covariance_model_v1",
                "dataset": dataset,
                "layer": layer,
                "representation": "separate 32-row READ/WRITE PCA; joint standardized coordinate covariance",
                "fit": fit_payload(fit),
                "selected_cv": selected,
                "calibration_sample_ids": sample_ids,
                "seeds": {"base": COVARIANCE_SEED},
            },
            model_path,
        )
        reloaded = fit_from_payload(torch.load(model_path, map_location="cpu", weights_only=False)["fit"], device)
        target_index = derived_seed(COVARIANCE_SEED, dataset, layer, "fidelity") % len(sample_ids)
        target_read = read[target_index]
        target_write = write[target_index]
        read_norms = target_read.norm(dim=1)
        write_norms = target_write.norm(dim=1)
        generated = generate_joint_path_null(
            reloaded,
            read_norms,
            write_norms,
            derived_seed(COVARIANCE_SEED, dataset, layer, "reload"),
        )
        repeat = generate_joint_path_null(
            reloaded,
            read_norms,
            write_norms,
            derived_seed(COVARIANCE_SEED, dataset, layer, "reload"),
        )
        final_read_subspace_error = reconstruction_error([generated[0]], reloaded.read)
        final_write_subspace_error = reconstruction_error([generated[1]], reloaded.write)
        score_covariance_relative_error = joint_score_covariance_error(
            reloaded,
            derived_seed(COVARIANCE_SEED, dataset, layer, "joint-score-fidelity"),
            JOINT_SCORE_FIDELITY_DRAWS,
        )
        fidelity_rows.append(
            {
                "dataset": dataset,
                "layer": layer,
                "read_rank": fit.read.rank,
                "write_rank": fit.write.rank,
                "read_explained_variance": fit.read.explained_variance,
                "write_explained_variance": fit.write.explained_variance,
                "joint_condition_number": float(torch.linalg.cond(fit.score_covariance).item()),
                "read_row_norm_max_abs": float((generated[0].norm(dim=1) - read_norms).abs().max().item()),
                "write_row_norm_max_abs": float((generated[1].norm(dim=1) - write_norms).abs().max().item()),
                "final_native_read_subspace_relative_error": final_read_subspace_error,
                "final_native_write_subspace_relative_error": final_write_subspace_error,
                "joint_score_covariance_relative_error": score_covariance_relative_error,
                "reload_seed_repeat_exact": bool(torch.equal(generated[0], repeat[0]) and torch.equal(generated[1], repeat[1])),
            }
        )
        cv_rows.append({"dataset": dataset, "layer": layer, "selected": selected, "candidates": candidates})
        model_paths.append(model_path)
        print(f"fitted {dataset} layer {layer}", flush=True)

    metadata = [metadata_from_summary(row) for row in summaries.values()]
    calipers, coverage = fit_paired_donor_calipers(metadata, DONOR_COUNT)
    donor_rows = []
    for target in sorted(metadata, key=lambda item: (item.dataset, item.sample_id, item.layer)):
        caliper = calipers[(target.dataset, target.layer)]
        seed = derived_seed(DONOR_SEED, target.sample_id, target.layer)
        selected = select_paired_donors(target, metadata, DONOR_COUNT, seed, caliper)
        donor_rows.append(
            {
                "sample_id": target.sample_id,
                "image_id": target.image_id,
                "dataset": target.dataset,
                "layer": target.layer,
                "caliper": caliper,
                "tie_seed": seed,
                "donors": [
                    {
                        "sample_id": donor.sample_id,
                        "image_id": donor.image_id,
                        "distance": paired_geometry_distance(target, donor),
                    }
                    for donor in selected
                ],
            }
        )
    donor_index_path = donor_root / "paired_donor_index.jsonl"
    jsonl_write(donor_index_path, donor_rows)
    coverage_by_stratum = {}
    weak_targets = []
    stratum_gates = []
    for key, cap in sorted(calipers.items()):
        values = [
            row["distance_to_required_donor"]
            for row in coverage
            if row["dataset"] == key[0] and int(row["layer"]) == key[1]
        ]
        above_tight = sum(value > TIGHT_DONOR_CALIPER for value in values)
        tail_fraction = above_tight / len(values)
        local_repair = (
            cap <= LOCAL_REPAIR_MAX_CALIPER
            and tail_fraction <= LOCAL_REPAIR_MAX_TAIL_FRACTION
        )
        stratum_pass = cap <= TIGHT_DONOR_CALIPER or local_repair
        stratum_gates.append(stratum_pass)
        coverage_by_stratum[f"{key[0]}:layer_{key[1]}"] = {
            "minimum_caliper": cap,
            "required_eighth_distance_quantiles": quantiles(values),
            "targets_above_1_5": above_tight,
            "fraction_above_1_5": tail_fraction,
            "targets_above_2_0": sum(value > 2.0 for value in values),
            "classification": (
                "tight" if cap <= TIGHT_DONOR_CALIPER
                else "minimal_local_repair" if local_repair
                else "substantively_weak"
            ),
            "gate_pass": stratum_pass,
        }
        weak_targets.extend(row for row in coverage if row["dataset"] == key[0] and int(row["layer"]) == key[1] and row["distance_to_required_donor"] > TIGHT_DONOR_CALIPER)
    donor_gate = all(stratum_gates)
    coverage_payload = {
        "schema_version": "v3_paired_donor_coverage_v1",
        "outcome_blind": True,
        "terminal_answer_or_action_outcomes_used": False,
        "donor_count": DONOR_COUNT,
        "distance": "maximum multiplicative ratio across paired READ/WRITE norms, rows, image/prompt tokens, RMSNorm scale ratios, and row-norm CVs",
        "caliper_rule": "smallest dataset-layer leave-one-image-out cap covering the eighth nearest eligible donor for every calibration target",
        "calipers": {f"{dataset}:layer_{layer}": value for (dataset, layer), value in sorted(calipers.items())},
        "coverage_by_stratum": coverage_by_stratum,
        "weak_tail_definition": "eighth-nearest distance > 1.5",
        "substantive_weakening_rule": {
            "tight_cap": TIGHT_DONOR_CALIPER,
            "local_repair_max_cap": LOCAL_REPAIR_MAX_CALIPER,
            "local_repair_max_fraction_above_tight_cap": LOCAL_REPAIR_MAX_TAIL_FRACTION,
            "rule": "a stratum above 1.5 is acceptable only when its exact cap is <=1.6 and <=1% of targets exceed 1.5; otherwise stop for approval",
        },
        "target_coverage": sorted(
            coverage,
            key=lambda row: (row["dataset"], row["sample_id"], int(row["layer"])),
        ),
        "weak_targets": weak_targets,
        "gate_pass": donor_gate,
    }
    json_write(Path(args.coverage_output), coverage_payload)
    json_write(
        donor_root / "manifest.json",
        {
            "schema_version": "v3_paired_donor_manifest_v1",
            "calibration_pool": "400 inspected Stage B records, dataset/layer matched",
            "donor_count": DONOR_COUNT,
            "donor_seed": DONOR_SEED,
            "calipers": coverage_payload["calipers"],
            "index_sha256": sha256(donor_index_path),
            "coverage_sha256": sha256(Path(args.coverage_output)),
        },
    )

    precision_ids = {}
    for dataset in ("gqa", "textvqa"):
        dataset_ids = sorted(
            {sample_id for (sample_id, layer), row in summaries.items() if row["dataset"] == dataset},
            key=lambda sample_id: (derived_seed(2026080611, sample_id), sample_id),
        )
        precision_ids[dataset] = dataset_ids[:16]
    proxy_rows = []
    isotropic_proxy_rows = []
    donor_proxy_rows = []
    donor_lookup = {(row["sample_id"], row["layer"]): row for row in donor_rows}
    for dataset, sample_ids in precision_ids.items():
        for sample_id in sample_ids:
            actual_layers = {
                layer: {
                    key: value.float().to(device)
                    for key, value in tensors[sample_id]["layers"][layer].items()
                }
                for layer in LAYERS
            }
            per_draw = []
            isotropic_per_draw = []
            donor_draws = []
            for draw in range(MAX_PRECISION_DRAWS):
                cells = []
                isotropic_cells = []
                for layer in LAYERS:
                    actual = actual_layers[layer]
                    read = actual["read"]
                    write = actual["write"]
                    generated = generate_joint_path_null(
                        fitted[(dataset, layer)],
                        actual["read_row_norms"],
                        actual["write_row_norms"],
                        derived_seed(COVARIANCE_SEED, sample_id, layer, draw),
                    )
                    cells.append(geometry_proxy(read, write, *generated))
                    isotropic = generate_paired_isotropic_null(
                        int(read.shape[1]),
                        actual["read_row_norms"],
                        actual["write_row_norms"],
                        derived_seed(ISOTROPIC_SEED, sample_id, layer, draw),
                    )
                    isotropic_cells.append(geometry_proxy(read, write, *isotropic))
                per_draw.append(max(cells))
                isotropic_per_draw.append(max(isotropic_cells))
            for donor_position in range(DONOR_COUNT):
                cells = []
                for layer in LAYERS:
                    actual = actual_layers[layer]
                    donor_id = donor_lookup[(sample_id, layer)]["donors"][donor_position]["sample_id"]
                    donor = tensors[donor_id]["layers"][layer]
                    generated = generate_paired_real_null(
                        donor["read"].float().to(device),
                        donor["write"].float().to(device),
                        actual["read_row_norms"],
                        actual["write_row_norms"],
                    )
                    cells.append(
                        geometry_proxy(
                            actual["read"],
                            actual["write"],
                            *generated,
                        )
                    )
                donor_draws.append(max(cells))
            proxy_rows.append({"dataset": dataset, "sample_id": sample_id, "draws": per_draw})
            isotropic_proxy_rows.append(
                {"dataset": dataset, "sample_id": sample_id, "draws": isotropic_per_draw}
            )
            donor_proxy_rows.append({"dataset": dataset, "sample_id": sample_id, "draws": donor_draws})
            print(f"precision {dataset} {sample_id}", flush=True)

    def precision_grid(proxy: torch.Tensor, family: str) -> tuple[list[dict[str, Any]], int | None]:
        reference = proxy.mean(dim=1)
        reference_ci = bootstrap_ci(
            -reference, 1000, derived_seed(PRECISION_BOOTSTRAP_SEED, family, "reference")
        )
        candidates = []
        for count in DRAW_CANDIDATES:
            panels = []
            for start in range(0, MAX_PRECISION_DRAWS, count):
                stop = start + count
                if stop > MAX_PRECISION_DRAWS:
                    break
                estimate = proxy[:, start:stop].mean(dim=1)
                ci = bootstrap_ci(
                    -estimate,
                    1000,
                    derived_seed(PRECISION_BOOTSTRAP_SEED, family, count, start),
                )
                panels.append(
                    {
                        "start": start,
                        "mean_error": abs(float(estimate.mean().item() - reference.mean().item())),
                        "q95_error": abs(float(torch.quantile(estimate, 0.95).item() - torch.quantile(reference, 0.95).item())),
                        "paired_geometry_ci_endpoint_max_error": max(abs(ci[0] - reference_ci[0]), abs(ci[1] - reference_ci[1])),
                    }
                )
            passed = all(
                row["mean_error"] <= 0.01
                and row["q95_error"] <= 0.02
                and row["paired_geometry_ci_endpoint_max_error"] <= 0.015
                for row in panels
            )
            candidates.append({"draw_count": count, "panels": panels, "pass": passed})
        passing = [row["draw_count"] for row in candidates if row["pass"]]
        return candidates, min(passing) if passing else None

    proxy = torch.tensor([row["draws"] for row in proxy_rows], dtype=torch.float64)
    isotropic_proxy = torch.tensor(
        [row["draws"] for row in isotropic_proxy_rows], dtype=torch.float64
    )
    precision_candidates, covariance_draws = precision_grid(proxy, "joint_covariance")
    isotropic_precision_candidates, isotropic_draws = precision_grid(
        isotropic_proxy, "isotropic"
    )
    donor_proxy = torch.tensor([row["draws"] for row in donor_proxy_rows], dtype=torch.float64)
    donor_reference = donor_proxy.mean(dim=1)
    donor_prefix = []
    for count in (2, 4, 8):
        estimate = donor_proxy[:, :count].mean(dim=1)
        donor_prefix.append(
            {
                "draw_count": count,
                "mean_error_vs_all_eight": abs(float(estimate.mean().item() - donor_reference.mean().item())),
                "q95_error_vs_all_eight": abs(float(torch.quantile(estimate, 0.95).item() - torch.quantile(donor_reference, 0.95).item())),
            }
        )
    precision_payload = {
        "schema_version": "v3_null_draw_precision_v1",
        "outcome_blind": True,
        "terminal_answer_or_action_outcomes_used": False,
        "geometry_proxy": "maximum over the same 21 layer/action cells of normalized final-native residual-pair distance",
        "limitation": "geometry precision cannot establish terminal-score null stability; the draw count is frozen prospectively and may not be retuned after held-out scoring",
        "precision_sample_ids": precision_ids,
        "maximum_simulated_draws": MAX_PRECISION_DRAWS,
        "criteria": {"mean_error": 0.01, "upper_q95_error": 0.02, "paired_geometry_ci_endpoint_error": 0.015},
        "covariance_candidates": precision_candidates,
        "isotropic_candidates": isotropic_precision_candidates,
        "frozen_joint_covariance_draws": covariance_draws,
        "frozen_isotropic_draws": isotropic_draws,
        "real_donor_precision": donor_prefix,
        "frozen_real_donor_draws": DONOR_COUNT,
        "paired_draw_shared_across_actions_within_layer": True,
        "draws_independent_across_layers": True,
        "maximum_over_21": search_budget_cells(LAYERS, ACTIONS),
        "gate_pass": covariance_draws is not None and isotropic_draws is not None,
    }
    json_write(Path(args.precision_output), precision_payload)

    cv_path = covariance_root / "cross_validation.json"
    json_write(cv_path, {"folds": CV_FOLDS, "grid_rows": GRID_ROWS, "strata": cv_rows})
    fidelity_path = covariance_root / "fidelity_validation.json"
    json_write(
        fidelity_path,
        {
            "validation_draws": JOINT_SCORE_FIDELITY_DRAWS,
            "tolerances": {
                "joint_score_covariance_relative_error": JOINT_SCORE_COVARIANCE_RELATIVE_TOLERANCE,
                "final_native_subspace_relative_error": FINAL_NATIVE_SUBSPACE_RELATIVE_TOLERANCE,
                "row_norm_max_abs": 1e-4,
            },
            "rows": fidelity_rows,
            "gate_pass": all(
                row["joint_condition_number"] <= 1e5
                and row["read_row_norm_max_abs"] <= 1e-4
                and row["write_row_norm_max_abs"] <= 1e-4
                and row["final_native_read_subspace_relative_error"]
                <= FINAL_NATIVE_SUBSPACE_RELATIVE_TOLERANCE
                and row["final_native_write_subspace_relative_error"]
                <= FINAL_NATIVE_SUBSPACE_RELATIVE_TOLERANCE
                and row["joint_score_covariance_relative_error"]
                <= JOINT_SCORE_COVARIANCE_RELATIVE_TOLERANCE
                and row["reload_seed_repeat_exact"]
                for row in fidelity_rows
            ),
        },
    )
    model_manifest_path = covariance_root / "manifest.json"
    json_write(
        model_manifest_path,
        {
            "schema_version": "v3_joint_covariance_manifest_v1",
            "calibration_pool": "400 inspected Stage B records; 200 per dataset/layer",
            "layers": LAYERS,
            "grid_rows": GRID_ROWS,
            "cv_folds": CV_FOLDS,
            "variance_candidates": VARIANCE_CANDIDATES,
            "shrinkage_candidates": SHRINKAGE_CANDIDATES,
            "model_checksums": {str(path): sha256(path) for path in model_paths},
            "cross_validation_sha256": sha256(cv_path),
            "fidelity_sha256": sha256(fidelity_path),
            "precision_sha256": sha256(Path(args.precision_output)),
            "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        },
    )
    for root in (covariance_root, donor_root):
        paths = sorted(path for path in root.iterdir() if path.name != "SHA256SUMS")
        (root / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path}\n" for path in paths), encoding="utf-8"
        )
    gate = (
        donor_gate
        and precision_payload["gate_pass"]
        and json.loads(fidelity_path.read_text())["gate_pass"]
    )
    print(
        json.dumps(
            {
                "gate_pass": gate,
                "covariance_draws": covariance_draws,
                "donor_count": DONOR_COUNT,
                "max_caliper": max(calipers.values()),
                "model_manifest": str(model_manifest_path),
            }
        )
    )
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
