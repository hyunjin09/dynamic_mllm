from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nulls.joint_four_action import (
    JointPathCovariance,
    _decode_path,
    _path_scores,
    generate_joint_path_null,
    fit_joint_path_covariance,
)
from nulls.native_row_joint import (
    DirectionBasis,
    NativeRowJointModel,
    final_native_projection_error,
    fit_native_row_joint_model,
    generate_native_row_joint_null,
)
from nulls.structured_read import FixedGridCovariance, fit_fixed_grid_covariance, map_rows


LAYERS = [0, 4, 8, 12, 16, 20, 24]
GRID_ROWS = 32
VARIANCE_TARGET = 0.95
EIGEN_SHRINKAGE = 0.05
JOINT_SHRINKAGE = 0.05
NATIVE_MIN_GROUP = 32
NATIVE_C_ROWS_PER_SAMPLE = 8
NATIVE_C_VARIANCE_TARGET = 0.85
NATIVE_C_MAXIMUM_RANK = 512
NATIVE_C_POSITION_BINS = 8
CV_VALIDATION_FRACTION = 0.20
CV_SEED = 2026080703
FIT_SEED = 2026080704
FIDELITY_DRAWS = 32768
FINAL_NATIVE_TOLERANCE = 0.50
JOINT_COVARIANCE_TOLERANCE = 0.15
NORM_TOLERANCE = 1e-4
CONDITION_TOLERANCE = 1e5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare outcome-blind v3 covariance representations.")
    parser.add_argument(
        "--geometry-root", default="artifacts/v3_null_redesign/read_write_geometry_v2"
    )
    parser.add_argument(
        "--output",
        default="outputs/v3_null_redesign/covariance_representation_comparison.json",
    )
    parser.add_argument(
        "--model-root", default="artifacts/v3_null_redesign/joint_covariance_models_v2"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base: int, *parts: Any) -> int:
    value = ":".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def stable_validation(sample_id: str) -> bool:
    value = derived_seed(CV_SEED, sample_id) / (2**63 - 1)
    return value < CV_VALIDATION_FRACTION


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def fixed_grid_projection_errors(
    residuals: Sequence[torch.Tensor], fit: FixedGridCovariance
) -> tuple[list[float], float]:
    errors = []
    error_square = 0.0
    norm_square = 0.0
    for value in residuals:
        score = _path_scores([value], fit)[0]
        decoded = _decode_path(fit, score)
        actual = map_rows(value.float(), fit.grid_rows)
        error_square += float((decoded - actual).square().sum().item())
        norm_square += float(actual.square().sum().item())
        errors.append(
            float((decoded - actual).norm().item() / actual.norm().clamp_min(1e-12).item())
        )
    return errors, math.sqrt(error_square / max(norm_square, 1e-24))


def native_projection_errors(
    residuals: Sequence[torch.Tensor], fit: DirectionBasis
) -> tuple[list[float], float]:
    values = [final_native_projection_error(item, fit) for item in residuals]
    error_square = 0.0
    norm_square = 0.0
    for item, relative in zip(residuals, values):
        energy = float(item.float().square().sum().item())
        error_square += relative * relative * energy
        norm_square += energy
    return values, math.sqrt(error_square / max(norm_square, 1e-24))


def joint_mc_error(mean: torch.Tensor, covariance: torch.Tensor, factor: torch.Tensor, seed: int) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = factor.device
    sum_values = torch.zeros_like(mean, dtype=torch.float64, device=device)
    cross = torch.zeros_like(covariance, dtype=torch.float64, device=device)
    batch = 2048
    remaining = FIDELITY_DRAWS
    while remaining:
        current = min(batch, remaining)
        noise = torch.randn(current, factor.shape[1], generator=generator, device="cpu").to(device)
        values = mean.float().unsqueeze(0) + noise @ factor.float().T
        double = values.double()
        sum_values += double.sum(dim=0)
        cross += double.T @ double
        remaining -= current
    empirical_mean = sum_values / FIDELITY_DRAWS
    empirical = (cross - FIDELITY_DRAWS * empirical_mean[:, None] * empirical_mean[None, :]) / (FIDELITY_DRAWS - 1)
    return float(
        ((empirical.float() - covariance.float()).norm() / covariance.float().norm().clamp_min(1e-12)).item()
    )


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


def fixed_model_payload(model: JointPathCovariance) -> dict[str, Any]:
    return {
        "read": path_payload(model.read),
        "write": path_payload(model.write),
        "score_mean": model.score_mean.detach().cpu(),
        "score_covariance": model.score_covariance.detach().cpu(),
        "score_factor": model.score_factor.detach().cpu(),
        "joint_shrinkage": model.joint_shrinkage,
    }


def direction_payload(path: DirectionBasis) -> dict[str, Any]:
    return {
        "basis": path.basis.detach().cpu(),
        "rank": path.rank,
        "hidden_size": path.hidden_size,
        "variance_target": path.variance_target,
        "sampled_explained_variance": path.sampled_explained_variance,
        "sampled_row_count": path.sampled_row_count,
        "target_reached": path.target_reached,
    }


def native_model_payload(model: NativeRowJointModel) -> dict[str, Any]:
    return {
        "read": direction_payload(model.read),
        "write": direction_payload(model.write),
        "joint_mean": model.joint_mean.detach().cpu(),
        "joint_covariance": model.joint_covariance.detach().cpu(),
        "joint_factor": model.joint_factor.detach().cpu(),
        "read_within_variance": model.read_within_variance.detach().cpu(),
        "write_within_variance": model.write_within_variance.detach().cpu(),
        "position_bins": model.position_bins,
        "joint_shrinkage": model.joint_shrinkage,
        "calibration_samples": model.calibration_samples,
    }


def custom_native_shape_joint(
    read: Sequence[torch.Tensor], write: Sequence[torch.Tensor]
) -> JointPathCovariance:
    read_fit = fit_fixed_grid_covariance(
        read, int(read[0].shape[0]), VARIANCE_TARGET, EIGEN_SHRINKAGE
    )
    write_fit = fit_fixed_grid_covariance(
        write, int(write[0].shape[0]), VARIANCE_TARGET, EIGEN_SHRINKAGE
    )
    scores = torch.cat([_path_scores(read, read_fit), _path_scores(write, write_fit)], dim=1)
    mean = scores.mean(dim=0)
    centered = scores - mean
    covariance = centered.T @ centered / (scores.shape[0] - 1)
    identity = torch.eye(covariance.shape[0], device=covariance.device)
    covariance = (1 - JOINT_SHRINKAGE) * covariance + JOINT_SHRINKAGE * identity
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(1e-6)
    covariance = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
    factor = eigenvectors @ torch.diag(eigenvalues.sqrt())
    return JointPathCovariance(read_fit, write_fit, mean, covariance, factor, JOINT_SHRINKAGE)


def fit_a(
    dataset: str,
    layer: int,
    sample_ids: list[str],
    read: list[torch.Tensor],
    write: list[torch.Tensor],
    model_root: Path,
) -> dict[str, Any]:
    train = [index for index, item in enumerate(sample_ids) if not stable_validation(item)]
    test = [index for index, item in enumerate(sample_ids) if stable_validation(item)]
    fit = fit_joint_path_covariance(
        [read[index] for index in train],
        [write[index] for index in train],
        GRID_ROWS,
        VARIANCE_TARGET,
        JOINT_SHRINKAGE,
        marginal_eigen_shrinkage=EIGEN_SHRINKAGE,
    )
    read_errors, read_pooled = fixed_grid_projection_errors([read[index] for index in test], fit.read)
    write_errors, write_pooled = fixed_grid_projection_errors([write[index] for index in test], fit.write)
    final = fit_joint_path_covariance(
        read,
        write,
        GRID_ROWS,
        VARIANCE_TARGET,
        JOINT_SHRINKAGE,
        marginal_eigen_shrinkage=EIGEN_SHRINKAGE,
    )
    target_index = derived_seed(FIT_SEED, "a", dataset, layer, "target") % len(read)
    read_norms = read[target_index].norm(dim=1)
    write_norms = write[target_index].norm(dim=1)
    generated = generate_joint_path_null(
        final, read_norms, write_norms, derived_seed(FIT_SEED, "a", dataset, layer, "draw")
    )
    repeat = generate_joint_path_null(
        final, read_norms, write_norms, derived_seed(FIT_SEED, "a", dataset, layer, "draw")
    )
    generated_read_errors, generated_read_pooled = fixed_grid_projection_errors([generated[0]], final.read)
    generated_write_errors, generated_write_pooled = fixed_grid_projection_errors([generated[1]], final.write)
    model_path = model_root / "representation_a_32_row" / f"{dataset}_layer_{layer:02d}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "v3_null_redesign_representation_a_v1",
            "dataset": dataset,
            "layer": layer,
            "sample_ids": sample_ids,
            "fit": fixed_model_payload(final),
        },
        model_path,
    )
    row = {
        "dataset": dataset,
        "layer": layer,
        "train_count": len(train),
        "validation_count": len(test),
        "read_rank": final.read.rank,
        "write_rank": final.write.rank,
        "read_cv": {**quantiles(read_errors), "pooled_relative_error": read_pooled},
        "write_cv": {**quantiles(write_errors), "pooled_relative_error": write_pooled},
        "final_native_read_subspace_relative_error": generated_read_pooled,
        "final_native_write_subspace_relative_error": generated_write_pooled,
        "joint_condition_number": float(torch.linalg.cond(final.score_covariance).item()),
        "joint_covariance_relative_error": joint_mc_error(
            final.score_mean,
            final.score_covariance,
            final.score_factor,
            derived_seed(FIT_SEED, "a", dataset, layer, "mc"),
        ),
        "read_row_norm_max_abs": float((generated[0].norm(dim=1) - read_norms).abs().max().item()),
        "write_row_norm_max_abs": float((generated[1].norm(dim=1) - write_norms).abs().max().item()),
        "deterministic_repeat_exact": bool(torch.equal(generated[0], repeat[0]) and torch.equal(generated[1], repeat[1])),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
    }
    row["gate_pass"] = (
        read_pooled <= FINAL_NATIVE_TOLERANCE
        and write_pooled <= FINAL_NATIVE_TOLERANCE
        and covariance_row_pass(row)
    )
    return row


def fit_b(
    dataset: str,
    layer: int,
    sample_ids: list[str],
    read: list[torch.Tensor],
    write: list[torch.Tensor],
) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (read_value, write_value) in enumerate(zip(read, write)):
        groups[(int(read_value.shape[0]), int(write_value.shape[0]))].append(index)
    populated = {key: indices for key, indices in groups.items() if len(indices) >= NATIVE_MIN_GROUP}
    group_rows = []
    for (read_rows, write_rows), indices in sorted(populated.items()):
        train = [index for index in indices if not stable_validation(sample_ids[index])]
        test = [index for index in indices if stable_validation(sample_ids[index])]
        if len(train) < 3 or not test:
            group_rows.append(
                {
                    "read_rows": read_rows,
                    "write_rows": write_rows,
                    "sample_count": len(indices),
                    "gate_pass": False,
                    "failure": "deterministic train/validation split insufficient",
                }
            )
            continue
        fit = custom_native_shape_joint(
            [read[index] for index in train], [write[index] for index in train]
        )
        read_errors, read_pooled = fixed_grid_projection_errors([read[index] for index in test], fit.read)
        write_errors, write_pooled = fixed_grid_projection_errors([write[index] for index in test], fit.write)
        row = {
            "read_rows": read_rows,
            "write_rows": write_rows,
            "sample_count": len(indices),
            "train_count": len(train),
            "validation_count": len(test),
            "read_rank": fit.read.rank,
            "write_rank": fit.write.rank,
            "read_cv": {**quantiles(read_errors), "pooled_relative_error": read_pooled},
            "write_cv": {**quantiles(write_errors), "pooled_relative_error": write_pooled},
            "joint_condition_number": float(torch.linalg.cond(fit.score_covariance).item()),
            "joint_covariance_relative_error": joint_mc_error(
                fit.score_mean,
                fit.score_covariance,
                fit.score_factor,
                derived_seed(FIT_SEED, "b", dataset, layer, read_rows, write_rows),
            ),
        }
        row["gate_pass"] = (
            read_pooled <= FINAL_NATIVE_TOLERANCE
            and write_pooled <= FINAL_NATIVE_TOLERANCE
            and row["joint_condition_number"] <= CONDITION_TOLERANCE
            and row["joint_covariance_relative_error"] <= JOINT_COVARIANCE_TOLERANCE
        )
        group_rows.append(row)
        del fit
        torch.cuda.empty_cache()
    covered = sum(len(indices) for indices in populated.values())
    return {
        "dataset": dataset,
        "layer": layer,
        "exact_shape_group_count": len(groups),
        "adequately_populated_group_count": len(populated),
        "minimum_group_size": NATIVE_MIN_GROUP,
        "covered_sample_count": covered,
        "coverage_fraction": covered / len(sample_ids),
        "shape_counts": {
            f"read_{key[0]}:write_{key[1]}": len(value)
            for key, value in sorted(groups.items())
        },
        "modeled_groups": group_rows,
        "gate_pass": covered == len(sample_ids) and all(row["gate_pass"] for row in group_rows),
    }


def fit_c(
    dataset: str,
    layer: int,
    sample_ids: list[str],
    read: list[torch.Tensor],
    write: list[torch.Tensor],
    model_root: Path,
    maximum_rank: int = NATIVE_C_MAXIMUM_RANK,
) -> dict[str, Any]:
    train = [index for index, item in enumerate(sample_ids) if not stable_validation(item)]
    test = [index for index, item in enumerate(sample_ids) if stable_validation(item)]
    fit = fit_native_row_joint_model(
        [read[index] for index in train],
        [write[index] for index in train],
        NATIVE_C_ROWS_PER_SAMPLE,
        NATIVE_C_VARIANCE_TARGET,
        maximum_rank,
        NATIVE_C_POSITION_BINS,
        JOINT_SHRINKAGE,
        derived_seed(FIT_SEED, "c", dataset, layer, "cv"),
    )
    read_errors, read_pooled = native_projection_errors([read[index] for index in test], fit.read)
    write_errors, write_pooled = native_projection_errors([write[index] for index in test], fit.write)
    final = fit_native_row_joint_model(
        read,
        write,
        NATIVE_C_ROWS_PER_SAMPLE,
        NATIVE_C_VARIANCE_TARGET,
        maximum_rank,
        NATIVE_C_POSITION_BINS,
        JOINT_SHRINKAGE,
        derived_seed(FIT_SEED, "c", dataset, layer, "final"),
    )
    target_index = derived_seed(FIT_SEED, "c", dataset, layer, "target") % len(read)
    read_norms = read[target_index].norm(dim=1)
    write_norms = write[target_index].norm(dim=1)
    generated = generate_native_row_joint_null(
        final, read_norms, write_norms, derived_seed(FIT_SEED, "c", dataset, layer, "draw")
    )
    repeat = generate_native_row_joint_null(
        final, read_norms, write_norms, derived_seed(FIT_SEED, "c", dataset, layer, "draw")
    )
    model_path = model_root / "representation_c_native_row" / f"{dataset}_layer_{layer:02d}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "v3_null_redesign_representation_c_v1",
            "dataset": dataset,
            "layer": layer,
            "sample_ids": sample_ids,
            "fit": native_model_payload(final),
        },
        model_path,
    )
    row = {
        "dataset": dataset,
        "layer": layer,
        "train_count": len(train),
        "validation_count": len(test),
        "read_rank": final.read.rank,
        "write_rank": final.write.rank,
        "read_sampled_explained_variance": final.read.sampled_explained_variance,
        "write_sampled_explained_variance": final.write.sampled_explained_variance,
        "read_variance_target_reached": final.read.target_reached,
        "write_variance_target_reached": final.write.target_reached,
        "maximum_rank": maximum_rank,
        "read_cv": {**quantiles(read_errors), "pooled_relative_error": read_pooled},
        "write_cv": {**quantiles(write_errors), "pooled_relative_error": write_pooled},
        "final_native_read_subspace_relative_error": final_native_projection_error(generated[0], final.read),
        "final_native_write_subspace_relative_error": final_native_projection_error(generated[1], final.write),
        "joint_condition_number": float(torch.linalg.cond(final.joint_covariance).item()),
        "joint_covariance_relative_error": joint_mc_error(
            final.joint_mean,
            final.joint_covariance,
            final.joint_factor,
            derived_seed(FIT_SEED, "c", dataset, layer, "mc"),
        ),
        "read_row_norm_max_abs": float((generated[0].norm(dim=1) - read_norms).abs().max().item()),
        "write_row_norm_max_abs": float((generated[1].norm(dim=1) - write_norms).abs().max().item()),
        "deterministic_repeat_exact": bool(torch.equal(generated[0], repeat[0]) and torch.equal(generated[1], repeat[1])),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
    }
    row["gate_pass"] = (
        row["read_variance_target_reached"]
        and row["write_variance_target_reached"]
        and read_pooled <= FINAL_NATIVE_TOLERANCE
        and write_pooled <= FINAL_NATIVE_TOLERANCE
        and covariance_row_pass(row)
    )
    return row


def covariance_row_pass(row: dict[str, Any]) -> bool:
    return (
        row["final_native_read_subspace_relative_error"] <= FINAL_NATIVE_TOLERANCE
        and row["final_native_write_subspace_relative_error"] <= FINAL_NATIVE_TOLERANCE
        and row["joint_condition_number"] <= CONDITION_TOLERANCE
        and row["joint_covariance_relative_error"] <= JOINT_COVARIANCE_TOLERANCE
        and row["read_row_norm_max_abs"] <= NORM_TOLERANCE
        and row["write_row_norm_max_abs"] <= NORM_TOLERANCE
        and row["deterministic_repeat_exact"]
    )


def load_dataset_layer(
    index: list[dict[str, str]], dataset: str, layer: int, device: torch.device
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
    rows = []
    for row in index:
        if not Path(row["path"]).name.startswith(f"{dataset}__"):
            continue
        payload = torch.load(row["path"], map_location="cpu", weights_only=False)
        if payload["dataset"] != dataset:
            raise RuntimeError(f"Tensor filename/payload dataset mismatch: {row['path']}")
        rows.append(
            (
                payload["sample_id"],
                payload["layers"][layer]["read"].float().to(device),
                payload["layers"][layer]["write"].float().to(device),
            )
        )
        del payload
    rows.sort(key=lambda item: item[0])
    return (
        [item[0] for item in rows],
        [item[1] for item in rows],
        [item[2] for item in rows],
    )


def execute(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    geometry_root = Path(args.geometry_root)
    manifest_path = geometry_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["sample_count"] != 2000 or manifest["dataset_counts"] != {"gqa": 1000, "textvqa": 1000}:
        raise RuntimeError("Expected complete independent 2,000-record geometry")
    index = json.loads((geometry_root / "tensor_index.json").read_text())
    model_root = Path(args.model_root)
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {model_root}")
    model_root.mkdir(parents=True, exist_ok=True)

    rows_a = []
    rows_b = []
    rows_c = []
    for dataset in ("gqa", "textvqa"):
        for layer in LAYERS:
            sample_ids, read, write = load_dataset_layer(index, dataset, layer, device)
            if len(sample_ids) != 1000:
                raise RuntimeError(f"Expected 1,000 {dataset} samples")
            rows_a.append(fit_a(dataset, layer, sample_ids, read, write, model_root))
            torch.cuda.empty_cache()
            rows_b.append(fit_b(dataset, layer, sample_ids, read, write))
            torch.cuda.empty_cache()
            rows_c.append(fit_c(dataset, layer, sample_ids, read, write, model_root))
            del read, write
            torch.cuda.empty_cache()
            print(f"completed {dataset} layer {layer}", flush=True)
        torch.cuda.empty_cache()

    gate_a = all(row["gate_pass"] for row in rows_a)
    gate_b = all(row["gate_pass"] for row in rows_b)
    gate_c = all(row["gate_pass"] for row in rows_c)
    selected = (
        "A_fixed_32_row" if gate_a else "B_native_shape_stratified" if gate_b else "C_native_row_distribution" if gate_c else None
    )
    model_paths = sorted(model_root.rglob("*.pt"))
    payload = {
        "schema_version": "v3_null_redesign_covariance_representation_comparison_v1",
        "outcome_blind": True,
        "answer_likelihood_correctness_or_action_values_loaded": False,
        "prospective_candidates_only": [
            "A fixed 32-row baseline",
            "B exact native-row-count stratified",
            "C common feature basis with direct native-row reconstruction",
        ],
        "selection_rule": "select the first and simplest A, then B, then C that passes every dataset/layer gate",
        "tolerances": {
            "final_native_subspace_relative_error": FINAL_NATIVE_TOLERANCE,
            "joint_covariance_relative_error": JOINT_COVARIANCE_TOLERANCE,
            "row_norm_max_abs": NORM_TOLERANCE,
            "condition_number": CONDITION_TOLERANCE,
        },
        "shared_hyperparameters": {
            "variance_target_a_b": VARIANCE_TARGET,
            "marginal_eigen_shrinkage": EIGEN_SHRINKAGE,
            "joint_shrinkage": JOINT_SHRINKAGE,
            "fidelity_draws": FIDELITY_DRAWS,
            "cv_validation_fraction": CV_VALIDATION_FRACTION,
            "cv_seed": CV_SEED,
            "fit_seed": FIT_SEED,
        },
        "representation_a": {
            "description": "path-specific 32-row PCA with paired standardized-score covariance and native remap",
            "rows": rows_a,
            "gate_pass": gate_a,
        },
        "representation_b": {
            "description": "exact paired native shape strata with no row remapping; minimum 32 calibration pairs per shape",
            "rows": rows_b,
            "gate_pass": gate_b,
        },
        "representation_c": {
            "description": "zero-centered row-direction feature bases, paired sample-pooled coefficient covariance, position-bin within-row variance, direct target-row generation",
            "rows_per_sample": NATIVE_C_ROWS_PER_SAMPLE,
            "variance_target": NATIVE_C_VARIANCE_TARGET,
            "maximum_rank": NATIVE_C_MAXIMUM_RANK,
            "position_bins": NATIVE_C_POSITION_BINS,
            "rows": rows_c,
            "gate_pass": gate_c,
        },
        "selected_representation": selected,
        "model_checksums": {str(path): sha256(path) for path in model_paths},
        "geometry_manifest_sha256": sha256(manifest_path),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    model_manifest = model_root / "manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "schema_version": "v3_null_redesign_covariance_models_v1",
                "selected_representation": selected,
                "comparison_sha256": sha256(output),
                "model_checksums": payload["model_checksums"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths = [*model_paths, model_manifest]
    (model_root / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in paths)
    )
    print(json.dumps({"output": str(output), "selected": selected, "gates": {"A": gate_a, "B": gate_b, "C": gate_c}}, sort_keys=True))


if __name__ == "__main__":
    execute(parse_args())
