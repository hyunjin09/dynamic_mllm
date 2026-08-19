from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nulls.structured_read import FixedGridCovariance, fit_fixed_grid_covariance, map_rows


@dataclass
class JointPathCovariance:
    read: FixedGridCovariance
    write: FixedGridCovariance
    score_mean: torch.Tensor
    score_covariance: torch.Tensor
    score_factor: torch.Tensor
    joint_shrinkage: float


@dataclass(frozen=True)
class PairedGeometryMetadata:
    sample_id: str
    image_id: str
    dataset: str
    layer: int
    read_norm: float
    write_norm: float
    read_rows: int
    write_rows: int
    image_tokens: int
    prompt_tokens: int
    read_scale_ratio: float
    write_scale_ratio: float
    read_row_cv: float
    write_row_cv: float


def _path_scores(residuals: Sequence[torch.Tensor], fit: FixedGridCovariance) -> torch.Tensor:
    grids = torch.stack([map_rows(item.float(), fit.grid_rows) for item in residuals])
    centered = grids.reshape(len(grids), -1) - fit.mean.float().reshape(1, -1)
    basis = fit.basis.float().reshape(fit.rank, -1)
    coefficients = centered @ basis.transpose(0, 1)
    return coefficients / fit.eigenvalues.float().sqrt().clamp_min(1e-12)


def fit_joint_path_covariance(
    read_residuals: Sequence[torch.Tensor],
    write_residuals: Sequence[torch.Tensor],
    grid_rows: int,
    variance_target: float,
    joint_shrinkage: float,
    marginal_eigen_shrinkage: float | None = None,
) -> JointPathCovariance:
    """Fit path-specific subspaces and correlate only their standardized scores.

    READ text rows and WRITE visual rows are never flattened into one native
    tensor. Each path is mapped and decomposed separately; only the paired,
    dimensionless PCA scores enter the joint covariance.
    """
    if len(read_residuals) != len(write_residuals) or len(read_residuals) < 3:
        raise ValueError("READ and WRITE calibration lists must have equal length >= 3")
    if not 0.0 <= joint_shrinkage < 1.0:
        raise ValueError("joint_shrinkage must lie in [0, 1)")
    marginal_shrinkage = (
        joint_shrinkage
        if marginal_eigen_shrinkage is None
        else float(marginal_eigen_shrinkage)
    )
    if not 0.0 <= marginal_shrinkage < 1.0:
        raise ValueError("marginal_eigen_shrinkage must lie in [0, 1)")
    read_fit = fit_fixed_grid_covariance(
        read_residuals, grid_rows, variance_target, eigen_shrinkage=marginal_shrinkage
    )
    write_fit = fit_fixed_grid_covariance(
        write_residuals, grid_rows, variance_target, eigen_shrinkage=marginal_shrinkage
    )
    scores = torch.cat(
        [_path_scores(read_residuals, read_fit), _path_scores(write_residuals, write_fit)],
        dim=1,
    )
    score_mean = scores.mean(dim=0)
    centered = scores - score_mean
    covariance = centered.transpose(0, 1) @ centered / (scores.shape[0] - 1)
    identity = torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    covariance = (1.0 - joint_shrinkage) * covariance + joint_shrinkage * identity
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(1e-6)
    covariance = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    factor = eigenvectors @ torch.diag(eigenvalues.sqrt())
    return JointPathCovariance(
        read=read_fit,
        write=write_fit,
        score_mean=score_mean,
        score_covariance=covariance,
        score_factor=factor,
        joint_shrinkage=joint_shrinkage,
    )


def _match_row_norms(candidate: torch.Tensor, target_row_norms: torch.Tensor) -> torch.Tensor:
    if candidate.ndim != 2 or target_row_norms.ndim != 1:
        raise ValueError("candidate must be [rows, hidden] and norms must be [rows]")
    if candidate.shape[0] != target_row_norms.shape[0]:
        raise ValueError("target row norms do not match candidate rows")
    if not torch.isfinite(target_row_norms).all() or (target_row_norms < 0).any():
        raise ValueError("target row norms must be finite and nonnegative")
    current = candidate.float().norm(dim=1)
    positive_target = target_row_norms.float() > 0
    if (current[positive_target] <= 1e-12).any():
        raise ValueError("generated path has a degenerate row with positive target norm")
    scale = torch.zeros_like(current)
    scale[positive_target] = target_row_norms.float()[positive_target] / current[positive_target]
    return candidate.float() * scale.unsqueeze(1)


def _decode_path(fit: FixedGridCovariance, standardized: torch.Tensor) -> torch.Tensor:
    coefficients = standardized * fit.eigenvalues.float().sqrt()
    return fit.mean.float() + torch.einsum(
        "r,rgh->gh", coefficients, fit.basis.float()
    )


def generate_joint_path_null(
    fit: JointPathCovariance,
    target_read_row_norms: torch.Tensor,
    target_write_row_norms: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(fit.score_factor.shape[1], generator=generator).to(
        fit.score_factor.device
    )
    scores = fit.score_mean.float() + fit.score_factor.float() @ noise
    read_scores = scores[: fit.read.rank]
    write_scores = scores[fit.read.rank :]
    if write_scores.numel() != fit.write.rank:
        raise RuntimeError("joint score dimensions do not match path ranks")
    read_grid = _decode_path(fit.read, read_scores)
    write_grid = _decode_path(fit.write, write_scores)
    read_native = map_rows(read_grid, int(target_read_row_norms.numel()))
    write_native = map_rows(write_grid, int(target_write_row_norms.numel()))
    return (
        _match_row_norms(read_native, target_read_row_norms),
        _match_row_norms(write_native, target_write_row_norms),
    )


def generate_paired_isotropic_null(
    hidden_size: int,
    target_read_row_norms: torch.Tensor,
    target_write_row_norms: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a paired isotropic baseline with exact native row-norm matching."""
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    read = torch.randn(
        int(target_read_row_norms.numel()), hidden_size, generator=generator
    ).to(target_read_row_norms.device)
    write = torch.randn(
        int(target_write_row_norms.numel()), hidden_size, generator=generator
    ).to(target_write_row_norms.device)
    return (
        _match_row_norms(read, target_read_row_norms),
        _match_row_norms(write, target_write_row_norms),
    )


def generate_paired_real_null(
    donor_read: torch.Tensor,
    donor_write: torch.Tensor,
    target_read_row_norms: torch.Tensor,
    target_write_row_norms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a coherent donor pair and match each target row's residual norm."""
    read_native = map_rows(donor_read.float(), int(target_read_row_norms.numel()))
    write_native = map_rows(donor_write.float(), int(target_write_row_norms.numel()))
    return (
        _match_row_norms(read_native, target_read_row_norms),
        _match_row_norms(write_native, target_write_row_norms),
    )


def _ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0 or not math.isfinite(first) or not math.isfinite(second):
        return math.inf
    return max(first / second, second / first)


def paired_geometry_distance(
    target: PairedGeometryMetadata, donor: PairedGeometryMetadata
) -> float:
    if target.dataset != donor.dataset or target.layer != donor.layer:
        return math.inf
    return max(
        _ratio(target.read_norm, donor.read_norm),
        _ratio(target.write_norm, donor.write_norm),
        _ratio(target.read_rows, donor.read_rows),
        _ratio(target.write_rows, donor.write_rows),
        _ratio(target.image_tokens, donor.image_tokens),
        _ratio(target.prompt_tokens, donor.prompt_tokens),
        _ratio(target.read_scale_ratio, donor.read_scale_ratio),
        _ratio(target.write_scale_ratio, donor.write_scale_ratio),
        _ratio(target.read_row_cv, donor.read_row_cv),
        _ratio(target.write_row_cv, donor.write_row_cv),
    )


def _tie_hash(seed: int, target_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{seed}:{target_id}:{donor_id}".encode()).hexdigest()


def _eligible_sorted(
    target: PairedGeometryMetadata,
    donors: Sequence[PairedGeometryMetadata],
    seed: int,
) -> list[tuple[float, PairedGeometryMetadata]]:
    rows = [
        (paired_geometry_distance(target, donor), donor)
        for donor in donors
        if donor.dataset == target.dataset
        and donor.layer == target.layer
        and donor.sample_id != target.sample_id
        and donor.image_id != target.image_id
    ]
    rows.sort(
        key=lambda item: (
            item[0],
            _tie_hash(seed, target.sample_id, item[1].sample_id),
        )
    )
    return rows


def fit_paired_donor_calipers(
    calibration: Sequence[PairedGeometryMetadata], donor_count: int
) -> tuple[dict[tuple[str, int], float], list[dict[str, Any]]]:
    if donor_count < 1:
        raise ValueError("donor_count must be positive")
    grouped: dict[tuple[str, int], list[PairedGeometryMetadata]] = defaultdict(list)
    for item in calibration:
        grouped[(item.dataset, item.layer)].append(item)
    calipers: dict[tuple[str, int], float] = {}
    coverage: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        required = []
        for target in rows:
            distances = _eligible_sorted(target, rows, seed=0)
            if len(distances) < donor_count:
                raise ValueError(
                    f"Only {len(distances)} eligible donors for {target.sample_id}"
                )
            kth = float(distances[donor_count - 1][0])
            required.append(kth)
            coverage.append(
                {
                    "sample_id": target.sample_id,
                    "image_id": target.image_id,
                    "dataset": target.dataset,
                    "layer": target.layer,
                    "distance_to_required_donor": kth,
                    "nearest_distances": [float(item[0]) for item in distances[: donor_count + 1]],
                }
            )
        calipers[key] = max(required)
    return calipers, coverage


def select_paired_donors(
    target: PairedGeometryMetadata,
    donors: Sequence[PairedGeometryMetadata],
    donor_count: int,
    seed: int,
    caliper: float,
) -> list[PairedGeometryMetadata]:
    if not math.isfinite(caliper) or caliper < 1:
        raise ValueError("caliper must be finite and at least one")
    eligible = [item for item in _eligible_sorted(target, donors, seed) if item[0] <= caliper]
    if len(eligible) < donor_count:
        raise ValueError(
            f"Only {len(eligible)} eligible donors for {target.sample_id}; {donor_count} required"
        )
    return [item[1] for item in eligible[:donor_count]]


def search_budget_cells(layers: Sequence[int], actions: Sequence[str]) -> list[dict[str, Any]]:
    if len(set(layers)) != len(layers) or len(set(actions)) != len(actions):
        raise ValueError("layers and actions must be unique")
    if not layers or not actions:
        raise ValueError("layers and actions must be nonempty")
    return [{"layer": int(layer), "action": str(action)} for layer in layers for action in actions]
