from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nulls.joint_four_action import _match_row_norms


@dataclass
class DirectionBasis:
    basis: torch.Tensor  # [rank, hidden]
    rank: int
    hidden_size: int
    variance_target: float
    sampled_explained_variance: float
    sampled_row_count: int
    target_reached: bool


@dataclass
class NativeRowJointModel:
    read: DirectionBasis
    write: DirectionBasis
    joint_mean: torch.Tensor
    joint_covariance: torch.Tensor
    joint_factor: torch.Tensor
    read_within_variance: torch.Tensor  # [position_bins, read_rank]
    write_within_variance: torch.Tensor  # [position_bins, write_rank]
    position_bins: int
    joint_shrinkage: float
    calibration_samples: int


def _unit_rows(value: torch.Tensor) -> torch.Tensor:
    norms = value.float().norm(dim=1, keepdim=True)
    return torch.where(norms > 1e-12, value.float() / norms.clamp_min(1e-12), 0.0)


def _sample_rows(value: torch.Tensor, maximum: int) -> torch.Tensor:
    rows = _unit_rows(value)
    if rows.shape[0] <= maximum:
        return rows
    indices = torch.linspace(
        0, rows.shape[0] - 1, maximum, device=rows.device
    ).round().long()
    return rows[indices]


def fit_direction_basis(
    residuals: Sequence[torch.Tensor],
    rows_per_sample: int,
    variance_target: float,
    maximum_rank: int,
    seed: int,
) -> DirectionBasis:
    if len(residuals) < 3:
        raise ValueError("At least three residuals are required")
    if rows_per_sample < 1 or maximum_rank < 1:
        raise ValueError("rows_per_sample and maximum_rank must be positive")
    if not 0 < variance_target <= 1:
        raise ValueError("variance_target must be in (0, 1]")
    hidden = {int(item.shape[1]) for item in residuals if item.ndim == 2}
    if len(hidden) != 1 or len(hidden) != len({int(item.shape[1]) for item in residuals}):
        raise ValueError("Residuals must be nonempty 2D tensors with one hidden size")
    sampled = torch.cat([_sample_rows(item, rows_per_sample) for item in residuals], dim=0)
    q = min(maximum_rank, sampled.shape[0] - 1, sampled.shape[1])
    if q < 1:
        raise ValueError("Insufficient sampled rows")
    prior_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed) % (2**63 - 1))
    try:
        _, singular, vectors = torch.pca_lowrank(
            sampled, q=q, center=False, niter=4
        )
    finally:
        torch.random.set_rng_state(prior_state)
    energy = singular.float().square()
    total = sampled.float().square().sum().clamp_min(1e-12)
    cumulative = torch.cumsum(energy, dim=0) / total
    reached = bool((cumulative >= variance_target).any().item())
    if reached:
        rank = int(torch.where(cumulative >= variance_target)[0][0].item()) + 1
    else:
        rank = q
    return DirectionBasis(
        basis=vectors[:, :rank].T.contiguous(),
        rank=rank,
        hidden_size=int(sampled.shape[1]),
        variance_target=float(variance_target),
        sampled_explained_variance=float(cumulative[rank - 1].item()),
        sampled_row_count=int(sampled.shape[0]),
        target_reached=reached,
    )


def project_rows(value: torch.Tensor, fit: DirectionBasis) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != fit.hidden_size:
        raise ValueError("Residual shape does not match the direction basis")
    return _unit_rows(value) @ fit.basis.float().T


def final_native_projection_error(value: torch.Tensor, fit: DirectionBasis) -> float:
    row_norms = value.float().norm(dim=1)
    coefficients = project_rows(value, fit)
    reconstruction = coefficients @ fit.basis.float()
    positive = row_norms > 0
    if positive.any():
        reconstructed_norms = reconstruction.norm(dim=1)
        if (reconstructed_norms[positive] <= 1e-12).any():
            return float("inf")
        reconstruction = _match_row_norms(reconstruction, row_norms)
    error = (reconstruction - value.float()).norm() / value.float().norm().clamp_min(1e-12)
    return float(error.item())


def _position_bin_indices(rows: int, bins: int, device: torch.device) -> torch.Tensor:
    if rows < 1 or bins < 1:
        raise ValueError("rows and bins must be positive")
    positions = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    return torch.clamp((positions * bins).long(), max=bins - 1)


def _path_statistics(
    residuals: Sequence[torch.Tensor], fit: DirectionBasis, bins: int, rows_per_sample: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = []
    sums = torch.zeros(bins, fit.rank, device=fit.basis.device)
    squares = torch.zeros_like(sums)
    counts = torch.zeros(bins, 1, device=fit.basis.device)
    for value in residuals:
        sampled = _sample_rows(value, rows_per_sample)
        coefficients = sampled @ fit.basis.float().T
        mean = coefficients.mean(dim=0)
        pooled.append(mean)
        centered = coefficients - mean
        indices = _position_bin_indices(sampled.shape[0], bins, sampled.device)
        sums.index_add_(0, indices, centered)
        squares.index_add_(0, indices, centered.square())
        counts[:, 0] += torch.bincount(indices, minlength=bins).to(counts.dtype)
    mean = sums / counts.clamp_min(1)
    variance = squares / counts.clamp_min(1) - mean.square()
    variance = variance.clamp_min(1e-8)
    return torch.stack(pooled), variance


def fit_native_row_joint_model(
    read_residuals: Sequence[torch.Tensor],
    write_residuals: Sequence[torch.Tensor],
    rows_per_sample: int,
    variance_target: float,
    maximum_rank: int,
    position_bins: int,
    shrinkage: float,
    seed: int,
) -> NativeRowJointModel:
    if len(read_residuals) != len(write_residuals) or len(read_residuals) < 3:
        raise ValueError("READ/WRITE residual lists must be paired and have length >= 3")
    if not 0 <= shrinkage < 1:
        raise ValueError("shrinkage must be in [0, 1)")
    read = fit_direction_basis(
        read_residuals, rows_per_sample, variance_target, maximum_rank, seed
    )
    write = fit_direction_basis(
        write_residuals, rows_per_sample, variance_target, maximum_rank, seed + 1
    )
    read_pooled, read_within = _path_statistics(
        read_residuals, read, position_bins, rows_per_sample
    )
    write_pooled, write_within = _path_statistics(
        write_residuals, write, position_bins, rows_per_sample
    )
    pooled = torch.cat([read_pooled, write_pooled], dim=1)
    mean = pooled.mean(dim=0)
    centered = pooled - mean
    covariance = centered.T @ centered / (pooled.shape[0] - 1)
    diagonal = torch.diag(torch.diag(covariance))
    covariance = (1 - shrinkage) * covariance + shrinkage * diagonal
    scale_floor = torch.diag(covariance).mean().clamp_min(1e-12) * 1e-6
    covariance = covariance + torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    ) * scale_floor
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(scale_floor)
    covariance = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
    factor = eigenvectors @ torch.diag(eigenvalues.sqrt())
    return NativeRowJointModel(
        read=read,
        write=write,
        joint_mean=mean,
        joint_covariance=covariance,
        joint_factor=factor,
        read_within_variance=read_within,
        write_within_variance=write_within,
        position_bins=position_bins,
        joint_shrinkage=shrinkage,
        calibration_samples=len(read_residuals),
    )


def _generate_path(
    pooled: torch.Tensor,
    within_variance: torch.Tensor,
    basis: DirectionBasis,
    row_norms: torch.Tensor,
    bins: int,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = int(row_norms.numel())
    indices = _position_bin_indices(rows, bins, pooled.device)
    noise = torch.randn(rows, basis.rank, generator=generator, device="cpu").to(
        pooled.device
    )
    noise = noise * within_variance[indices].sqrt()
    noise = noise - noise.mean(dim=0, keepdim=True)
    coefficients = pooled.unsqueeze(0) + noise
    candidate = coefficients @ basis.basis.float()
    return _match_row_norms(candidate, row_norms)


def generate_native_row_joint_null(
    model: NativeRowJointModel,
    target_read_row_norms: torch.Tensor,
    target_write_row_norms: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(model.joint_factor.shape[1], generator=generator).to(
        model.joint_factor.device
    )
    pooled = model.joint_mean.float() + model.joint_factor.float() @ noise
    read_pooled = pooled[: model.read.rank]
    write_pooled = pooled[model.read.rank :]
    if write_pooled.numel() != model.write.rank:
        raise RuntimeError("Joint pooled coefficient dimensions do not match")
    return (
        _generate_path(
            read_pooled,
            model.read_within_variance,
            model.read,
            target_read_row_norms,
            model.position_bins,
            generator,
        ),
        _generate_path(
            write_pooled,
            model.write_within_variance,
            model.write,
            target_write_row_norms,
            model.position_bins,
            generator,
        ),
    )
