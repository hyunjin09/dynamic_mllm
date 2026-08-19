from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DonorMetadata:
    sample_id: str
    image_id: str
    residual_norm: float
    postvisual_rows: int
    visual_tokens: int
    prompt_tokens: int


@dataclass
class FixedGridCovariance:
    mean: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    rank: int
    explained_variance: float
    grid_rows: int
    hidden_size: int
    calibration_samples: int
    variance_target: float
    eigen_shrinkage: float


def map_rows(residual: torch.Tensor, target_rows: int) -> torch.Tensor:
    if residual.ndim != 2 or residual.shape[0] < 1 or residual.shape[1] < 1:
        raise ValueError("Residual must have shape [rows, hidden] with nonzero dimensions")
    if target_rows < 1:
        raise ValueError("Target row count must be positive")
    if residual.shape[0] == target_rows:
        return residual.clone()
    if residual.shape[0] == 1:
        return residual.expand(target_rows, -1).clone()
    values = residual.float().transpose(0, 1).unsqueeze(0)
    mapped = F.interpolate(values, size=target_rows, mode="linear", align_corners=True)
    return mapped.squeeze(0).transpose(0, 1).to(residual.dtype)


def fit_fixed_grid_covariance(
    residuals: Sequence[torch.Tensor],
    grid_rows: int,
    variance_target: float,
    eigen_shrinkage: float,
) -> FixedGridCovariance:
    if len(residuals) < 3:
        raise ValueError("At least three calibration residuals are required")
    if not 0.0 < variance_target <= 1.0:
        raise ValueError("Variance target must lie in (0, 1]")
    if not 0.0 <= eigen_shrinkage < 1.0:
        raise ValueError("Eigenvalue shrinkage must lie in [0, 1)")
    if any(item.ndim != 2 or item.shape[0] < 1 or item.shape[1] < 1 for item in residuals):
        raise ValueError("Calibration residuals must all have shape [rows, hidden]")
    hidden_sizes = {int(item.shape[1]) for item in residuals}
    if len(hidden_sizes) != 1:
        raise ValueError("Calibration residuals must share one hidden size")

    grids = torch.stack([map_rows(item.float(), grid_rows) for item in residuals])
    flat = grids.reshape(len(residuals), -1)
    mean_flat = flat.mean(dim=0)
    centered = flat - mean_flat
    gram = centered @ centered.transpose(0, 1) / (len(residuals) - 1)
    eigenvalues, sample_vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    sample_vectors = sample_vectors[:, order]
    positive = eigenvalues > max(float(eigenvalues[0].item()) * 1e-10, 1e-12)
    eigenvalues = eigenvalues[positive]
    sample_vectors = sample_vectors[:, positive]
    if eigenvalues.numel() == 0:
        raise ValueError("Calibration covariance is degenerate")
    cumulative = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
    threshold = torch.tensor(
        variance_target, dtype=cumulative.dtype, device=cumulative.device
    )
    rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
    retained_values = eigenvalues[:rank]
    retained_vectors = sample_vectors[:, :rank]
    denominators = torch.sqrt((len(residuals) - 1) * retained_values).unsqueeze(1)
    basis_flat = retained_vectors.transpose(0, 1) @ centered / denominators
    shrink_target = retained_values.mean()
    shrunk = (1.0 - eigen_shrinkage) * retained_values + eigen_shrinkage * shrink_target
    hidden_size = int(grids.shape[-1])
    return FixedGridCovariance(
        mean=mean_flat.reshape(grid_rows, hidden_size),
        basis=basis_flat.reshape(rank, grid_rows, hidden_size),
        eigenvalues=shrunk,
        rank=rank,
        explained_variance=float(cumulative[rank - 1].item()),
        grid_rows=grid_rows,
        hidden_size=hidden_size,
        calibration_samples=len(residuals),
        variance_target=variance_target,
        eigen_shrinkage=eigen_shrinkage,
    )


def _match_norm(candidate: torch.Tensor, target_norm: float) -> torch.Tensor:
    if target_norm < 0.0 or not math.isfinite(target_norm):
        raise ValueError("Target norm must be finite and nonnegative")
    if target_norm == 0.0:
        return torch.zeros_like(candidate)
    current = float(candidate.float().norm().item())
    if current <= 1e-12 or not math.isfinite(current):
        raise ValueError("Generated residual is degenerate")
    return candidate.float() * (target_norm / current)


def generate_covariance_null(
    fit: FixedGridCovariance,
    target_rows: int,
    target_norm: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    coefficients = torch.randn(fit.rank, generator=generator) * fit.eigenvalues.sqrt()
    sampled = fit.mean.float() + torch.einsum(
        "r,rgh->gh", coefficients, fit.basis.float()
    )
    return _match_norm(map_rows(sampled, target_rows), target_norm)


def _ratio(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return math.inf
    return max(a / b, b / a)


def _tie_hash(seed: int, target_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{seed}:{target_id}:{donor_id}".encode("utf-8")).hexdigest()


def donor_matching_ratio(target: DonorMetadata, donor: DonorMetadata) -> float:
    """Maximum multiplicative mismatch across frozen residual-geometry covariates."""
    return max(
        _ratio(donor.residual_norm, target.residual_norm),
        _ratio(donor.postvisual_rows, target.postvisual_rows),
        _ratio(donor.visual_tokens, target.visual_tokens),
    )


def fit_real_donor_caliper(
    calibration: Sequence[DonorMetadata], draws: int
) -> float:
    """Fit the smallest common cap covering `draws` donors for every target."""
    if draws < 1:
        raise ValueError("At least one real-residual draw is required")
    required: list[float] = []
    for target in calibration:
        distances = sorted(
            donor_matching_ratio(target, donor)
            for donor in calibration
            if donor.sample_id != target.sample_id and donor.image_id != target.image_id
        )
        if len(distances) < draws:
            raise ValueError(
                f"Only {len(distances)} non-identical donors for {target.sample_id}; {draws} required"
            )
        required.append(distances[draws - 1])
    return max(required)


def real_donor_candidates(
    target: DonorMetadata,
    donors: Sequence[DonorMetadata],
    seed: int,
    matching_ratio_cap: float,
) -> list[DonorMetadata]:
    if matching_ratio_cap < 1.0 or not math.isfinite(matching_ratio_cap):
        raise ValueError("Matching-ratio cap must be finite and at least one")
    eligible = [
        donor
        for donor in donors
        if donor.sample_id != target.sample_id
        and donor.image_id != target.image_id
        and donor_matching_ratio(target, donor) <= matching_ratio_cap
    ]
    eligible.sort(
        key=lambda donor: (
            donor_matching_ratio(target, donor),
            _ratio(donor.residual_norm, target.residual_norm),
            _ratio(donor.postvisual_rows, target.postvisual_rows),
            _ratio(donor.visual_tokens, target.visual_tokens),
            _tie_hash(seed, target.sample_id, donor.sample_id),
        )
    )
    return eligible


def select_real_donors(
    target: DonorMetadata,
    donors: Sequence[DonorMetadata],
    draws: int,
    seed: int,
    matching_ratio_cap: float,
) -> list[DonorMetadata]:
    eligible = real_donor_candidates(
        target,
        donors,
        seed,
        matching_ratio_cap,
    )
    if len(eligible) < draws:
        raise ValueError(
            f"Only {len(eligible)} eligible donors for {target.sample_id}; {draws} required"
        )
    return eligible[:draws]


def generate_real_residual_null(
    donor_residual: torch.Tensor, target_rows: int, target_norm: float
) -> torch.Tensor:
    return _match_norm(map_rows(donor_residual.float(), target_rows), target_norm)


def compose_null_read_output(
    full_output: torch.Tensor,
    off_output: torch.Tensor,
    null_delta: torch.Tensor,
    visual_token_mask: torch.Tensor,
) -> torch.Tensor:
    if full_output.shape != off_output.shape or full_output.shape != null_delta.shape:
        raise ValueError("FULL, OFF, and null tensors must have identical shapes")
    if visual_token_mask.shape != full_output.shape[:-1]:
        raise ValueError("Visual mask must match batch and sequence dimensions")
    replacement = (off_output.float() + null_delta.float()).to(full_output.dtype)
    return torch.where(visual_token_mask.unsqueeze(-1), full_output, replacement)
