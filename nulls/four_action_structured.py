from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ResidualPairMetadata:
    sample_id: str
    image_id: str
    read_norm: float
    write_norm: float
    postvisual_rows: int
    visual_rows: int
    prompt_tokens: int


def _ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return math.inf
    return max(first / second, second / first)


def _exact_norm_match(value: torch.Tensor, target_norm: float) -> torch.Tensor:
    if not math.isfinite(target_norm) or target_norm < 0:
        raise ValueError("target_norm must be finite and nonnegative")
    if target_norm == 0:
        return torch.zeros_like(value, dtype=torch.float32)
    current = float(value.float().norm().item())
    if not math.isfinite(current) or current <= 1e-12:
        raise ValueError("null draw is degenerate")
    return value.float() * (target_norm / current)


def generate_isotropic_null(
    shape: Sequence[int], target_norm: float, seed: int
) -> torch.Tensor:
    if not shape or any(int(size) < 1 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(tuple(int(size) for size in shape), generator=generator)
    return _exact_norm_match(value, target_norm)


def pair_matching_ratio(
    target: ResidualPairMetadata, donor: ResidualPairMetadata
) -> float:
    return max(
        _ratio(target.read_norm, donor.read_norm),
        _ratio(target.write_norm, donor.write_norm),
        _ratio(target.postvisual_rows, donor.postvisual_rows),
        _ratio(target.visual_rows, donor.visual_rows),
        _ratio(target.prompt_tokens, donor.prompt_tokens),
    )


def _tie_hash(seed: int, target_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{seed}:{target_id}:{donor_id}".encode("utf-8")).hexdigest()


def select_pair_donors(
    target: ResidualPairMetadata,
    donors: Sequence[ResidualPairMetadata],
    draws: int,
    seed: int,
    caliper: float,
) -> list[ResidualPairMetadata]:
    if draws < 1:
        raise ValueError("draws must be positive")
    if not math.isfinite(caliper) or caliper < 1:
        raise ValueError("caliper must be finite and at least one")
    eligible = [
        donor
        for donor in donors
        if donor.sample_id != target.sample_id
        and donor.image_id != target.image_id
        and pair_matching_ratio(target, donor) <= caliper
    ]
    eligible.sort(
        key=lambda donor: (
            pair_matching_ratio(target, donor),
            _tie_hash(seed, target.sample_id, donor.sample_id),
        )
    )
    if len(eligible) < draws:
        raise ValueError(
            f"Only {len(eligible)} eligible paired donors for {target.sample_id}; {draws} required"
        )
    return eligible[:draws]
