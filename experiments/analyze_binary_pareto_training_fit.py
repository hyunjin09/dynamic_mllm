#!/usr/bin/env python3
"""Evaluate saved Pareto-router checkpoints on frozen train/validation inputs.

This is a read-only diagnostic.  It reuses the exact full10 set collator,
frozen question encoder, visual-feature cache, predictor, and threshold decoder.
No optimizer or model parameter is updated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml

from binary_policy.dataset import BinaryPolicyManifestDataset
from binary_policy.evaluation import mask_diversity_metrics
from binary_policy.multimodal import make_multimodal_set_collator
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256, seed_worker, validate_gate
from experiments.train_binary_polar_full10 import (
    BENCHMARKS,
    MetricAccumulator,
    forward_logits,
    move_batch,
)


def weighted_bce_oracle(
    valid_masks: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the exact thresholded weighted bit-marginal BCE optimum."""
    if valid_masks.ndim != 3 or valid_mask.shape != valid_masks.shape[:2]:
        raise ValueError("valid route tensors have incompatible shapes")
    if route_weights.shape != valid_mask.shape:
        raise ValueError("route_weights must align with valid_mask")
    weights = torch.where(
        valid_mask,
        route_weights.to(dtype=valid_masks.dtype),
        torch.zeros_like(route_weights, dtype=valid_masks.dtype),
    )
    totals = weights.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("every sample requires positive valid-route weight")
    marginals = (valid_masks * (weights / totals).unsqueeze(-1)).sum(dim=1)
    return (marginals >= 0.5).to(torch.int64)


def factorized_route_log_probabilities(
    logits: torch.Tensor, valid_masks: torch.Tensor
) -> torch.Tensor:
    """Complete-mask log probabilities under the factorized binary head."""
    if logits.ndim != 2 or valid_masks.ndim != 3:
        raise ValueError("logits must be [B,L] and masks [B,K,L]")
    if logits.shape[0] != valid_masks.shape[0] or logits.shape[1] != valid_masks.shape[2]:
        raise ValueError("logits and masks do not align")
    on = F.logsigmoid(logits).unsqueeze(1)
    off = F.logsigmoid(-logits).unsqueeze(1)
    masks = valid_masks.to(dtype=logits.dtype)
    return (masks * on + (1.0 - masks) * off).sum(dim=-1)


def batch_probability_diagnostics(
    logits: torch.Tensor,
    valid_masks: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict[str, float | int]:
    """Return additive sufficient statistics for confidence/route-mass metrics."""
    probabilities = torch.sigmoid(logits)
    decoded = (logits >= 0).to(torch.int64)
    oracle = weighted_bce_oracle(valid_masks, valid_mask, route_weights)
    route_log_prob = factorized_route_log_probabilities(logits, valid_masks)
    masked_route_log_prob = route_log_prob.masked_fill(~valid_mask, -torch.inf)
    normalized_weights = torch.where(
        valid_mask,
        route_weights.to(dtype=logits.dtype),
        torch.zeros_like(route_weights, dtype=logits.dtype),
    )
    normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True)
    weighted_log_mass = torch.logsumexp(
        masked_route_log_prob
        + torch.where(
            valid_mask,
            normalized_weights.clamp_min(torch.finfo(logits.dtype).tiny).log(),
            torch.full_like(normalized_weights, -torch.inf),
        ),
        dim=1,
    )
    total_log_mass = torch.logsumexp(masked_route_log_prob, dim=1)
    max_log_prob = masked_route_log_prob.max(dim=1).values
    valid_sets = [
        {tuple(int(bit) for bit in row) for row in valid_masks[i, valid_mask[i]].tolist()}
        for i in range(logits.shape[0])
    ]
    decoded_cpu = decoded.detach().cpu().tolist()
    oracle_cpu = oracle.detach().cpu().tolist()
    route_counts = valid_mask.sum(dim=1)
    singleton_indices = torch.nonzero(route_counts == 1, as_tuple=False).flatten()
    result: dict[str, float | int] = {
        "sample_count": int(logits.shape[0]),
        "bit_count": int(probabilities.numel()),
        "margin_sum": float((probabilities - 0.5).abs().sum()),
        "near_half_count": int(((probabilities >= 0.45) & (probabilities <= 0.55)).sum()),
        "above_point9_count": int((probabilities > 0.9).sum()),
        "below_point1_count": int((probabilities < 0.1).sum()),
        "pareto_hit_count": sum(
            tuple(mask) in valid_set for mask, valid_set in zip(decoded_cpu, valid_sets)
        ),
        "oracle_pareto_hit_count": sum(
            tuple(mask) in valid_set for mask, valid_set in zip(oracle_cpu, valid_sets)
        ),
        "oracle_exact_count": int((decoded == oracle).all(dim=1).sum()),
        "oracle_hamming_sum": int((decoded != oracle).sum()),
        "oracle_on_sum": int(oracle.sum()),
        "predicted_on_sum": int(decoded.sum()),
        "max_route_probability_sum": float(max_log_prob.exp().sum()),
        "total_pareto_probability_sum": float(total_log_mass.exp().sum()),
        "weighted_pareto_mass_sum": float(weighted_log_mass.exp().sum()),
        "singleton_sample_count": int(singleton_indices.numel()),
        "singleton_bit_count": 0,
        "singleton_correct_bit_count": 0,
        "singleton_target_probability_sum": 0.0,
        "singleton_confident_correct_bit_count": 0,
    }
    if singleton_indices.numel():
        singleton_probabilities = probabilities.index_select(0, singleton_indices)
        singleton_decoded = decoded.index_select(0, singleton_indices)
        singleton_masks = valid_masks.index_select(0, singleton_indices)[:, 0].to(torch.int64)
        target_probabilities = torch.where(
            singleton_masks.bool(), singleton_probabilities, 1.0 - singleton_probabilities
        )
        result.update(
            {
                "singleton_bit_count": int(singleton_masks.numel()),
                "singleton_correct_bit_count": int(
                    (singleton_decoded == singleton_masks).sum()
                ),
                "singleton_target_probability_sum": float(target_probabilities.sum()),
                "singleton_confident_correct_bit_count": int(
                    (target_probabilities > 0.9).sum()
                ),
            }
        )
    return result


class ProbabilityAccumulator:
    def __init__(self) -> None:
        self.sums: Counter[str] = Counter()

    def update(self, diagnostics: dict[str, float | int]) -> None:
        self.sums.update(diagnostics)

    def finalize(self) -> dict[str, float | int | None]:
        samples = int(self.sums["sample_count"])
        bits = int(self.sums["bit_count"])
        singleton_bits = int(self.sums["singleton_bit_count"])
        if samples < 1 or bits < 1:
            raise ValueError("cannot finalize empty probability diagnostics")
        return {
            "examples": samples,
            "mean_probability_margin_from_0_5": self.sums["margin_sum"] / bits,
            "fraction_bits_0_45_to_0_55": self.sums["near_half_count"] / bits,
            "fraction_bits_above_0_9": self.sums["above_point9_count"] / bits,
            "fraction_bits_below_0_1": self.sums["below_point1_count"] / bits,
            "pareto_valid_hit_at_1": self.sums["pareto_hit_count"] / samples,
            "bce_oracle_pareto_hit_at_1": self.sums["oracle_pareto_hit_count"] / samples,
            "predictor_bce_oracle_exact_agreement": self.sums["oracle_exact_count"] / samples,
            "predictor_bce_oracle_mean_hamming": self.sums["oracle_hamming_sum"] / samples,
            "bce_oracle_mean_visual_on": self.sums["oracle_on_sum"] / samples,
            "predicted_mean_visual_on": self.sums["predicted_on_sum"] / samples,
            "mean_best_pareto_route_probability": self.sums[
                "max_route_probability_sum"
            ]
            / samples,
            "mean_total_pareto_set_probability": self.sums[
                "total_pareto_probability_sum"
            ]
            / samples,
            "mean_weighted_pareto_mass": self.sums["weighted_pareto_mass_sum"] / samples,
            "singleton_examples": int(self.sums["singleton_sample_count"]),
            "singleton_bit_accuracy": (
                self.sums["singleton_correct_bit_count"] / singleton_bits
                if singleton_bits
                else None
            ),
            "singleton_mean_target_bit_probability": (
                self.sums["singleton_target_probability_sum"] / singleton_bits
                if singleton_bits
                else None
            ),
            "singleton_fraction_bits_confidently_correct": (
                self.sums["singleton_confident_correct_bit_count"] / singleton_bits
                if singleton_bits
                else None
            ),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_objective_loss(result: dict[str, Any], objective: str) -> dict[str, Any]:
    key = "set_nll" if objective == "exact_set_nll" else "duplicated_bce"
    for current in [
        result["overall"],
        *result["by_benchmark"].values(),
        *result["by_pareto_multiplicity"].values(),
        *result["by_supervision_group"].values(),
    ]:
        current["objective_loss"] = current[key]
    return result


def finalize_nonempty(accumulators: dict[str, Any]) -> dict[str, Any]:
    """Finalize only strata represented in a bounded technical subset."""
    return {
        name: accumulator.finalize()
        for name, accumulator in accumulators.items()
        if accumulator.examples
    }


@torch.no_grad()
def evaluate_split(
    predictor: torch.nn.Module,
    encoder: torch.nn.Module,
    loader: DataLoader,
    metadata: dict[str, dict[str, Any]],
    *,
    modality: str,
    device: torch.device,
    epoch: int,
    split: str,
    objective: str,
) -> dict[str, Any]:
    predictor.eval()
    encoder.eval()
    overall = MetricAccumulator()
    benchmarks = {name: MetricAccumulator() for name in BENCHMARKS}
    multiplicity = {
        "singleton": MetricAccumulator(),
        "doubleton": MetricAccumulator(),
        "three_or_more": MetricAccumulator(),
    }
    groups = {name: MetricAccumulator() for name in ("A", "B", "C")}
    probability = ProbabilityAccumulator()
    original_valid_masks = {
        uid: {tuple(int(bit) for bit in key) for key in item["original_valid_mask_keys"]}
        for uid, item in metadata.items()
    }

    def update_metric(accumulator: MetricAccumulator, indices: torch.Tensor, batch, logits):
        selected_uids = [batch["uids"][index] for index in indices.tolist()]
        accumulator.update(
            logits.index_select(0, indices),
            batch["valid_masks"].index_select(0, indices),
            batch["valid_mask"].index_select(0, indices),
            batch["route_weights"].index_select(0, indices),
            uids=selected_uids,
            original_valid_masks=original_valid_masks,
        )

    progress = tqdm(
        loader,
        desc=f"{objective} {split} e{epoch:02d}",
        unit="batch",
        dynamic_ncols=True,
    )
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = forward_logits(predictor, encoder, batch, modality).float()
        all_indices = torch.arange(logits.shape[0], device=device)
        update_metric(overall, all_indices, batch, logits)
        probability.update(
            batch_probability_diagnostics(
                logits, batch["valid_masks"], batch["valid_mask"], batch["route_weights"]
            )
        )
        for name in BENCHMARKS:
            indices = torch.tensor(
                [i for i, uid in enumerate(batch["uids"]) if metadata[uid]["benchmark"] == name],
                device=device,
                dtype=torch.long,
            )
            if indices.numel():
                update_metric(benchmarks[name], indices, batch, logits)
        for name, predicate in (
            ("singleton", lambda row: row["pareto_efficient_route_count"] == 1),
            ("doubleton", lambda row: row["pareto_efficient_route_count"] == 2),
            ("three_or_more", lambda row: row["pareto_efficient_route_count"] >= 3),
        ):
            indices = torch.tensor(
                [i for i, uid in enumerate(batch["uids"]) if predicate(metadata[uid])],
                device=device,
                dtype=torch.long,
            )
            if indices.numel():
                update_metric(multiplicity[name], indices, batch, logits)
        for name in groups:
            indices = torch.tensor(
                [
                    i
                    for i, uid in enumerate(batch["uids"])
                    if metadata[uid]["supervision_group"] == name
                ],
                device=device,
                dtype=torch.long,
            )
            if indices.numel():
                update_metric(groups[name], indices, batch, logits)
        progress.set_postfix(hit=f"{overall.sample_sums.get('top1_valid_route_coverage', 0.0) / overall.examples:.3f}")
    result = {
        "overall": overall.finalize(),
        "by_benchmark": finalize_nonempty(benchmarks),
        "by_pareto_multiplicity": finalize_nonempty(multiplicity),
        "by_supervision_group": finalize_nonempty(groups),
        "probability_diagnostics": probability.finalize(),
    }
    return add_objective_loss(result, objective)


def validation_reproduction_delta(computed: dict[str, Any], logged: dict[str, Any]) -> dict[str, float]:
    keys = (
        "pareto_valid_hit_at_1",
        "original_valid_hit_at_1",
        "nearest_valid_hamming",
        "average_predicted_visual_on",
        "fraction_top1_all_on",
        "fraction_top1_all_off",
        "top1_mask_entropy_nats",
        "objective_loss",
    )
    return {key: abs(float(computed[key]) - float(logged[key])) for key in keys}


VALIDATION_REPRODUCTION_TOLERANCES = {
    "exact": {
        key: 1e-8
        for key in (
            "pareto_valid_hit_at_1",
            "original_valid_hit_at_1",
            "nearest_valid_hamming",
            "average_predicted_visual_on",
            "fraction_top1_all_on",
            "fraction_top1_all_off",
            "top1_mask_entropy_nats",
            "objective_loss",
        )
    },
    # The frozen checkpoints were trained/evaluated on an A6000. This profile
    # permits only the small, predeclared BF16 threshold sensitivity observed
    # when re-evaluating on an A4000; the original logged validation trajectory
    # remains the scientific validation result.
    "ampere_diagnostic": {
        "pareto_valid_hit_at_1": 1 / 874,
        "original_valid_hit_at_1": 1 / 874,
        "nearest_valid_hamming": 0.02,
        "average_predicted_visual_on": 0.02,
        "fraction_top1_all_on": 1 / 874,
        "fraction_top1_all_off": 1 / 874,
        "top1_mask_entropy_nats": 0.02,
        "objective_loss": 1e-4,
    },
}
VALIDATION_REPRODUCTION_RELATIVE_TOLERANCES = {
    "exact": {},
    "ampere_diagnostic": {"objective_loss": 5e-5},
}


def validation_reproduction_passes(
    deltas: dict[str, float], profile: str, reference: dict[str, Any] | None = None
) -> bool:
    """Apply a frozen, metric-specific validation reproduction gate."""
    if profile not in VALIDATION_REPRODUCTION_TOLERANCES:
        raise ValueError(f"unknown validation reproduction profile: {profile}")
    tolerances = VALIDATION_REPRODUCTION_TOLERANCES[profile]
    if set(deltas) != set(tolerances):
        raise ValueError("validation reproduction metrics differ from the frozen gate")
    relative = VALIDATION_REPRODUCTION_RELATIVE_TOLERANCES[profile]
    return all(
        float(deltas[key])
        <= max(
            tolerance,
            relative.get(key, 0.0) * abs(float(reference[key])) if reference else 0.0,
        )
        for key, tolerance in tolerances.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("duplicated_bce", "exact_set_nll"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", default="1-10", help="Inclusive range such as 1-10 or one epoch")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Technical smoke only: evaluate the first N records of each split.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Evaluation-only loader override; batch membership and model math are unchanged.",
    )
    parser.add_argument(
        "--validation-reproduction-profile",
        choices=tuple(VALIDATION_REPRODUCTION_TOLERANCES),
        default="exact",
        help="Frozen checkpoint-validation parity gate.",
    )
    parser.add_argument(
        "--evaluation-splits",
        choices=("train", "train_validation"),
        default="train_validation",
        help="Evaluate train only when frozen logged validation metrics are authoritative.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint fitting analysis requires a scheduled GPU")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["training"]["objective"] != args.objective:
        raise RuntimeError("objective differs from frozen config")
    for name, specification in config["gates"].items():
        validate_gate(name, specification)
    for path_value, expected in config["source_sha256"].items():
        if file_sha256(Path(path_value)) != expected:
            raise RuntimeError(f"frozen source checksum mismatch: {path_value}")
    manifest_path = Path(config["data"]["manifest"])
    feature_manifest_path = Path(config["visual_features"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("Pareto manifest checksum mismatch")
    if file_sha256(feature_manifest_path) != config["visual_features"]["manifest_sha256"]:
        raise RuntimeError("visual feature manifest checksum mismatch")

    start_text, separator, stop_text = args.epochs.partition("-")
    start = int(start_text)
    stop = int(stop_text) if separator else start
    epochs = list(range(start, stop + 1))
    if not epochs or min(epochs) < 1 or max(epochs) > 10:
        raise ValueError("epochs must be within 1-10")

    history = json.loads((args.training_dir / "history.json").read_text(encoding="utf-8"))
    if len(history) != 10 or [int(row["epoch"]) for row in history] != list(range(1, 11)):
        raise RuntimeError("training history is not the complete frozen ten-epoch trajectory")
    feature_index = {row["uid"]: row for row in read_jsonl(feature_manifest_path)}
    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    datasets = {
        split: BinaryPolicyManifestDataset(manifest_path, split, max_valid_routes=route_cap)
        for split in ("train", "validation")
    }
    if len(datasets["train"]) != 6043 or len(datasets["validation"]) != 874:
        raise RuntimeError("positive split population differs from 6043/874")
    if args.max_samples_per_split < 0:
        raise ValueError("max-samples-per-split cannot be negative")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.max_samples_per_split:
        for dataset in datasets.values():
            dataset.rows = dataset.rows[: args.max_samples_per_split]
    if set(row["uid"] for dataset in datasets.values() for row in dataset.rows) - feature_index.keys():
        raise RuntimeError("visual feature cache is incomplete")

    seed = int(config["training"]["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    predictor = BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
        image_dim=int(config["visual_features"]["feature_width"]),
    ).to(device)
    collator = make_multimodal_set_collator(
        tokenizer,
        feature_index,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=config["data"]["route_weighting"],
    )
    requested_splits = (
        ("train", "validation")
        if args.evaluation_splits == "train_validation"
        else ("train",)
    )
    loaders = {
        split: DataLoader(
            dataset,
            shuffle=False,
            batch_size=int(config["training"]["physical_batch_size"]),
            num_workers=(
                int(config["training"]["num_workers"])
                if args.num_workers is None
                else args.num_workers
            ),
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=collator,
        )
        for split, dataset in datasets.items()
        if split in requested_splits
    }
    metadata = {
        split: {str(row["uid"]): row for row in dataset.rows}
        for split, dataset in datasets.items()
    }

    output = {
        "schema_version": "binary_pareto_training_fit_v1",
        "passed": False,
        "objective": args.objective,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "training_dir": str(args.training_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "max_samples_per_split": args.max_samples_per_split,
        "num_workers": (
            int(config["training"]["num_workers"])
            if args.num_workers is None
            else args.num_workers
        ),
        "validation_reproduction_profile": args.validation_reproduction_profile,
        "validation_reproduction_tolerances": VALIDATION_REPRODUCTION_TOLERANCES[
            args.validation_reproduction_profile
        ],
        "validation_reproduction_relative_tolerances":
            VALIDATION_REPRODUCTION_RELATIVE_TOLERANCES[
                args.validation_reproduction_profile
            ],
        "epochs": [],
        "evaluation_splits": list(requested_splits),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        checkpoint_path = args.training_dir / f"epoch_{epoch:02d}/checkpoint.pt"
        expected_sha = (args.training_dir / f"epoch_{epoch:02d}/checkpoint.sha256").read_text().split()[0]
        actual_sha = file_sha256(checkpoint_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"checkpoint checksum mismatch at epoch {epoch}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint["epoch"]) != epoch:
            raise RuntimeError(f"checkpoint epoch mismatch at {epoch}")
        predictor.load_state_dict(checkpoint["predictor"], strict=True)
        current = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": actual_sha,
            "logged_online_train": history[epoch - 1]["train"],
            "train": evaluate_split(
                predictor,
                encoder,
                loaders["train"],
                metadata["train"],
                modality="image_question",
                device=device,
                epoch=epoch,
                split="train",
                objective=args.objective,
            ),
        }
        if "validation" in requested_splits:
            current["validation"] = evaluate_split(
                predictor,
                encoder,
                loaders["validation"],
                metadata["validation"],
                modality="image_question",
                device=device,
                epoch=epoch,
                split="validation",
                objective=args.objective,
            )
        if "validation" not in requested_splits or args.max_samples_per_split:
            current["validation_reproduction_max_abs"] = None
            current["validation_reproduction_by_metric"] = None
        else:
            deltas = validation_reproduction_delta(
                current["validation"]["overall"], history[epoch - 1]["validation"]["overall"]
            )
            current["validation_reproduction_max_abs"] = max(deltas.values())
            current["validation_reproduction_by_metric"] = deltas
            if not validation_reproduction_passes(
                deltas,
                args.validation_reproduction_profile,
                history[epoch - 1]["validation"]["overall"],
            ):
                raise RuntimeError(
                    "validation reproduction mismatch at epoch "
                    f"{epoch} under {args.validation_reproduction_profile}: {deltas}"
                )
        output["epochs"].append(current)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "objective": args.objective,
                    "epoch": epoch,
                    "train_hit": current["train"]["overall"]["pareto_valid_hit_at_1"],
                    "validation_hit": history[epoch - 1]["validation"]["overall"][
                        "pareto_valid_hit_at_1"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output["passed"] = True
    output["completed_epochs"] = epochs
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "output": str(args.output), "sha256": digest}))


if __name__ == "__main__":
    main()
