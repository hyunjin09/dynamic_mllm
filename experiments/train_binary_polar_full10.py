#!/usr/bin/env python3
"""Run one frozen full10 Question or Image+Question predictor trajectory."""

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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from binary_policy.dataset import (
    BinaryPolicyManifestDataset,
    make_duplicated_path_collator,
    make_set_collator,
)
from binary_policy.evaluation import batch_offline_metrics, mask_diversity_metrics
from binary_policy.losses import multi_valid_set_nll, polar_path_bce_per_path
from binary_policy.multimodal import (
    make_multimodal_duplicated_path_collator,
    make_multimodal_set_collator,
)
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import file_sha256, seed_worker, validate_gate


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def optimizer_steps_per_epoch(sample_count: int, physical_batch: int, accumulation: int) -> int:
    if min(sample_count, physical_batch, accumulation) < 1:
        raise ValueError("sample_count, physical_batch, and accumulation must be positive")
    return math.ceil(math.ceil(sample_count / physical_batch) / accumulation)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_logits(predictor, encoder, batch: dict, modality: str) -> torch.Tensor:
    with torch.no_grad():
        question = encoder(batch["input_ids"], batch["attention_mask"])
    if modality == "question":
        return predictor(question, batch["attention_mask"])
    return predictor(
        question,
        batch["attention_mask"],
        batch["image_features"],
        batch["image_attention_mask"],
    )


def forward_from_features(
    predictor,
    question: torch.Tensor,
    batch: dict,
    modality: str,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the predictor from one unique-input encoder pass.

    ``indices`` expands unique inputs into duplicated valid-route rows for the
    POLAR-style BCE objective without recomputing the frozen question encoder.
    """
    if indices is not None:
        question = question.index_select(0, indices)
        attention_mask = batch["attention_mask"].index_select(0, indices)
    else:
        attention_mask = batch["attention_mask"]
    if modality == "question":
        return predictor(question, attention_mask)
    image_features = batch["image_features"]
    image_attention_mask = batch["image_attention_mask"]
    if indices is not None:
        image_features = image_features.index_select(0, indices)
        image_attention_mask = image_attention_mask.index_select(0, indices)
    return predictor(question, attention_mask, image_features, image_attention_mask)


def scale_gradients_to_sample_mean(module, sample_count: int) -> None:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


def train_epoch(
    predictor,
    encoder,
    loader,
    optimizer,
    scheduler,
    *,
    modality: str,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip_norm: float,
    epoch: int,
    global_step: int,
    objective: str,
    duplicated_route_microbatch_size: int,
) -> tuple[dict, int]:
    if objective not in {"exact_set_nll", "duplicated_bce"}:
        raise ValueError(f"unsupported full10 objective: {objective}")
    if duplicated_route_microbatch_size < 1:
        raise ValueError("duplicated_route_microbatch_size must be positive")
    predictor.train()
    encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    accumulated_examples = 0
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(loader, desc=f"{modality} train e{epoch:02d}", unit="batch", dynamic_ncols=True)
    for batch_index, raw_batch in enumerate(progress, start=1):
        batch = move_batch(raw_batch, device)
        with torch.no_grad():
            question = encoder(batch["input_ids"], batch["attention_mask"])
        if objective == "exact_set_nll":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_from_features(predictor, question, batch, modality)
                loss = multi_valid_set_nll(
                    logits,
                    batch["valid_masks"],
                    valid_mask=batch["valid_mask"],
                    route_weights=batch["route_weights"],
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"nonfinite {modality} loss at epoch {epoch} batch {batch_index}"
                )
            batch_size = int(logits.shape[0])
            batch_loss_sum = loss * batch_size
            batch_loss_sum.backward()
            batch_loss_value = float(batch_loss_sum.detach())
        else:
            batch_size = int(batch["unique_examples"])
            route_count = int(batch["targets"].shape[0])
            batch_loss_value = 0.0
            for start in range(0, route_count, duplicated_route_microbatch_size):
                stop = min(start + duplicated_route_microbatch_size, route_count)
                indices = batch["route_sample_index"][start:stop]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = forward_from_features(
                        predictor, question, batch, modality, indices=indices
                    )
                    per_path = polar_path_bce_per_path(logits, batch["targets"][start:stop])
                    chunk_loss_sum = (
                        per_path
                        * batch["sample_weights"][start:stop].to(per_path.dtype)
                    ).sum()
                if not bool(torch.isfinite(chunk_loss_sum)):
                    raise FloatingPointError(
                        f"nonfinite {modality} BCE at epoch {epoch} batch {batch_index}"
                    )
                chunk_loss_sum.backward()
                batch_loss_value += float(chunk_loss_sum.detach())
        # Each input's normalized route weights sum to one. Divide accumulated
        # gradient sums by the exact unique-input count before the optimizer
        # step, including the final partial accumulation group.
        accumulated_examples += batch_size
        total_examples += batch_size
        total_loss += batch_loss_value
        should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            scale_gradients_to_sample_mean(predictor, accumulated_examples)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_examples = 0
            global_step += 1
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            step=global_step,
        )
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("frozen question encoder received gradients")
    return {
        "loss": total_loss / total_examples,
        "objective": objective,
        ("set_nll" if objective == "exact_set_nll" else "duplicated_bce"):
            total_loss / total_examples,
        "examples": total_examples,
        "optimizer_steps": global_step,
    }, global_step


class MetricAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.loss_sum = 0.0
        self.bce_loss_sum = 0.0
        self.sample_sums: dict[str, float] = {}
        self.mask_counts: Counter[tuple[int, ...]] = Counter()
        self.original_valid_hits = 0
        self.original_valid_examples = 0
        self.original_valid_hamming_sum = 0.0

    def update(
        self,
        logits,
        valid_masks,
        valid_mask,
        route_weights,
        *,
        uids: list[str] | None = None,
        original_valid_masks: dict[str, set[tuple[int, ...]]] | None = None,
    ) -> None:
        count = int(logits.shape[0])
        loss = multi_valid_set_nll(
            logits,
            valid_masks,
            valid_mask=valid_mask,
            route_weights=route_weights,
        )
        expanded_logits = logits.unsqueeze(1).expand_as(valid_masks)
        per_route_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            expanded_logits,
            valid_masks.to(dtype=logits.dtype),
            reduction="none",
        ).mean(dim=-1)
        normalized_weights = torch.where(
            valid_mask,
            route_weights.to(dtype=logits.dtype),
            torch.zeros_like(route_weights, dtype=logits.dtype),
        )
        normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True)
        bce_loss = (per_route_bce * normalized_weights).sum(dim=1).mean()
        metrics = batch_offline_metrics(logits, valid_masks, valid_mask, top_k=5)
        self.examples += count
        self.loss_sum += float(loss) * count
        self.bce_loss_sum += float(bce_loss) * count
        for serialized, occurrences in metrics["top1_mask_counts"].items():
            self.mask_counts[tuple(int(bit) for bit in serialized)] += int(occurrences)
        for key in (
            "top1_valid_route_coverage",
            "topk_valid_route_coverage",
            "nearest_valid_hamming",
            "nearest_valid_on_count_error",
        ):
            self.sample_sums[key] = self.sample_sums.get(key, 0.0) + float(metrics[key]) * count
        if original_valid_masks is not None:
            if uids is None or len(uids) != count:
                raise ValueError("UIDs must align with logits for original-valid metrics")
            decoded = (logits >= 0).to(torch.int64).detach().cpu().tolist()
            for uid, mask in zip(uids, decoded):
                predicted = tuple(mask)
                candidates = original_valid_masks[uid]
                self.original_valid_hits += int(predicted in candidates)
                self.original_valid_hamming_sum += min(
                    sum(left != right for left, right in zip(predicted, candidate))
                    for candidate in candidates
                )
            self.original_valid_examples += count

    def finalize(self) -> dict:
        if self.examples < 1:
            raise ValueError("cannot finalize an empty metric accumulator")
        result = {
            "examples": self.examples,
            "set_nll": self.loss_sum / self.examples,
            "duplicated_bce": self.bce_loss_sum / self.examples,
            **{key: value / self.examples for key, value in self.sample_sums.items()},
            **mask_diversity_metrics(self.mask_counts),
        }
        result["pareto_valid_hit_at_1"] = result["top1_valid_route_coverage"]
        result["original_valid_hit_at_1"] = (
            self.original_valid_hits / self.original_valid_examples
            if self.original_valid_examples
            else result["top1_valid_route_coverage"]
        )
        result["nearest_original_valid_hamming"] = (
            self.original_valid_hamming_sum / self.original_valid_examples
            if self.original_valid_examples
            else result["nearest_valid_hamming"]
        )
        return result


@torch.no_grad()
def validate_epoch(
    predictor,
    encoder,
    loader,
    *,
    modality: str,
    device: torch.device,
    epoch: int,
    validation_metadata: dict[str, dict] | None = None,
    objective: str = "exact_set_nll",
) -> dict:
    if objective not in {"exact_set_nll", "duplicated_bce"}:
        raise ValueError(f"unsupported validation objective: {objective}")
    predictor.eval()
    encoder.eval()
    accumulators = {"overall": MetricAccumulator(), **{name: MetricAccumulator() for name in BENCHMARKS}}
    multiplicity = {
        "singleton": MetricAccumulator(),
        "doubleton": MetricAccumulator(),
        "three_or_more": MetricAccumulator(),
    }
    groups = {name: MetricAccumulator() for name in ("A", "B", "C")}
    original_valid_masks = (
        {
            uid: {tuple(int(bit) for bit in key) for key in item["original_valid_mask_keys"]}
            for uid, item in validation_metadata.items()
        }
        if validation_metadata is not None
        else None
    )

    def update_subset(accumulator, indices, batch, logits):
        selected_uids = [batch["uids"][index] for index in indices.tolist()]
        accumulator.update(
            logits.index_select(0, indices),
            batch["valid_masks"].index_select(0, indices),
            batch["valid_mask"].index_select(0, indices),
            batch["route_weights"].index_select(0, indices),
            uids=selected_uids,
            original_valid_masks=original_valid_masks,
        )
    progress = tqdm(loader, desc=f"{modality} valid e{epoch:02d}", unit="batch", dynamic_ncols=True)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = forward_logits(predictor, encoder, batch, modality).float()
        accumulators["overall"].update(
            logits,
            batch["valid_masks"],
            batch["valid_mask"],
            batch["route_weights"],
            uids=batch["uids"],
            original_valid_masks=original_valid_masks,
        )
        for benchmark in BENCHMARKS:
            indices = torch.tensor(
                [index for index, uid in enumerate(batch["uids"]) if uid.startswith(f"{benchmark}:")],
                device=device,
                dtype=torch.long,
            )
            if indices.numel() == 0:
                continue
            update_subset(accumulators[benchmark], indices, batch, logits)
        if validation_metadata is not None:
            for name, condition in (
                ("singleton", lambda item: item.get("supervision_route_count", item.get("pareto_efficient_route_count")) == 1),
                ("doubleton", lambda item: item.get("supervision_route_count", item.get("pareto_efficient_route_count")) == 2),
                ("three_or_more", lambda item: item.get("supervision_route_count", item.get("pareto_efficient_route_count")) >= 3),
            ):
                indices = torch.tensor(
                    [
                        index
                        for index, uid in enumerate(batch["uids"])
                        if condition(validation_metadata[uid])
                    ],
                    device=device,
                    dtype=torch.long,
                )
                if indices.numel():
                    update_subset(multiplicity[name], indices, batch, logits)
            for name in groups:
                indices = torch.tensor(
                    [
                        index
                        for index, uid in enumerate(batch["uids"])
                        if validation_metadata[uid]["supervision_group"] == name
                    ],
                    device=device,
                    dtype=torch.long,
                )
                if indices.numel():
                    update_subset(groups[name], indices, batch, logits)
        current = accumulators["overall"]
        progress.set_postfix(nll=f"{current.loss_sum / current.examples:.4f}")
    result = {
        "overall": accumulators["overall"].finalize(),
        "by_benchmark": {name: accumulators[name].finalize() for name in BENCHMARKS},
        "by_pareto_multiplicity": {
            name: accumulator.finalize()
            for name, accumulator in multiplicity.items()
            if accumulator.examples
        },
        "by_supervision_group": {
            name: accumulator.finalize()
            for name, accumulator in groups.items()
            if accumulator.examples
        },
    }
    result["by_supervision_multiplicity"] = result["by_pareto_multiplicity"]
    for current in [
        result["overall"],
        *result["by_benchmark"].values(),
        *result["by_pareto_multiplicity"].values(),
        *result["by_supervision_group"].values(),
    ]:
        current["objective_loss"] = current[
            "set_nll" if objective == "exact_set_nll" else "duplicated_bce"
        ]
    return result


def checkpoint_key(row: dict) -> tuple:
    metrics = row["validation"]["overall"]
    return (
        float(metrics["top1_valid_route_coverage"]),
        -float(metrics["nearest_valid_hamming"]),
        -float(metrics["objective_loss"]),
        -int(row["epoch"]),
    )


def save_epoch_checkpoint(
    output_dir: Path,
    predictor,
    optimizer,
    scheduler,
    *,
    epoch: int,
    global_step: int,
    config: dict,
    config_sha256: str,
    metrics: dict,
) -> dict:
    epoch_dir = output_dir / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=False, exist_ok=False)
    path = epoch_dir / "checkpoint.pt"
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "config_sha256": config_sha256,
            "metrics": metrics,
        },
        path,
    )
    digest = file_sha256(path)
    record = {
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "metrics": metrics,
    }
    (epoch_dir / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (epoch_dir / "checkpoint.sha256").write_text(f"{digest}  checkpoint.pt\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--modality", choices=("question", "image_question"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument(
        "--objective", choices=("exact_set_nll", "duplicated_bce"), required=True
    )
    parser.add_argument("--confirm-full10", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full10:
        raise RuntimeError("full10 training requires --confirm-full10")
    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.objective != config["training"]["objective"]:
        raise RuntimeError("CLI objective differs from the frozen full10 config")
    if config["source_plan"]["sha256"] != file_sha256(Path(config["source_plan"]["path"])):
        raise RuntimeError("full10 source-plan checksum mismatch")
    if config["authorization"] not in {
        "full10_question_and_image_question_only",
        "binary_pareto_full10_image_question_only",
        "binary_cap_sweep_full10_image_question_only",
        "binary_cap_nll5_executed_validation_image_question_only",
    }:
        raise RuntimeError("config does not authorize this full10 action")
    for name, specification in config["gates"].items():
        validate_gate(name, specification)
    for source_value, expected in config["source_sha256"].items():
        if file_sha256(Path(source_value)) != expected:
            raise RuntimeError(f"full10 source checksum mismatch: {source_value}")
    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or preflight.get("config_sha256") != config_sha:
        raise RuntimeError("full10 runtime preflight failed or used another config")
    if not torch.cuda.is_available():
        raise RuntimeError("full10 training requires a scheduled GPU")

    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    feature_manifest_path = Path(config["visual_features"]["manifest"])
    if file_sha256(feature_manifest_path) != config["visual_features"]["manifest_sha256"]:
        raise RuntimeError("full visual-feature manifest checksum mismatch")
    feature_index = {row["uid"]: row for row in read_jsonl(feature_manifest_path)}

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
    architecture = dict(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    )
    seed_everything(seed)
    shared_reference = BinaryPolarBackbone(**architecture)
    shared_state = shared_reference.state_dict()
    shared_sha = predictor_state_sha256(shared_reference)
    seed_everything(seed)
    predictor = BinaryPolarBackbone(
        **architecture,
        **(
            {"image_dim": int(config["visual_features"]["feature_width"])}
            if args.modality == "image_question"
            else {}
        ),
    ).to(device)
    predictor_state = predictor.state_dict()
    mismatches = [
        name for name, tensor in shared_state.items() if not torch.equal(tensor, predictor_state[name].cpu())
    ]
    if mismatches:
        raise RuntimeError(f"full10 shared initialization mismatch: {mismatches[:3]}")
    del shared_reference

    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    train_dataset = BinaryPolicyManifestDataset(manifest_path, "train", max_valid_routes=route_cap)
    validation_dataset = BinaryPolicyManifestDataset(
        manifest_path, "validation", max_valid_routes=route_cap
    )
    expected_train = int(config["data"].get("train_positive_records", 6043))
    expected_validation = int(config["data"].get("validation_positive_records", 874))
    if len(train_dataset) != expected_train or len(validation_dataset) != expected_validation:
        raise RuntimeError(
            "full10 positive population differs from config: "
            f"{len(train_dataset)}/{len(validation_dataset)} != "
            f"{expected_train}/{expected_validation}"
        )
    if set(row["uid"] for row in train_dataset.rows + validation_dataset.rows) - feature_index.keys():
        raise RuntimeError("full10 feature cache is incomplete")
    validation_metadata = {str(row["uid"]): row for row in validation_dataset.rows}
    train_metadata = {str(row["uid"]): row for row in train_dataset.rows}
    if args.modality == "image_question":
        unique_tensors = {
            row["path"]: row["sha256"] for row in feature_index.values()
        }
        progress = tqdm(
            sorted(unique_tensors.items()),
            desc="verify visual tensors",
            unit="tensor",
            dynamic_ncols=True,
        )
        for path_value, expected in progress:
            if file_sha256(Path(path_value)) != expected:
                raise RuntimeError(f"full10 visual tensor checksum mismatch: {path_value}")
    validation_collator = (
        make_set_collator(
            tokenizer,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
        if args.modality == "question"
        else make_multimodal_set_collator(
            tokenizer,
            feature_index,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
    )
    if args.objective == "exact_set_nll":
        train_collator = validation_collator
    elif args.modality == "question":
        train_collator = make_duplicated_path_collator(
            tokenizer,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
    else:
        train_collator = make_multimodal_duplicated_path_collator(
            tokenizer,
            feature_index,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
    physical_batch = int(config["training"]["physical_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if physical_batch * accumulation != int(config["training"]["effective_batch_size"]):
        raise RuntimeError("effective batch size is not physical_batch * accumulation")
    common_loader = dict(
        batch_size=physical_batch,
        num_workers=int(config["training"]["num_workers"]),
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, collate_fn=validation_collator, **common_loader
    )
    train_evaluation_loader = DataLoader(
        train_dataset, shuffle=False, collate_fn=validation_collator, **common_loader
    )
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = optimizer_steps_per_epoch(len(train_dataset), physical_batch, accumulation)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"]["warmup_steps"]),
        num_training_steps=total_steps,
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite full10 output: {output_dir}")
    output_dir.mkdir(parents=True)
    initialization = {
        "modality": args.modality,
        "seed": seed,
        "full_initialization_sha256": predictor_state_sha256(predictor),
        "shared_initialization_sha256": shared_sha,
        "shared_initialization_matches": True,
        "config": str(config_path),
        "config_sha256": config_sha,
        "physical_batch_size": physical_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": physical_batch * accumulation,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_scheduler_steps": total_steps,
        "warmup_steps": int(config["training"]["warmup_steps"]),
        "objective": args.objective,
    }
    (output_dir / "initialization.json").write_text(
        json.dumps(initialization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history = []
    checkpoints = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        # Epoch-specific seeding makes an epoch-boundary repair reproducible.
        seed_everything(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            collate_fn=train_collator,
            **common_loader,
        )
        train_metrics, global_step = train_epoch(
            predictor,
            encoder,
            train_loader,
            optimizer,
            scheduler,
            modality=args.modality,
            device=device,
            accumulation_steps=accumulation,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            epoch=epoch,
            global_step=global_step,
            objective=args.objective,
            duplicated_route_microbatch_size=int(
                config["training"]["duplicated_route_microbatch_size"]
            ),
        )
        validation = validate_epoch(
            predictor,
            encoder,
            validation_loader,
            modality=args.modality,
            device=device,
            epoch=epoch,
            validation_metadata=validation_metadata,
            objective=args.objective,
        )
        train_evaluation = validate_epoch(
            predictor,
            encoder,
            train_evaluation_loader,
            modality=args.modality,
            device=device,
            epoch=epoch,
            validation_metadata=train_metadata,
            objective=args.objective,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "train_evaluation": train_evaluation,
            "validation": validation,
        }
        history.append(row)
        checkpoint = save_epoch_checkpoint(
            output_dir,
            predictor,
            optimizer,
            scheduler,
            epoch=epoch,
            global_step=global_step,
            config=config,
            config_sha256=config_sha,
            metrics=row,
        )
        checkpoints.append(checkpoint)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"modality": args.modality, **row}, sort_keys=True), flush=True)
    if global_step != total_steps:
        raise RuntimeError(f"optimizer-step mismatch: {global_step} != {total_steps}")
    best_hit = max(history, key=checkpoint_key)
    best_loss = min(
        history,
        key=lambda row: (row["validation"]["overall"]["objective_loss"], row["epoch"]),
    )
    diversity = max(
        history,
        key=lambda row: (
            -row["validation"]["overall"]["fraction_top1_all_on"],
            row["validation"]["overall"]["unique_top1_masks"],
            row["validation"]["overall"]["top1_mask_entropy_nats"],
            -row["epoch"],
        ),
    )
    selections = {
        "best_hit_at_1": int(best_hit["epoch"]),
        "best_validation_loss": int(best_loss["epoch"]),
        "best_set_nll": int(
            min(history, key=lambda row: (row["validation"]["overall"]["set_nll"], row["epoch"]))[
                "epoch"
            ]
        ),
        "lowest_all_on_highest_diversity_diagnostic": int(diversity["epoch"]),
        "final": epochs,
    }
    summary = {
        "schema_version": "binary_polar_full10_training_v1",
        "passed": True,
        "modality": args.modality,
        "objective": args.objective,
        "initialization": initialization,
        "epochs_completed": len(history),
        "global_steps": global_step,
        "selections": selections,
        "checkpoints": [
            {key: value for key, value in record.items() if key != "metrics"}
            for record in checkpoints
        ],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "training_summary.json.sha256").write_text(
        f"{file_sha256(output_dir / 'training_summary.json')}  training_summary.json\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
