#!/usr/bin/env python3
"""Run the amended BP-0A exact-valid-set-NLL implementation sanity check."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from binary_policy.losses import multi_valid_set_nll
from binary_policy.objective_audit import optimize_complete_mask_logits


def file_sha256(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    label_audit_path = Path(args.label_audit)
    label_audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
    label_integrity = {
        "records_parse": label_audit.get("source_record_count") == 4000
        and label_audit.get("geometry", {}).get("samples") == 4000
        and not label_audit.get("invalid_records"),
        "source_counts_match": label_audit.get("source_audit_match", {}).get("passed") is True,
        "group_split_feasible": label_audit.get("group_split_audit", {}).get("passed") is True,
    }
    label_integrity["passed"] = all(label_integrity.values())

    logits = torch.tensor([[0.7, -1.2, 0.2, -0.4]], dtype=torch.float64)
    masks = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.float64)
    raw_weights = torch.tensor([[0.25, 1.0]], dtype=torch.float64)
    normalized_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)
    log_on = F.logsigmoid(logits).unsqueeze(1)
    log_off = F.logsigmoid(-logits).unsqueeze(1)
    manual_complete_mask_logp = (masks * log_on + (1.0 - masks) * log_off).sum(dim=-1)
    manual_loss = -torch.logsumexp(manual_complete_mask_logp + normalized_weights.log(), dim=1).mean()
    implemented_loss = multi_valid_set_nll(logits, masks, route_weights=raw_weights)
    formula_error = abs(float(manual_loss - implemented_loss))

    padded_masks = torch.cat([masks, torch.zeros(1, 1, 4, dtype=torch.float64)], dim=1)
    padded_weights = torch.tensor([[0.25, 1.0, 0.0]], dtype=torch.float64)
    valid_rows = torch.tensor([[True, True, False]])
    padded_loss = multi_valid_set_nll(
        logits, padded_masks, valid_mask=valid_rows, route_weights=padded_weights
    )
    padded_route_error = abs(float(padded_loss - implemented_loss))

    optimization_runs = [
        optimize_complete_mask_logits(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            weights=[0.5, 0.5],
            seed=seed,
            steps=300,
            learning_rate=0.1,
        )
        for seed in (7, 13, 29, 43)
    ]
    optimization_passed = all(
        row["finite_gradients"]
        and row["top1_is_valid"]
        and row["final_loss"] < row["initial_loss"]
        for row in optimization_runs
    )
    objective_passed = formula_error <= 1e-12 and padded_route_error <= 1e-12 and optimization_passed
    report = {
        "schema_version": "binary_polar_bp0a_exact_set_nll_v1",
        "scientific_generalization_evidence": False,
        "runs_model_inference": False,
        "trains_shared_predictor": False,
        "label_audit": str(label_audit_path.resolve()),
        "label_audit_sha256": file_sha256(label_audit_path),
        "old_empirical_marginal_result_role": "label_structure_diagnostic_only",
        "label_integrity": label_integrity,
        "objective": {
            "complete_mask_formula": "-log sum_m normalized(w_m) * P_theta(m|x)",
            "raw_route_weights": raw_weights[0].tolist(),
            "normalized_route_weights": normalized_weights[0].tolist(),
            "manual_loss": float(manual_loss),
            "implemented_loss": float(implemented_loss),
            "manual_formula_absolute_error": formula_error,
            "padded_route_absolute_error": padded_route_error,
            "optimization_runs": optimization_runs,
            "passed": objective_passed,
        },
        "passed": label_integrity["passed"] and objective_passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError("amended BP-0A exact-valid-set-NLL gate failed")


if __name__ == "__main__":
    main()
