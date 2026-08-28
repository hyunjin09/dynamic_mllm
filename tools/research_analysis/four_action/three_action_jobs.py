from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


THREE_ACTION_CODE_PATHS = (
    "binary_policy/actions.py",
    "binary_policy/executor/__init__.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/runtime.py",
    "reference/dvr_qwen/eval_metrics.py",
    "tools/research_analysis/four_action/label_runtime.py",
    "tools/research_analysis/four_action/label_jobs.py",
    "tools/research_analysis/four_action/targets.py",
    "tools/research_analysis/four_action/three_action_labels.py",
    "tools/research_analysis/four_action/three_action_jobs.py",
    "experiments/run_three_action_label_conversion.py",
    "experiments/finalize_three_action_noise_calibration.py",
    "experiments/audit_three_action_label_pilot.py",
    "experiments/estimate_three_action_label_conversion.py",
    "experiments/run_three_action_calibration_pilot.sh",
    "experiments/run_three_action_full.sh",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_three_action_execution_contract(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    epsilon_path: Path | None,
    config: dict[str, Any],
    git_commit: str,
    torch_version: str,
    transformers_version: str,
    mode: str,
) -> dict[str, Any]:
    """Bind a calibration/conversion run to code, data, model, and epsilon."""
    if mode not in {"calibrate", "pilot", "full"}:
        raise ValueError(f"unsupported three-action run mode: {mode}")
    root = Path(project_root).resolve()
    code_hashes = {}
    for relative in THREE_ACTION_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_hashes[relative] = file_sha256(path)
    model = config["model"]
    snapshot = Path(model["snapshot_path"])
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    contract: dict[str, Any] = {
        "schema_version": "three_action_answer_aligned_execution_contract_v1",
        "mode": mode,
        "git_commit": str(git_commit),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": file_sha256(Path(manifest_path)),
        "config": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(Path(config_path)),
        "model_snapshot_path": str(snapshot.resolve()),
        "model_revision": str(model["revision"]),
        "attention_implementation": str(model["attention_implementation"]),
        "seed": int(config["seed"]),
        "beam_width": int(config["beam_width"]),
        "beam_validation_width": int(config["beam_validation_width"]),
        "layer_count": int(config["layer_count"]),
        "action_aliases": {
            "READ_OFF": "WRITE_ONLY",
            "WRITE_OFF": "READ_ONLY",
            "BOTH_OFF": "IGNORE",
            "FULL": "FULL_REFERENCE",
        },
        "torch_version": str(torch_version),
        "transformers_version": str(transformers_version),
        "code_sha256": code_hashes,
    }
    if epsilon_path is None:
        if mode != "calibrate":
            raise ValueError("pilot/full contracts require a frozen epsilon artifact")
        noise = config["noise_calibration"]
        contract["noise_calibration"] = {
            "repetitions": int(noise["repetitions"]),
            "absolute_quantile": float(noise["absolute_quantile"]),
            "mean_score_floor": float(noise["mean_score_floor"]),
        }
    else:
        epsilon_file = Path(epsilon_path)
        payload = json.loads(epsilon_file.read_text(encoding="utf-8"))
        contract.update(
            {
                "epsilon_artifact": str(epsilon_file.resolve()),
                "epsilon_sha256": file_sha256(epsilon_file),
                "epsilon": float(payload["epsilon"]),
                "epsilon_selection_rule": str(payload.get("selection_rule", "")),
            }
        )
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = sha256(encoded.encode()).hexdigest()
    return contract
