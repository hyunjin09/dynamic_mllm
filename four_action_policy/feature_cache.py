"""Fail-closed loading and audit of projected visual-row caches."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import torch

from experiments.train_binary_polar import file_sha256


def visual_cache_contract(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "four_action_polar_visual_cache_contract_v1",
        "predictor_manifest_sha256": config["data"]["manifest_sha256"],
        "records": int(config["data"]["train_records"])
        + int(config["data"]["validation_records"]),
        "unique_image_groups": int(config["data"]["unique_image_groups"]),
        "model_revision": str(config["base_model"]["revision"]),
        "feature_source": str(config["visual_features"]["source"]),
        "feature_width": int(config["visual_features"]["feature_width"]),
        "dtype": str(config["visual_features"]["dtype"]),
        "unpooled": bool(config["visual_features"]["unpooled"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": sha256(encoded).hexdigest()}


def read_feature_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verified_feature_index(
    manifest_path: str | Path,
    *,
    manifest_sha256: str,
    expected_uids: Iterable[str],
    expected_feature_width: int,
    expected_dtype: str = "torch.bfloat16",
    verify_tensors: bool = True,
) -> dict[str, dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"visual feature manifest is missing: {path}")
    if file_sha256(path) != manifest_sha256:
        raise RuntimeError("visual feature manifest checksum mismatch")
    rows = read_feature_manifest(path)
    index: dict[str, dict[str, Any]] = {}
    group_records: dict[str, tuple[str, str, tuple[int, ...], str]] = {}
    for row in rows:
        uid = str(row.get("uid") or "")
        group = str(row.get("split_group") or "")
        if not uid or not group or uid in index:
            raise RuntimeError(f"invalid or duplicate visual feature identity: {uid!r}")
        tensor_path = Path(row.get("path") or "")
        shape = tuple(int(value) for value in row.get("shape", []))
        dtype = str(row.get("dtype") or "")
        width = int(row.get("feature_width", -1))
        if len(shape) != 2 or shape[0] < 1 or shape[1] != expected_feature_width:
            raise RuntimeError(f"visual feature shape metadata mismatch for {uid}")
        if width != expected_feature_width or dtype != expected_dtype:
            raise RuntimeError(f"visual feature dtype/width mismatch for {uid}")
        declaration = (str(tensor_path), str(row.get("sha256") or ""), shape, dtype)
        previous = group_records.setdefault(group, declaration)
        if previous != declaration:
            raise RuntimeError(f"visual feature group has inconsistent declarations: {group}")
        index[uid] = row
    expected = {str(uid) for uid in expected_uids}
    observed = set(index)
    if observed != expected:
        raise RuntimeError(
            "visual feature UID coverage mismatch: "
            f"missing={len(expected - observed)} unexpected={len(observed - expected)}"
        )
    if verify_tensors:
        for tensor_path_value, expected_sha, expected_shape, expected_tensor_dtype in sorted(
            set(group_records.values())
        ):
            tensor_path = Path(tensor_path_value)
            if not tensor_path.is_file() or file_sha256(tensor_path) != expected_sha:
                raise RuntimeError(f"visual feature tensor checksum mismatch: {tensor_path}")
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
                raise RuntimeError(f"visual feature tensor shape mismatch: {tensor_path}")
            if str(tensor.dtype) != expected_tensor_dtype or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"visual feature tensor dtype/finite mismatch: {tensor_path}")
    return index
