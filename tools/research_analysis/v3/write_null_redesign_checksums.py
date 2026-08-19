from __future__ import annotations

import hashlib
from pathlib import Path


PATHS = [
    Path("outputs/v3_null_redesign/calibration_pool_manifest.json"),
    Path("outputs/v3_null_redesign/calibration_pool_manifest_v2.json"),
    Path("outputs/v3_null_redesign/donor_coverage.json"),
    Path("outputs/v3_null_redesign/donor_coverage_v2.json"),
    Path("outputs/v3_null_redesign/donor_pool_size_curve.json"),
    Path("outputs/v3_null_redesign/covariance_representation_comparison.json"),
    Path("outputs/v3_null_redesign/covariance_representation_c_rank_extension.json"),
    Path("artifacts/v3_null_redesign/read_write_geometry_combined_v3/manifest.json"),
    Path("artifacts/v3_null_redesign/paired_donor_index_v3/manifest.json"),
    Path("artifacts/v3_null_redesign/joint_covariance_models_v2/manifest.json"),
    Path("artifacts/v3_null_redesign/joint_covariance_models_c_rank1024_v1/manifest.json"),
    Path("data_manifests/v3_null_redesign_calibration_2000_v1.jsonl"),
    Path("data_manifests/v3_null_redesign_calibration_4000_v2.jsonl"),
    Path("data_manifests/v3_null_redesign_calibration_delta_2000_v2.jsonl"),
    Path("workspace/v3_structured_null_spec_v3_candidate.md"),
    Path("reports/v3_null_redesign_v2.md"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute() -> None:
    missing = [path for path in PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checksum inputs: {missing}")
    output = Path("outputs/v3_null_redesign/SHA256SUMS")
    output.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in PATHS), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    execute()
