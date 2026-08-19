from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable


ARCHIVE_NAME = "stage_b_stage_c_artifacts_v1.tar.gz"
MANIFEST_NAME = "artifact_manifest_v1.jsonl"
SUMMARY_NAME = "archive_summary_v1.json"
CHECKSUM_NAME = "archive_checksums_v1.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the frozen Stage B/C Outcome B archive."
    )
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--output-dir",
        default="archives/stage_b_stage_c_frozen_outcome_b_v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        if child.is_file() and "__pycache__" not in child.parts and child.suffix != ".pyc":
            yield child


def selected_files(project: Path) -> list[Path]:
    selected: set[Path] = set()
    for root_name in ("outputs", "runs"):
        root = project / root_name
        if root.is_dir():
            for child in root.iterdir():
                if root_name == "runs" and child.name.startswith(
                    "stage_b_c_archive_outcome_b_"
                ):
                    continue
                if child.name.startswith(("stage_b", "stage_c")):
                    selected.update(_regular_files(child))

    for root_name in ("reports", "workspace", "configs", "data_manifests"):
        root = project / root_name
        if root.is_dir():
            for child in root.rglob("*"):
                if child.is_file() and child.name.startswith(("stage_b", "stage_c")):
                    selected.add(child)

    phase_root = project / "workspace" / "phase_memory"
    if phase_root.is_dir():
        for child in phase_root.iterdir():
            if child.is_file() and child.name.startswith(("phase_02_", "phase_03_")):
                selected.add(child)

    for root_name in ("experiments", "analysis", "tests"):
        root = project / root_name
        if root.is_dir():
            for child in root.rglob("*.py"):
                if child.name.startswith(("stage_b", "stage_c", "test_stage_b", "test_stage_c")):
                    selected.add(child)

    explicit = (
        "ACCESS_POLICY.md",
        "AGENTS.md",
        ".agents/skills/research-control/SKILL.md",
        "plans/dynamic_mllm_read_write_causal_analysis_plan_v2.md",
        "workspace/research_plan.md",
        "workspace/workflow_state.md",
        "workspace/decision_log.md",
        "workspace/env_state.md",
        "workspace/dataset_inventory.md",
        "configs/model.yaml",
        "requirements.txt",
        "scoring/reference_likelihood.py",
        "scoring/contextual_reference_likelihood.py",
        "scoring/benchmark_metrics.py",
        "nulls/structured_read.py",
        "interventions/read_path.py",
        "interventions/four_state.py",
        "interventions/prompt_cache.py",
        "infra/archive_stage_b_c.py",
    )
    for raw_path in explicit:
        path = project / raw_path
        if path.is_file():
            selected.add(path)

    ordered = sorted(selected, key=lambda path: path.relative_to(project).as_posix())
    for path in ordered:
        resolved = path.resolve()
        if not resolved.is_relative_to(project.resolve()):
            raise RuntimeError(f"Archive candidate leaves project root: {path}")
    return ordered


def manifest_rows(project: Path, paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(project).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def write_archive(
    project: Path,
    archive_path: Path,
    paths: list[Path],
    manifest_payload: bytes,
) -> None:
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for path in paths:
                    relative = path.relative_to(project).as_posix()
                    info = _tar_info(relative, path.stat().st_size)
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                manifest_info = _tar_info(
                    f"ARCHIVE_METADATA/{MANIFEST_NAME}", len(manifest_payload)
                )
                archive.addfile(manifest_info, io.BytesIO(manifest_payload))


def main() -> int:
    args = parse_args()
    project = Path(args.project).resolve()
    output_dir = (project / args.output_dir).resolve()
    if not output_dir.is_relative_to(project):
        raise RuntimeError("Archive output must stay inside the project")
    paths = selected_files(project)
    total_bytes = sum(path.stat().st_size for path in paths)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "artifact_count": len(paths),
                    "total_uncompressed_bytes": total_bytes,
                    "output_dir": output_dir.relative_to(project).as_posix(),
                },
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    summary_path = output_dir / SUMMARY_NAME
    checksum_path = output_dir / CHECKSUM_NAME
    if any(path.exists() for path in (archive_path, manifest_path, summary_path, checksum_path)):
        raise FileExistsError("Refusing to overwrite the frozen Stage B/C archive")

    rows = manifest_rows(project, paths)
    manifest_payload = manifest_bytes(rows)
    manifest_path.write_bytes(manifest_payload)
    write_archive(project, archive_path, paths, manifest_payload)
    summary = {
        "schema_version": "stage_b_stage_c_frozen_outcome_b_archive_v1",
        "status": "frozen",
        "closed_hypothesis": "harmful layer-0 visual READ under the frozen protocol",
        "stage_c_outcome": "Outcome B",
        "supported_conclusion": [
            "held-out TextVQA layer-0 reference-support effect replicated",
            "effect was heavy-tailed and prompt-sensitive",
            "actual READ removal did not outperform either structured residual null",
            "no confirmed answer-misaligned READ or harmful visual participation claim",
        ],
        "secondary_descriptive_findings": {
            "reference_vs_original_wrong_margin_shift": "positive",
            "corrections": 22,
            "regressions": 12,
            "net_correct_over_800": 10,
            "accuracy_or_mechanism_claim": False,
        },
        "stage_d_authorized": False,
        "read_harm_mechanism_search_authorized": False,
        "artifact_count": len(rows),
        "total_uncompressed_bytes": total_bytes,
        "archive_path": archive_path.relative_to(project).as_posix(),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "manifest_path": manifest_path.relative_to(project).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "selection_scope": [
            "all outputs/stage_b* and outputs/stage_c* files",
            "all runs/stage_b* and runs/stage_c* files",
            "Stage B/C reports, workspace protocols, configs, and data manifests",
            "phase 02/03 memory and compact global research state",
            "approved source plan and directly used execution/scoring/null code",
        ],
        "original_artifacts_moved_or_deleted": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum_rows = [
        (archive_path, sha256(archive_path)),
        (manifest_path, sha256(manifest_path)),
        (summary_path, sha256(summary_path)),
        (project / "reports/stage_c_frozen_outcome_b_closure.md", sha256(project / "reports/stage_c_frozen_outcome_b_closure.md")),
    ]
    checksum_path.write_text(
        "".join(
            f"{digest}  {path.relative_to(project).as_posix()}\n"
            for path, digest in checksum_rows
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
