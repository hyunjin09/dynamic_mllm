#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tools.research_analysis.four_action.route_conditioned import flatten_route_conditioned_samples


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    sidecar(path)


def write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar(path)


def write_parquet_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    sidecar(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge a route-conditioned stage.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mode_directory = args.mode if not args.output_tag else f"{args.mode}__{args.output_tag}"
    stage_dir = Path(config[f"{args.mode}_root"]) / mode_directory
    manifest_path = (
        Path(config["output_root"]) / "pilot_manifest.jsonl"
        if args.mode == "pilot"
        else Path(config["anchor_manifest"])
    )
    expected = read_jsonl(manifest_path)
    result_paths = sorted(stage_dir.glob("shard_*/results*.jsonl"))
    runtime_paths = sorted(stage_dir.glob("shard_*/runtime*.json"))
    failure_paths = sorted(stage_dir.glob("shard_*/failures*.jsonl"))
    results = [row for path in result_paths for row in read_jsonl(path)]
    runtimes = [json.loads(path.read_text(encoding="utf-8")) for path in runtime_paths]
    failures = [row for path in failure_paths for row in read_jsonl(path)]
    expected_ids = {row["uid"] for row in expected}
    observed_ids = [row["uid"] for row in results]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("stage results contain duplicate sample UIDs")
    if set(observed_ids) != expected_ids:
        missing = sorted(expected_ids - set(observed_ids))
        extra = sorted(set(observed_ids) - expected_ids)
        raise RuntimeError(f"stage coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    passed_uids = {row["uid"] for row in results if row["sample_gate"]["passed"]}
    disqualifying_failures = [row for row in failures if row.get("uid") not in passed_uids]
    if disqualifying_failures:
        raise RuntimeError(f"stage contains {len(disqualifying_failures)} unrecovered failures")
    if not all(row["sample_gate"]["passed"] for row in results):
        raise RuntimeError("one or more route-conditioned sample gates failed")
    replicas = {int(row["replicas_per_gpu"]) for row in runtimes}
    if len(replicas) != 1:
        raise RuntimeError("runtime manifests disagree on replicas per GPU")
    replicas_per_gpu = next(iter(replicas))
    expected_workers = {
        (gpu, replica) for gpu in range(8) for replica in range(replicas_per_gpu)
    }
    observed_workers = {(int(row["gpu_index"]), int(row["replica_index"])) for row in runtimes}
    if observed_workers != expected_workers or len(runtimes) != len(expected_workers):
        raise RuntimeError("stage is missing the exact all-eight-GPU worker contract")
    for sample in results:
        if len(sample["cells"]) != int(sample["anchor_off_count"]):
            raise RuntimeError(f"cell count mismatch for {sample['uid']}")
        if {row["target_layer"] for row in sample["cells"]} != set(sample["anchor_off_layers"]):
            raise RuntimeError(f"OFF-layer coverage mismatch for {sample['uid']}")
        if args.mode == "pilot" and not all(
            row["m00_reproduction"] is not None and row["m00_reproduction"]["passed"]
            for row in sample["cells"]
        ):
            raise RuntimeError(f"pilot M00 reproduction failed for {sample['uid']}")
    results.sort(key=lambda row: row["uid"])
    flat = flatten_route_conditioned_samples(results)
    worker_seconds = defaultdict(float)
    for row in results:
        key = (
            int(row["worker"]["gpu_index"]),
            int(row["worker"]["replica_index"]),
        )
        worker_seconds[key] += float(row["elapsed_seconds"])
    useful_wall_seconds = max(worker_seconds.values()) if worker_seconds else 0.0
    new_cells = sum(int(row["new_intervention_cell_count"]) for row in results)
    summary = {
        "schema_version": "route_conditioned_stage_summary_v1",
        "mode": args.mode,
        "output_tag": args.output_tag,
        "passed": True,
        "sample_count": len(results),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in results).items())),
        "anchor_off_position_count": sum(len(row["cells"]) for row in results),
        "new_intervention_cell_count": new_cells,
        "flat_action_row_count": len(flat),
        "taxonomy_counts": dict(
            sorted(Counter(cell["taxonomy"] for row in results for cell in row["cells"]).items())
        ),
        "replicas_per_gpu": replicas_per_gpu,
        "worker_count": len(runtimes),
        "all_eight_gpu_worker_contract": True,
        "all_sample_gates_pass": True,
        "all_pilot_m00_reproductions_pass": args.mode != "pilot"
        or all(cell["m00_reproduction"]["passed"] for row in results for cell in row["cells"]),
        "failure_artifact_count": len(failures),
        "disqualifying_failure_count": len(disqualifying_failures),
        "useful_worker_wall_seconds": useful_wall_seconds,
        "useful_new_cells_per_second": new_cells / useful_wall_seconds if useful_wall_seconds else None,
        "useful_samples_per_second": len(results) / useful_wall_seconds if useful_wall_seconds else None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    merged_path = stage_dir / "merged_results.jsonl"
    cells_jsonl = stage_dir / "route_conditioned_cells.jsonl"
    cells_parquet = stage_dir / "route_conditioned_cells.parquet"
    write_jsonl_once(merged_path, results)
    write_jsonl_once(cells_jsonl, flat)
    write_parquet_once(cells_parquet, flat)
    summary.update(
        {
            "merged_results_sha256": sha256_file(merged_path),
            "route_conditioned_cells_jsonl_sha256": sha256_file(cells_jsonl),
            "route_conditioned_cells_parquet_sha256": sha256_file(cells_parquet),
        }
    )
    write_json_once(stage_dir / "stage_summary.json", summary)
    if args.mode == "full":
        output_root = Path(config["output_root"])
        write_jsonl_once(output_root / "route_conditioned_cells.jsonl", flat)
        write_parquet_once(output_root / "route_conditioned_cells.parquet", flat)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
