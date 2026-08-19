#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="HF dataset path, e.g. squad or lmms-lab/DocVQA")
    p.add_argument("--name", default=None, help="Optional HF dataset config name")
    p.add_argument("--split", default=None, help="Optional split to materialize")
    p.add_argument("--cache-dir", default="/data/dataset/huggingface/datasets")
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--report", required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    if args.num_proc >= 4:
        raise SystemExit("num_proc must be <= 3 for this project policy.")
    if args.num_proc < 1:
        raise SystemExit("num_proc must be >= 1.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.cache_dir).resolve()))

    from datasets import load_dataset

    kwargs = {
        "path": args.dataset,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.name:
        kwargs["name"] = args.name
    if args.split:
        kwargs["split"] = args.split
    # datasets.load_dataset accepts num_proc for supported builders.
    kwargs["num_proc"] = args.num_proc

    ds = load_dataset(**kwargs)

    report = {
        "dataset": args.dataset,
        "name": args.name,
        "split": args.split,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "num_proc": args.num_proc,
        "trust_remote_code": args.trust_remote_code,
        "status": "downloaded_or_available",
        "type": str(type(ds)),
    }
    if hasattr(ds, "num_rows"):
        report["num_rows"] = ds.num_rows
    elif isinstance(ds, dict):
        report["splits"] = {k: getattr(v, "num_rows", None) for k, v in ds.items()}

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
