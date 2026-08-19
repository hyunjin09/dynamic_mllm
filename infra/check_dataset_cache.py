#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_limited(root: Path, query: str, max_depth: int, max_results: int):
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    query_l = query.lower()
    results = []
    base_depth = len(root.parts)
    stack = [root]
    while stack and len(results) < max_results:
        current = stack.pop()
        try:
            rel_depth = len(current.parts) - base_depth
            name_match = query_l in current.name.lower()
            if name_match:
                results.append(str(current))
                if len(results) >= max_results:
                    break
            if current.is_dir() and rel_depth < max_depth:
                for child in current.iterdir():
                    if child.name.startswith("."):
                        continue
                    stack.append(child)
        except PermissionError:
            continue
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-results", type=int, default=50)
    p.add_argument("--report", required=True)
    args = p.parse_args()

    hits = {}
    for root in args.roots:
        hits[root] = iter_limited(Path(root), args.dataset, args.max_depth, args.max_results)

    report = {
        "dataset": args.dataset,
        "roots": args.roots,
        "hits": hits,
        "found": any(v for v in hits.values()),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
