#!/usr/bin/env python3
"""Download the exact pinned Qwen2.5-VL snapshot and print its local path."""

from __future__ import annotations

import argparse
from huggingface_hub import snapshot_download


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-dir")
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
