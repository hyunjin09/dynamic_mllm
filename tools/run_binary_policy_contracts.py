#!/usr/bin/env python3
"""Run binary-policy contract tests without requiring pytest."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_binary_executor, test_binary_policy


def main() -> None:
    count = 0
    for module in (test_binary_policy, test_binary_executor):
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                getattr(module, name)()
                count += 1
    print(f"{count} binary-policy contract tests passed")


if __name__ == "__main__":
    main()
