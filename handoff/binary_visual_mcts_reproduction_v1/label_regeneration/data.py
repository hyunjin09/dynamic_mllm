"""Dataset-independent helpers used by the portable MCTS runner."""

from __future__ import annotations

from hashlib import sha256


def safe_sample_filename(uid: str) -> str:
    readable = str(uid).replace(":", "__").replace("/", "_")
    suffix = sha256(str(uid).encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{suffix}.json"
