"""Best-effort Linux process naming for long-running experiment workers."""

from __future__ import annotations

import ctypes
import os


def set_process_name(name: str | None) -> str | None:
    """Set the short name shown by ``ps -o comm``/``nvidia-smi``.

    Linux limits ``PR_SET_NAME`` to 15 visible bytes. The full run and shard IDs
    remain present in argv and the job-state JSON; this short name is only for
    operational monitoring.
    """

    if not name:
        return None
    value = str(name).strip()
    if not value:
        return None
    encoded = value.encode("ascii", errors="ignore")[:15]
    if not encoded:
        return None
    try:
        libc = ctypes.CDLL(None)
        result = libc.prctl(15, ctypes.c_char_p(encoded), 0, 0, 0)
        if result != 0:
            return None
    except (AttributeError, OSError):
        return None
    os.environ["EXPERIMENT_PROCESS_NAME"] = encoded.decode("ascii")
    return encoded.decode("ascii")
