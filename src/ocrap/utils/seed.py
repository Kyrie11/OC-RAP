from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*items: Any) -> int:
    """Return a process-independent uint32 seed for arbitrary values."""
    s = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest()[:8], 16)
