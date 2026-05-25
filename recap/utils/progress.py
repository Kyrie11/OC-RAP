from __future__ import annotations

import os
import sys
from typing import Iterable, TypeVar

T = TypeVar("T")


def tqdm(iterable: Iterable[T] | None = None, *args, **kwargs):
    """Small tqdm wrapper with a no-dependency fallback.

    The wrapper keeps tqdm visible in normal terminal runs and in log files.  It
    can be disabled explicitly with ``RECAP_DISABLE_TQDM=1``.  Keeping the
    default on is important for long teacher-label builds, where the process may
    be redirected to a per-worker log by a parallel launcher.
    """
    kwargs.setdefault("disable", os.environ.get("RECAP_DISABLE_TQDM", "").lower() in {"1", "true", "yes"})
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("file", sys.stderr)
    try:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm(iterable, *args, **kwargs)
    except Exception:
        if iterable is None:
            class _Dummy:
                def update(self, n: int = 1) -> None: pass
                def set_postfix(self, *args, **kwargs) -> None: pass
                def close(self) -> None: pass
                def __enter__(self): return self
                def __exit__(self, *exc): self.close()
            return _Dummy()
        return iterable
