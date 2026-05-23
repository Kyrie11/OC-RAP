from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def tqdm(iterable: Iterable[T] | None = None, *args, **kwargs):
    """Small tqdm wrapper with a no-dependency fallback."""
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
