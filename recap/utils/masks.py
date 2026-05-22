from __future__ import annotations

import numpy as np


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("masked_argmax called with all-False mask")
    v = np.where(mask, values, -np.inf)
    return int(np.argmax(v))


def ensure_bool_mask(mask, shape=None):
    m = np.asarray(mask, dtype=bool)
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"mask shape {m.shape} != expected {shape}")
    return m
