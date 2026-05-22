from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def write_channel_pngs(bev_frame: np.ndarray, channel_names: list[str], out_dir: str | Path) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    if bev_frame.ndim == 4:
        frame = bev_frame[-1]
    else:
        frame = bev_frame
    for i, name in enumerate(channel_names[: frame.shape[0]]):
        arr = np.nan_to_num(frame[i].astype(np.float32))
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        Image.fromarray((arr * 255).astype(np.uint8)).save(p / f"{i:02d}_{name}.png")
