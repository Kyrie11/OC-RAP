from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.serialization import load_npz, parse_json_field, write_json
from ocrap.data.validation import missing_fields


def iter_sample_paths(dataset: str | Path, max_samples: int | None = None) -> list[Path]:
    root = Path(dataset)
    if (root / "samples").exists():
        paths = sorted((root / "samples").glob("*.npz"))
    else:
        paths = sorted(root.glob("*.npz"))
    return paths[:max_samples] if max_samples else paths


def diagnose_dataset(dataset: str | Path, output: str | Path | None = None, max_samples: int | None = None) -> dict:
    paths = iter_sample_paths(dataset, max_samples)
    failures: list[str] = []
    split_by_scene: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}
    for p in paths:
        d = load_npz(p)
        miss = missing_fields(set(d.keys()))
        if miss:
            failures.append(f"missing required fields in {p.name}: {','.join(miss[:8])}")
        scene = str(np.asarray(d.get("scene_id", "")).item()) if "scene_id" in d else p.stem
        split = str(np.asarray(d.get("split_id", "unknown")).item()) if "split_id" in d else "unknown"
        split_by_scene.setdefault(scene, set()).add(split)
        for s in np.asarray(d.get("future_sources", []), dtype=str).reshape(-1):
            source_counts[str(s)] = source_counts.get(str(s), 0) + 1
        if "root_probs" in d:
            rp = np.asarray(d["root_probs"], dtype=float)
            if not np.isclose(rp.sum(), 1.0, atol=1e-3):
                failures.append(f"root_probs not normalized in {p.name}")
    leakage = [s for s, splits in split_by_scene.items() if len(splits) > 1]
    if leakage:
        failures.append(f"scenario split leakage: {leakage[:5]}")
    result = {"num_samples": len(paths), "future_source_coverage": source_counts, "num_scenes": len(split_by_scene), "failures": failures}
    if output:
        write_json(result, output)
    return result
