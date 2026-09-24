from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import jax  # type: ignore
        value = jax.device_get(value)
    except Exception:
        pass
    return np.asarray(value)


def simulator_state_fingerprint(state: Any) -> str:
    """Hash the treatment-boundary simulator state in a platform-stable form.

    The raw scenario/target key already fixes the static map and log playback.
    This digest therefore covers the dynamic state that can differ after a
    pre-treatment rollout: current timestep plus all current object poses,
    velocities, validity, and object identity metadata.
    """
    t = int(np.asarray(getattr(state, "timestep")).reshape(()))
    tr = getattr(state, "sim_trajectory")
    h = hashlib.sha256()
    h.update(b"ocrap-contact-anchor-state-v1\0")
    h.update(np.asarray([t], dtype="<i8").tobytes())
    for name in ("x", "y", "z", "yaw", "vel_x", "vel_y", "length", "width", "height"):
        if not hasattr(tr, name):
            continue
        arr = _as_numpy(getattr(tr, name))
        if arr.ndim >= 2:
            tt = min(max(t, 0), arr.shape[-1] - 1)
            arr = arr[..., tt]
        arr = np.asarray(arr, dtype="<f8")
        h.update(name.encode("utf-8") + b"\0")
        h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        h.update(arr.tobytes(order="C"))
    if hasattr(tr, "valid"):
        arr = _as_numpy(getattr(tr, "valid"))
        if arr.ndim >= 2:
            tt = min(max(t, 0), arr.shape[-1] - 1)
            arr = arr[..., tt]
        arr = np.asarray(arr, dtype=np.uint8)
        h.update(b"valid\0")
        h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        h.update(arr.tobytes(order="C"))
    meta = getattr(state, "object_metadata", None)
    if meta is not None:
        for name in ("ids", "object_types", "is_sdc"):
            if not hasattr(meta, name):
                continue
            arr = _as_numpy(getattr(meta, name))
            if name == "is_sdc":
                arr = np.asarray(arr, dtype=np.uint8)
            else:
                arr = np.asarray(arr, dtype="<i8")
            h.update(name.encode("utf-8") + b"\0")
            h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
            h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def load_anchor_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path in {None, ""}:
        return None
    p = Path(path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"contact anchor manifest must be an object: {p}")
    if value.get("schema") != "ocrap-contact-anchor-manifest-v1" or value.get("valid") is not True:
        raise ValueError(f"invalid contact anchor manifest: {p}")
    return value


def anchor_manifest_index(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    anchors = manifest.get("anchors") or []
    if not isinstance(anchors, list):
        raise ValueError("contact anchor manifest anchors must be a list")
    out: dict[str, dict[str, Any]] = {}
    for row in anchors:
        if not isinstance(row, dict):
            raise ValueError("contact anchor manifest row must be an object")
        key = str(row.get("target_key") or "")
        if not key or key in out:
            raise ValueError(f"duplicate/empty contact anchor target_key: {key!r}")
        out[key] = row
    return out


def select_one_anchor_per_scene(
    mining_result: dict[str, Any], *, min_post_steps: int = 10
) -> dict[str, Any]:
    """Build a scene-disjoint anchor manifest from exact-a0 mining output.

    Selection uses only pre-treatment quantities.  Among valid anchors from the
    same scene we prefer the state with the largest remaining simulator horizon;
    ties use earlier anchor time and then target key for deterministic behavior.
    """
    scenes = mining_result.get("scenes") or []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for s in scenes:
        if not isinstance(s, dict) or not bool(s.get("contact_anchor_found")):
            continue
        rem = int(s.get("contact_anchor_remaining_steps") or 0)
        fp = str(s.get("contact_anchor_fingerprint") or "")
        key = str(s.get("target_key") or "")
        sid = str(s.get("scene_id") or "")
        if not sid or not key or len(fp) != 64 or rem < int(min_post_steps):
            continue
        by_scene.setdefault(sid, []).append(s)
    anchors: list[dict[str, Any]] = []
    for sid, rows in sorted(by_scene.items()):
        best = sorted(
            rows,
            key=lambda s: (
                -int(s.get("contact_anchor_remaining_steps") or 0),
                int(s.get("contact_anchor_time_index") or 10**9),
                str(s.get("target_key") or ""),
            ),
        )[0]
        anchors.append({
            "scene_id": sid,
            "target_key": str(best["target_key"]),
            "target_time_index": int(best.get("target_time_index") or 0),
            "contact_anchor_time_index": int(best.get("contact_anchor_time_index") or 0),
            "contact_anchor_prelude_env_steps": int(best.get("contact_anchor_prelude_env_steps") or 0),
            "contact_anchor_remaining_steps": int(best.get("contact_anchor_remaining_steps") or 0),
            "contact_anchor_fingerprint": str(best["contact_anchor_fingerprint"]),
        })
    source_keys = sorted(str(s.get("target_key") or "") for s in scenes if s.get("target_key"))
    selected_keys = sorted(a["target_key"] for a in anchors)
    payload = {
        "schema": "ocrap-contact-anchor-manifest-v1",
        "valid": bool(anchors),
        "selection_policy": "scene_disjoint_max_remaining_horizon_then_earliest_anchor_then_target_key",
        "pre_treatment_policy": "exact_a0",
        "min_post_steps": int(min_post_steps),
        "num_mining_targets": len(source_keys),
        "num_selected_anchors": len(anchors),
        "num_selected_scenes": len({a["scene_id"] for a in anchors}),
        "target_keys": selected_keys,
        "anchors": anchors,
    }
    return payload
