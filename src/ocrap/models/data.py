from __future__ import annotations

import csv
import os
import zlib
import hashlib
import json
import time
import fcntl
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz_selected
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.simulation.teacher.controllers import rollout_recovery_controller


RECOVERY_MODE_VOCAB = [
    "stop",
    "brake_lane",
    "lateral_escape",
    "yield_rejoin",
    "pull_over",
    "mitigate_contact",
    "post_contact_stabilize",
    "avoid_secondary",
]
RECOVERY_MODE_TO_ID = {name: i for i, name in enumerate(RECOVERY_MODE_VOCAB)}
BUCKET_TO_ID = {"safe": 0, "near_contact": 1, "contact": 2, "other": 3}
OPTION_PARAM_DIM = 3
OPTION_FEATURE_DIM = len(RECOVERY_MODE_VOCAB) + OPTION_PARAM_DIM + 2

# v48.45.6 engineering-only I/O fast path.  These are exactly the NPZ members
# consumed by OCRAPSampleDataset/sample_to_feature/fix_sample_geometry and the
# training labels returned by __getitem__.  Keeping this list explicit makes the
# optimization auditable: it changes no tensor value, sample order, loss, or model
# input; it only avoids decompressing unrelated archive members.
MODEL_SAMPLE_NPZ_KEYS: frozenset[str] = frozenset({
    "agent_history", "agent_valid", "ego_state", "bev_occ", "route",
    "map_polylines", "dynamic_map", "prefix_param", "prefix_states",
    "prefix_controls", "prefix_macro_type_id", "prefix_macro_id",
    "utility", "hard_violation", "harm_proxy", "feasible",
    "is_nominal", "scene_id", "time_index", "candidate_index",
    "root_probs", "m_star", "c_star", "y_obs", "root_valid",
    "option_valid", "root_signature", "root_future_signature",
    "recovery_params", "recovery_modes", "r_dep_star", "r_orc_star",
    "i_art_star",
})

# v48.56 engineering-only teacher-index fast path.  The teacher index never
# constructs model features; it only consumes OC-MERO matrices/probabilities,
# cached recovery scalars and grouping metadata.  Keeping this subset separate
# avoids decompressing BEV/map/history/prefix/signature tensors for every
# calibration sample.  This changes no teacher value or row ordering.
TEACHER_PCD_NPZ_KEYS: frozenset[str] = frozenset({
    "m_star", "root_probs", "c_star", "root_valid", "option_valid",
    "r_dep_star", "r_orc_star", "hard_violation", "harm_proxy",
    "scene_id", "time_index", "candidate_index", "prefix_macro_type_id",
    "prefix_macro_id", "is_nominal",
})


NOMINAL_DEVIATION_NPZ_KEYS: frozenset[str] = frozenset({
    "scene_id", "time_index", "is_nominal", "prefix_states",
})


# v48.16 ANCHOR: protocol-role aliases. Dedicated calibration intentionally
# renames split_id to make train/dev/certificate roles explicit. All readers
# must agree on these aliases instead of silently discarding every sample.
SPLIT_ROLE_ALIASES: dict[str, frozenset[str]] = {
    "train": frozenset({"train", "evidence_adapt_train"}),
    "val": frozenset({"val", "evidence_adapt_dev"}),
    "calibration": frozenset({"calibration", "certificate_pool"}),
    "certificate_pool": frozenset({"certificate_pool"}),
    "evidence_adapt_train": frozenset({"evidence_adapt_train"}),
    "evidence_adapt_dev": frozenset({"evidence_adapt_dev"}),
}


def expand_split_roles(splits: str | set[str] | tuple[str, ...] | list[str]) -> set[str]:
    """Expand semantic split roles into concrete manifest split_id values."""
    raw = {splits} if isinstance(splits, str) else {str(x) for x in splits}
    out: set[str] = set()
    for item in raw:
        item = str(item).strip()
        if not item:
            continue
        out.update(SPLIT_ROLE_ALIASES.get(item, frozenset({item})))
    return out


def split_id_matches(split_id: object, allowed: str | set[str] | tuple[str, ...] | list[str]) -> bool:
    """Return whether a manifest/NPZ split id belongs to an allowed role."""
    return str(split_id).strip() in expand_split_roles(allowed)


def _path_key(path: str | Path) -> str:
    """Stable absolute key for matching sample paths to manifest rows."""
    return os.path.abspath(os.fspath(path))


def _dataset_root_for_sample(path: str | Path) -> Path:
    """Return the OC-RAP dataset root for either root/samples/x.npz or root/x.npz."""
    p = Path(path)
    return p.parent.parent if p.parent.name == "samples" else p.parent


@lru_cache(maxsize=128)
def _manifest_metadata_map(dataset_root_key: str) -> dict[str, dict[str, str]]:
    """Load lightweight per-sample metadata from dataset_root/manifest.csv.

    Training only needs metadata such as split_id and i_art_star for pre-flight
    filtering/sampling.  Reading those fields from the manifest avoids opening
    every compressed .npz before the first epoch.
    """
    root = Path(dataset_root_key)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_path = str(row.get("path", "")).strip()
                if not raw_path:
                    continue
                sample_path = Path(raw_path)
                if not sample_path.is_absolute():
                    sample_path = root / sample_path
                out[_path_key(sample_path)] = {str(k): "" if v is None else str(v) for k, v in row.items()}
    except Exception:
        return {}
    return out


def manifest_metadata_for_path(path: str | Path) -> dict[str, str] | None:
    """Return manifest metadata for a sample path when available."""
    root = _dataset_root_for_sample(path)
    table = _manifest_metadata_map(_path_key(root))
    return table.get(_path_key(path)) if table else None


def npz_scalar(path: str | Path, key: str, default: Any = None) -> Any:
    """Read a single scalar-like value from an .npz without materializing arrays.

    This is intentionally different from load_npz(), which expands every array in
    the archive.  It keeps split filtering and sampler construction cheap even
    for large compressed samples containing BEV/map/history tensors.
    """
    try:
        with np.load(path, allow_pickle=True) as z:
            if key not in z.files:
                return default
            arr = np.asarray(z[key])
            if arr.shape == ():
                return arr.item()
            if arr.size == 1:
                return arr.reshape(-1)[0].item()
            return arr.tolist()
    except Exception:
        return default


def scalar_metadata_for_path(path: str | Path, key: str, default: Any = None) -> Any:
    """Read metadata from manifest.csv first, falling back to one-key NPZ read."""
    row = manifest_metadata_for_path(path)
    if row is not None and key in row and row.get(key, "") != "":
        return row.get(key, default)
    return npz_scalar(path, key, default)


def iter_sample_paths_many(dataset: str | Path, max_samples: int | None = None) -> list[Path]:
    """Return sample paths from one or more OC-RAP dataset directories.

    The CLI still exposes a single --dataset argument for backwards
    compatibility.  To train on multiple sharded builds, pass a comma-separated
    list, e.g. /path/w0,/path/w1,/path/w2,/path/w3.
    """
    if isinstance(dataset, Path):
        specs = [dataset]
    else:
        raw = str(dataset)
        sep_specs: list[str] = []
        for chunk in raw.split(','):
            chunk = chunk.strip()
            if chunk:
                sep_specs.append(chunk)
        # Also accept the platform path separator when it is unambiguous.
        if len(sep_specs) == 1 and os.pathsep in sep_specs[0] and not Path(sep_specs[0]).exists():
            sep_specs = [x.strip() for x in sep_specs[0].split(os.pathsep) if x.strip()]
        specs = [Path(x) for x in sep_specs]
    paths: list[Path] = []
    for spec in specs:
        paths.extend(iter_sample_paths(spec))
    paths = sorted(dict.fromkeys(paths))
    return paths[:max_samples] if max_samples else paths


def _arr(d: dict[str, Any], key: str, default_shape: tuple[int, ...] = (0,)) -> np.ndarray:
    if key not in d:
        return np.zeros(default_shape, dtype=np.float32)
    try:
        return np.asarray(d[key], dtype=np.float32)
    except Exception:
        return np.zeros(default_shape, dtype=np.float32)


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return default


def _int_scalar(d: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(round(float(np.asarray(d.get(key, default)).item())))
    except Exception:
        return default


def _pad_flat(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float32)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    out[: min(n, v.size)] = v[:n]
    return out


def _finite_stats(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([v.mean(), v.std(), v.min(), np.quantile(v, 0.25), np.quantile(v, 0.75), v.max()], dtype=np.float32)


def _one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=np.float32)
    if 0 <= int(index) < int(width):
        out[int(index)] = 1.0
    return out


def _string_array(d: dict[str, Any], key: str) -> list[str]:
    if key not in d:
        return []
    arr = np.asarray(d[key])
    out: list[str] = []
    for v in arr.reshape(-1).tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def option_features_from_sample(d: dict[str, Any], cfg: dict | None = None) -> np.ndarray:
    """Encode recovery option identity and parameters for the margin decoder.

    The paper's margin head is conditioned on the recovery option ``g_l``.  Older
    code used only learned option-index embeddings, which makes two runs with the
    same option count but different option modes/parameters indistinguishable to
    the model.  This compact feature keeps the model architecture independent of
    Python dataclasses while preserving the semantic option conditioning saved in
    each ``.npz`` sample.
    """
    del cfg  # reserved for future feature scaling knobs
    params = _arr(d, "recovery_params")
    if params.ndim == 1:
        params = params.reshape(-1, OPTION_PARAM_DIM) if params.size else np.zeros((0, OPTION_PARAM_DIM), dtype=np.float32)
    if params.ndim != 2:
        params = np.zeros((0, OPTION_PARAM_DIM), dtype=np.float32)
    L = int(params.shape[0]) if params.size else int(np.asarray(d.get("m_star", np.zeros((1, 0)))).shape[-1])
    out = np.zeros((L, OPTION_FEATURE_DIM), dtype=np.float32)
    modes = _string_array(d, "recovery_modes")
    valid = np.asarray(d.get("option_valid", np.ones((L,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    for l in range(L):
        mode = modes[l] if l < len(modes) else ""
        mid = RECOVERY_MODE_TO_ID.get(mode, -1)
        if mid >= 0:
            out[l, mid] = 1.0
        if l < params.shape[0]:
            p = np.asarray(params[l], dtype=np.float32).reshape(-1)
            out[l, len(RECOVERY_MODE_VOCAB) : len(RECOVERY_MODE_VOCAB) + OPTION_PARAM_DIM] = np.pad(p[:OPTION_PARAM_DIM], (0, max(0, OPTION_PARAM_DIM - min(OPTION_PARAM_DIM, p.size))))[:OPTION_PARAM_DIM]
        out[l, -2] = float(valid[l] > 0.5) if l < valid.size else 1.0
        out[l, -1] = float(l) / max(float(max(L - 1, 1)), 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _agent_features(d: dict[str, Any], max_agents: int) -> np.ndarray:
    hist = _arr(d, "agent_history")
    valid = _arr(d, "agent_valid")
    ego = _arr(d, "ego_state", (9,)).reshape(-1)
    if hist.ndim != 3 or valid.ndim < 2 or hist.shape[1] == 0:
        return np.zeros(8 + max_agents * 10, dtype=np.float32)
    last = hist[-1]
    vmask = valid[-1].astype(bool) if valid.shape[0] else np.zeros((last.shape[0],), dtype=bool)
    ego_xy = ego[:2] if ego.size >= 2 else np.zeros(2, dtype=np.float32)
    rows = []
    dists = []
    speeds = []
    # Agent feature layout follows ocrap.data.schema.AGENT_FEATURES.
    for a in range(1, min(last.shape[0], len(vmask))):
        if not vmask[a]:
            continue
        s = last[a]
        if s.size < 16:
            continue
        dx = float(s[0] - ego_xy[0])
        dy = float(s[1] - ego_xy[1])
        dist = float(np.hypot(dx, dy))
        speed = float(np.hypot(s[3], s[4]))
        dists.append(dist)
        speeds.append(speed)
        rows.append((dist, np.array([
            dx / 80.0,
            dy / 80.0,
            float(s[3]) / 20.0,
            float(s[4]) / 20.0,
            speed / 20.0,
            float(s[8]) if s.size > 8 else 0.0,
            float(s[9]) if s.size > 9 else 1.0,
            float(s[10]) / 10.0 if s.size > 10 else 0.0,
            float(s[11]) / 5.0 if s.size > 11 else 0.0,
            float(s[13]) / 10.0 if s.size > 13 else 0.0,
        ], dtype=np.float32)))
    rows.sort(key=lambda x: x[0])
    packed = np.zeros((max_agents, 10), dtype=np.float32)
    for i, (_, feat) in enumerate(rows[:max_agents]):
        packed[i] = feat
    d_arr = np.asarray(dists, dtype=np.float32)
    s_arr = np.asarray(speeds, dtype=np.float32)
    summary = np.array([
        float(len(rows)) / max(float(max_agents), 1.0),
        float(d_arr.min() / 80.0) if d_arr.size else 0.0,
        float(d_arr.mean() / 80.0) if d_arr.size else 0.0,
        float(d_arr.std() / 80.0) if d_arr.size else 0.0,
        float((d_arr < 8.0).mean()) if d_arr.size else 0.0,
        float((d_arr < 20.0).mean()) if d_arr.size else 0.0,
        float(s_arr.mean() / 20.0) if s_arr.size else 0.0,
        float(s_arr.max() / 20.0) if s_arr.size else 0.0,
    ], dtype=np.float32)
    return np.concatenate([summary, packed.reshape(-1)], axis=0)




# v48.60.1 engineering correction.  CPHR must use the complete executable
# prefix, not the historically truncated 80-D encoder block.  Keep the feature
# schema explicit so stale v48.60.0 checkpoints/caches cannot be reused silently.
DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA = 2
DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_DIM = 6


def direct_absolute_physical_headroom_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None
) -> np.ndarray:
    """Compute the six CPHR coordinates from full observable sample tensors.

    This is deliberately a side-channel *to the CPHR source only*.  The frozen
    Stage-I encoder continues to consume the exact historical flat feature row.
    Inputs are restricted to decision-time observation plus the executable
    candidate prefix: ``ego_state``, ``agent_history[-1]``, ``agent_valid[-1]``,
    ``prefix_states`` and ``prefix_controls``.  No future/teacher tensor is read.

    Feature order is unchanged from v48.60.0:
      0 signed minimum-clearance reserve
      1 signed terminal-clearance reserve
      2 clearance recovery gain
      3 stopping reserve
      4 control-envelope reserve
      5 stability reserve
    """
    cfg = cfg or {}
    states = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=np.float32)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 4))), dtype=np.float32)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=np.float32).reshape(-1)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=np.float32)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)

    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 9:
        raise ValueError(
            f"CPHR requires full prefix_states[T,>=9], got shape={getattr(states, 'shape', None)}"
        )
    if ego.size < 9:
        raise ValueError(f"CPHR requires ego_state[>=9], got shape={ego.shape}")
    if controls.ndim != 2:
        controls = np.zeros((0, 4), dtype=np.float32)
    if controls.shape[1] < 4 and controls.shape[0] > 0:
        raise ValueError(f"CPHR requires prefix_controls[T,>=4], got shape={controls.shape}")

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    max_agents = int(model_cfg.get("feature_max_agents", 32))
    ego_xy = ego[:2].astype(np.float64, copy=False)
    ego_len = max(abs(float(ego[7])), 1.0e-3)
    ego_wid = max(abs(float(ego[8])), 1.0e-3)
    ego_rad = 0.5 * float(np.hypot(ego_len, ego_wid))

    # Match the encoder's observable-neighbour contract: current valid non-ego
    # agents, nearest-first, capped at feature_max_agents.  Unlike v48.60.0 we do
    # not round-trip through normalized packed tokens, so no prefix information
    # is lost before the CPHR calculation.
    agents: list[tuple[float, np.ndarray]] = []
    if hist.ndim == 3 and hist.shape[0] > 0 and hist.shape[1] > 1 and valid.ndim >= 2 and valid.shape[0] > 0:
        last = hist[-1]
        vmask = valid[-1].reshape(-1)
        for aidx in range(1, min(last.shape[0], vmask.size)):
            if not bool(vmask[aidx]):
                continue
            row = np.asarray(last[aidx], dtype=np.float64).reshape(-1)
            if row.size < 12 or not np.isfinite(row[:12]).all():
                continue
            rel = row[:2] - ego_xy
            agents.append((float(np.linalg.norm(rel)), row))
    agents.sort(key=lambda z: z[0])
    agents = agents[:max_agents]

    T = int(states.shape[0])
    sample_rate = float(cfg.get("sample_rate_hz", 10.0) or 10.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"invalid sample_rate_hz for CPHR: {sample_rate}")
    dt = 1.0 / sample_rate
    # prefix_generation._rollout stores the first executable state after one
    # integration interval, so its physical timestamp is dt rather than 0.
    times = (np.arange(T, dtype=np.float64) + 1.0) * dt
    prefix_xy_rel = states[:, :2].astype(np.float64) - ego_xy[None, :]

    if agents:
        rel0 = np.stack([row[:2] - ego_xy for _, row in agents], axis=0)
        vel = np.stack([row[3:5] for _, row in agents], axis=0)
        alen = np.asarray([max(abs(float(row[10])), 1.0e-3) for _, row in agents], dtype=np.float64)
        awid = np.asarray([max(abs(float(row[11])), 1.0e-3) for _, row in agents], dtype=np.float64)
        arad = 0.5 * np.hypot(alen, awid)
        agent_future = rel0[None, :, :] + times[:, None, None] * vel[None, :, :]
        delta = prefix_xy_rel[:, None, :] - agent_future
        signed_clearance = np.linalg.norm(delta, axis=-1) - ego_rad - arad[None, :]
        c_min = float(np.min(signed_clearance))
        c_terminal = float(np.min(signed_clearance[-1]))
        c0 = float(np.min(np.linalg.norm(rel0, axis=-1) - ego_rad - arad))
    else:
        c_min = c_terminal = c0 = 40.0

    speed = np.abs(states[:, 6].astype(np.float64))
    max_speed = float(np.max(speed))
    terminal_speed = float(speed[-1])
    d_safe0 = float(cfg.get("d_safe0_m", 1.0))
    headway = float(cfg.get("safe_time_headway_s", 0.5))
    scales = cfg.get("margin_scales", {}) if isinstance(cfg.get("margin_scales", {}), dict) else {}
    distance_scale = max(float(scales.get("distance", 2.0)), 1.0e-6)
    stop_scale = max(float(scales.get("stop", 5.0)), 1.0e-6)
    d_safe = d_safe0 + headway * max_speed

    h_min_clear = np.tanh((c_min - d_safe) / distance_scale)
    h_terminal_clear = np.tanh((c_terminal - d_safe) / distance_scale)
    h_clear_gain = np.tanh((c_terminal - c0) / distance_scale)

    limits = cfg.get("control_limits", {}) if isinstance(cfg.get("control_limits", {}), dict) else {}
    a_max = float(limits.get("a_max", 3.0))
    a_min = float(limits.get("a_min", -6.0))
    delta_max = float(limits.get("delta_max", 0.55))
    jerk_max = float(limits.get("j_max", 6.0))
    steer_rate_max = float(limits.get("steer_rate_max", 0.5))
    stop_decel = max(abs(a_min), 1.0e-3)
    stop_required = terminal_speed * terminal_speed / (2.0 * stop_decel) + d_safe0
    h_stop = np.tanh((c_terminal - stop_required) / stop_scale)

    if controls.shape[0] > 0:
        c = controls[:, :4].astype(np.float64)
        accel_scale = max(float(scales.get("accel", 1.0)), 1.0e-6)
        decel_scale = max(float(scales.get("decel", 1.0)), 1.0e-6)
        steer_scale = max(float(scales.get("steer", 0.1)), 1.0e-6)
        jerk_scale = max(float(scales.get("jerk", 2.0)), 1.0e-6)
        rate_scale = max(float(scales.get("steer_rate", 0.1)), 1.0e-6)
        ctrl_terms = np.asarray([
            (a_max - float(np.max(np.maximum(c[:, 0], 0.0)))) / accel_scale,
            (abs(a_min) - float(np.max(np.maximum(-c[:, 0], 0.0)))) / decel_scale,
            (delta_max - float(np.max(np.abs(c[:, 1])))) / steer_scale,
            (jerk_max - float(np.max(np.abs(c[:, 2])))) / jerk_scale,
            (steer_rate_max - float(np.max(np.abs(c[:, 3])))) / rate_scale,
        ], dtype=np.float64)
    else:
        # No control action consumes no envelope; this matches the old padded-zero
        # semantics while avoiding a fabricated extra timestep.
        ctrl_terms = np.asarray([
            a_max / max(float(scales.get("accel", 1.0)), 1.0e-6),
            abs(a_min) / max(float(scales.get("decel", 1.0)), 1.0e-6),
            delta_max / max(float(scales.get("steer", 0.1)), 1.0e-6),
            jerk_max / max(float(scales.get("jerk", 2.0)), 1.0e-6),
            steer_rate_max / max(float(scales.get("steer_rate", 0.1)), 1.0e-6),
        ], dtype=np.float64)
    h_control = np.tanh(float(np.min(ctrl_terms)))

    yaw_rate_max = float(cfg.get("yaw_rate_max_rps", 0.6))
    yaw_scale = max(float(scales.get("yaw", 0.2)), 1.0e-6)
    yaw_rate = float(np.max(np.abs(states[:, 5].astype(np.float64))))
    h_stability = np.tanh((yaw_rate_max - yaw_rate) / yaw_scale)

    out = np.asarray([
        h_min_clear,
        h_terminal_clear,
        h_clear_gain,
        h_stop,
        h_control,
        h_stability,
    ], dtype=np.float32)
    if out.shape != (DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_DIM,) or not np.isfinite(out).all():
        raise ValueError(f"invalid CPHR feature vector: {out}")
    return out




# v48.61 ERWF (Executable Recovery Witness Field).  Unlike CPHR's one vector
# per candidate, ERWF builds one signed observable continuation-witness vector
# per *recovery option*.  The side channel therefore has shape [L, 6] and is
# consumed only by the Stage-II absolute source.  Stage-I features remain byte-
# semantically unchanged.
DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA = 1
DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_DIM = 6


def _decode_recovery_mode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    return str(value)


def direct_executable_recovery_witness_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None, *, num_options: int | None = None,
    include_recovery_stability_tail: bool = False,
) -> np.ndarray:
    """Return the v48.61 option-resolved executable recovery witness field.

    For every valid recovery option ``g_l`` this function rolls the existing
    deterministic recovery controller *after the candidate prefix*, starting at
    the true terminal prefix state.  Surrounding agents are extrapolated only
    from the current observable history using constant velocity.  No
    counterfactual future, root identity, teacher margin/component, regime ID or
    held-out label is read.

    Each option receives six signed, tanh-bounded coordinates, all with the same
    physical zero semantics as CPHR but evaluated over the recovery
    continuation rather than over the candidate prefix itself:

      0 minimum clearance reserve over the recovery continuation
      1 terminal clearance reserve after the recovery continuation
      2 clearance recovery gain relative to candidate-prefix terminal clearance
      3 terminal stopping reserve
      4 recovery-controller control-envelope reserve
      5 recovery-controller stability reserve

    The field is padded to ``num_options`` with zeros; padded/invalid options are
    separately masked by ``option_valid`` and never enter OC-MERO.  This makes
    ERWF candidate x option structured while preserving one regime-agnostic
    shared physical field and the frozen Stage-I representation.
    """
    cfg = cfg or {}
    states = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=np.float32)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 4))), dtype=np.float32)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=np.float32).reshape(-1)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=np.float32)
    valid_hist = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    modes = np.asarray(d.get("recovery_modes", []), dtype=object).reshape(-1)
    params = np.asarray(d.get("recovery_params", np.zeros((0, 3))), dtype=np.float32)
    option_valid_raw = np.asarray(d.get("option_valid", np.ones((len(modes),), dtype=bool)), dtype=bool).reshape(-1)

    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 9:
        raise ValueError(
            f"ERWF requires full prefix_states[T,>=9], got shape={getattr(states, 'shape', None)}"
        )
    if ego.size < 9:
        raise ValueError(f"ERWF requires ego_state[>=9], got shape={ego.shape}")
    if controls.ndim != 2:
        controls = np.zeros((0, 4), dtype=np.float32)
    if controls.shape[0] > 0 and controls.shape[1] < 4:
        raise ValueError(f"ERWF requires prefix_controls[T,>=4], got shape={controls.shape}")
    if params.ndim != 2:
        raise ValueError(f"ERWF requires recovery_params[L,P], got shape={params.shape}")
    raw_L = int(max(len(modes), params.shape[0], option_valid_raw.size))
    if raw_L <= 0:
        raise ValueError("ERWF requires recovery_modes/recovery_params/option_valid")
    L = int(num_options if num_options is not None else raw_L)
    if L < raw_L:
        raise ValueError(f"ERWF num_options={L} cannot shrink raw recovery option count={raw_L}")

    sample_rate = float(cfg.get("sample_rate_hz", 10.0) or 10.0)
    recovery_horizon_s = float(cfg.get("recovery_horizon_s", 4.0) or 4.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"invalid sample_rate_hz for ERWF: {sample_rate}")
    if not np.isfinite(recovery_horizon_s) or recovery_horizon_s <= 0.0:
        raise ValueError(f"invalid recovery_horizon_s for ERWF: {recovery_horizon_s}")
    dt = 1.0 / sample_rate
    horizon_steps = max(2, int(round(recovery_horizon_s * sample_rate)))

    prefix = CandidatePrefix(
        macro_id=int(np.asarray(d.get("prefix_macro_id", 0)).reshape(-1)[0]),
        macro_name=str(np.asarray(d.get("prefix_macro_name", "candidate")).reshape(-1)[0]),
        params=np.asarray(d.get("prefix_param", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
        prefix_states=states,
        prefix_controls=controls,
        utility=float(np.asarray(d.get("utility", 0.0)).reshape(-1)[0]),
        feasible=bool(float(np.asarray(d.get("feasible", 1.0)).reshape(-1)[0]) > 0.5),
        hard_violation=float(np.asarray(d.get("hard_violation", 0.0)).reshape(-1)[0]),
        harm_proxy=float(np.asarray(d.get("harm_proxy", 0.0)).reshape(-1)[0]),
    )

    # Observable agents: same nearest-current-agent contract as CPHR.  Use a
    # circle support approximation for efficient vectorized distance evolution;
    # the scientific intervention in v48.61 is continuation/option resolution,
    # not a new collision-geometry estimator.
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    max_agents = int(model_cfg.get("feature_max_agents", 32))
    ego_xy = ego[:2].astype(np.float64, copy=False)
    ego_len = max(abs(float(ego[7])), 1.0e-3)
    ego_wid = max(abs(float(ego[8])), 1.0e-3)
    ego_rad = 0.5 * float(np.hypot(ego_len, ego_wid))
    agents: list[tuple[float, np.ndarray]] = []
    if hist.ndim == 3 and hist.shape[0] > 0 and hist.shape[1] > 1 and valid_hist.ndim >= 2 and valid_hist.shape[0] > 0:
        last = hist[-1]
        vmask = valid_hist[-1].reshape(-1)
        for aidx in range(1, min(last.shape[0], vmask.size)):
            if not bool(vmask[aidx]):
                continue
            row = np.asarray(last[aidx], dtype=np.float64).reshape(-1)
            if row.size < 12 or not np.isfinite(row[:12]).all():
                continue
            agents.append((float(np.linalg.norm(row[:2] - ego_xy)), row))
    agents.sort(key=lambda z: z[0])
    agents = agents[:max_agents]

    if agents:
        rel0 = np.stack([row[:2] - ego_xy for _, row in agents], axis=0)
        vel = np.stack([row[3:5] for _, row in agents], axis=0)
        alen = np.asarray([max(abs(float(row[10])), 1.0e-3) for _, row in agents], dtype=np.float64)
        awid = np.asarray([max(abs(float(row[11])), 1.0e-3) for _, row in agents], dtype=np.float64)
        arad = 0.5 * np.hypot(alen, awid)
    else:
        rel0 = np.zeros((0, 2), dtype=np.float64)
        vel = np.zeros((0, 2), dtype=np.float64)
        arad = np.zeros((0,), dtype=np.float64)

    prefix_duration = float(states.shape[0]) * dt
    terminal_xy_rel = states[-1, :2].astype(np.float64) - ego_xy
    if agents:
        agent_at_prefix_end = rel0 + prefix_duration * vel
        terminal_delta = terminal_xy_rel[None, :] - agent_at_prefix_end
        prefix_terminal_clear = float(
            np.min(np.linalg.norm(terminal_delta, axis=-1) - ego_rad - arad)
        )
    else:
        prefix_terminal_clear = 40.0

    scales = cfg.get("margin_scales", {}) if isinstance(cfg.get("margin_scales", {}), dict) else {}
    distance_scale = max(float(scales.get("distance", 2.0)), 1.0e-6)
    stop_scale = max(float(scales.get("stop", 5.0)), 1.0e-6)
    accel_scale = max(float(scales.get("accel", 1.0)), 1.0e-6)
    decel_scale = max(float(scales.get("decel", 1.0)), 1.0e-6)
    steer_scale = max(float(scales.get("steer", 0.1)), 1.0e-6)
    jerk_scale = max(float(scales.get("jerk", 2.0)), 1.0e-6)
    rate_scale = max(float(scales.get("steer_rate", 0.1)), 1.0e-6)
    yaw_scale = max(float(scales.get("yaw", 0.2)), 1.0e-6)
    d_safe0 = float(cfg.get("d_safe0_m", 1.0))
    headway = float(cfg.get("safe_time_headway_s", 0.5))
    limits = cfg.get("control_limits", {}) if isinstance(cfg.get("control_limits", {}), dict) else {}
    a_max = float(limits.get("a_max", 3.0))
    a_min = float(limits.get("a_min", -6.0))
    delta_max = float(limits.get("delta_max", 0.55))
    jerk_max = float(limits.get("j_max", 6.0))
    steer_rate_max = float(limits.get("steer_rate_max", 0.5))
    yaw_rate_max = float(cfg.get("yaw_rate_max_rps", 0.6))
    stop_decel = max(abs(a_min), 1.0e-3)

    feature_dim = 10 if include_recovery_stability_tail else DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_DIM
    field = np.zeros((L, feature_dim), dtype=np.float32)
    for l in range(raw_L):
        valid_l = bool(option_valid_raw[l]) if l < option_valid_raw.size else True
        if not valid_l:
            continue
        if l >= len(modes) or l >= params.shape[0]:
            raise ValueError(f"ERWF option {l} missing mode/params")
        mode = _decode_recovery_mode(modes[l])
        option = RecoveryOption(
            option_id=l,
            mode=mode,
            params=np.asarray(params[l], dtype=np.float32).reshape(-1),
            valid=True,
        )
        rec_states, rec_controls, _diag = rollout_recovery_controller(
            prefix, option, horizon_steps, cfg
        )
        rec_states = np.asarray(rec_states, dtype=np.float64)
        rec_controls = np.asarray(rec_controls, dtype=np.float64)
        if rec_states.ndim != 2 or rec_states.shape[0] < 1 or rec_states.shape[1] < 9:
            raise ValueError(f"ERWF recovery rollout invalid for option={l}: shape={rec_states.shape}")

        max_speed = float(np.max(np.abs(rec_states[:, 6])))
        terminal_speed = float(abs(rec_states[-1, 6]))
        d_safe = d_safe0 + headway * max_speed
        if agents:
            # rec_states[0] is exactly the candidate terminal state at
            # prefix_duration; later rows are separated by dt.
            times = prefix_duration + np.arange(rec_states.shape[0], dtype=np.float64) * dt
            agent_future = rel0[None, :, :] + times[:, None, None] * vel[None, :, :]
            rec_xy_rel = rec_states[:, :2] - ego_xy[None, :]
            delta = rec_xy_rel[:, None, :] - agent_future
            signed_clearance = np.linalg.norm(delta, axis=-1) - ego_rad - arad[None, :]
            c_min = float(np.min(signed_clearance))
            c_terminal = float(np.min(signed_clearance[-1]))
        else:
            c_min = c_terminal = 40.0

        h_min_clear = np.tanh((c_min - d_safe) / distance_scale)
        h_terminal_clear = np.tanh((c_terminal - d_safe) / distance_scale)
        h_clear_gain = np.tanh((c_terminal - prefix_terminal_clear) / distance_scale)
        stop_required = terminal_speed * terminal_speed / (2.0 * stop_decel) + d_safe0
        h_stop = np.tanh((c_terminal - stop_required) / stop_scale)

        if rec_controls.ndim == 2 and rec_controls.shape[0] > 0 and rec_controls.shape[1] >= 4:
            rc = rec_controls[:, :4]
            ctrl_terms = np.asarray([
                (a_max - float(np.max(np.maximum(rc[:, 0], 0.0)))) / accel_scale,
                (abs(a_min) - float(np.max(np.maximum(-rc[:, 0], 0.0)))) / decel_scale,
                (delta_max - float(np.max(np.abs(rc[:, 1])))) / steer_scale,
                (jerk_max - float(np.max(np.abs(rc[:, 2])))) / jerk_scale,
                (steer_rate_max - float(np.max(np.abs(rc[:, 3])))) / rate_scale,
            ], dtype=np.float64)
            h_control = np.tanh(float(np.min(ctrl_terms)))
        else:
            h_control = np.float64(1.0)
        yaw = float(np.max(np.abs(rec_states[:, 5])))
        h_stability = np.tanh((yaw_rate_max - yaw) / yaw_scale)
        values = [
            h_min_clear,
            h_terminal_clear,
            h_clear_gain,
            h_stop,
            h_control,
            h_stability,
        ]
        if include_recovery_stability_tail:
            terminal_yaw = float(abs(rec_states[-1, 5]))
            initial_yaw = float(abs(rec_states[0, 5]))
            h_terminal_stability = np.tanh((yaw_rate_max - terminal_yaw) / yaw_scale)
            h_stability_gain = np.tanh((initial_yaw - terminal_yaw) / yaw_scale)
            # Path-preservation witnesses prevent a terminal recovery from hiding
            # a worse secondary excursion during the recovery continuation.
            h_clearance_floor_gain = np.tanh((c_min - prefix_terminal_clear) / distance_scale)
            h_stability_floor_gain = np.tanh((initial_yaw - yaw) / yaw_scale)
            values.extend([
                h_terminal_stability, h_stability_gain,
                h_clearance_floor_gain, h_stability_floor_gain,
            ])
        field[l] = np.asarray(values, dtype=np.float32)

    if field.shape != (L, feature_dim) or not np.isfinite(field).all():
        raise ValueError(f"invalid ERWF feature field shape/value: shape={field.shape}")
    return field


# v48.62 OC-CWRF (Observation-Consistent Common-Witness Recovery Field).
# The first six coordinates are execution-identical to ERWF.  Two recovery-tail
# stability coordinates make post-contact recovery non-compensatory without
# treating the already-violated initial state as permanently infeasible.
DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA = 1
DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_DIM = 10


def direct_common_recovery_witness_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None, *, num_options: int | None = None
) -> np.ndarray:
    """Return the v48.62 option-resolved finite-time recovery witness field.

    Feature order:
      0 continuation minimum-clearance reserve
      1 continuation terminal-clearance reserve
      2 clearance recovery gain from candidate terminal
      3 terminal stopping reserve
      4 controller control-envelope reserve
      5 minimum stability reserve over the continuation
      6 terminal stability reserve
      7 stability recovery gain (initial |yaw| - terminal |yaw|)
      8 clearance floor gain (minimum continuation clearance - initial clearance)
      9 stability floor gain (initial |yaw| - maximum continuation |yaw|)

    No regime ID, latent-root identity, teacher component/future or held-out label
    is read.  This is the same deterministic executable continuation used by
    ERWF, augmented only with finite-time recovery semantics for stability.
    """
    out = direct_executable_recovery_witness_features_from_sample(
        d, cfg, num_options=num_options, include_recovery_stability_tail=True
    )
    if out.ndim != 2 or out.shape[1] != DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_DIM:
        raise ValueError(f"invalid OC-CWRF feature field shape: {out.shape}")
    return out


def _feature_layout_values(cfg: dict | None = None) -> dict[str, int]:
    cfg = cfg or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    return {
        "max_agents": int(model_cfg.get("feature_max_agents", 32)),
        "num_macros": int(model_cfg.get("num_macros", 16)),
        "prefix_flat": int(model_cfg.get("feature_prefix_flat_dim", 80)),
        "control_flat": int(model_cfg.get("feature_control_flat_dim", 40)),
        "route_flat": int(model_cfg.get("feature_route_flat_dim", 64)),
        "map_flat": int(model_cfg.get("feature_map_flat_dim", 64)),
        "dyn_flat": int(model_cfg.get("feature_dynamic_map_flat_dim", 32)),
    }


def _candidate_feature_parts(d: dict[str, Any], cfg: dict | None = None) -> list[np.ndarray]:
    """Candidate-dependent prefix features in the exact historical layout."""
    cfg = cfg or {}
    layout = _feature_layout_values(cfg)
    ego = _pad_flat(_arr(d, "ego_state", (9,)), 9)
    prefix_param = _pad_flat(_arr(d, "prefix_param", (5,)), int(cfg.get("prefix_param_dim", 5)))
    macro_type_id = _int_scalar(d, "prefix_macro_type_id", _int_scalar(d, "prefix_macro_id", 0))
    macro = _one_hot(macro_type_id, layout["num_macros"])
    scalar_feat = np.array([
        _scalar(d, "utility") / 20.0,
        _scalar(d, "hard_violation") / 5.0,
        _scalar(d, "harm_proxy") / 5.0,
        _scalar(d, "feasible"),
        _scalar(d, "is_nominal"),
        _scalar(d, "time_index") / 1000.0,
    ], dtype=np.float32)
    return [
        ego,
        prefix_param,
        macro,
        scalar_feat,
        _pad_flat(_arr(d, "prefix_states"), layout["prefix_flat"]),
        _pad_flat(_arr(d, "prefix_controls"), layout["control_flat"]),
    ]


def _shared_scene_feature_parts(d: dict[str, Any], cfg: dict | None = None) -> list[np.ndarray]:
    """History/map/BEV features shared by all candidates of one replan."""
    cfg = cfg or {}
    layout = _feature_layout_values(cfg)
    bev = _arr(d, "bev_occ")
    route = _arr(d, "route")
    maps = _arr(d, "map_polylines")
    dyn = _arr(d, "dynamic_map")

    bev_stats: list[np.ndarray] = []
    if bev.ndim >= 3:
        ch = bev.reshape(bev.shape[0], -1)
        bev_stats.append(ch.mean(axis=1))
        bev_stats.append(ch.std(axis=1))
    else:
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
    bev_feat = _pad_flat(np.concatenate(bev_stats, axis=0), 2 * int(cfg.get("bev_channels", 7)))

    return [
        _agent_features(d, layout["max_agents"]),
        bev_feat,
        _finite_stats(route),
        _pad_flat(route[..., :2] if route.ndim >= 2 else route, layout["route_flat"]),
        _finite_stats(maps),
        _pad_flat(maps[..., :2] if maps.ndim >= 3 else maps, layout["map_flat"]),
        _finite_stats(dyn),
        _pad_flat(dyn, layout["dyn_flat"]),
    ]


def sample_to_feature(d: dict[str, Any], cfg: dict | None = None) -> np.ndarray:
    """Convert one sample to the flat model feature without changing layout."""
    parts = [*_candidate_feature_parts(d, cfg), *_shared_scene_feature_parts(d, cfg)]
    x = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def samples_to_feature_matrix(
    ds: list[dict[str, Any]],
    cfg: dict | None = None,
    *,
    shared_scene: bool = False,
) -> np.ndarray:
    """Vectorize feature extraction and optionally reuse scene-static work.

    All candidates created at one closed-loop replan share history, map, route,
    BEV and dynamic-map tensors.  The previous implementation recomputed agent
    sorting, BEV statistics, map statistics and flattening once per candidate.
    ``shared_scene=True`` computes those exact arrays once and concatenates them
    with each candidate's prefix-dependent features.  The resulting rows are
    numerically identical to repeated :func:`sample_to_feature` calls.
    """
    if not ds:
        return np.zeros((0, 0), dtype=np.float32)
    if not shared_scene:
        return np.stack([sample_to_feature(d, cfg) for d in ds], axis=0)
    shared = _shared_scene_feature_parts(ds[0], cfg)
    rows = []
    for d in ds:
        x = np.concatenate([*_candidate_feature_parts(d, cfg), *shared], axis=0).astype(np.float32)
        rows.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(rows, axis=0)


def _target_dim_from_cfg(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _model_target_dim(cfg: dict, key: str, default: int) -> int:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        return int(model_cfg.get(key, default))
    except Exception:
        return int(default)


def _fix_1d(x: np.ndarray, n: int, *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((int(n),), fill, dtype=dtype)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=dtype).reshape(-1)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _fix_bool_1d(x: np.ndarray, n: int, *, fill: bool = False) -> np.ndarray:
    out = np.full((int(n),), bool(fill), dtype=bool)
    if n <= 0:
        return out
    v = np.asarray(x).reshape(-1).astype(bool)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return out


def _fix_2d(x: np.ndarray, shape: tuple[int, int], *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    out = np.full((rows, cols), fill, dtype=dtype)
    if rows <= 0 or cols <= 0:
        return out
    v = np.asarray(x, dtype=dtype)
    if v.ndim == 1:
        if rows == 1:
            v = v.reshape(1, -1)
        else:
            v = v.reshape(-1, cols) if v.size and v.size % cols == 0 else v.reshape(1, -1)
    if v.ndim != 2:
        return out
    r = min(rows, v.shape[0])
    c = min(cols, v.shape[1])
    if r > 0 and c > 0:
        out[:r, :c] = v[:r, :c]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _fix_square(x: np.ndarray, n: int, *, fill_offdiag: float = 0.0, diag: float = 1.0, dtype=np.float32) -> np.ndarray:
    out = np.full((int(n), int(n)), fill_offdiag, dtype=dtype)
    if n > 0:
        np.fill_diagonal(out, diag)
    v = np.asarray(x, dtype=dtype)
    if v.ndim == 2:
        r = min(int(n), v.shape[0])
        c = min(int(n), v.shape[1])
        if r > 0 and c > 0:
            out[:r, :c] = v[:r, :c]
            np.fill_diagonal(out, diag)
    return np.nan_to_num(out, nan=fill_offdiag, posinf=fill_offdiag, neginf=fill_offdiag)




def bucket_id_for_path(path: str | Path) -> int:
    """Coarse regime id inferred from the current dataset root path.

    This is used only for regime-aware training losses and diagnostics.  The
    selector must still make decisions from model predictions and candidate
    metadata; it should not require teacher labels at test time.
    """
    try:
        name = str(_dataset_root_for_sample(path).name).lower()
    except Exception:
        name = str(path).lower()
    if "near" in name:
        return int(BUCKET_TO_ID["near_contact"])
    if "contact" in name:
        return int(BUCKET_TO_ID["contact"])
    if "safe" in name or "normal" in name or "background" in name:
        return int(BUCKET_TO_ID["safe"])
    return int(BUCKET_TO_ID["other"])


def stable_scene_hash(scene_id: object) -> int:
    """Stable non-negative 31-bit hash for grouping scene-time candidates."""
    b = str(scene_id).encode("utf-8", errors="ignore")
    return int(zlib.adler32(b) & 0x7FFFFFFF)

def _geometry_from_sample(d: dict[str, Any]) -> tuple[int, int, int, int]:
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=np.float32)
    K = int(m.shape[0]) if m.ndim >= 1 else 0
    L = int(m.shape[1]) if m.ndim >= 2 else 0
    rs = np.asarray(d.get("root_signature", np.zeros((K, 0))), dtype=np.float32)
    fs = np.asarray(d.get("root_future_signature", np.zeros((K, 0))), dtype=np.float32)
    return K, L, int(rs.shape[-1]) if rs.ndim >= 2 else 0, int(fs.shape[-1]) if fs.ndim >= 2 else 0


def fix_sample_geometry(d: dict[str, Any], *, num_roots: int, num_options: int, d_signature: int = 0, d_future_signature: int = 0) -> dict[str, np.ndarray]:
    """Return training/evaluation arrays padded to checkpoint geometry.

    The four intended OC-RAP dataset families are intentionally heterogeneous:
    proof-artifact sets often use fewer roots/options than natural/strict/post
    sets.  A single neural model still needs fixed tensor shapes, so padded roots
    and options are marked invalid and excluded by masks, losses, OC-MERO, and
    metrics.
    """
    K = int(num_roots)
    L = int(num_options)
    root_probs = _fix_1d(np.asarray(d.get("root_probs", []), dtype=np.float32), K, fill=0.0)
    root_valid = _fix_bool_1d(np.asarray(d.get("root_valid", root_probs > 0), dtype=bool), K, fill=False)
    if root_valid.any():
        root_probs = np.where(root_valid, np.clip(root_probs, 0.0, None), 0.0).astype(np.float32)
        s = float(root_probs.sum())
        if s > 1e-8:
            root_probs = (root_probs / s).astype(np.float32)
    option_valid = _fix_bool_1d(np.asarray(d.get("option_valid", np.ones((L,), dtype=bool)), dtype=bool), L, fill=False)
    return {
        "root_probs": root_probs,
        "m_star": _fix_2d(np.asarray(d.get("m_star", []), dtype=np.float32), (K, L), fill=-1e9),
        "c_star": _fix_square(np.asarray(d.get("c_star", []), dtype=np.float32), K, fill_offdiag=0.0, diag=1.0),
        "y_obs": _fix_square(np.asarray(d.get("y_obs", []), dtype=np.float32), K, fill_offdiag=0.0, diag=1.0),
        "option_valid": option_valid,
        "root_valid": root_valid,
        "root_signature": _fix_2d(np.asarray(d.get("root_signature", []), dtype=np.float32), (K, int(d_signature)), fill=0.0),
        "root_future_signature": _fix_2d(np.asarray(d.get("root_future_signature", []), dtype=np.float32), (K, int(d_future_signature)), fill=0.0),
        "option_features": _fix_2d(option_features_from_sample(d), (L, OPTION_FEATURE_DIM), fill=0.0),
    }


def _nominal_deviation_by_path(paths: list[Path]) -> list[float]:
    """Compute the certificate's candidate-vs-nominal prefix deviation.

    v48.31 keeps validation/checkpoint selection on exactly the same eligible
    population as policy calibration.  The value is precomputed once because a
    normal Dataset item has no access to the other candidates in its scene-time
    group.
    """

    records: list[tuple[tuple[int, str, int], bool, np.ndarray | None]] = []
    grouped: dict[tuple[int, str, int], list[int]] = {}
    for index, path in enumerate(paths):
        d = load_npz_selected(path, NOMINAL_DEVIATION_NPZ_KEYS)
        scene = str(np.asarray(d.get("scene_id", path.stem)).item())
        time_index = int(np.asarray(d.get("time_index", 0)).item())
        key = (bucket_id_for_path(path), scene, time_index)
        is_nominal = bool(float(np.asarray(d.get("is_nominal", 0.0)).item()) > 0.5)
        try:
            prefix = np.asarray(d.get("prefix_states"), dtype=np.float64)[:, :2]
        except Exception:
            prefix = None
        records.append((key, is_nominal, prefix))
        grouped.setdefault(key, []).append(index)
    out = [0.0] * len(paths)
    for indices in grouped.values():
        nominal_index = next((i for i in indices if records[i][1]), None)
        if nominal_index is None:
            continue
        reference = records[nominal_index][2]
        if reference is None:
            continue
        for i in indices:
            prefix = records[i][2]
            if prefix is None:
                continue
            length = min(len(reference), len(prefix))
            if length <= 0:
                continue
            out[i] = float(
                np.sqrt(np.mean(np.sum((prefix[:length] - reference[:length]) ** 2, axis=-1)))
                / 5.0
            )
    return out


_PERSISTENT_TENSOR_CACHE_SCHEMA = 5

def _load_persistent_tensor_cache_payload(cache_path: Path) -> dict[str, Any] | None:
    """Load an immutable decoded-tensor cache with mmap when supported.

    Multiple v48.46 arms/variants may read the same cache concurrently.  mmap
    keeps tensor storages file-backed so the OS page cache can be shared across
    processes instead of materialising another full host-RAM copy per trainer.
    The payload contains tensors plus primitive provenance only, so
    ``weights_only=True`` is sufficient on modern PyTorch.  Older PyTorch
    versions fall back to the historical loader.  A malformed/truncated cache
    is treated as a cache miss and rebuilt under the existing flock rather than
    becoming an experiment-level RC=30.
    """
    attempts = (
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
        {"weights_only": False},
    )
    for extra in attempts:
        try:
            payload = torch.load(cache_path, map_location="cpu", **extra)
            return payload if isinstance(payload, dict) else None
        except TypeError:
            # ``mmap``/``weights_only`` are version-dependent keyword args.
            continue
        except Exception:
            # A second loader mode may still work (for example mmap unsupported
            # by the filesystem).  If every mode fails the caller rebuilds.
            continue
    return None

def _dataset_manifest_fingerprint(paths: list[Path]) -> list[dict[str, Any]]:
    """Stable provenance for a decoded-tensor cache without opening every NPZ."""
    roots = sorted({_dataset_root_for_sample(p).resolve() for p in paths}, key=lambda x: str(x))
    rows: list[dict[str, Any]] = []
    for root in roots:
        manifest = root / "manifest.csv"
        if manifest.is_file():
            raw = manifest.read_bytes()
            rows.append({
                "root": str(root),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "manifest_size": len(raw),
            })
        else:
            # Fail-safe fallback for non-manifest datasets used by older tests.
            member = [p for p in paths if _dataset_root_for_sample(p).resolve() == root]
            rows.append({
                "root": str(root),
                "members": [(str(p.resolve()), int(p.stat().st_size), int(p.stat().st_mtime_ns)) for p in member],
            })
    return rows

def _persistent_tensor_cache_key(
    paths: list[Path], cfg: dict, *, num_roots: int, num_options: int,
    d_signature: int, d_future_signature: int, feature_dim: int,
) -> str:
    training = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    cphr_features_enabled = bool(model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False))
    erwf_features_enabled = bool(model_cfg.get("direct_recovery_absolute_executable_witness_correction", False))
    common_witness_features_enabled = bool(model_cfg.get("direct_recovery_absolute_common_witness_correction", False))
    payload = {
        "schema": _PERSISTENT_TENSOR_CACHE_SCHEMA,
        "manifests": _dataset_manifest_fingerprint(paths),
        "path_count": len(paths),
        "paths": [str(p.resolve()) for p in paths],
        "geometry": {
            "num_roots": int(num_roots), "num_options": int(num_options),
            "d_signature": int(d_signature), "d_future_signature": int(d_future_signature),
            "feature_dim": int(feature_dim),
        },
        # Only tensor-construction settings belong in this key.  Model-head,
        # optimizer, ROCT and option-execution settings do not change any
        # OCRAPSampleDataset tensor and previously forced redundant 90--250 s
        # decompression passes between witness/factor stages and ablation arms.
        "feature_layout": _feature_layout_values(cfg),
        "prefix_param_dim": int(cfg.get("prefix_param_dim", 5)),
        "bev_channels": int(cfg.get("bev_channels", 7)),
        "exact_eligibility": bool(training.get("direct_policy_metric_exact_eligibility", False)),
        "cphr_full_prefix_features": cphr_features_enabled,
        "cphr_feature_schema": (DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA if cphr_features_enabled else 0),
        "erwf_option_resolved_features": erwf_features_enabled,
        "erwf_feature_schema": (DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA if erwf_features_enabled else 0),
        "erwf_recovery_horizon_s": (float(cfg.get("recovery_horizon_s", 4.0)) if erwf_features_enabled else 0.0),
        "erwf_sample_rate_hz": (float(cfg.get("sample_rate_hz", 10.0)) if erwf_features_enabled else 0.0),
        "common_witness_option_resolved_features": common_witness_features_enabled,
        "common_witness_feature_schema": (DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA if common_witness_features_enabled else 0),
        "common_witness_recovery_horizon_s": (float(cfg.get("recovery_horizon_s", 4.0)) if common_witness_features_enabled else 0.0),
        "common_witness_sample_rate_hz": (float(cfg.get("sample_rate_hz", 10.0)) if common_witness_features_enabled else 0.0),
        "npz_keys": sorted(MODEL_SAMPLE_NPZ_KEYS),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _stack_tensor_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        return {}
    keys = tuple(items[0].keys())
    if any(tuple(x.keys()) != keys for x in items):
        raise ValueError("persistent tensor cache item-key mismatch")
    return {k: torch.stack([x[k] for x in items], dim=0).contiguous() for k in keys}



class OCRAPSampleDataset(Dataset):
    def __init__(self, paths: list[Path], cfg: dict | None = None):
        self.paths = list(paths)
        self.cfg = cfg or {}
        if not self.paths:
            raise ValueError("OCRAPSampleDataset requires at least one sample path")
        first = load_npz_selected(self.paths[0], MODEL_SAMPLE_NPZ_KEYS)
        first_K, first_L, first_sig, first_fsig = _geometry_from_sample(first)
        # Default to the paper/build geometry in config, but never shrink below
        # the first sample.  This makes the README's mixed proof/natural/stress
        # training command work without requiring per-run shape bookkeeping.
        self.num_roots = max(_target_dim_from_cfg(self.cfg, "num_roots", first_K or 1), first_K or 1)
        self.num_options = max(_target_dim_from_cfg(self.cfg, "num_recovery_options", first_L or 1), first_L or 1)
        self.d_signature = max(_model_target_dim(self.cfg, "d_signature", first_sig), first_sig)
        self.d_future_signature = max(_model_target_dim(self.cfg, "d_future_signature", first_fsig), first_fsig)
        self.feature_dim = int(sample_to_feature(first, self.cfg).shape[0])
        training_cfg = self.cfg.get("training", {}) if isinstance(self.cfg.get("training", {}), dict) else {}
        self.nominal_deviation = (
            _nominal_deviation_by_path(self.paths)
            if bool(training_cfg.get("direct_policy_metric_exact_eligibility", False))
            else [0.0] * len(self.paths)
        )
        # v48.45.6 engineering-only fast path.  A decoded model item is only a
        # few KB (flat feature + recovery labels), whereas the source NPZ can
        # contain large compressed map/BEV/debug arrays.  Caching the final CPU
        # tensors once removes repeated ZIP decompression across 8--20 epochs.
        # It is opt-in so non-v48 workflows keep their historical memory profile.
        self.persistent_tensor_cache = bool(training_cfg.get("persistent_tensor_cache", False))
        cache_dir_raw = str(training_cfg.get("persistent_tensor_cache_dir", "") or "").strip()
        self.persistent_tensor_cache_dir = Path(cache_dir_raw).expanduser() if cache_dir_raw else None
        self.cache_samples_in_memory = bool(training_cfg.get("cache_samples_in_memory", False)) or self.persistent_tensor_cache
        self._item_cache: list[dict[str, torch.Tensor]] | None = None
        self._stacked_item_cache: dict[str, torch.Tensor] | None = None
        self.tensor_cache_event: dict[str, Any] = {"enabled": self.persistent_tensor_cache, "hit": False}
        if self.persistent_tensor_cache:
            if self.persistent_tensor_cache_dir is None:
                raise ValueError("training.persistent_tensor_cache=true requires persistent_tensor_cache_dir")
            self._load_or_build_persistent_tensor_cache()
        elif self.cache_samples_in_memory:
            self._item_cache = [self._build_item(i) for i in range(len(self.paths))]

    def _load_or_build_persistent_tensor_cache(self) -> None:
        assert self.persistent_tensor_cache_dir is not None
        cache_dir = self.persistent_tensor_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _persistent_tensor_cache_key(
            self.paths, self.cfg, num_roots=self.num_roots, num_options=self.num_options,
            d_signature=self.d_signature, d_future_signature=self.d_future_signature,
            feature_dim=self.feature_dim,
        )
        cache_path = cache_dir / f"ocrap_tensor_items_{key}.pt"
        lock_path = cache_dir / f"ocrap_tensor_items_{key}.lock"
        t0 = time.perf_counter()
        with lock_path.open("a+b") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                if cache_path.is_file():
                    payload = _load_persistent_tensor_cache_payload(cache_path)
                    if (
                        isinstance(payload, dict)
                        and int(payload.get("schema", -1)) == _PERSISTENT_TENSOR_CACHE_SCHEMA
                        and payload.get("key") == key
                        and int(payload.get("num_items", -1)) == len(self.paths)
                        and isinstance(payload.get("tensors"), dict)
                    ):
                        tensors = payload["tensors"]
                        if tensors and all(int(v.shape[0]) == len(self.paths) for v in tensors.values()):
                            self._stacked_item_cache = tensors
                            self.tensor_cache_event = {
                                "enabled": True, "hit": True, "key": key, "path": str(cache_path),
                                "seconds": float(time.perf_counter() - t0),
                            }
                            return
                    # Corrupt/stale same-name artifact should never be trusted.
                    cache_path.unlink(missing_ok=True)
                items = [self._build_item(i) for i in range(len(self.paths))]
                tensors = _stack_tensor_items(items)
                payload = {
                    "schema": _PERSISTENT_TENSOR_CACHE_SCHEMA, "key": key,
                    "num_items": len(self.paths), "tensors": tensors,
                }
                tmp = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
                torch.save(payload, tmp)
                os.replace(tmp, cache_path)
                self._stacked_item_cache = tensors
                self.tensor_cache_event = {
                    "enabled": True, "hit": False, "key": key, "path": str(cache_path),
                    "seconds": float(time.perf_counter() - t0),
                }
            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

    def cached_tensor_bytes(self) -> int:
        if self._stacked_item_cache is not None:
            return int(sum(v.numel() * v.element_size() for v in self._stacked_item_cache.values()))
        if self._item_cache is not None:
            return int(sum(t.numel() * t.element_size() for item in self._item_cache for t in item.values()))
        return 0

    def __len__(self) -> int:
        return len(self.paths)

    def _build_item(self, idx: int) -> dict[str, torch.Tensor]:
        d = load_npz_selected(self.paths[idx], MODEL_SAMPLE_NPZ_KEYS)
        x = sample_to_feature(d, self.cfg)
        fixed = fix_sample_geometry(
            d,
            num_roots=self.num_roots,
            num_options=self.num_options,
            d_signature=self.d_signature,
            d_future_signature=self.d_future_signature,
        )
        out = {
            "x": torch.from_numpy(x),
            "root_probs": torch.from_numpy(fixed["root_probs"]),
            "m_star": torch.from_numpy(fixed["m_star"]),
            "c_star": torch.from_numpy(fixed["c_star"]),
            "y_obs": torch.from_numpy(fixed["y_obs"]),
            "option_valid": torch.from_numpy(fixed["option_valid"]),
            "root_valid": torch.from_numpy(fixed["root_valid"]),
            "r_dep_star": torch.tensor(float(np.asarray(d["r_dep_star"]).item()), dtype=torch.float32),
            "r_orc_star": torch.tensor(float(np.asarray(d["r_orc_star"]).item()), dtype=torch.float32),
            "i_art_star": torch.tensor(float(np.asarray(d["i_art_star"]).item()), dtype=torch.float32),
            "utility": torch.tensor(float(np.asarray(d.get("utility", 0.0)).item()), dtype=torch.float32),
            "hard_violation": torch.tensor(float(np.asarray(d.get("hard_violation", 0.0)).item()), dtype=torch.float32),
            "harm_proxy": torch.tensor(float(np.asarray(d.get("harm_proxy", 0.0)).item()), dtype=torch.float32),
            "feasible": torch.tensor(float(np.asarray(d.get("feasible", 1.0)).item()), dtype=torch.float32),
            "nominal_deviation": torch.tensor(float(self.nominal_deviation[idx]), dtype=torch.float32),
            "scene_hash": torch.tensor(stable_scene_hash(d.get("scene_id", "")), dtype=torch.long),
            "time_index": torch.tensor(int(np.asarray(d.get("time_index", 0)).item()), dtype=torch.long),
            "candidate_index": torch.tensor(int(np.asarray(d.get("candidate_index", 0)).item()), dtype=torch.long),
            # Semantic macro id used by macro-conditioned recovery losses.
            # Older datasets only stored prefix_macro_id, so keep the same
            # fallback convention as sample_to_feature().
            "prefix_macro_type_id": torch.tensor(
                int(np.asarray(d.get("prefix_macro_type_id", d.get("prefix_macro_id", 0))).item()),
                dtype=torch.long,
            ),
            "is_nominal": torch.tensor(float(np.asarray(d.get("is_nominal", 0)).item()), dtype=torch.float32),
            "bucket_id": torch.tensor(bucket_id_for_path(self.paths[idx]), dtype=torch.long),
            "root_signature": torch.from_numpy(fixed["root_signature"]),
            "root_future_signature": torch.from_numpy(fixed["root_future_signature"]),
            "option_features": torch.from_numpy(fixed["option_features"]),
        }
        model_cfg = self.cfg.get("model", {}) if isinstance(self.cfg.get("model", {}), dict) else {}
        if bool(model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False)):
            out["direct_absolute_physical_headroom_features"] = torch.from_numpy(
                direct_absolute_physical_headroom_features_from_sample(d, self.cfg)
            )
        if bool(model_cfg.get("direct_recovery_absolute_executable_witness_correction", False)):
            out["direct_absolute_executable_witness_features"] = torch.from_numpy(
                direct_executable_recovery_witness_features_from_sample(
                    d, self.cfg, num_options=self.num_options
                )
            )
        if bool(model_cfg.get("direct_recovery_absolute_common_witness_correction", False)):
            out["direct_absolute_common_witness_features"] = torch.from_numpy(
                direct_common_recovery_witness_features_from_sample(
                    d, self.cfg, num_options=self.num_options
                )
            )
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._stacked_item_cache is not None:
            return {k: v[idx] for k, v in self._stacked_item_cache.items()}
        if self._item_cache is not None:
            return self._item_cache[idx]
        return self._build_item(idx)


def split_paths_by_npz_split(paths: list[Path], split: str | set[str]) -> list[Path]:
    requested = {split} if isinstance(split, str) else set(split)
    if "all" in requested:
        return list(paths)
    allowed = expand_split_roles(requested)
    keep: list[Path] = []
    for p in paths:
        sid = str(scalar_metadata_for_path(p, "split_id", ""))
        if sid in allowed:
            keep.append(p)
    return keep
