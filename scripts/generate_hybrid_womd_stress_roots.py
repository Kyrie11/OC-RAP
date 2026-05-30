#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REGIMES = ["normal_high_headroom", "low_headroom", "near_contact", "contact_post_contact"]


def _load_roots(root_dir: Path, split: str) -> list[Path]:
    if split != "all" and (root_dir / "splits.json").exists():
        ids = json.loads((root_dir / "splits.json").read_text()).get(split, [])
        return [root_dir / f"{rid}.json" for rid in ids if (root_dir / f"{rid}.json").exists()]
    return sorted(p for p in root_dir.glob("*.json") if p.name not in ("metadata.json", "splits.json"))


def _ego_xy(root: dict) -> tuple[float, float, float]:
    e = root.get("ego_state", {}) or {}
    return float(e.get("x", 0.0)), float(e.get("y", 0.0)), float(e.get("heading", 0.0))


def _actor_template(actor_id: str, x: float, y: float, heading: float, vx: float, vy: float, length: float = 4.7, width: float = 1.9) -> dict:
    return {
        "actor_id": actor_id, "x": float(x), "y": float(y), "heading": float(heading),
        "vx": float(vx), "vy": float(vy), "length": float(length), "width": float(width),
        "actor_type": "vehicle", "dynamic": True,
    }


def _ensure_actors(root: dict) -> list[dict]:
    actors = root.setdefault("actor_states", [])
    if not isinstance(actors, list):
        actors = []
        root["actor_states"] = actors
    return actors


def _set_regime(root: dict, regime: str) -> None:
    root["regime"] = regime
    root.setdefault("traffic_config", {})["hybrid_stress_regime"] = regime


def _append_history_actor(root: dict, actor: dict, *, visible_history: bool = True, dt: float = 0.2) -> None:
    """Append a kinematically consistent stress-actor history.

    The old generator copied the current stress actor into every past frame,
    which made an occluded/cut-in actor appear frozen in the BEV history.  For
    visible stressors we back-propagate with the actor velocity; for occluded
    release we intentionally omit it from history so it is not observable before
    release.
    """
    hist = root.get("history", []) or []
    if not visible_history:
        root.setdefault("traffic_config", {})["stress_actor_hidden_in_history"] = True
        return
    n = len(hist)
    for idx, h in enumerate(hist):
        h.setdefault("actor_states", [])
        if not isinstance(h["actor_states"], list):
            continue
        age = float(max(0, n - 1 - idx)) * float(dt)
        past = copy.deepcopy(actor)
        past["x"] = float(actor.get("x", 0.0) - actor.get("vx", 0.0) * age)
        past["y"] = float(actor.get("y", 0.0) - actor.get("vy", 0.0) * age)
        h["actor_states"].append(past)


def _apply_lead_brake(root: dict, severity: float) -> str:
    ex, ey, eh = _ego_xy(root)
    e = root.get("ego_state", {}) or {}
    ev = float(e.get("v", 8.0))
    dist = max(5.0, 16.0 - 8.0 * severity)
    x = ex + dist * math.cos(eh)
    y = ey + dist * math.sin(eh)
    v = max(0.0, ev * (0.65 - 0.35 * severity))
    actor = _actor_template(f"stress_lead_brake_{root.get('root_id','root')}", x, y, eh, v * math.cos(eh), v * math.sin(eh))
    _ensure_actors(root).append(actor)
    _append_history_actor(root, actor, visible_history=True)
    root.setdefault("traffic_config", {})["stress_type"] = "lead_brake"
    root["traffic_config"]["lead_brake_decel_proxy"] = float(3.0 + 4.0 * severity)
    _set_regime(root, "near_contact" if severity >= 0.55 else "low_headroom")
    return "lead_brake"


def _apply_cut_in(root: dict, severity: float) -> str:
    ex, ey, eh = _ego_xy(root)
    e = root.get("ego_state", {}) or {}
    ev = float(e.get("v", 8.0))
    dist = max(4.0, 18.0 - 10.0 * severity)
    lane_offset = 3.6 * (1.0 if random.random() < 0.5 else -1.0)
    x = ex + dist * math.cos(eh) - lane_offset * math.sin(eh)
    y = ey + dist * math.sin(eh) + lane_offset * math.cos(eh)
    v = max(1.0, ev * (0.8 - 0.25 * severity))
    actor = _actor_template(f"stress_cut_in_{root.get('root_id','root')}", x, y, eh, v * math.cos(eh), v * math.sin(eh))
    actor["vy"] += -math.copysign(1.0 + 1.5 * severity, lane_offset) * math.cos(eh)
    actor["vx"] += math.copysign(1.0 + 1.5 * severity, lane_offset) * math.sin(eh)
    _ensure_actors(root).append(actor)
    _append_history_actor(root, actor, visible_history=True)
    root.setdefault("traffic_config", {})["stress_type"] = "cut_in"
    root["traffic_config"]["cut_in_lateral_speed_proxy"] = float(1.0 + 1.5 * severity)
    _set_regime(root, "near_contact" if severity >= 0.5 else "low_headroom")
    return "cut_in"


def _apply_occluded_release(root: dict, severity: float) -> str:
    ex, ey, eh = _ego_xy(root)
    e = root.get("ego_state", {}) or {}
    ev = float(e.get("v", 8.0))
    side = 1.0 if random.random() < 0.5 else -1.0
    x = ex + max(3.5, 12.0 - 6.0 * severity) * math.cos(eh) - side * 5.0 * math.sin(eh)
    y = ey + max(3.5, 12.0 - 6.0 * severity) * math.sin(eh) + side * 5.0 * math.cos(eh)
    heading = eh - side * math.pi / 2.0
    v = max(1.0, 4.0 + 5.0 * severity)
    actor = _actor_template(f"stress_occluded_release_{root.get('root_id','root')}", x, y, heading, v * math.cos(heading), v * math.sin(heading), length=1.8, width=0.8)
    actor["actor_type"] = "pedestrian" if severity > 0.65 else "cyclist"
    _ensure_actors(root).append(actor)
    _append_history_actor(root, actor, visible_history=False)
    root.setdefault("traffic_config", {})["stress_type"] = "occluded_release"
    root["traffic_config"]["occlusion_release_time_proxy"] = float(max(0.1, 1.2 - severity))
    _set_regime(root, "near_contact")
    return "occluded_release"


def _apply_contact_proxy(root: dict, severity: float) -> str:
    ex, ey, eh = _ego_xy(root)
    e = root.get("ego_state", {}) or {}
    ev = float(e.get("v", 5.0))
    x = ex + max(0.5, 2.2 - severity) * math.cos(eh)
    y = ey + max(0.0, 0.3 * (1.0 - severity))
    actor = _actor_template(f"stress_contact_proxy_{root.get('root_id','root')}", x, y, eh, 0.25 * ev * math.cos(eh), 0.25 * ev * math.sin(eh))
    _ensure_actors(root).append(actor)
    _append_history_actor(root, actor, visible_history=True)
    root.setdefault("traffic_config", {})["stress_type"] = "contact_proxy"
    root["traffic_config"]["post_contact_stabilization_required"] = True
    _set_regime(root, "contact_post_contact")
    return "contact_proxy"


def _apply_friction_delay(root: dict, severity: float) -> str:
    root.setdefault("traffic_config", {})["stress_type"] = "friction_delay"
    root["traffic_config"]["friction_scale"] = float(max(0.35, 1.0 - 0.55 * severity))
    root["traffic_config"]["actuation_delay_s"] = float(0.1 + 0.35 * severity)
    _set_regime(root, "low_headroom" if severity < 0.65 else "near_contact")
    return "friction_delay"


STRESS_FNS = [_apply_lead_brake, _apply_cut_in, _apply_occluded_release, _apply_contact_proxy, _apply_friction_delay]


def main():
    ap = argparse.ArgumentParser(description="Create hybrid WOMD/ScenarioNet stress roots compatible with OC-RAP teacher generation.")
    ap.add_argument("--input-root-dir", required=True)
    ap.add_argument("--output-root-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-roots", type=int, default=1000)
    ap.add_argument("--copies-per-root", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stress-types", default="all", help="Comma list: lead_brake,cut_in,occluded_release,contact_proxy,friction_delay, or all")
    args = ap.parse_args()
    random.seed(args.seed)
    in_dir = Path(args.input_root_dir)
    out_dir = Path(args.output_root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = _load_roots(in_dir, args.split)
    if not roots:
        raise FileNotFoundError(f"no roots found in {in_dir} split={args.split}")
    selected = [random.choice(roots) for _ in range(int(args.num_roots))]
    allowed = {"lead_brake": _apply_lead_brake, "cut_in": _apply_cut_in, "occluded_release": _apply_occluded_release, "contact_proxy": _apply_contact_proxy, "friction_delay": _apply_friction_delay}
    if args.stress_types.strip().lower() == "all":
        fns = list(allowed.values())
    else:
        fns = [allowed[x.strip()] for x in args.stress_types.split(",") if x.strip()]
    ids = []
    counts = Counter()
    for i, p in enumerate(selected):
        base = json.loads(p.read_text())
        for c in range(int(args.copies_per_root)):
            root = copy.deepcopy(base)
            severity = random.random()
            fn = random.choice(fns)
            stress_type = fn(root, severity)
            old_id = str(root.get("root_id", p.stem))
            new_id = f"hybrid_{old_id}_{stress_type}_{i:06d}_{c:02d}"
            root["root_id"] = new_id
            root["seed"] = int((int(root.get("seed", 0)) + 1000003 * (i + 1) + 9176 * (c + 1)) % (2**31 - 1))
            root.setdefault("scenario_data", {})["hybrid_source_root_id"] = old_id
            root["scenario_data"]["hybrid_womd_stress"] = True
            root["scenario_data"]["hybrid_stress_type"] = stress_type
            root["scenario_data"]["hybrid_stress_severity"] = float(severity)
            root["scenario_data"]["hybrid_stress_requires_simulator_injection"] = True
            root["scenario_data"]["paper_final_ready"] = False
            root.setdefault("map_config", {})["source"] = root.get("map_config", {}).get("source", "scenarionet")
            (out_dir / f"{new_id}.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
            ids.append(new_id)
            counts[stress_type] += 1
    # deterministic split: 80/10/10
    random.Random(args.seed).shuffle(ids)
    n = len(ids)
    splits = {"train": ids[: int(0.8*n)], "calib": ids[int(0.8*n): int(0.9*n)], "test": ids[int(0.9*n):], "all": ids}
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    metadata = {
        "is_synthetic": False,
        "is_hybrid_womd_stress": True,
        "source_root_dir": str(in_dir),
        "source_split": args.split,
        "num_roots": len(ids),
        "stress_counts": dict(counts),
        "backend": "metadrive_scenarionet_hybrid_stress",
        "paper_final_ready": False,
        "hybrid_stress_requires_simulator_injection": True,
        "paper_final_note": "Hybrid stress roots are JSON-level scenario perturbations. For paper-final MetaDrive rollouts, write them into ScenarioNet files or spawn/control stress actors during rollout; report separately from natural WOMD/ScenarioNet roots.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "num_roots": len(ids), "stress_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
