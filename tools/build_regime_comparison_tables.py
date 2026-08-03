#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ocrap.external_baselines.provenance import find_provenance


SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "safe": [
        ("num_scenes", "Scenes", "count"),
        ("collision_scene_rate", "Collision scene rate ↓", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("minimum_clearance_m", "Minimum clearance [m] ↑", "float"),
        ("minimum_ttc_s", "Minimum TTC [s] ↑", "float"),
        ("closed_loop_bounded_NUP", "Bounded NUP ↑", "float"),
        ("intervention_rate", "Intervention rate", "rate"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
    "near": [
        ("num_scenes", "Scenes", "count"),
        ("collision_scene_rate", "Collision scene rate ↓", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("scene_min_clearance_m_p05", "Scene clearance p05 [m] ↑", "float"),
        ("scene_ttc_s_p05", "Scene TTC p05 [s] ↑", "float"),
        ("terminal_clearance_m", "Terminal clearance [m] ↑", "float"),
        ("terminal_ttc_s", "Terminal TTC [s] ↑", "float"),
        ("critical_ttc_exposure_duration_s", "Critical-TTC exposure [s] ↓", "float"),
        ("closed_loop_DRS", "DRS ↑", "float"),
        ("closed_loop_ODG", "ODG ↓", "float"),
        ("closed_loop_FRA_exec", "FRA-exec ↓", "rate"),
        ("closed_loop_bounded_NUP", "Bounded NUP ↑", "float"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
    "contact": [
        ("num_scenes", "Scenes", "count"),
        ("post_contact_terminal_clearance_m", "Terminal clearance [m] ↑", "float"),
        ("post_contact_free_space_auc_normalized_m", "Free-space AUC [m] ↑", "float"),
        ("post_contact_clearance_gain_m", "Clearance gain [m] ↑", "float"),
        ("post_contact_escape_scene_rate", "Escape scene rate ↑", "rate"),
        ("recontact_scene_rate", "Re-contact scene rate ↓", "rate"),
        ("secondary_overlap_scene_rate", "Secondary-overlap rate ↓", "rate"),
        ("new_stable_stop_quality_scene_rate", "Stable-stop-quality rate ↑", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("post_contact_overlap_duration_s", "Post-contact overlap [s] ↓", "float"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
}


def _scene_journal(path: Path) -> Path | None:
    candidates = [Path(str(path) + ".scenes.jsonl"), path.with_suffix(path.suffix + ".scenes.jsonl")]
    return next((x for x in candidates if x.is_file()), None)


def _scene_keys(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line); s = e.get("scene", e)
            key = str(s.get("target_key") or e.get("resume_key") or "")
            if not key:
                scene_id = str(s.get("scene_id") or ""); t = s.get("target_time_index")
                key = f"{scene_id}:t{t}" if scene_id and t is not None else scene_id
            if key:
                out.add(key)
    return out


def _finite(x: Any) -> float | int | None:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return int(v) if v.is_integer() else v
    except Exception:
        return None


def _get(doc: dict[str, Any], key: str) -> float | int | None:
    if key == "decision_latency_ms":
        timing = doc.get("timing", {}) or {}
        per = timing.get("per_decision_s", {}) or {}
        if "total" in per:
            return _finite(1000.0 * float(per["total"]))
        values = [float(v) for v in per.values() if v is not None and math.isfinite(float(v))]
        return _finite(1000.0 * sum(values)) if values else None
    if key in doc:
        return _finite(doc.get(key))
    wm = doc.get("waymax_metrics", {}) or {}
    return _finite(wm.get(key))


def _format(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    v = float(value)
    if kind == "count":
        return str(int(round(v)))
    if kind == "rate":
        return f"{v:.4f}"
    return f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build regime-specific OC-RAP/external-baseline comparison tables.")
    ap.add_argument("--regime", choices=tuple(SCHEMA), required=True)
    ap.add_argument("--input", action="append", required=True, metavar="METHOD=RESULT.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--allow-unpaired", action="store_true", help="Do not fail when scene journals exist but target sets differ.")
    args = ap.parse_args()

    entries: list[tuple[str, Path, dict[str, Any]]] = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"invalid --input {spec!r}; expected METHOD=PATH")
        method, raw = spec.split("=", 1); path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        entries.append((method.strip(), path, json.loads(path.read_text(encoding="utf-8"))))

    journals = [(m, _scene_journal(p)) for m, p, _ in entries]
    present = [(m, j) for m, j in journals if j is not None]
    missing_journals = [m for m, j in journals if j is None]
    if missing_journals and not args.allow_unpaired:
        raise SystemExit(
            "missing scene journals required for paired comparison: "
            + ", ".join(missing_journals)
        )
    paired = False
    paired_count = None
    if len(present) == len(entries) and present:
        key_sets = {m: _scene_keys(j) for m, j in present if j is not None}
        reference_method = entries[0][0]; reference = key_sets[reference_method]
        mismatch = {m: {"only_reference": sorted(reference - ks)[:10], "only_method": sorted(ks - reference)[:10]} for m, ks in key_sets.items() if ks != reference}
        if mismatch and not args.allow_unpaired:
            raise SystemExit(f"unpaired closed-loop target sets: {json.dumps(mismatch, ensure_ascii=False)}")
        paired = not mismatch
        paired_count = len(reference) if paired else None

    rows: list[dict[str, Any]] = []
    for method, path, doc in entries:
        prov = find_provenance(method)
        row: dict[str, Any] = {
            "method": method,
            "reporting_name": prov.reporting_name if prov else method,
            "implementation_kind": prov.implementation_kind if prov else ("OC-RAP" if method.lower().startswith("ocrap") else "unknown"),
            "fidelity": prov.fidelity if prov else ("proposed method" if method.lower().startswith("ocrap") else "unknown"),
            "source_result": str(path),
        }
        for key, _, _ in SCHEMA[args.regime]:
            row[key] = _get(doc, key)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.regime}_comparison"
    json_doc = {
        "schema_version": 1, "regime": args.regime, "paired_scene_set": paired,
        "paired_scene_count": paired_count, "contact_protocol": "physical post-contact metrics only; certificate metrics intentionally omitted" if args.regime == "contact" else None,
        "metrics": [{"key": k, "label": label, "kind": kind} for k, label, kind in SCHEMA[args.regime]],
        "rows": rows,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(json_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["method", "reporting_name", "implementation_kind", "fidelity"] + [x[0] for x in SCHEMA[args.regime]]
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in rows])

    headers = ["Method"] + [x[1] for x in SCHEMA[args.regime]]
    lines = ["# " + args.regime.capitalize() + " regime comparison", "", f"Paired target set: **{paired}**" + (f" ({paired_count} scenes)" if paired_count is not None else ""), ""]
    if args.regime == "contact":
        lines += ["> Contact is evaluated with post-contact physical recovery/escape metrics; FRA/DRS/ODG are intentionally excluded.", ""]
    lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        cells = [str(r["reporting_name"])] + [_format(r.get(k), kind) for k, _, kind in SCHEMA[args.regime]]
        lines.append("| " + " | ".join(cells) + " |")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "regime_comparison_table", "regime": args.regime, "methods": len(rows), "paired": paired, "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
