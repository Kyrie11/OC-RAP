#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.fixed_main_stability import contact_construct_validity_gate


def load(path: Path) -> dict[str, Any]:
    v = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v, dict):
        raise ValueError(f"expected object: {path}")
    return v


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, name: str) -> Path:
    p = root / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def verify_manifest(root: Path) -> dict[str, Any]:
    p = resolve(root, "OC-RAP-v48.124-OC-FMSA-result-bundle-manifest.json")
    doc = load(p)
    mismatches: list[str] = []
    checked = 0
    # Support both mapping- and list-shaped retained manifests.
    records = doc.get("artifacts") or doc.get("members") or doc.get("files") or []
    if isinstance(records, dict):
        it = [(name, rec) for name, rec in records.items()]
    elif isinstance(records, list):
        it = [(None, rec) for rec in records]
    else:
        it = []
    for map_name, rec in it:
        if not isinstance(rec, dict):
            continue
        raw = rec.get("bundle_name") or rec.get("name") or rec.get("path") or map_name
        want_sha = rec.get("sha256")
        want_size = rec.get("size")
        if not raw or not want_sha:
            continue
        q = root / Path(str(raw)).name
        if not q.is_file():
            mismatches.append(f"missing:{q.name}")
            continue
        checked += 1
        if sha(q) != str(want_sha):
            mismatches.append(f"sha:{q.name}")
        if want_size is not None and q.stat().st_size != int(want_size):
            mismatches.append(f"size:{q.name}")
    return {
        "manifest_path": str(p),
        "manifest_declared_valid": bool(doc.get("valid", True)),
        "num_payloads_checked": checked,
        "mismatches": mismatches,
        "go": bool(doc.get("valid", True) and not mismatches),
    }


def exact_a0_gate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    go = True
    for regime, d in results.items():
        rate = float(d.get("intervention_rate", float("nan")))
        reasons = d.get("selection_reason_counts") or {}
        scene_rates = []
        for sc in d.get("scenes") or []:
            try:
                scene_rates.append(float(sc.get("intervention_rate", 0.0)))
            except Exception:
                scene_rates.append(float("nan"))
        ok = rate == rate and abs(rate) <= 1e-12 and set(reasons) == {"nominal_prefix_exact_a0"} and all(x == x and abs(x) <= 1e-12 for x in scene_rates)
        rows[regime] = {"go": ok, "intervention_rate": rate, "selection_reason_counts": reasons}
        go = go and ok
    return {"go": go, "regimes": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-audit V48.124.5 without treating a post-treatment Contact subset as a valid post-contact cohort.")
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = a.bundle_dir.resolve()

    nominal = {r: load(resolve(root, f"OC-RAP-v48.124-nominal-{r}-closed-loop.json")) for r in ("safe", "near", "contact")}
    balanced = {r: load(resolve(root, f"OC-RAP-v48.124-balanced-{r}-closed-loop.json")) for r in ("safe", "near", "contact")}
    precision = {r: load(resolve(root, f"OC-RAP-v48.124-precision-{r}-closed-loop.json")) for r in ("safe", "near", "contact")}
    original = load(resolve(root, "OC-RAP-v48.124-fixed-main-adjudication.json"))
    pipeline = load(resolve(root, "OC-RAP-v48.124-PIPELINE_COMPLETE.json"))
    runtime = load(resolve(root, "OC-RAP-v48.124-full-population-runtime-code-contract.json"))

    manifest = verify_manifest(root)
    exact = exact_a0_gate(nominal)
    construct = contact_construct_validity_gate({
        "nominal": nominal["contact"],
        "balanced": balanced["contact"],
        "precision": precision["contact"],
    })
    d = original.get("preregistered_decision") or {}

    # Safe/Near gates remain interpretable because their endpoints do not
    # condition on a treatment-induced event. Contact does not.
    corrected = {
        "coverage": bool((d.get("coverage_gate") or {}).get("go")),
        "determinism": bool((d.get("determinism_gate") or {}).get("go")),
        "safe": bool((d.get("safe_noninterference_gate") or {}).get("go")),
        "near": bool((d.get("near_closed_loop_validity_gate") or {}).get("go")),
        "contact_construct_validity": bool(construct.get("go")),
        "contact_algorithm_gate_entered": False if not construct.get("go") else True,
    }
    reliable = bool(
        manifest["go"]
        and runtime.get("valid") and runtime.get("attribution_ready")
        and pipeline.get("valid") and pipeline.get("attribution_ready")
        and exact["go"]
    )
    full_closed = bool(reliable and all(corrected[k] for k in ("coverage", "determinism", "safe", "near", "contact_construct_validity")))
    out = {
        "schema": "ocrap-v48.124.5-independent-scientific-validity-reaudit-v1",
        "input_engineering_version": runtime.get("engineering_version"),
        "scientific_version": runtime.get("scientific_version"),
        "artifact_provenance_reliability": manifest,
        "runtime_valid": bool(runtime.get("valid") and runtime.get("attribution_ready")),
        "pipeline_valid": bool(pipeline.get("valid") and pipeline.get("attribution_ready")),
        "exact_a0_control": exact,
        "original_formal_status": d.get("status"),
        "original_formal_go": d.get("go"),
        "corrected_scientific_gates": corrected,
        "contact_construct_validity_gate": construct,
        "reliability_for_safe_near_attribution": reliable,
        "five_gate_scientific_adjudication_closed": full_closed,
        "external_baseline_ready": False,
        "deployed_main_freeze_authorized": False,
        "algorithm_or_mechanism_change_authorized": False,
        "next_branch": "keep_theory_and_mechanism_search_frozen; fix_contact_evaluation_construct; preserve_safe_and_near_evidence; then re-enter Contact adjudication",
        "interpretation": {
            "safe": "GO is attributable under exact-a0 control.",
            "near": "STOP is attributable as a frozen-system closed-loop failure axis.",
            "contact": "NOT IDENTIFIED as an algorithm STOP: current post-contact endpoints condition on a policy-dependent observed-contact subset rather than a common pre-treatment Contact anchor.",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "reliability_for_safe_near_attribution": out["reliability_for_safe_near_attribution"],
        "original_formal_status": out["original_formal_status"],
        "contact_construct_validity": corrected["contact_construct_validity"],
        "external_baseline_ready": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
