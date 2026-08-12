#!/usr/bin/env python3
"""Validate and seal the deterministic v48.45 dedicated calibration protocol.

This is an engineering/provenance contract only.  It verifies that the SOWR/OCAF
calibration protocol is derived *only* from calibration_near_contact/contact plus
calibration_safe, with the preregistered v48.14 scene-disjoint partition rule.
No test_* root is accepted or read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROLE_LEAVES = {
    "evidence_adapt_train_near_contact": ("near_contact", "evidence_adapt_train"),
    "evidence_adapt_dev_near_contact": ("near_contact", "evidence_adapt_dev"),
    "certificate_pool_near_contact": ("near_contact", "certificate_pool"),
    "evidence_adapt_train_contact": ("contact", "evidence_adapt_train"),
    "evidence_adapt_dev_contact": ("contact", "evidence_adapt_dev"),
    "certificate_pool_contact": ("contact", "certificate_pool"),
}


def atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score(scene: str, seed: int) -> float:
    h = hashlib.sha256(f"v48.14|{seed}|{scene}".encode("utf-8", errors="replace")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def read_manifest(root: Path, *, require_protocol_role: str | None = None,
                  require_source_split: str | None = None, check_files: bool = True) -> dict:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")

    canonical_scenes: set[str] = set()
    scene_ids: set[str] = set()
    original_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    paths: list[str] = []
    missing_files: list[str] = []
    duplicate_paths: list[str] = []
    seen_paths: set[str] = set()
    for row in rows:
        original = str(row.get("original_scenario_id") or "").strip()
        scene = str(row.get("scene_id") or "").strip()
        canonical = original or scene
        if not canonical:
            raise ValueError(f"manifest row missing original_scenario_id/scene_id: {manifest}")
        canonical_scenes.add(canonical)
        if scene:
            scene_ids.add(scene)
        if original:
            original_ids.add(original)
        split = str(row.get("split_id") or "").strip()
        role = str(row.get("calibration_protocol_role") or "").strip()
        split_counts[split] += 1
        if role:
            protocol_counts[role] += 1
        raw = str(row.get("path") or "").strip()
        if not raw:
            raise ValueError(f"manifest row missing path: {manifest}")
        p = Path(raw)
        candidates = [p] if p.is_absolute() else [root / p, root / "samples" / p.name]
        resolved = next((q for q in candidates if q.is_file()), candidates[0])
        normalized = str(resolved.resolve(strict=False))
        if normalized in seen_paths:
            duplicate_paths.append(normalized)
        seen_paths.add(normalized)
        paths.append(normalized)
        if check_files and not resolved.is_file():
            missing_files.append(str(resolved))

    if require_source_split is not None:
        bad = {k: v for k, v in split_counts.items() if k != require_source_split}
        if bad:
            raise ValueError(
                f"unexpected split_id in source {root}; expected only {require_source_split}: {dict(split_counts)}"
            )
    if require_protocol_role is not None:
        bad_split = {k: v for k, v in split_counts.items() if k != require_protocol_role}
        bad_role = {k: v for k, v in protocol_counts.items() if k != require_protocol_role}
        if bad_split or bad_role or not protocol_counts:
            raise ValueError(
                f"bad protocol role in {root}: expected={require_protocol_role} "
                f"split={dict(split_counts)} role={dict(protocol_counts)}"
            )
    if duplicate_paths:
        raise ValueError(f"duplicate sample paths in {manifest}: first={duplicate_paths[:3]}")
    if missing_files:
        raise FileNotFoundError(
            f"missing sample files referenced by {manifest}: count={len(missing_files)} first={missing_files[:3]}"
        )
    return {
        "root": str(root.resolve(strict=False)),
        "manifest": str(manifest.resolve(strict=False)),
        "manifest_sha256": sha256_file(manifest),
        "num_samples": len(rows),
        "canonical_scenes": canonical_scenes,
        "scene_ids": scene_ids,
        "original_ids": original_ids,
        "num_scenes": len(canonical_scenes),
        "split_counts": dict(split_counts),
        "protocol_role_counts": dict(protocol_counts),
        "sample_paths_checked": len(paths) if check_files else 0,
    }


def expected_roles(scenes: Iterable[str], *, seed: int, train_fraction: float, dev_fraction: float) -> dict[str, set[str]]:
    roles = {"evidence_adapt_train": set(), "evidence_adapt_dev": set(), "certificate_pool": set()}
    for scene in scenes:
        x = score(scene, seed)
        if x < train_fraction:
            roles["evidence_adapt_train"].add(scene)
        elif x < train_fraction + dev_fraction:
            roles["evidence_adapt_dev"].add(scene)
        else:
            roles["certificate_pool"].add(scene)
    return roles


def stripped(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in {"canonical_scenes", "scene_ids", "original_ids"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--near-source", type=Path, required=True)
    ap.add_argument("--contact-source", type=Path, required=True)
    ap.add_argument("--safe-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=4814)
    ap.add_argument("--adapt-train-fraction", type=float, default=0.45)
    ap.add_argument("--adapt-dev-fraction", type=float, default=0.15)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-sample-file-check", action="store_true")
    args = ap.parse_args()

    checks: dict[str, bool] = {}
    errors: list[str] = []
    details: dict[str, object] = {}
    protocol_root = args.protocol_root.resolve(strict=False)
    near_source = args.near_source.resolve(strict=False)
    contact_source = args.contact_source.resolve(strict=False)
    safe_root = args.safe_root.resolve(strict=False)
    check_files = not args.skip_sample_file_check

    checks["fractions_valid"] = (
        0 < args.adapt_train_fraction < 1
        and 0 < args.adapt_dev_fraction < 1
        and args.adapt_train_fraction + args.adapt_dev_fraction < 1
    )
    checks["canonical_protocol_leaf"] = protocol_root.name == "calibration_v48_14_prism_4814"
    checks["no_test_root_argument"] = all("test" not in p.name.lower() for p in (near_source, contact_source, safe_root))
    checks["near_contact_contact_distinct"] = near_source != contact_source

    sources: dict[str, dict] = {}
    roles: dict[str, dict] = {}
    try:
        sources["near_contact"] = read_manifest(
            near_source, require_source_split="calibration", check_files=check_files
        )
        sources["contact"] = read_manifest(
            contact_source, require_source_split="calibration", check_files=check_files
        )
        sources["safe"] = read_manifest(
            safe_root, require_source_split="calibration", check_files=check_files
        )
        checks["source_manifests_valid"] = True
    except Exception as exc:
        checks["source_manifests_valid"] = False
        errors.append(f"source_manifest: {type(exc).__name__}: {exc}")

    complete_path = protocol_root / "CALIBRATION_PROTOCOL_COMPLETE.json"
    complete: dict = {}
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        checks["protocol_complete_marker"] = complete.get("event") == "v48_14_scene_disjoint_dedicated_protocol_complete"
        checks["protocol_seed_matches"] = int(complete.get("seed")) == args.seed
        checks["protocol_marker_no_test_read"] = complete.get("test_roots_read") is False
    except Exception as exc:
        checks["protocol_complete_marker"] = False
        checks["protocol_seed_matches"] = False
        checks["protocol_marker_no_test_read"] = False
        errors.append(f"protocol_complete: {type(exc).__name__}: {exc}")

    for leaf, (_regime, expected_role) in ROLE_LEAVES.items():
        try:
            roles[leaf] = read_manifest(
                protocol_root / leaf, require_protocol_role=expected_role, check_files=check_files
            )
        except Exception as exc:
            errors.append(f"{leaf}: {type(exc).__name__}: {exc}")
    checks["all_six_roles_valid"] = len(roles) == len(ROLE_LEAVES)

    if checks.get("source_manifests_valid") and checks.get("all_six_roles_valid"):
        for regime, source_key in (("near_contact", "near_contact"), ("contact", "contact")):
            expected = expected_roles(
                sources[source_key]["canonical_scenes"],
                seed=args.seed,
                train_fraction=args.adapt_train_fraction,
                dev_fraction=args.adapt_dev_fraction,
            )
            actual = {
                role: roles[f"{role}_{regime}"]["canonical_scenes"]
                for role in ("evidence_adapt_train", "evidence_adapt_dev", "certificate_pool")
            }
            checks[f"{regime}_roles_nonempty"] = all(bool(v) for v in actual.values())
            checks[f"{regime}_scene_disjoint"] = (
                not (actual["evidence_adapt_train"] & actual["evidence_adapt_dev"])
                and not (actual["evidence_adapt_train"] & actual["certificate_pool"])
                and not (actual["evidence_adapt_dev"] & actual["certificate_pool"])
            )
            checks[f"{regime}_scene_union_exact"] = set().union(*actual.values()) == sources[source_key]["canonical_scenes"]
            checks[f"{regime}_deterministic_assignment_exact"] = all(actual[k] == expected[k] for k in actual)
            details[f"{regime}_expected_scene_counts"] = {k: len(v) for k, v in expected.items()}
            details[f"{regime}_actual_scene_counts"] = {k: len(v) for k, v in actual.items()}

    # Provenance files are secondary evidence: validate source/role/seed/fractions.
    prov_ok = True
    provenance: dict[str, object] = {}
    for leaf, (regime, expected_role) in ROLE_LEAVES.items():
        p = protocol_root / leaf / "split_provenance.json"
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            source_expected = near_source if regime == "near_contact" else contact_source
            frac = doc.get("fractions") or {}
            role_ok = (
                Path(str(doc.get("source", ""))).resolve(strict=False) == source_expected
                and doc.get("role") == expected_role
                and int(doc.get("seed")) == args.seed
                and abs(float(frac.get("evidence_adapt_train")) - args.adapt_train_fraction) < 1e-12
                and abs(float(frac.get("evidence_adapt_dev")) - args.adapt_dev_fraction) < 1e-12
                and abs(float(frac.get("certificate_pool")) - (1.0 - args.adapt_train_fraction - args.adapt_dev_fraction)) < 1e-12
                and doc.get("scene_disjoint_by_construction") is True
                and doc.get("test_roots_read") is False
            )
            prov_ok = prov_ok and role_ok
            provenance[leaf] = {
                "valid": role_ok,
                "source": doc.get("source"),
                "role": doc.get("role"),
                "seed": doc.get("seed"),
                "num_scenes": doc.get("num_scenes"),
                "num_samples": doc.get("num_samples"),
            }
        except Exception as exc:
            prov_ok = False
            provenance[leaf] = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    checks["split_provenance_valid"] = prov_ok

    valid = all(checks.values()) and not errors
    doc = {
        "event": "v48_45_dedicated_protocol_seal",
        "version": "v48.45.5-engineering-protocol-bootstrap",
        "created_unix": time.time(),
        "valid": valid,
        "protocol_root": str(protocol_root),
        "inputs": {
            "near_source": str(near_source),
            "contact_source": str(contact_source),
            "safe_root": str(safe_root),
            "seed": args.seed,
            "adapt_train_fraction": args.adapt_train_fraction,
            "adapt_dev_fraction": args.adapt_dev_fraction,
            "certificate_fraction": 1.0 - args.adapt_train_fraction - args.adapt_dev_fraction,
        },
        "checks": checks,
        "errors": errors,
        "sources": {k: stripped(v) for k, v in sources.items()},
        "roles": {k: stripped(v) for k, v in roles.items()},
        "details": details,
        "provenance": provenance,
        "test_roots_read": False,
    }
    atomic(args.output, doc)
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
