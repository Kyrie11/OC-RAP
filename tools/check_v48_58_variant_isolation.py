#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    out: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k=k.strip(); v=v.strip()
        if k in out:
            duplicates.setdefault(k, [out[k]]).append(v)
        out[k] = v
    return out, duplicates


def _resolved(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="v48.58 fail-closed balanced/precision provenance isolation")
    ap.add_argument("--reference-run", type=Path, required=True)
    ap.add_argument("--reference-contract", type=Path, required=True)
    ap.add_argument("--native-run", type=Path, required=True)
    ap.add_argument("--learned-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    reference_contract = _load_json(args.reference_contract)
    if not bool(reference_contract.get("valid")):
        errors.append("reference reuse contract is not valid")
    if _resolved(reference_contract.get("reference_run", "")) != _resolved(args.reference_run):
        errors.append("reference reuse contract points to a different reference run")
    reference_snapshot = reference_contract.get("reference_candidate_checkpoint_sha256") or {}
    variants: dict[str, dict] = {}
    learned_checkpoint_paths: list[str] = []
    for variant in ("balanced", "precision"):
        ref_dir = args.reference_run / "candidates" / variant
        native_dir = args.native_run / "candidates" / variant
        learned_dir = args.learned_run / "candidates" / variant
        ref_ckpt = ref_dir / "model_v48_trac_sr" / "best.pt"
        native_ckpt = native_dir / "model_v48_trac_sr" / "best.pt"
        learned_ckpt = learned_dir / "model_v48_trac_sr" / "best.pt"
        learned_summary_path = learned_dir / "model_v48_trac_sr" / "train_summary.json"
        state_path = learned_dir / "V48_58_STAGE_I_STATE_ISOLATION.json"
        native_policy_path = native_dir / "POLICY_CONTRACT.env"
        learned_policy_path = learned_dir / "POLICY_CONTRACT.env"

        required = [ref_ckpt, native_ckpt, learned_ckpt, learned_summary_path, state_path, native_policy_path, learned_policy_path]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            errors.append(f"{variant}: missing required artifacts: {missing}")
            variants[variant] = {"valid": False, "missing": missing}
            continue

        ref_sha = _sha256(ref_ckpt)
        native_sha = _sha256(native_ckpt)
        summary = _load_json(learned_summary_path)
        state = _load_json(state_path)
        native_policy, native_duplicates = _env(native_policy_path)
        learned_policy, learned_duplicates = _env(learned_policy_path)

        expected_ref = _resolved(ref_ckpt)
        expected_learned = _resolved(learned_ckpt)
        init_ok = _resolved(summary.get("init_checkpoint", "")) == expected_ref
        checkpoint_ok = _resolved(summary.get("checkpoint", "")) == expected_learned
        trainable = list(summary.get("trainable_param_prefixes") or [])
        trainable_ok = trainable == ["direct_absolute_feasibility_head"]
        native_copy_ok = ref_sha == native_sha
        reference_snapshot_ok = str(reference_snapshot.get(variant, "")) == ref_sha
        state_ok = bool(state.get("valid")) and bool(state.get("stage_i_bitwise_identity"))
        state_ref_ok = _resolved(state.get("reference", "")) == expected_ref
        state_adapted_ok = _resolved(state.get("adapted", "")) == expected_learned
        expected_order = "rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank"
        critical_keys={"ABSOLUTE_FEASIBILITY_MODE","ABSOLUTE_FEASIBILITY_THRESHOLD","SELECTION_SEMANTICS"}
        native_duplicate_critical=sorted(k for k in native_duplicates if k in critical_keys)
        learned_duplicate_critical=sorted(k for k in learned_duplicates if k in critical_keys)
        native_policy_ok = (
            native_policy.get("ABSOLUTE_FEASIBILITY_MODE") == "native"
            and native_policy.get("ABSOLUTE_FEASIBILITY_THRESHOLD") == "0.5"
            and native_policy.get("SELECTION_SEMANTICS") == expected_order
            and not native_duplicate_critical
        )
        learned_policy_ok = (
            learned_policy.get("ABSOLUTE_FEASIBILITY_MODE") == "learned"
            and learned_policy.get("ABSOLUTE_FEASIBILITY_THRESHOLD") == "0.5"
            and learned_policy.get("SELECTION_SEMANTICS") == expected_order
            and not learned_duplicate_critical
        )
        valid = all((init_ok, checkpoint_ok, trainable_ok, native_copy_ok, reference_snapshot_ok, state_ok,
                     state_ref_ok, state_adapted_ok, native_policy_ok, learned_policy_ok))
        if not valid:
            errors.append(f"{variant}: provenance/isolation contract failed")
        learned_checkpoint_paths.append(expected_learned)
        variants[variant] = {
            "valid": valid,
            "reference_checkpoint": expected_ref,
            "reference_sha256": ref_sha,
            "native_checkpoint": _resolved(native_ckpt),
            "native_sha256": native_sha,
            "native_is_bitwise_reference": native_copy_ok,
            "reference_snapshot_sha256": reference_snapshot.get(variant),
            "reference_snapshot_matches_current": reference_snapshot_ok,
            "learned_checkpoint": expected_learned,
            "learned_init_checkpoint": _resolved(summary.get("init_checkpoint", "")),
            "learned_init_matches_same_variant_reference": init_ok,
            "learned_checkpoint_path_matches_variant": checkpoint_ok,
            "trainable_param_prefixes": trainable,
            "trainable_prefix_contract_valid": trainable_ok,
            "stage_i_state_isolation_valid": state_ok,
            "state_reference_matches_variant": state_ref_ok,
            "state_adapted_matches_variant": state_adapted_ok,
            "native_policy_contract_valid": native_policy_ok,
            "learned_policy_contract_valid": learned_policy_ok,
            "native_policy_duplicate_critical_keys": native_duplicate_critical,
            "learned_policy_duplicate_critical_keys": learned_duplicate_critical,
        }

    distinct_learned_paths = len(learned_checkpoint_paths) == 2 and len(set(learned_checkpoint_paths)) == 2
    if not distinct_learned_paths:
        errors.append("balanced/precision learned checkpoint paths are not distinct")
    valid = not errors and all(v.get("valid", False) for v in variants.values()) and distinct_learned_paths
    doc = {
        "schema": "ocrap-v48.58-parallel-variant-isolation-v1",
        "valid": valid,
        "reference_run": _resolved(args.reference_run),
        "native_run": _resolved(args.native_run),
        "learned_run": _resolved(args.learned_run),
        "distinct_learned_checkpoint_paths": distinct_learned_paths,
        "variants": variants,
        "errors": errors,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "v48_58_parallel_variant_isolation", "valid": valid, "output": str(args.output)}))
    return 0 if valid else 30


if __name__ == "__main__":
    raise SystemExit(main())
