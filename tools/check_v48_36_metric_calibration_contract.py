#!/usr/bin/env python3
"""Fail-closed identity check for the v48.36 OCAF one-rule protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _same(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_metric_row(train_summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    """Resolve the validation row without fabricating a training epoch.

    A normal trained stage is self-contained.  v48.38+ reserve-only stages are
    deliberately materialized with zero optimizer steps and an empty history;
    in that exact case the audited metric source is the byte-pinned factor-stage
    train summary.  Anything else fails closed.
    """
    summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    best_epoch = int(summary["best_epoch"])
    best = next(
        (x for x in summary.get("history", []) if int(x.get("epoch", -1)) == best_epoch),
        None,
    )
    if best is not None:
        return summary, best, best_epoch, {
            "kind": "stage_training_history",
            "stage_train_summary": str(train_summary_path),
            "stage_train_summary_sha256": _sha256(train_summary_path),
            "materialized_without_training": False,
        }

    no_training_contract = (
        summary.get("materialized_without_training") is True
        and summary.get("parameter_update_performed") is False
        and int(summary.get("epochs_completed", -1)) == 0
        and int(summary.get("total_train_steps", -1)) == 0
        and summary.get("history") == []
        and summary.get("factor_checkpoint_reused_without_parameter_update") is True
    )
    if not no_training_contract:
        raise SystemExit(f"best epoch {best_epoch} not found")

    source_raw = str(summary.get("metric_source_train_summary") or "").strip()
    source_checkpoint_raw = str(summary.get("source_factor_checkpoint") or "").strip()
    expected_summary_sha = str(summary.get("metric_source_train_summary_sha256") or "").strip()
    expected_metric_ckpt_sha = str(summary.get("metric_source_checkpoint_sha256") or "").strip()
    source_factor_ckpt_sha = str(summary.get("source_factor_checkpoint_sha256") or "").strip()
    if not source_raw or not source_checkpoint_raw or not expected_summary_sha or not expected_metric_ckpt_sha:
        raise SystemExit("reserve-only metric source provenance is incomplete")
    if expected_metric_ckpt_sha != source_factor_ckpt_sha:
        raise SystemExit("reserve-only metric source checkpoint hash mismatch")

    source_path = Path(source_raw)
    if not source_path.is_absolute():
        source_path = (train_summary_path.parent / source_path).resolve()
    if not source_path.is_file():
        raise SystemExit(f"metric source train summary missing: {source_path}")
    source_checkpoint = Path(source_checkpoint_raw)
    if not source_checkpoint.is_absolute():
        source_checkpoint = (train_summary_path.parent / source_checkpoint).resolve()
    if not source_checkpoint.is_file():
        raise SystemExit(f"metric source checkpoint missing: {source_checkpoint}")
    if source_path.parent != source_checkpoint.parent:
        raise SystemExit("metric source summary/checkpoint are not from the same model directory")
    if _sha256(source_checkpoint) != source_factor_ckpt_sha:
        raise SystemExit("source factor checkpoint SHA256 mismatch")
    actual_summary_sha = _sha256(source_path)
    if actual_summary_sha != expected_summary_sha:
        raise SystemExit("metric source train summary SHA256 mismatch")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_best_epoch = int(source["best_epoch"])
    declared_source_best = summary.get("source_factor_best_epoch")
    if declared_source_best is not None and int(declared_source_best) != source_best_epoch:
        raise SystemExit("materialized/source best epoch mismatch")
    source_best = next(
        (x for x in source.get("history", []) if int(x.get("epoch", -1)) == source_best_epoch),
        None,
    )
    if source_best is None:
        raise SystemExit(f"source best epoch {source_best_epoch} not found")
    return source, source_best, source_best_epoch, {
        "kind": "factor_training_history_for_zero_update_stage",
        "stage_train_summary": str(train_summary_path),
        "stage_train_summary_sha256": _sha256(train_summary_path),
        "metric_source_train_summary": str(source_path),
        "metric_source_train_summary_sha256": actual_summary_sha,
        "metric_source_checkpoint_sha256": expected_metric_ckpt_sha,
        "materialized_without_training": True,
        "stage_epochs_completed": 0,
        "stage_optimizer_steps": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summary", type=Path, required=True)
    ap.add_argument("--near-dev", type=Path, required=True)
    ap.add_argument("--contact-dev", type=Path, required=True)
    ap.add_argument("--shared-rule", type=Path, required=True)
    ap.add_argument("--gate-spec", type=Path, required=True)
    ap.add_argument("--policy-contract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    summary, best, best_epoch, metric_provenance = _load_metric_row(args.train_summary)
    val = best.get("val") or {}
    dev = {
        "near": json.loads(args.near_dev.read_text(encoding="utf-8")),
        "contact": json.loads(args.contact_dev.read_text(encoding="utf-8")),
    }
    shared = json.loads(args.shared_rule.read_text(encoding="utf-8"))
    gate_doc = json.loads(args.gate_spec.read_text(encoding="utf-8"))
    protocol = gate_doc.get("protocol") or {}
    policy = _env(args.policy_contract)

    failures: list[str] = []
    checks: dict[str, Any] = {}
    expected_order = "rank_topk_then_filter_then_evidence_rerank"
    expected_topk = int(policy.get("PROPOSAL_TOP_K", "-1"))
    checks["single_shared_rule"] = (
        int(shared.get("shared_rule_count", 0)) == 1
        and shared.get("strategy_regime_conditioning") is False
        and sorted(shared.get("audit_strata_only") or []) == ["contact", "near"]
    )
    selected_rule = shared.get("rule") or shared.get("diagnostic_fit_rule") or {}
    semantic_domain = ((shared.get("constraints") or {}).get("semantic_rule_domain") or {})
    checks["noncompensatory_semantic_rule_domain"] = (
        float(selected_rule.get("opportunity_threshold", -1.0)) >= float(semantic_domain.get("min_opportunity_threshold", 0.5))
        and float(selected_rule.get("harm_threshold", 2.0)) <= float(semantic_domain.get("max_harm_threshold", 0.5))
        and float(selected_rule.get("score_threshold", -1.0)) >= float(semantic_domain.get("min_score_threshold", 0.0))
    )
    checks["selection_semantics"] = (
        policy.get("SELECTION_SEMANTICS") == expected_order
        and ((protocol.get("policy") or {}).get("selection_semantics")) == expected_order
    )
    checks["proposal_top_k"] = (
        expected_topk > 0
        and int((protocol.get("policy") or {}).get("proposal_top_k", -1)) == expected_topk
        and all(int(d.get("proposal_top_k", -1)) == expected_topk for d in dev.values())
    )
    source_meta = shared.get("sources") or {}
    for stratum, path in (("near", args.near_dev), ("contact", args.contact_dev)):
        # shared fitter consumes proposal rows, not the dev summary. Verify its
        # declared stratum population against the dev certificate population.
        train_groups = int(round(float(val.get(f"direct_group_count_{stratum}", -1))))
        dev_groups = int(dev[stratum].get("num_groups", -2))
        source_groups = int((source_meta.get(stratum) or {}).get("group_count", -3))
        safe_train = int(round(float(val.get(f"direct_safe_opportunity_group_count_{stratum}", -1))))
        safe_dev = int((((dev[stratum].get("proposal_constrained_oracle_gate") or {}).get("fit") or {}).get("proposal_safe_positive_groups", -2)))
        fit_spec = ((protocol.get(stratum) or {}).get("fit") or {})
        shared_constraints = shared.get("constraints") or {}
        threshold_checks = {
            "min_selected": int((shared_constraints.get("min_selected") or {}).get(stratum, -1)) == int(fit_spec.get("min_selected", -2)),
            "min_precision_lcb": _same((shared_constraints.get("min_precision_lcb") or {}).get(stratum), fit_spec.get("min_precision_lcb")),
            "max_harmful_group_ucb": _same((shared_constraints.get("max_harmful_group_ucb") or {}).get(stratum), fit_spec.get("max_harmful_group_ucb")),
            "max_harmful_selected_ucb": _same((shared_constraints.get("max_harmful_selected_ucb") or {}).get(stratum), fit_spec.get("max_harmful_selected_ucb")),
        }
        checks[stratum] = {
            "train_groups": train_groups,
            "dev_groups": dev_groups,
            "shared_source_groups": source_groups,
            "train_safe_opportunity_groups": safe_train,
            "dev_safe_opportunity_groups": safe_dev,
            "group_identity": train_groups == dev_groups == source_groups,
            "safe_opportunity_identity": safe_train == safe_dev,
            "fit_threshold_identity": threshold_checks,
            "proposal_oracle_fit_feasible": bool((((dev[stratum].get("proposal_constrained_oracle_gate") or {}).get("fit") or {}).get("feasible", False))),
        }
    for name, ok in checks.items():
        if isinstance(ok, bool) and not ok:
            failures.append(name)
    for stratum in ("near", "contact"):
        row = checks[stratum]
        if not row["group_identity"]:
            failures.append(f"{stratum}:group_identity")
        if not row["safe_opportunity_identity"]:
            failures.append(f"{stratum}:safe_opportunity_identity")
        if not all(row["fit_threshold_identity"].values()):
            failures.append(f"{stratum}:fit_threshold_identity")
        if not row["proposal_oracle_fit_feasible"]:
            failures.append(f"{stratum}:proposal_oracle_fit_infeasible")

    doc = {
        "version": "v48.36-OCAF",
        "valid": not failures,
        "best_epoch": best_epoch,
        "best_metric": summary.get("best_metric"),
        "metric_summary_provenance": metric_provenance,
        "gate_spec_sha256": hashlib.sha256(args.gate_spec.read_bytes()).hexdigest(),
        "shared_rule_sha256": hashlib.sha256(args.shared_rule.read_bytes()).hexdigest(),
        "single_deployment_rule": True,
        "strategy_regime_conditioning": False,
        "checks": checks,
        "failure_reasons": failures,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
