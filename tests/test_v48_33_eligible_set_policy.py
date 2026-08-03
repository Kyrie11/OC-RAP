from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_multigroup_eligible_policy_preflight_has_all_head_gradients(tmp_path: Path) -> None:
    output = tmp_path / "contract.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_33_multigroup_loss_contract.py"), "--output", str(output)],
        check=False,
        env=env,
    )
    doc = json.loads(output.read_text())
    assert proc.returncode == 0
    assert doc["valid"] is True
    assert doc["eligible_set_policy"] is True
    assert doc["admission_gradient_l1"] > 0
    assert doc["opportunity_gradient_l1"] > 0
    assert doc["harm_gradient_l1"] > 0


def test_checkpoint_metric_filters_before_evidence_rerank() -> None:
    source = (ROOT / "src" / "ocrap" / "cli" / "train.py").read_text()
    block = source[source.index("proposal_eligible_mask = (") : source.index("chosen_idx = recs[cert_j]", source.index("proposal_eligible_mask = ("))]
    assert "eligible_local = proposal_local[proposal_eligible_mask]" in block
    assert "eligible_evidence = evidence_all[eligible_local]" in block
    assert block.index("eligible_local =") < block.index("eligible_evidence =")
    assert "joint_gate_available" in block


def test_soft_checkpoint_policy_includes_eligibility_before_evidence() -> None:
    source = (ROOT / "src" / "ocrap" / "cli" / "train.py").read_text()
    start = source.index("proposal_log_soft_eligibility = (")
    end = source.index("policy_prob_full_soft =", start)
    block = source[start:end]
    assert "opp_delta_all[proposal_local_soft]" in block
    assert "harm_delta_all[proposal_local_soft]" in block
    assert "proposal_evidence_soft / metric_soft_temperature" in block
    assert "+ proposal_log_soft_eligibility" in block


def test_calibration_uses_preregistered_fit_thresholds() -> None:
    script = (ROOT / "scripts" / "calibrate_v48_33_certificate_pool.sh").read_text()
    assert '--min-fit-selected="$NEAR_MIN_FIT_SELECTED"' in script
    assert '--min-fit-precision-lcb="$NEAR_MIN_FIT_PRECISION_LCB"' in script
    assert '--max-fit-harmful-group-ucb="$NEAR_MAX_FIT_HARM_UCB"' in script
    assert '--min-fit-selected="$CONTACT_MIN_FIT_SELECTED"' in script
    assert '--min-fit-precision-lcb="$CONTACT_MIN_FIT_PRECISION_LCB"' in script
    assert '"$NEAR_MIN_VERIFY_SELECTED" --min-fit' not in script
    assert '"$CONTACT_MIN_VERIFY_SELECTED" --min-fit' not in script


def test_main_contract_is_unified_top5_and_skips_ineffective_stage3() -> None:
    dedicated = (ROOT / "scripts" / "run_v48_33_eligible_set_policy_dedicated.sh").read_text()
    variant = (ROOT / "scripts" / "adapt_ocrap_v48_33_eligible_set_policy_variant.sh").read_text()
    assert 'PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"' in dedicated
    assert "V4833_ENABLE_FINAL_CALIBRATION=0" in dedicated
    assert "V4833_ADAPTIVE_IDENTITY_MARGIN=0" in dedicated
    assert 'ORDINAL_EVIDENCE_ELIGIBLE_POLICY_WEIGHT="${IDENTITY_ELIGIBLE_POLICY_WEIGHT:-1.25}"' in variant
    assert "regime_id" not in variant.lower()


def test_metric_calibration_checker_rejects_fit_threshold_drift(tmp_path: Path) -> None:
    summary = {
        "best_epoch": 1,
        "best_metric": "direct_contract_safe_rank_risk",
        "history": [{"epoch": 1, "val": {
            "direct_group_count_near": 10, "direct_safe_opportunity_group_count_near": 3,
            "direct_group_count_contact": 12, "direct_safe_opportunity_group_count_contact": 4,
        }}],
    }
    gate = {"protocol_sha256": "x", "protocol": {
        "policy": {"proposal_top_k": 5, "selection_semantics": "rank_topk_then_filter_then_evidence_rerank"},
        "near": {"fit": {"min_selected": 10, "min_precision_lcb": .5, "max_harmful_group_ucb": .12, "max_harmful_selected_ucb": .22}},
        "contact": {"fit": {"min_selected": 16, "min_precision_lcb": .5, "max_harmful_group_ucb": .14, "max_harmful_selected_ucb": .22}},
    }}
    def rule(groups: int, safe: int, min_selected: int) -> dict:
        return {
            "num_groups": groups, "proposal_top_k": 5,
            "constraints": {"proposal_top_k": 5, "evidence_rerank_top_k": True,
                "min_fit_selected": min_selected, "min_fit_precision_lcb": .5,
                "max_fit_harmful_group_ucb": .12 if groups == 10 else .14,
                "max_fit_harmful_selected_ucb": .22},
            "proposal_constrained_oracle_gate": {"fit": {"proposal_top_k": 5,
                "proposal_safe_positive_groups": safe, "feasible": True}},
        }
    paths = {}
    for name, doc in {
        "summary": summary, "gate": gate,
        "near": rule(10, 3, 8),  # deliberately drifted
        "contact": rule(12, 4, 16),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        paths[name] = path
    policy = tmp_path / "policy.env"
    policy.write_text("PROPOSAL_TOP_K=5\nEVIDENCE_RERANK_TOP_K=true\n")
    output = tmp_path / "out.json"
    proc = subprocess.run([
        sys.executable, str(ROOT / "tools" / "check_v48_33_metric_calibration_contract.py"),
        "--train-summary", str(paths["summary"]), "--near-rule", str(paths["near"]),
        "--contact-rule", str(paths["contact"]), "--gate-spec", str(paths["gate"]),
        "--policy-contract", str(policy), "--output", str(output),
    ], check=False)
    doc = json.loads(output.read_text())
    assert proc.returncode == 31
    assert doc["valid"] is False
    assert any("thresholds" in reason for reason in doc["failure_reasons"])


def test_ablation_requires_valid_main_rc20() -> None:
    script = (ROOT / "scripts" / "run_v48_33_eligible_set_policy_ablations.sh").read_text()
    assert "ablations are authorized only after main RC=20" in script
    assert "pipeline_valid" in script and "gate_evaluated" in script
    assert "V4833_ENABLE_FINAL_CALIBRATION=0" in script
    assert "PROPOSAL_TOP_K=5" in script
