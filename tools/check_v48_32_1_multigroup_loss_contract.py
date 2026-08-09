#!/usr/bin/env python3
"""Fail-fast smoke test for the v48.32.1 multi-group identity loss contract."""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
import sys

# Self-contained preflight: do not depend on an ambient PYTHONPATH when the
# contract is invoked directly by tests, recovery tooling, or operators.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss


def _loss_inputs() -> tuple[dict, torch.Tensor]:
    n = 6
    admission = torch.tensor([0.0, 1.0, -1.0, 0.0, 1.0, -1.0], requires_grad=True)
    kwargs = dict(
        pred_logit=torch.zeros(n), pred_logvar=torch.zeros(n),
        pred_rank_logit=torch.tensor([0.0, 2.0, 1.0, 0.0, 2.0, 1.0]),
        pred_opportunity_logit=torch.tensor([0.0, 1.0, -1.0, 0.0, 1.0, -1.0]),
        pred_harm_logit=torch.zeros(n), pred_component_harm_logits=torch.zeros((n, 5)),
        pred_admission_logit=admission,
        teacher_r_dep=torch.tensor([0.0, 1.4, 0.2, 0.0, 1.2, 0.1]),
        teacher_r_orc=torch.tensor([0.0, 1.4, 0.2, 0.0, 1.2, 0.1]),
        teacher_q=torch.ones((n, 5, 1)),
        teacher_m_star=torch.tensor([
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            [[-1.0], [1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            [[-1.0], [1.0], [1.0], [1.0], [1.0]],
        ]),
        teacher_hard_violation=torch.zeros(n), teacher_harm_proxy=torch.zeros(n),
        root_probs=torch.full((n, 5), 0.2), root_valid=torch.ones((n, 5), dtype=torch.bool),
        option_valid=torch.ones((n, 1), dtype=torch.bool),
        scene_hash=torch.tensor([10, 10, 10, 20, 20, 20]),
        time_index=torch.zeros(n, dtype=torch.long),
        macro_type_id=torch.tensor([0, 2, 3, 0, 2, 3]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        bucket_id=torch.ones(n, dtype=torch.long), macro_ids=(2, 3), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, selective_risk_weight=0.0, selective_coverage_weight=0.0,
        policy_distill_weight=0.0, policy_regret_weight=0.0, preference_weight=0.0,
        preference_regret_weight=0.0, preference_listwise_weight=0.0, preference_gap_weight=0.0,
        preference_set_weight=0.0, preference_all_group_set_weight=0.0, delta_nll_weight=0.0,
        ordinal_evidence_independent_tails=True, ordinal_evidence_factorized_harm=True,
        ordinal_evidence_component_tail_weight=0.0, ordinal_evidence_safe_benefit_target=False,
        ordinal_evidence_group_opportunity_weight=0.0, ordinal_evidence_admission_weight=0.0,
        ordinal_evidence_batch_balanced=False, ordinal_evidence_ordered_nll_top1_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=0.0, ordinal_evidence_proposal_topk_weight=0.0,
        ordinal_evidence_proposal_topk=2, ordinal_evidence_intragroup_benefit_weight=0.0,
        ordinal_evidence_intragroup_harm_weight=0.0,
        ordinal_evidence_safe_utility_regression_weight=0.75,
        ordinal_evidence_safe_utility_listwise_weight=0.75,
        ordinal_evidence_safe_hard_negative_weight=2.0,
        ordinal_evidence_safe_hard_negative_margin=0.04,
        ordinal_evidence_safe_hard_negative_teacher_scale=0.75,
        ordinal_evidence_frontier_pairwise_weight=0.35,
        strict_shape_contract=True,
    )
    return kwargs, admission


def _shadowing_check(losses_path: Path) -> list[dict[str, int | str]]:
    tree = ast.parse(losses_path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "direct_uncertainty_recovery_value_loss")
    loop = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
        and n.target.id == 'key' and n.lineno >= 1600
    )
    before: dict[str, int] = {}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.lineno < loop.lineno:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    before.setdefault(target.id, node.lineno)
    collisions = []
    for node in ast.walk(loop):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in before:
                    collisions.append({"name": target.id, "outer_line": before[target.id], "inner_line": node.lineno})
    return collisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {"event": "v48_32_1_multigroup_loss_contract", "created_unix": time.time(), "valid": False}
    try:
        collisions = _shadowing_check(root / "src" / "ocrap" / "models" / "losses.py")
        if collisions:
            raise RuntimeError(f"outer tensor overwritten inside group loop: {collisions}")
        kwargs, admission = _loss_inputs()
        loss = direct_uncertainty_recovery_value_loss(**kwargs)
        loss.backward()
        gradient = admission.grad
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite multigroup loss")
        if gradient is None or not bool(torch.isfinite(gradient).all().item()) or float(gradient.abs().sum()) <= 0.0:
            raise RuntimeError("adaptive multigroup loss produced no finite admission gradient")
        report.update({
            "valid": True, "loss": float(loss.detach()),
            "admission_gradient_l1": float(gradient.detach().abs().sum()),
            "group_count": 2, "adaptive_margin": True, "factorized_harm": True,
            "strict_shape_contract": True, "outer_loop_collisions": collisions,
        })
    except Exception as exc:
        report.update({"error_type": type(exc).__name__, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
