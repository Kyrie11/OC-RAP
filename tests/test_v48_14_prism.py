from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss

ROOT = Path(__file__).parents[1]


def _loss(harm_logits: torch.Tensor, **extra) -> torch.Tensor:
    kwargs = dict(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 2.0, 1.0]),
        pred_opportunity_logit=torch.tensor([0.0, -2.0, 2.0]),
        pred_harm_logit=harm_logits,
        teacher_r_dep=torch.zeros(3), teacher_r_orc=torch.zeros(3),
        teacher_q=torch.ones((3, 1, 1)),
        # nominal/dead, harmful, beneficial under exact PCD.
        teacher_m_star=torch.tensor([[[1.0]], [[-1.0]], [[1.0]]]),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([11, 11, 11]), time_index=torch.ones(3, dtype=torch.long),
        macro_type_id=torch.tensor([0, 5, 3]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.ones(3, dtype=torch.long), macro_ids=(3, 5), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
        advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0, setwise_admission_weight=0.0,
        policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0,
        preference_listwise_weight=0.0, preference_gap_weight=0.0,
        preference_set_weight=0.0, preference_all_group_set_weight=0.0,
        delta_nll_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=1.0,
        ordinal_evidence_proposal_topk_weight=1.0,
        ordinal_evidence_proposal_topk=2,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_hard_harm_mining_increases_false_safe_penalty() -> None:
    # Candidate 1 is harmful but has a very low predicted harm probability.
    logits = torch.tensor([0.0, -4.0, -2.0])
    base = _loss(logits, ordinal_evidence_hard_harm_weight=0.0)
    hard = _loss(logits, ordinal_evidence_hard_harm_weight=3.0, ordinal_evidence_hard_example_gamma=2.0)
    assert hard.item() > base.item()


def test_partition_tool_creates_three_scene_disjoint_roles(tmp_path: Path) -> None:
    source = tmp_path / "calibration_near_contact"
    (source / "samples").mkdir(parents=True)
    rows = []
    for i in range(30):
        p = source / "samples" / f"s{i}.npz"
        np.savez_compressed(p, scene_id=f"scene-{i}", time_index=0)
        rows.append({"path": f"samples/{p.name}", "scene_id": f"scene-{i}"})
    with (source / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "scene_id"]); w.writeheader(); w.writerows(rows)
    out = tmp_path / "protocol"
    spec = importlib.util.spec_from_file_location("part", ROOT / "tools" / "partition_dedicated_calibration_v48_14.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    report = mod._partition_regime(source, out, "near_contact", 0.45, 0.15, 4814, "copy", True)
    assert report["scene_overlap"] == 0
    role_scenes = []
    for role in ("evidence_adapt_train", "evidence_adapt_dev", "certificate_pool"):
        with (out / f"{role}_near_contact" / "manifest.csv").open(newline="", encoding="utf-8") as f:
            role_scenes.append({r["scene_id"] for r in csv.DictReader(f)})
    assert not (role_scenes[0] & role_scenes[1])
    assert not (role_scenes[0] & role_scenes[2])
    assert not (role_scenes[1] & role_scenes[2])
    assert set().union(*role_scenes) == {f"scene-{i}" for i in range(30)}


def test_safe_nominal_probe_does_not_require_gamma_or_calibration() -> None:
    text = (ROOT / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    assert 'if [[ "$SAFE_NOMINAL_ONLY" != "1" ]]; then' in text
    assert '--set closed_loop.require_gamma_by_bucket=false' in text


def test_ablation_script_does_not_use_bash_groups_special_variable() -> None:
    text = (ROOT / "scripts" / "run_v48_14_parallel_ablations.sh").read_text()
    assert "ABLATION_SPECS=(" in text
    assert "GROUPS=(" not in text
