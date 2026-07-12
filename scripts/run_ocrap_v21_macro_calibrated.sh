#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX="$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"
export VAL_MIX="$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"
export SAFE_TEST="$OCRAP_ROOT/test_safe"
export NEAR_TEST="$OCRAP_ROOT/test_near_contact"
export CONTACT_TEST="$OCRAP_ROOT/test_contact"
export WOMD_VAL=${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}
export WOMD_VAL_INTERACTIVE=${WOMD_VAL_INTERACTIVE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}

export RUN=${RUN:-runs/ocrap_v21_macro_calibrated}
export V21_RUN=${V21_RUN:-runs/ocrap_v21_macro_calibrated}
export CKPT=${CKPT:-$V21_RUN/model_v21/best.pt}
export CAL=${CAL:-$V21_RUN/calibration/calibration_mix_v21.json}
export GAMMA=${GAMMA:-$V21_RUN/calibration/gamma_rec_by_bucket_v21.json}
mkdir -p "$RUN"

[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT; set CKPT=/path/to/best.pt" >&2; exit 2; }
[[ -f "$GAMMA" ]] || { echo "missing gamma map $GAMMA; set GAMMA=/path/to/gamma.json" >&2; exit 2; }
[[ -f "$CAL" ]] || { echo "missing calibration $CAL; set CAL=/path/to/calibration.json" >&2; exit 2; }

make_sel() {
  local tag="$1"  # scalar | v21
  local rel_near=false
  local rel_contact=false
  local rec_pool=false
  case "$tag" in
    scalar) rel_near=false; rel_contact=false; rec_pool=false ;;
    v21) rel_near=true; rel_contact=true; rec_pool=true ;;
    *) echo "unknown selector tag: $tag" >&2; exit 2 ;;
  esac

  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$GAMMA"
    --set selection.ocrap_selector=calibrated_constrained
    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true

    # Paper invariant: never execute an unadmitted recovery fallback.
    --set selection.require_admitted_intervention_by_bucket.safe=true
    --set selection.require_admitted_intervention_by_bucket.safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.test_safe=true
    --set selection.require_admitted_intervention_by_bucket.test_safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.near_contact=true
    --set selection.require_admitted_intervention_by_bucket.test_near_contact=true
    --set selection.require_admitted_intervention_by_bucket.contact=true
    --set selection.require_admitted_intervention_by_bucket.test_contact=true
    --set selection.unadmitted_fallback_to_nominal=true

    # Safe regime remains nominal-preserving.
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe_v2=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe_v2=true
    --set selection.safe_force_nominal_mode_by_bucket.safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.safe_v2=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2=feasible

    # Keep v16/v17 option-DRS shortcut off in the main run. It increased
    # intervention but degraded DRS/FRA in previous results.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact=false
    --set selection.option_drs_certificate_by_bucket.test_near_contact=false
    --set selection.option_drs_certificate_by_bucket.contact=false
    --set selection.option_drs_certificate_by_bucket.test_contact=false

    # v21 macro-gated Pareto plus protective-macro recovery certificate.
    # Relative interventions are admitted only when they are on a deployability
    # Pareto frontier relative to nominal: they must improve shared-option DRS or
    # close the deployability gap while satisfying non-inferiority guards.
    --set selection.relative_recovery_certificate_by_bucket.safe=false
    --set selection.relative_recovery_certificate_by_bucket.safe_v2=false
    --set selection.relative_recovery_certificate_by_bucket.test_safe=false
    --set selection.relative_recovery_certificate_by_bucket.near_contact="$rel_near"
    --set selection.relative_recovery_certificate_by_bucket.test_near_contact="$rel_near"
    --set selection.relative_recovery_certificate_by_bucket.contact="$rel_contact"
    --set selection.relative_recovery_certificate_by_bucket.test_contact="$rel_contact"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_contact="$rec_pool"
    --set selection.relative_recovery_counts_as_evidence=true
    --set selection.relative_recovery_require_macro_by_bucket.near_contact=true
    --set selection.relative_recovery_require_macro_by_bucket.test_near_contact=true
    --set selection.relative_recovery_require_macro_by_bucket.contact=true
    --set selection.relative_recovery_require_macro_by_bucket.test_contact=true
    --set selection.relative_recovery_macro_allowlist_by_bucket.near_contact=brake,stabilize,yield,merge
    --set selection.relative_recovery_macro_allowlist_by_bucket.test_near_contact=brake,stabilize,yield,merge
    --set selection.relative_recovery_macro_allowlist_by_bucket.contact=yield,merge
    --set selection.relative_recovery_macro_allowlist_by_bucket.test_contact=yield,merge
    --set selection.relative_recovery_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over
    --set selection.relative_recovery_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over

    # v21 narrow protective channel. v19 proved macro gating prevents the v18
    # perturb_nominal failure, but it over-abstained on audited brake recoveries.
    # Only brake/stabilize can use this relaxed protective certificate; merge/yield
    # remain under the stricter Pareto relative-recovery gate above.
    --set selection.protective_macro_certificate_by_bucket.safe=false
    --set selection.protective_macro_certificate_by_bucket.test_safe=false
    --set selection.protective_macro_certificate_by_bucket.near_contact=false
    --set selection.protective_macro_certificate_by_bucket.test_near_contact=false
    --set selection.protective_macro_certificate_by_bucket.contact="$rel_contact"
    --set selection.protective_macro_certificate_by_bucket.test_contact="$rel_contact"
    --set selection.protective_macro_allowlist_by_bucket.contact=brake,stabilize
    --set selection.protective_macro_allowlist_by_bucket.test_contact=brake,stabilize
    --set selection.protective_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,merge,yield
    --set selection.protective_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,merge,yield
    --set selection.protective_macro_nominal_rec_lcb_max_by_bucket.contact=-0.42
    --set selection.protective_macro_nominal_rec_lcb_max_by_bucket.test_contact=-0.42
    --set selection.protective_macro_nominal_gap_min_by_bucket.contact=0.95
    --set selection.protective_macro_nominal_gap_min_by_bucket.test_contact=0.95
    --set selection.protective_macro_nominal_drs_max_by_bucket.contact=0.86
    --set selection.protective_macro_nominal_drs_max_by_bucket.test_contact=0.86
    # v21: protective brake/stabilize are judged by a deployability-vector score
    # rather than by mandatory gap reduction.  This tests the actual hypothesis
    # revealed by v20: missed brake candidates were teacher-deployable but the
    # model/selector over-penalized their predicted gap.
    --set selection.protective_macro_gate_by_bucket.contact=score
    --set selection.protective_macro_gate_by_bucket.test_contact=score
    --set selection.protective_macro_score_min_gain_by_bucket.contact=-0.010
    --set selection.protective_macro_score_min_gain_by_bucket.test_contact=-0.010
    --set selection.protective_macro_score_rec_weight_by_bucket.contact=0.25
    --set selection.protective_macro_score_rec_weight_by_bucket.test_contact=0.25
    --set selection.protective_macro_score_drs_weight_by_bucket.contact=0.65
    --set selection.protective_macro_score_drs_weight_by_bucket.test_contact=0.65
    --set selection.protective_macro_score_gap_weight_by_bucket.contact=0.10
    --set selection.protective_macro_score_gap_weight_by_bucket.test_contact=0.10
    --set selection.protective_macro_min_drs_by_bucket.contact=0.70
    --set selection.protective_macro_min_drs_by_bucket.test_contact=0.70
    --set selection.protective_macro_min_gap_reduction_by_bucket.contact=-1.0
    --set selection.protective_macro_min_gap_reduction_by_bucket.test_contact=-1.0
    --set selection.protective_macro_min_rec_gain_by_bucket.contact=-1.0
    --set selection.protective_macro_min_rec_gain_by_bucket.test_contact=-1.0
    --set selection.protective_macro_min_drs_gain_by_bucket.contact=-0.035
    --set selection.protective_macro_min_drs_gain_by_bucket.test_contact=-0.035
    --set selection.protective_macro_min_improvement_axes_by_bucket.contact=1
    --set selection.protective_macro_min_improvement_axes_by_bucket.test_contact=1
    --set selection.protective_macro_max_drs_drop_by_bucket.contact=0.04
    --set selection.protective_macro_max_drs_drop_by_bucket.test_contact=0.04
    --set selection.protective_macro_max_rec_lcb_drop_by_bucket.contact=0.25
    --set selection.protective_macro_max_rec_lcb_drop_by_bucket.test_contact=0.25
    --set selection.protective_macro_max_gap_by_bucket.contact=2.10
    --set selection.protective_macro_max_gap_by_bucket.test_contact=2.10
    --set selection.protective_macro_max_gap_increase_by_bucket.contact=0.70
    --set selection.protective_macro_max_gap_increase_by_bucket.test_contact=0.70
    --set selection.protective_macro_max_hard_by_bucket.contact=1.0
    --set selection.protective_macro_max_hard_by_bucket.test_contact=1.0
    --set selection.protective_macro_max_harm_by_bucket.contact=0.70
    --set selection.protective_macro_max_harm_by_bucket.test_contact=0.70
    --set selection.protective_macro_bonus_by_bucket.contact=0.28
    --set selection.protective_macro_bonus_by_bucket.test_contact=0.28
    --set selection.protective_macro_counts_as_evidence=true

    # Trigger relative reasoning only when nominal has low headroom/high gap.
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.near_contact=-0.30
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_near_contact=-0.30
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.contact=-0.42
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_contact=-0.42
    --set selection.relative_recovery_nominal_gap_min_by_bucket.near_contact=1.10
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_near_contact=1.10
    --set selection.relative_recovery_nominal_gap_min_by_bucket.contact=0.95
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_contact=0.95
    --set selection.relative_recovery_nominal_drs_max_by_bucket.near_contact=0.82
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_near_contact=0.82
    --set selection.relative_recovery_nominal_drs_max_by_bucket.contact=0.86
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_contact=0.86

    # Near-contact is a cautious anticipatory regime: require two-axis Pareto
    # evidence and no DRS/LCB deterioration. This should prevent v17's near-contact
    # offline degradation while preserving the story that intervention is possible
    # only under strong headroom evidence.
    --set selection.relative_recovery_gate_by_bucket.near_contact=pareto
    --set selection.relative_recovery_gate_by_bucket.test_near_contact=pareto
    --set selection.relative_recovery_min_improvement_axes_by_bucket.near_contact=2
    --set selection.relative_recovery_min_improvement_axes_by_bucket.test_near_contact=2
    --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.08
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.near_contact=0.12
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_near_contact=0.12
    --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=0.025
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=0.025
    --set selection.relative_recovery_max_drs_drop_by_bucket.near_contact=0.00
    --set selection.relative_recovery_max_drs_drop_by_bucket.test_near_contact=0.00
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.near_contact=0.02
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.test_near_contact=0.02
    --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.80
    --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.80
    --set selection.relative_recovery_max_gap_by_bucket.near_contact=1.35
    --set selection.relative_recovery_max_gap_by_bucket.test_near_contact=1.35
    --set selection.relative_recovery_max_gap_increase_by_bucket.near_contact=0.00
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_near_contact=0.00

    # Contact is a low-headroom recovery regime, but not an aggressive trigger.
    # v21 only allows semantic recovery macros to use the recovery-feasibility pool
    # and requires multi-axis calibrated deployability evidence.
    --set selection.relative_recovery_gate_by_bucket.contact=pareto
    --set selection.relative_recovery_gate_by_bucket.test_contact=pareto
    --set selection.relative_recovery_min_improvement_axes_by_bucket.contact=2
    --set selection.relative_recovery_min_improvement_axes_by_bucket.test_contact=2
    --set selection.relative_recovery_min_rec_gain_by_bucket.contact=0.04
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_contact=0.04
    --set selection.relative_recovery_min_gap_reduction_by_bucket.contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_contact=0.08
    --set selection.relative_recovery_min_drs_gain_by_bucket.contact=0.035
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_contact=0.035
    --set selection.relative_recovery_max_drs_drop_by_bucket.contact=0.00
    --set selection.relative_recovery_max_drs_drop_by_bucket.test_contact=0.00
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.contact=0.02
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.test_contact=0.02
    --set selection.relative_recovery_min_drs_by_bucket.contact=0.70
    --set selection.relative_recovery_min_drs_by_bucket.test_contact=0.70
    --set selection.relative_recovery_max_gap_by_bucket.contact=1.45
    --set selection.relative_recovery_max_gap_by_bucket.test_contact=1.45
    --set selection.relative_recovery_max_gap_increase_by_bucket.contact=0.00
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_contact=0.00

    # Recovery-feasibility pool: separate normal-driving admissibility from
    # recovery admissibility, but do not make it completely unconstrained.
    --set selection.recovery_cert_max_hard_by_bucket.near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_contact=1.0
    --set selection.recovery_cert_max_harm_by_bucket.near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.test_near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.contact=0.70
    --set selection.recovery_cert_max_harm_by_bucket.test_contact=0.70

    # Certified-intervention evidence and ranking.
    --set selection.require_intervention_evidence_by_bucket.safe=true
    --set selection.require_intervention_evidence_by_bucket.test_safe=true
    --set selection.require_intervention_evidence_by_bucket.near_contact=true
    --set selection.require_intervention_evidence_by_bucket.test_near_contact=true
    --set selection.require_intervention_evidence_by_bucket.contact=true
    --set selection.require_intervention_evidence_by_bucket.test_contact=true
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.70
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.contact=2.10
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=2.10
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.00
    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=0.80
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=0.70
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.35
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.28
    --set selection.deployability_bonus_by_bucket.near_contact=0.50
    --set selection.deployability_bonus_by_bucket.contact=0.35
    --set selection.contact_deployability_bonus_by_bucket.contact=0.25
    --set selection.contact_gap_penalty_by_bucket.contact=0.18
    --set selection.relative_recovery_bonus_by_bucket.near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.test_near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.contact=0.15
    --set selection.relative_recovery_bonus_by_bucket.test_contact=0.15
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.10
    --set selection.intervention_budget_rate_by_bucket.contact=0.16
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=1.20
    --set selection.intervention_budget_penalty_by_bucket.contact=5.0
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.12
    --set selection.deviation_penalty_by_bucket.contact=0.18
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.030
    --set selection.intervention_penalty_by_bucket.contact=0.040
  )
}

run_eval() {
  local tag="$1"; local d="$2"; local gpu="$3"
  make_sel "$tag"
  local dataset="$SAFE_TEST"
  [[ "$d" == "near_contact" ]] && dataset="$NEAR_TEST"
  [[ "$d" == "contact" ]] && dataset="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
    --dataset "$dataset" --checkpoint "$CKPT" --calibration "$CAL" --split test \
    --output "$RUN/eval_${d}_v21_${tag}.json" \
    --set evaluation.delta=0.05 \
    --set evaluation.group_by_dataset=true \
    --set evaluation.fallback_to_all_if_empty_split=true \
    "${COMMON_SEL[@]}" \
    --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
    | tee "$RUN/eval_${d}_v21_${tag}.log"
}

run_audit() {
  local tag="$1"; local b="$2"; local gpu="$3"; local targets="${4:-32}"; local labels="${5:-384}"
  make_sel "$tag"
  local bucket="$NEAR_TEST"
  [[ "$b" == "contact" ]] && bucket="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL_INTERACTIVE@150" --checkpoint "$CKPT" \
    --output "$RUN/audit_${b}_selected_topk_v21_${tag}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$bucket" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets="$targets" \
    --set closed_loop.max_rollouts=12 \
    --set closed_loop.raw_max_scenarios=900 \
    --set closed_loop.max_steps=20 \
    --set closed_loop.num_candidate_prefixes=12 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=selected_topk \
    --set closed_loop.audit_every_n_steps=4 \
    --set closed_loop.audit_max_labels="$labels" \
    --set closed_loop.audit_top_k=10 \
    --set closed_loop.audit_max_extra_candidates=9 \
    --set closed_loop.progress_every_steps=1 \
    | tee "$RUN/audit_${b}_selected_topk_v21_${tag}.log"
}

run_safe_closed_loop() {
  make_sel v21
  CUDA_VISIBLE_DEVICES=${GPU_SAFE:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL@150" --checkpoint "$CKPT" \
    --output "$RUN/closed_loop_safe_fast_v21.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$SAFE_TEST" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=80 \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts=32 \
    --set closed_loop.raw_max_scenarios=600 \
    --set closed_loop.max_steps=40 \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.num_candidate_prefixes=16 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=fast \
    --set closed_loop.progress_every_steps=5 \
    | tee "$RUN/closed_loop_safe_fast_v21.log"
}

summarize() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v21.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print("\n===== Offline eval v21 =====")
for p in sorted(root.glob('eval_*_v21_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} reason={r.get('selection_reason_counts')}")
print("\n===== Closed-loop/audit v21 =====")
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_best_PCD','closed_loop_audit_selected_PCD_regret','closed_loop_audit_pcd_selector_miss_rate','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v21*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_bounded_NUP' not in d: continue
    print('\n', p.name)
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  macro_counts:', d.get('macro_counts'))
    print('  audit_best_macro_counts:', d.get('audit_best_macro_counts'))
    print('  audit_miss_best_macro_counts:', d.get('audit_miss_best_macro_counts'))
    print('  audit_pcd_best_macro_counts:', d.get('audit_pcd_best_macro_counts'))
    print('  audit_pcd_miss_best_macro_counts:', d.get('audit_pcd_miss_best_macro_counts'))
    print('  audit_miss_selected_macro_counts:', d.get('audit_miss_selected_macro_counts'))
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
}

# Offline selector grid: pair scalar and v21 on two GPUs per regime.
run_eval scalar safe 0 &
run_eval v21 safe 1 &
wait
run_eval scalar near_contact 0 &
run_eval v21 near_contact 1 &
wait
run_eval scalar contact 0 &
run_eval v21 contact 1 &
wait

# Stress audits. Use both GPUs concurrently.
run_audit scalar near_contact 0 32 384 &
run_audit v21 near_contact 1 32 384 &
wait
run_audit scalar contact 0 32 384 &
run_audit v21 contact 1 32 384 &
wait

# Optional deeper contact audit for the proposed main selector. Enable with RUN_DEEP_CONTACT=1.
if [[ "${RUN_DEEP_CONTACT:-0}" == "1" ]]; then
  run_audit v21 contact 0 48 640
fi

run_safe_closed_loop
summarize
