#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export SAFE_TEST=${SAFE_TEST:-$OCRAP_ROOT/test_safe}
export NEAR_TEST=${NEAR_TEST:-$OCRAP_ROOT/test_near_contact}
export CONTACT_TEST=${CONTACT_TEST:-$OCRAP_ROOT/test_contact}
export WOMD_VAL=${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}
export WOMD_VAL_INTERACTIVE=${WOMD_VAL_INTERACTIVE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}

# v36 is a selector-only residual contact correction on top of the v33 model. It keeps
# v34's residual brake-tail challenge, but separates broad tail admission from a
# stricter PCD-gain-bypass gate and adds hard budget control to avoid over-firing. Override CKPT/CAL/GAMMA if you place the model elsewhere.
export RUN=${RUN:-runs/ocrap_v36_calm_tail_eval}
export BASE_RUN=${BASE_RUN:-runs/ocrap_v33_brake_tail}
export CKPT=${CKPT:-$BASE_RUN/model_v33_brake_tail/best.pt}
export CAL=${CAL:-$BASE_RUN/calibration/calibration_mix_v33.json}
export GAMMA=${GAMMA:-$BASE_RUN/calibration/gamma_rec_by_bucket_v33.json}
mkdir -p "$RUN"

[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT; set CKPT=/path/to/best.pt" >&2; exit 2; }
[[ -f "$GAMMA" ]] || { echo "missing gamma map $GAMMA; set GAMMA=/path/to/gamma.json" >&2; exit 2; }
[[ -f "$CAL" ]] || { echo "missing calibration $CAL; set CAL=/path/to/calibration.json" >&2; exit 2; }

make_sel() {
  local tag="$1"  # scalar | v36
  local rel_near=false
  local rel_contact=false
  local rec_pool=false
  local brake_rescue=false
  local pcd_rescue=false
  local brake_tail=false
  local brake_tail_bypass_pcd=false
  local is_v27=false
  case "$tag" in
    scalar) rel_near=false; rel_contact=false; rec_pool=false; brake_rescue=false; pcd_rescue=false; brake_tail=false; brake_tail_bypass_pcd=false ;;
    v36|v27) rel_near=true; rel_contact=true; rec_pool=true; brake_rescue=${CONTACT_BRAKE_RESCUE:-false}; pcd_rescue=true; brake_tail=${CONTACT_BRAKE_TAIL:-true}; brake_tail_bypass_pcd=${CONTACT_TAIL_BYPASS_PCD_GAIN:-true}; is_v27=true ;;
    *) echo "unknown selector tag: $tag" >&2; exit 2 ;;
  esac

  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$GAMMA"
    --set selection.ocrap_selector=calibrated_constrained
    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true

    # Paper invariant: never execute unadmitted recovery, and never treat
    # exploration/coverage prefixes as recoveries.
    --set selection.require_admitted_intervention_by_bucket.safe=true
    --set selection.require_admitted_intervention_by_bucket.test_safe=true
    --set selection.require_admitted_intervention_by_bucket.near_contact=true
    --set selection.require_admitted_intervention_by_bucket.test_near_contact=true
    --set selection.require_admitted_intervention_by_bucket.contact=true
    --set selection.require_admitted_intervention_by_bucket.test_contact=true
    --set selection.unadmitted_fallback_to_nominal=true
    --set selection.intervention_macro_blocklist_by_bucket.safe=keep,brake,yield,lane_shift,merge,pull_over,stabilize,perturb_nominal
    --set selection.intervention_macro_blocklist_by_bucket.test_safe=keep,brake,yield,lane_shift,merge,pull_over,stabilize,perturb_nominal
    --set selection.intervention_macro_allowlist_by_bucket.near_contact=brake,stabilize,yield,merge
    --set selection.intervention_macro_allowlist_by_bucket.test_near_contact=brake,stabilize,yield,merge
    --set selection.intervention_macro_allowlist_by_bucket.contact=brake,stabilize,yield,merge
    --set selection.intervention_macro_allowlist_by_bucket.test_contact=brake,stabilize,yield,merge
    --set selection.intervention_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.intervention_require_macro_by_bucket.near_contact=true
    --set selection.intervention_require_macro_by_bucket.test_near_contact=true
    --set selection.intervention_require_macro_by_bucket.contact=true
    --set selection.intervention_require_macro_by_bucket.test_contact=true

    # Safe regime remains nominal-preserving.
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_mode_by_bucket.safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe=feasible

    # Keep old option shortcut off.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact=false
    --set selection.option_drs_certificate_by_bucket.test_near_contact=false
    --set selection.option_drs_certificate_by_bucket.contact=false
    --set selection.option_drs_certificate_by_bucket.test_contact=false

    # Relative Pareto channel: near contact permits semantic recovery macros;
    # contact uses it only for yield/merge.  Brake is routed through the v24
    # macro-stratified brake-rescue certificate below.
    --set selection.relative_recovery_certificate_by_bucket.safe=false
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
    --set selection.relative_recovery_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.relative_recovery_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake,stabilize
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake,stabilize

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

    --set selection.recovery_cert_max_hard_by_bucket.near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_contact=1.0
    --set selection.recovery_cert_max_harm_by_bucket.near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.test_near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.contact=0.70
    --set selection.recovery_cert_max_harm_by_bucket.test_contact=0.70

    # v24: disable the old generic brake/stabilize protective channel in contact.
    # Brake uses the macro-stratified rescue certificate; stabilize must satisfy
    # the stricter near-contact relative gate or abstain.
    --set selection.protective_macro_certificate_by_bucket.safe=false
    --set selection.protective_macro_certificate_by_bucket.test_safe=false
    --set selection.protective_macro_certificate_by_bucket.near_contact=false
    --set selection.protective_macro_certificate_by_bucket.test_near_contact=false
    --set selection.protective_macro_certificate_by_bucket.contact=false
    --set selection.protective_macro_certificate_by_bucket.test_contact=false

    # v24 macro-stratified brake-rescue certificate.  The candidate must be a
    # moderate-gap brake, not a nominal-like perturbation and not a high-gap artifact.
    --set selection.brake_rescue_certificate_by_bucket.near_contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.test_near_contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.test_contact="$brake_rescue"
    --set selection.brake_rescue_macro_name_by_bucket.contact=brake
    --set selection.brake_rescue_macro_name_by_bucket.test_contact=brake
    --set selection.brake_rescue_min_pred_drs_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MIN_DRS:-0.90}
    --set selection.brake_rescue_min_pred_drs_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MIN_DRS:-0.90}
    --set selection.brake_rescue_min_pred_r_dep_by_bucket.contact=-0.70
    --set selection.brake_rescue_min_pred_r_dep_by_bucket.test_contact=-0.70
    --set selection.brake_rescue_min_candidate_gap_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MIN_GAP:-0.18}
    --set selection.brake_rescue_min_candidate_gap_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MIN_GAP:-0.18}
    --set selection.brake_rescue_max_candidate_gap_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MAX_GAP:-0.34}
    --set selection.brake_rescue_max_candidate_gap_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MAX_GAP:-0.34}
    --set selection.brake_rescue_max_hard_by_bucket.contact=1.0
    --set selection.brake_rescue_max_hard_by_bucket.test_contact=1.0
    --set selection.brake_rescue_max_harm_by_bucket.contact=0.70
    --set selection.brake_rescue_max_harm_by_bucket.test_contact=0.70
    --set selection.brake_rescue_require_nominal_unadmitted_by_bucket.contact=true
    --set selection.brake_rescue_require_nominal_unadmitted_by_bucket.test_contact=true
    --set selection.brake_rescue_nominal_rec_lcb_max_by_bucket.contact=0.60
    --set selection.brake_rescue_nominal_rec_lcb_max_by_bucket.test_contact=0.60
    --set selection.brake_rescue_nominal_gap_min_by_bucket.contact=0.02
    --set selection.brake_rescue_nominal_gap_min_by_bucket.test_contact=0.02
    --set selection.brake_rescue_nominal_drs_max_by_bucket.contact=1.01
    --set selection.brake_rescue_nominal_drs_max_by_bucket.test_contact=1.01
    --set selection.brake_rescue_counts_as_evidence=true
    --set selection.brake_rescue_budget_bypass_by_bucket.near_contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.test_near_contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.test_contact=false

    # v34 residual brake-tail certificate: captures contact cases where paper-best
    # brake has high DRS but learned nominal PCD is over-confident, while avoiding
    # early nominal-best brake false positives by requiring high-gap or low-R_dep shape.
    --set selection.brake_tail_rescue_certificate_by_bucket.safe=false
    --set selection.brake_tail_rescue_certificate_by_bucket.test_safe=false
    --set selection.brake_tail_rescue_certificate_by_bucket.near_contact=${RUN_NEAR_TAIL:-false}
    --set selection.brake_tail_rescue_certificate_by_bucket.test_near_contact=${RUN_NEAR_TAIL:-false}
    --set selection.brake_tail_rescue_certificate_by_bucket.contact="$brake_tail"
    --set selection.brake_tail_rescue_certificate_by_bucket.test_contact="$brake_tail"
    --set selection.brake_tail_min_pred_drs_by_bucket.contact=${CONTACT_TAIL_MIN_DRS:-0.78}
    --set selection.brake_tail_min_pred_drs_by_bucket.test_contact=${CONTACT_TAIL_MIN_DRS:-0.78}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.contact=${CONTACT_TAIL_MIN_R_DEP:--0.55}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.test_contact=${CONTACT_TAIL_MIN_R_DEP:--0.55}
    --set selection.brake_tail_min_pred_pcd_by_bucket.contact=${CONTACT_TAIL_MIN_PCD:-0.24}
    --set selection.brake_tail_min_pred_pcd_by_bucket.test_contact=${CONTACT_TAIL_MIN_PCD:-0.24}
    --set selection.brake_tail_min_candidate_gap_by_bucket.contact=${CONTACT_TAIL_MIN_GAP:-0.08}
    --set selection.brake_tail_min_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_MIN_GAP:-0.08}
    --set selection.brake_tail_max_candidate_gap_by_bucket.contact=${CONTACT_TAIL_MAX_GAP:-0.42}
    --set selection.brake_tail_max_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_MAX_GAP:-0.42}
    --set selection.brake_tail_high_gap_min_by_bucket.contact=${CONTACT_TAIL_HIGH_GAP:-0.34}
    --set selection.brake_tail_high_gap_min_by_bucket.test_contact=${CONTACT_TAIL_HIGH_GAP:-0.34}
    --set selection.brake_tail_low_r_dep_max_by_bucket.contact=${CONTACT_TAIL_LOW_R_DEP_MAX:-0.00}
    --set selection.brake_tail_low_r_dep_max_by_bucket.test_contact=${CONTACT_TAIL_LOW_R_DEP_MAX:-0.00}
    --set selection.brake_tail_max_hard_by_bucket.contact=1.0
    --set selection.brake_tail_max_hard_by_bucket.test_contact=1.0
    --set selection.brake_tail_max_harm_by_bucket.contact=0.70
    --set selection.brake_tail_max_harm_by_bucket.test_contact=0.70
    --set selection.brake_tail_counts_as_evidence=true
    --set selection.brake_calm_tail_bypass_by_bucket.contact=false
    --set selection.brake_calm_tail_bypass_by_bucket.test_contact=false
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.contact="$brake_tail_bypass_pcd"
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.test_contact="$brake_tail_bypass_pcd"
    --set selection.brake_tail_challenge_min_pred_drs_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_DRS:-0.86}
    --set selection.brake_tail_challenge_min_pred_drs_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_DRS:-0.86}
    --set selection.brake_tail_challenge_min_pred_pcd_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_PCD:-0.32}
    --set selection.brake_tail_challenge_min_pred_pcd_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_PCD:-0.32}
    --set selection.brake_tail_challenge_min_candidate_gap_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_GAP:-0.085}
    --set selection.brake_tail_challenge_min_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_GAP:-0.085}
    --set selection.brake_tail_challenge_max_candidate_gap_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MAX_GAP:-0.42}
    --set selection.brake_tail_challenge_max_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MAX_GAP:-0.42}
    --set selection.brake_tail_challenge_high_gap_min_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_HIGH_GAP:-0.115}
    --set selection.brake_tail_challenge_high_gap_min_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_HIGH_GAP:-0.115}
    --set selection.brake_tail_challenge_low_r_dep_max_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_LOW_R_DEP_MAX:--0.14}
    --set selection.brake_tail_challenge_low_r_dep_max_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_LOW_R_DEP_MAX:--0.14}
    --set selection.brake_tail_min_pred_drs_by_bucket.near_contact=${NEAR_TAIL_MIN_DRS:-0.93}
    --set selection.brake_tail_min_pred_drs_by_bucket.test_near_contact=${NEAR_TAIL_MIN_DRS:-0.93}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.near_contact=${NEAR_TAIL_MIN_R_DEP:--0.20}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.test_near_contact=${NEAR_TAIL_MIN_R_DEP:--0.20}
    --set selection.brake_tail_min_pred_pcd_by_bucket.near_contact=${NEAR_TAIL_MIN_PCD:-0.33}
    --set selection.brake_tail_min_pred_pcd_by_bucket.test_near_contact=${NEAR_TAIL_MIN_PCD:-0.33}
    --set selection.brake_tail_min_candidate_gap_by_bucket.near_contact=${NEAR_TAIL_MIN_GAP:-0.14}
    --set selection.brake_tail_min_candidate_gap_by_bucket.test_near_contact=${NEAR_TAIL_MIN_GAP:-0.14}
    --set selection.brake_tail_max_candidate_gap_by_bucket.near_contact=${NEAR_TAIL_MAX_GAP:-0.36}
    --set selection.brake_tail_max_candidate_gap_by_bucket.test_near_contact=${NEAR_TAIL_MAX_GAP:-0.36}
    --set selection.brake_tail_high_gap_min_by_bucket.near_contact=${NEAR_TAIL_HIGH_GAP:-0.15}
    --set selection.brake_tail_high_gap_min_by_bucket.test_near_contact=${NEAR_TAIL_HIGH_GAP:-0.15}
    --set selection.brake_tail_low_r_dep_max_by_bucket.near_contact=${NEAR_TAIL_LOW_R_DEP_MAX:--0.05}
    --set selection.brake_tail_low_r_dep_max_by_bucket.test_near_contact=${NEAR_TAIL_LOW_R_DEP_MAX:--0.05}
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.near_contact=${NEAR_TAIL_BYPASS_PCD_GAIN:-false}
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.test_near_contact=${NEAR_TAIL_BYPASS_PCD_GAIN:-false}
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.near_contact=${NEAR_TAIL_CHALLENGE_BUDGET_BYPASS:-false}
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.test_near_contact=${NEAR_TAIL_CHALLENGE_BUDGET_BYPASS:-false}

    # v24 BMRC: a budget-aware post-contact-deployability rescue certificate.
    # It allows brake to challenge an over-confident nominal prediction in
    # near/contact, but repeated interventions must still pay the hard exposure
    # budget.  This targets the remaining v23 misses where audit-best PCD is
    # brake while selected nominal has over-confident DRS/gap predictions.
    --set selection.pcd_rescue_certificate_by_bucket.safe=false
    --set selection.pcd_rescue_certificate_by_bucket.test_safe=false
    --set selection.pcd_rescue_certificate_by_bucket.near_contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.test_near_contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.test_contact="$pcd_rescue"
    --set selection.pcd_rescue_macro_allowlist_by_bucket.near_contact=brake
    --set selection.pcd_rescue_macro_allowlist_by_bucket.test_near_contact=brake
    --set selection.pcd_rescue_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
    --set selection.pcd_rescue_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
    --set selection.pcd_rescue_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,merge,yield,stabilize
    --set selection.pcd_rescue_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,merge,yield,stabilize
    --set selection.pcd_rescue_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.pcd_rescue_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.near_contact=0.32
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.test_near_contact=0.32
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.contact=${CONTACT_FRONTIER_MIN_PCD:-0.28}
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_PCD:-0.28}
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.near_contact=-1.0
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_near_contact=-1.0
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.contact=${CONTACT_FRONTIER_MIN_PCD_GAIN:-0.0}
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_PCD_GAIN:-0.0}
    --set selection.pcd_rescue_min_pred_drs_by_bucket.near_contact=0.88
    --set selection.pcd_rescue_min_pred_drs_by_bucket.test_near_contact=0.88
    --set selection.pcd_rescue_min_pred_drs_by_bucket.contact=${CONTACT_FRONTIER_MIN_DRS:-0.62}
    --set selection.pcd_rescue_min_pred_drs_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_DRS:-0.62}
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.near_contact=-0.75
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.test_near_contact=-0.75
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.contact=${CONTACT_FRONTIER_MIN_R_DEP:--1.20}
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_R_DEP:--1.20}
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.near_contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.test_near_contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.test_contact=0.02
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.near_contact=0.34
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.test_near_contact=0.34
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.contact=${CONTACT_FRONTIER_MAX_GAP:-1.45}
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.test_contact=${CONTACT_FRONTIER_MAX_GAP:-1.45}
    --set selection.pcd_rescue_max_hard_by_bucket.near_contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.test_near_contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.test_contact=1.0
    --set selection.pcd_rescue_max_harm_by_bucket.near_contact=0.55
    --set selection.pcd_rescue_max_harm_by_bucket.test_near_contact=0.55
    --set selection.pcd_rescue_max_harm_by_bucket.contact=0.70
    --set selection.pcd_rescue_max_harm_by_bucket.test_contact=0.70
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.near_contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.test_near_contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.test_contact=true
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.near_contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.test_near_contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.test_contact=false
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.near_contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.test_near_contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.test_contact=0.65
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.near_contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.test_near_contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.test_contact=0.02
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.near_contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.test_near_contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.test_contact=1.01
    --set selection.pcd_rescue_max_utility_drop_by_bucket.near_contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.test_near_contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.test_contact=-1.0
    --set selection.pcd_rescue_bonus_by_bucket.near_contact=0.15
    --set selection.pcd_rescue_bonus_by_bucket.test_near_contact=0.15
    --set selection.pcd_rescue_bonus_by_bucket.contact=0.12
    --set selection.pcd_rescue_bonus_by_bucket.test_contact=0.12
    --set selection.pcd_rescue_counts_as_evidence=true
    --set selection.pcd_rescue_budget_bypass_by_bucket.near_contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.test_near_contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.test_contact=false

    # v26 NARC: a certified brake/PCD rescue may challenge nominal even when
    # the nominal learned LCB is over-confident, but exposure is controlled by
    # an explicit closed-loop cooldown.  Contact is enabled by default; near is
    # conservative by default to test whether the NUP drop was repeated braking.
    --set selection.stress_rescue_challenge_nominal_by_bucket.safe=false
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_safe=false
    --set selection.stress_rescue_challenge_nominal_by_bucket.near_contact=${RUN_NEAR_CHALLENGE:-false}
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_near_contact=${RUN_NEAR_CHALLENGE:-false}
    --set selection.stress_rescue_challenge_nominal_by_bucket.contact=true
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_contact=true
    --set selection.intervention_cooldown_steps_by_bucket.near_contact=${NEAR_COOLDOWN_STEPS:-12}
    --set selection.intervention_cooldown_steps_by_bucket.test_near_contact=${NEAR_COOLDOWN_STEPS:-12}
    --set selection.intervention_cooldown_steps_by_bucket.contact=${CONTACT_COOLDOWN_STEPS:-2}
    --set selection.intervention_cooldown_steps_by_bucket.test_contact=${CONTACT_COOLDOWN_STEPS:-2}

    # v26 Guarded-NARC: v25 proved that challenge itself works, but contact
    # over-selected brake because the challenge frontier was identical to the
    # coarse rescue certificate.  These extra guards restrict nominal challenge
    # to a moderate ambiguity band and cap high-utility false-brake artifacts.
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_GAP:-0.18}
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_GAP:-0.18}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_GAP:-0.32}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_GAP:-0.32}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_DRS:-0.90}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_DRS:-0.90}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_PCD:-0.30}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_PCD:-0.30}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_used_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_USED:-3}
    --set selection.rescue_challenge_max_used_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_USED:-3}

    --set selection.rescue_challenge_min_candidate_gap_by_bucket.contact=${CONTACT_CHALLENGE_MIN_GAP:-0.08}
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_GAP:-0.08}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.contact=${CONTACT_CHALLENGE_MAX_GAP:-1.45}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_GAP:-1.45}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.contact=${CONTACT_CHALLENGE_MIN_DRS:-0.62}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_DRS:-0.62}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.contact=${CONTACT_CHALLENGE_MIN_PCD:-0.28}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_PCD:-0.28}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.contact=${CONTACT_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_used_by_bucket.contact=${CONTACT_CHALLENGE_MAX_USED:-5}
    --set selection.rescue_challenge_max_used_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_USED:-5}
    --set selection.rescue_challenge_score_pcd_weight_by_bucket.contact=1.0
    --set selection.rescue_challenge_score_pcd_weight_by_bucket.test_contact=1.0
    --set selection.rescue_challenge_score_drs_weight_by_bucket.contact=0.20
    --set selection.rescue_challenge_score_drs_weight_by_bucket.test_contact=0.20
    --set selection.rescue_challenge_score_gap_weight_by_bucket.contact=0.30
    --set selection.rescue_challenge_score_gap_weight_by_bucket.test_contact=0.30
    --set selection.rescue_challenge_score_utility_weight_by_bucket.contact=0.02
    --set selection.rescue_challenge_score_utility_weight_by_bucket.test_contact=0.02

    --set selection.require_intervention_evidence_by_bucket.safe=true
    --set selection.require_intervention_evidence_by_bucket.test_safe=true
    --set selection.require_intervention_evidence_by_bucket.near_contact=true
    --set selection.require_intervention_evidence_by_bucket.test_near_contact=true
    --set selection.require_intervention_evidence_by_bucket.contact=true
    --set selection.require_intervention_evidence_by_bucket.test_contact=true
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.64
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.64
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.45
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.45
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
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.34
    --set selection.deployability_bonus_by_bucket.near_contact=0.50
    --set selection.deployability_bonus_by_bucket.contact=0.35
    --set selection.contact_deployability_bonus_by_bucket.contact=0.18
    --set selection.contact_gap_penalty_by_bucket.contact=0.30
    --set selection.relative_recovery_bonus_by_bucket.near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.test_near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.contact=0.15
    --set selection.relative_recovery_bonus_by_bucket.test_contact=0.15
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.06
    --set selection.intervention_budget_rate_by_bucket.contact=${CONTACT_BUDGET_RATE:-0.08}
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=6.0
    --set selection.intervention_budget_penalty_by_bucket.contact=6.0
    --set selection.intervention_budget_hard_by_bucket.contact=${CONTACT_BUDGET_HARD:-true}
    --set selection.intervention_budget_hard_by_bucket.test_contact=${CONTACT_BUDGET_HARD:-true}
    --set selection.intervention_budget_hard_min_rec_gain_by_bucket.contact=0.04
    --set selection.intervention_budget_hard_min_rec_gain_by_bucket.test_contact=0.04
    --set selection.intervention_budget_hard_min_drs_gain_by_bucket.contact=0.03
    --set selection.intervention_budget_hard_min_drs_gain_by_bucket.test_contact=0.03
    --set selection.intervention_budget_hard_min_gap_reduction_by_bucket.contact=0.08
    --set selection.intervention_budget_hard_min_gap_reduction_by_bucket.test_contact=0.08
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.12
    --set selection.deviation_penalty_by_bucket.contact=0.18
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.030
    --set selection.intervention_penalty_by_bucket.contact=0.060
  )

  if [[ "$is_v27" == "true" ]]; then
    COMMON_SEL+=(
      # v27 DDC principle:
      # safe is locked nominal; near-contact is nominal-preserving unless a
      # recovery macro has material deployability dominance; contact allows
      # post-contact recovery but still requires positive PCD/DRS/gap evidence.
      --set selection.stress_rescue_challenge_nominal_by_bucket.near_contact=${RUN_NEAR_CHALLENGE:-false}
      --set selection.stress_rescue_challenge_nominal_by_bucket.test_near_contact=${RUN_NEAR_CHALLENGE:-false}
      --set selection.stress_rescue_challenge_nominal_by_bucket.contact=true
      --set selection.stress_rescue_challenge_nominal_by_bucket.test_contact=true
      --set selection.brake_rescue_certificate_by_bucket.near_contact=${RUN_NEAR_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.test_near_contact=${RUN_NEAR_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.contact=${CONTACT_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE:-false}

      # Near-contact: preserve nominal utility; no repeated brake artifacts.
      --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.90
      --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.90
      --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=0.00
      --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=0.00
      --set selection.relative_recovery_min_gap_reduction_by_bucket.near_contact=0.08
      --set selection.relative_recovery_min_gap_reduction_by_bucket.test_near_contact=0.08
      --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.10
      --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.10
      --set selection.relative_recovery_min_improvement_axes_by_bucket.near_contact=2
      --set selection.relative_recovery_min_improvement_axes_by_bucket.test_near_contact=2
      --set selection.relative_recovery_macro_allowlist_by_bucket.near_contact=merge,yield,stabilize
      --set selection.relative_recovery_macro_allowlist_by_bucket.test_near_contact=merge,yield,stabilize
      --set selection.pcd_rescue_macro_allowlist_by_bucket.near_contact=merge,yield,stabilize
      --set selection.pcd_rescue_macro_allowlist_by_bucket.test_near_contact=merge,yield,stabilize
      --set selection.pcd_rescue_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.pcd_rescue_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.near_contact=0.020
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_near_contact=0.020
      --set selection.pcd_rescue_max_utility_drop_by_bucket.near_contact=0.12
      --set selection.pcd_rescue_max_utility_drop_by_bucket.test_near_contact=0.12
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.near_contact=2
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.test_near_contact=2
      --set selection.intervention_budget_rate_by_bucket.near_contact=0.035
      --set selection.intervention_budget_rate_by_bucket.test_near_contact=0.035
      --set selection.intervention_budget_hard_by_bucket.near_contact=true
      --set selection.intervention_budget_hard_by_bucket.test_near_contact=true
      --set selection.intervention_cooldown_steps_by_bucket.near_contact=${NEAR_COOLDOWN_STEPS:-8}
      --set selection.intervention_cooldown_steps_by_bucket.test_near_contact=${NEAR_COOLDOWN_STEPS:-8}
      --set selection.stress_nominal_anchor_by_bucket.near_contact=true
      --set selection.stress_nominal_anchor_by_bucket.test_near_contact=true
      --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.90
      --set selection.stress_anchor_drs_floor_by_bucket.test_near_contact=0.90
      --set selection.stress_anchor_max_gap_by_bucket.near_contact=0.25
      --set selection.stress_anchor_max_gap_by_bucket.test_near_contact=0.25
      --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.04
      --set selection.stress_anchor_min_drs_gain_by_bucket.test_near_contact=0.04
      --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.05
      --set selection.stress_anchor_min_rec_gain_by_bucket.test_near_contact=0.05
      --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.04
      --set selection.stress_anchor_min_gap_reduction_by_bucket.test_near_contact=0.04

      # Contact: prioritize post-contact deployability, but do not accept a
      # rescue challenge without deployability dominance over nominal.
      --set selection.pcd_rescue_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.pcd_rescue_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.pcd_rescue_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift
      --set selection.pcd_rescue_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.contact=${CONTACT_PCD_RESCUE_MIN_GAIN:-0.000}
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_contact=${CONTACT_PCD_RESCUE_MIN_GAIN:-0.000}
      --set selection.pcd_rescue_max_utility_drop_by_bucket.contact=${CONTACT_PCD_RESCUE_MAX_UTILITY_DROP:-0.35}
      --set selection.pcd_rescue_max_utility_drop_by_bucket.test_contact=${CONTACT_PCD_RESCUE_MAX_UTILITY_DROP:-0.35}
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.contact=1
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.test_contact=1
      --set selection.relative_recovery_macro_allowlist_by_bucket.contact=${CONTACT_RELATIVE_MACROS:-yield}
      --set selection.relative_recovery_macro_allowlist_by_bucket.test_contact=${CONTACT_RELATIVE_MACROS:-yield}
      --set selection.relative_recovery_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.relative_recovery_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.intervention_budget_rate_by_bucket.contact=${CONTACT_BUDGET_RATE:-0.08}
      --set selection.intervention_budget_rate_by_bucket.test_contact=${CONTACT_BUDGET_RATE:-0.08}
      --set selection.intervention_budget_hard_by_bucket.contact=${CONTACT_BUDGET_HARD:-true}
      --set selection.intervention_budget_hard_by_bucket.test_contact=${CONTACT_BUDGET_HARD:-true}
      --set selection.intervention_cooldown_steps_by_bucket.contact=${CONTACT_COOLDOWN_STEPS:-2}
      --set selection.intervention_cooldown_steps_by_bucket.test_contact=${CONTACT_COOLDOWN_STEPS:-2}

      # New code-level DDC guards.  These are the main v27 addition.
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.near_contact=0.035
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.test_near_contact=0.035
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.near_contact=0.060
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.test_near_contact=0.060
      --set selection.rescue_challenge_min_drs_gain_by_bucket.near_contact=0.000
      --set selection.rescue_challenge_min_drs_gain_by_bucket.test_near_contact=0.000
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.near_contact=0.040
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.test_near_contact=0.040
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.near_contact=3
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.test_near_contact=3
      --set selection.rescue_challenge_macro_allowlist_by_bucket.near_contact=${NEAR_CHALLENGE_MACROS:-merge,yield,stabilize}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.test_near_contact=${NEAR_CHALLENGE_MACROS:-merge,yield,stabilize}
      --set selection.rescue_challenge_max_used_by_bucket.near_contact=1
      --set selection.rescue_challenge_max_used_by_bucket.test_near_contact=1

      --set selection.rescue_challenge_min_pcd_gain_by_bucket.contact=${CONTACT_CHALLENGE_MIN_PCD_GAIN:-0.000}
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_PCD_GAIN:-0.000}
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.contact=${CONTACT_CHALLENGE_MIN_REC_GAIN:--1.000}
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_REC_GAIN:--1.000}
      --set selection.rescue_challenge_min_drs_gain_by_bucket.contact=0.000
      --set selection.rescue_challenge_min_drs_gain_by_bucket.test_contact=0.000
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.contact=${CONTACT_CHALLENGE_MIN_GAP_REDUCTION:--1.000}
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_GAP_REDUCTION:--1.000}
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.contact=${CONTACT_CHALLENGE_MIN_AXES:-1}
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_AXES:-1}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.rescue_challenge_max_used_by_bucket.contact=${CONTACT_CHALLENGE_MAX_USED:-5}
      --set selection.rescue_challenge_max_used_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_USED:-5}
      --set selection.rescue_challenge_score_pcd_weight_by_bucket.contact=0.20
      --set selection.rescue_challenge_score_pcd_weight_by_bucket.test_contact=0.20
      --set selection.rescue_challenge_score_drs_weight_by_bucket.contact=1.00
      --set selection.rescue_challenge_score_drs_weight_by_bucket.test_contact=1.00
      --set selection.rescue_challenge_score_gap_weight_by_bucket.contact=0.08
      --set selection.rescue_challenge_score_gap_weight_by_bucket.test_contact=0.08
      --set selection.rescue_challenge_nominal_guard_min_pcd_by_bucket.contact=-1.0
      --set selection.rescue_challenge_nominal_guard_min_pcd_by_bucket.test_contact=-1.0
      --set selection.rescue_challenge_nominal_guard_max_gap_by_bucket.contact=0.08
      --set selection.rescue_challenge_nominal_guard_max_gap_by_bucket.test_contact=0.08
    )
  fi
}

run_eval() {
  local tag="$1"; local d="$2"; local gpu="$3"
  make_sel "$tag"
  local dataset="$SAFE_TEST"
  [[ "$d" == "near_contact" ]] && dataset="$NEAR_TEST"
  [[ "$d" == "contact" ]] && dataset="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
    --dataset "$dataset" --checkpoint "$CKPT" --calibration "$CAL" --split test \
    --output "$RUN/eval_${d}_v36_${tag}.json" \
    --set evaluation.delta=0.05 \
    --set evaluation.group_by_dataset=true \
    --set evaluation.fallback_to_all_if_empty_split=true \
    "${COMMON_SEL[@]}" \
    --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
    | tee "$RUN/eval_${d}_v36_${tag}.log"
}

run_audit() {
  local tag="$1"; local b="$2"; local gpu="$3"; local targets="${4:-32}"; local labels="${5:-384}"
  make_sel "$tag"
  local bucket="$NEAR_TEST"
  [[ "$b" == "contact" ]] && bucket="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL_INTERACTIVE@150" --checkpoint "$CKPT" \
    --output "$RUN/audit_${b}_selected_topk_v36_${tag}.json" \
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
    | tee "$RUN/audit_${b}_selected_topk_v36_${tag}.log"
  assert_json "$RUN/audit_${b}_selected_topk_v36_${tag}.json"
}


assert_json() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "missing expected JSON output: $path" >&2
    exit 7
  fi
  python - <<'JSONCHECK' "$path"
import json, sys
with open(sys.argv[1]) as f:
    json.load(f)
JSONCHECK
}

run_safe_closed_loop() {
  make_sel v36
  CUDA_VISIBLE_DEVICES=${GPU_SAFE:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL@150" --checkpoint "$CKPT" \
    --output "$RUN/closed_loop_safe_fast_v36.json" \
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
    | tee "$RUN/closed_loop_safe_fast_v36.log"
  assert_json "$RUN/closed_loop_safe_fast_v36.json"
}

summarize() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v36.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print("\n===== Offline eval v36 =====")
for p in sorted(root.glob('eval_*_v36_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} reason={r.get('selection_reason_counts')}")
print("\n===== Closed-loop/audit v36 =====")
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_best_PCD','closed_loop_audit_selected_PCD_regret','closed_loop_audit_pcd_selector_miss_rate','closed_loop_audit_paper_best_PCD','closed_loop_audit_paper_selected_PCD_regret','closed_loop_audit_paper_pcd_selector_miss_rate','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v36*.json')):
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
    print('  audit_paper_pcd_best_macro_counts:', d.get('audit_paper_pcd_best_macro_counts'))
    print('  audit_paper_pcd_miss_best_macro_counts:', d.get('audit_paper_pcd_miss_best_macro_counts'))
    print('  audit_miss_selected_macro_counts:', d.get('audit_miss_selected_macro_counts'))
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
}

# Offline selector grid; each pair uses both A30s.
run_eval scalar safe 0 &
run_eval v36 safe 1 &
wait
run_eval scalar near_contact 0 &
run_eval v36 near_contact 1 &
wait
run_eval scalar contact 0 &
run_eval v36 contact 1 &
wait

# Stress audits. Two GPUs run clean scalar baseline and v34 concurrently.
run_audit scalar near_contact 0 32 384 &
run_audit v36 near_contact 1 ${AUDIT_TARGETS:-32} ${AUDIT_LABELS:-384} &
wait
run_audit scalar contact 0 32 384 &
run_audit v36 contact 1 ${AUDIT_TARGETS:-32} ${AUDIT_LABELS:-384} &
wait

if [[ "${RUN_NEAR_CHALLENGE_ABLATION:-0}" == "1" ]]; then
  RUN_NEAR_CHALLENGE=true RUN_NEAR_BRAKE_RESCUE=false run_eval v36 near_contact 0
  RUN_NEAR_CHALLENGE=true RUN_NEAR_BRAKE_RESCUE=false run_audit v36 near_contact 0 ${AUDIT_TARGETS:-32} ${AUDIT_LABELS:-384}
fi

if [[ "${RUN_DEEP_CONTACT:-0}" == "1" ]]; then
  run_audit v36 contact 0 ${DEEP_CONTACT_TARGETS:-48} ${DEEP_CONTACT_LABELS:-640}
fi

run_safe_closed_loop
summarize
