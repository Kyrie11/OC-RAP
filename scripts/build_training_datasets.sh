#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_TRAIN:=${WOMD_ROOT}/training/training_tfexample.tfrecord}"

REGIME=all
while (($#)); do
  case "$1" in
    --regime) REGIME="${2:?missing value for --regime}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/build_training_datasets.sh [--regime all|safe|near|contact]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OCRAP_ROOT"

build_safe() {
  echo "[DATA] building Safe train bucket from WOMD training"
  python -m ocrap.cli build-dataset   --set data_source=womd   --set simulation_backend=waymax_closed_loop   --set womd_patterns="${WOMD_TRAIN}@1000"   --set max_scenarios=800   --set split.force_id=train   --set max_times_per_scenario=3   --set max_biased_times_per_scenario=0   --set dataset_quality.min_uniform_times_per_scenario=3   --set num_candidate_prefixes=24   --set num_reactive_futures=2   --set num_targeted_futures=0   --set num_roots=8   --set num_recovery_options=12   --set waymax.compute_future_metrics=false   --set waymax.teacher_backend=hybrid   --set waymax.teacher_rollout_top_k_options=4   --set waymax.enable_augmented_hidden_roots=false   --set waymax.enable_visible_perturbation_roots=false   --set artifact.force_mine=false   --set artifact.mine_probability=0.0   --set artifact.use_margin_override=false   --set dataset_quality.balanced_two_pass=false   --set dataset_quality.artifact_pair_mode=tag   --set dataset_quality.max_accepted_prefixes_per_scene_time=8   --set dataset_quality.require_nominal_per_scene_time=true   --set dataset_quality.keep_nominal_even_if_quality_fails=true   --set dataset_quality.min_accepted_prefixes_per_scene_time=2   --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact]'   --set 'dataset_quality.forbid_any_regimes=[oracle_artifact]'   --set regime_thresholds.tau_occ=0.75   --set regime_thresholds.tau_normal_occ=0.90   --set regime_thresholds.include_prefix_collision_in_near=false   --set regime_thresholds.include_prefix_contact_in_post=false   --set regime_thresholds.use_paper_regime_definitions=true   --set io.compress_npz=false   --set io.fsync_npz=false   --output "$OCRAP_ROOT/train_safe"
}

build_near() {
  echo "[DATA] building Near-Contact train bucket from WOMD training"
  python -m ocrap.cli build-dataset   --set data_source=womd   --set simulation_backend=waymax_closed_loop   --set womd_patterns="${WOMD_TRAIN}@1000"   --set max_scenarios=600   --set split.force_id=train   --set max_times_per_scenario=3   --set max_biased_times_per_scenario=3   --set num_candidate_prefixes=24   --set num_reactive_futures=2   --set num_targeted_futures=8   --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,low_friction_braking,control_delay_noise]'   --set num_roots=8   --set num_recovery_options=12   --set waymax.compute_future_metrics=true   --set waymax.teacher_backend=hybrid   --set waymax.teacher_rollout_top_k_options=0   --set waymax.teacher_metrics_stride=0   --set waymax.use_jit_scan_rollouts=true   --set waymax.enable_augmented_hidden_roots=true   --set waymax.augmented_hidden_from_unknown_only=true   --set waymax.enable_visible_perturbation_roots=true   --set artifact.force_mine=true   --set artifact.mine_probability=0.30   --set artifact.use_margin_override=false   --set artifact.enable_branch_intent_margin=true   --set artifact.branch_intent_compatible_margin=1.0   --set artifact.branch_intent_incompatible_margin=-2.5   --set dataset_quality.balanced_two_pass=true   --set dataset_quality.artifact_pair_mode=balanced   --set dataset_quality.artifact_quota_uses_label=true   --set dataset_quality.max_accepted_prefixes_per_scene_time=8   --set dataset_quality.min_artifact_prefixes_per_scene_time=1   --set dataset_quality.max_artifact_prefixes_per_scene_time=2   --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4   --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6   --set dataset_quality.max_artifact_attempts_per_scene_time=24   --set dataset_quality.max_nonartifact_attempts_per_scene_time=12   --set dataset_quality.require_nominal_per_scene_time=true   --set dataset_quality.keep_nominal_even_if_quality_fails=true   --set dataset_quality.min_accepted_prefixes_per_scene_time=2   --set io.compress_npz=false   --set io.fsync_npz=false   --output "$OCRAP_ROOT/train_near_contact"
}

build_contact() {
  echo "[DATA] building Contact train bucket from WOMD training"
  python -m ocrap.cli build-dataset   --set data_source=womd   --set simulation_backend=waymax_closed_loop   --set womd_patterns="${WOMD_TRAIN}@1000"   --set max_scenarios=500   --set split.force_id=train   --set max_times_per_scenario=4   --set max_biased_times_per_scenario=4   --set num_candidate_prefixes=24   --set num_reactive_futures=2   --set num_targeted_futures=10   --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]'   --set num_roots=8   --set num_recovery_options=12   --set waymax.compute_future_metrics=true   --set waymax.teacher_backend=hybrid   --set waymax.teacher_rollout_top_k_options=0   --set waymax.teacher_metrics_stride=0   --set waymax.use_jit_scan_rollouts=true   --set waymax.enable_augmented_hidden_roots=true   --set waymax.augmented_hidden_from_unknown_only=true   --set waymax.enable_visible_perturbation_roots=true   --set artifact.force_mine=true   --set artifact.mine_probability=0.25   --set artifact.use_margin_override=false   --set artifact.enable_branch_intent_margin=true   --set artifact.branch_intent_compatible_margin=1.0   --set artifact.branch_intent_incompatible_margin=-2.5   --set dataset_quality.balanced_two_pass=true   --set dataset_quality.artifact_pair_mode=balanced   --set dataset_quality.artifact_quota_uses_label=true   --set dataset_quality.max_accepted_prefixes_per_scene_time=9   --set dataset_quality.min_artifact_prefixes_per_scene_time=1   --set dataset_quality.max_artifact_prefixes_per_scene_time=2   --set dataset_quality.min_nonartifact_prefixes_per_scene_time=5   --set dataset_quality.max_nonartifact_prefixes_per_scene_time=7   --set dataset_quality.max_artifact_attempts_per_scene_time=24   --set dataset_quality.max_nonartifact_attempts_per_scene_time=12   --set dataset_quality.require_nominal_per_scene_time=true   --set dataset_quality.keep_nominal_even_if_quality_fails=true   --set dataset_quality.min_accepted_prefixes_per_scene_time=2   --set io.compress_npz=false   --set io.fsync_npz=false   --output "$OCRAP_ROOT/train_contact"
}

case "$REGIME" in
  all) build_safe; build_near; build_contact ;;
  safe) build_safe ;;
  near|near_contact) build_near ;;
  contact) build_contact ;;
  *) echo "Invalid --regime: $REGIME" >&2; exit 2 ;;
esac
