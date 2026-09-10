#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
usage(){ cat <<'EOF'
Usage: scripts/run_external_baselines.sh [options]
  --regime all|safe|near|contact   default: all
  --out DIR                        default: runs/external_baselines_v48_111
  --gpus LIST                      default: 0,1
  --max-scenarios N                default: 0 (all bucket targets)
  --retrain                        force retraining learned Safe baselines
  --recalibrate                    force rebuilding Near CPSF calibration
  --offline                        also run offline metrics
  --test-only                      do not train/register/calibrate; require reusable artifacts
  --womd-role ROLE                 validation (default) or validation_interactive

Default behavior is resume-aware: learned checkpoints are reused only when best.pt is valid
and train_summary.json proves the configured epoch budget was completed. Otherwise missing/
incomplete learned checkpoints are trained, and then closed-loop testing runs.
EOF
}
REGIME=all; OUT="${OUT:-runs/external_baselines_v48_111}"; CUDA_DEVICES="${CUDA_DEVICES:-0,1}"; MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
FORCE_RETRAIN=false; FORCE_RECALIBRATE=false; DO_OFFLINE=false; TEST_ONLY=false; WOMD_ROLE="${PRIMARY_WOMD_ROLE:-validation}"
while (($#)); do case "$1" in
 --regime) REGIME="$2";shift 2;; --out) OUT="$2";shift 2;; --gpus) CUDA_DEVICES="$2";shift 2;; --max-scenarios) MAX_SCENARIOS="$2";shift 2;;
 --retrain) FORCE_RETRAIN=true;shift;; --recalibrate) FORCE_RECALIBRATE=true;shift;; --offline) DO_OFFLINE=true;shift;; --test-only) TEST_ONLY=true;shift;;
 --womd-role) WOMD_ROLE="$2";shift 2;; -h|--help) usage;exit 0;; *) echo "unknown option: $1" >&2;usage >&2;exit 2;; esac; done
case "$WOMD_ROLE" in validation|validation_interactive) ;; *) echo "invalid --womd-role $WOMD_ROLE" >&2; exit 2;; esac
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export CUDA_DEVICES MAX_SCENARIOS DO_OFFLINE DO_CLOSED_LOOP=true CL_WOMD_ROLE="$WOMD_ROLE" CALIB_WOMD_ROLE="$WOMD_ROLE" PRIMARY_WOMD_ROLE="$WOMD_ROLE"
# Unified publication entry always resolves replay from dataset provenance; stale shell
# CL_WOMD/CALIB_WOMD overrides must not silently bypass the requested WOMD role.
export CL_WOMD=auto CALIB_WOMD=auto
if [[ "$TEST_ONLY" == true ]]; then export DO_TRAIN_SAFE=false DO_TRAIN_NEAR=false DO_TRAIN_CONTACT=false DO_CALIBRATE_NEAR=false; else export DO_TRAIN_SAFE=true DO_TRAIN_NEAR=true DO_TRAIN_CONTACT=true DO_CALIBRATE_NEAR=true; fi
export FORCE_RETRAIN_ALL="$FORCE_RETRAIN" FORCE_RETRAIN_SAFE="$FORCE_RETRAIN" FORCE_REREGISTER="$FORCE_RETRAIN" FORCE_RECALIBRATE_NEAR="$FORCE_RECALIBRATE" FORCE_RECALIBRATE="$FORCE_RECALIBRATE"
case "$REGIME" in
 all) export OUT; bash scripts/run_external_baselines_all.sh;;
 safe) export RUN="$OUT/safe" DO_TRAIN="${DO_TRAIN_SAFE:-true}"; bash scripts/run_external_baselines_safe.sh;;
 near) export RUN="$OUT/near" DO_TRAIN="${DO_TRAIN_NEAR:-true}" DO_CALIBRATE="${DO_CALIBRATE_NEAR:-true}"; bash scripts/run_external_baselines_near.sh;;
 contact) export RUN="$OUT/contact" DO_TRAIN="${DO_TRAIN_CONTACT:-true}"; bash scripts/run_external_baselines_contact.sh;;
 *) echo "invalid --regime $REGIME" >&2; usage >&2; exit 2;; esac
