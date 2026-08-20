#!/usr/bin/env bash
set -Eeuo pipefail
# Build a scene-disjoint strict-min-slack shadow calibration protocol, rebuild the
# teacher indexes, and run TCSA.  It deliberately does not touch test roots.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHADOW_ROOT="${SHADOW_ROOT:-/data0/senzeyu2/dataset/OCRAP_v48_56_strict_teacher}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$SHADOW_ROOT/calibration_v48_14_prism_4814}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RUN_BUILD="${V4856_RUN_BUILD:-1}"
INDEX_PROGRESS_EVERY="${V4856_INDEX_PROGRESS_EVERY:-250}"

phase(){ printf '[v48.56][%s] %s\n' "$(date -Iseconds)" "$*"; }

if [[ "$RUN_BUILD" == 1 ]]; then
  phase "building strict-min-slack shadow datasets"
  OUTPUT_ROOT="$SHADOW_ROOT" GPU0="$GPU0" GPU1="$GPU1" \
    bash scripts/build_v48_56_strict_teacher_calibration_shadow.sh
fi

phase "preparing scene-disjoint calibration protocol"
OCRAP_ROOT="$SHADOW_ROOT" PROTOCOL_ROOT="$PROTOCOL_ROOT" \
CAL_SAFE="$SHADOW_ROOT/calibration_safe" \
CAL_NEAR="$SHADOW_ROOT/calibration_near_contact" \
CAL_CONTACT="$SHADOW_ROOT/calibration_contact" \
  bash scripts/prepare_v48_45_protocol.sh

TRAIN_DATA="$PROTOCOL_ROOT/evidence_adapt_train_near_contact,$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_DATA="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact,$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_INDEX="$PROTOCOL_ROOT/V48_56_STRICT_TEACHER_TRAIN_INDEX.jsonl"
DEV_INDEX="$PROTOCOL_ROOT/V48_56_STRICT_TEACHER_DEV_INDEX.jsonl"

phase "building train teacher index (fresh OC-MERO verification is computed inline)"
"$PYTHON_BIN" tools/build_teacher_pcd_index_v48.py \
  --dataset "$TRAIN_DATA" --option-execution-semantics observation_class \
  --quality-mode warn --progress-every "$INDEX_PROGRESS_EVERY" --output "$TRAIN_INDEX" \
  --summary-output "$PROTOCOL_ROOT/V48_56_STRICT_TEACHER_TRAIN_INDEX_SUMMARY.json"
phase "building dev teacher index (fresh OC-MERO verification is computed inline)"
"$PYTHON_BIN" tools/build_teacher_pcd_index_v48.py \
  --dataset "$DEV_DATA" --option-execution-semantics observation_class \
  --quality-mode warn --progress-every "$INDEX_PROGRESS_EVERY" --output "$DEV_INDEX" \
  --summary-output "$PROTOCOL_ROOT/V48_56_STRICT_TEACHER_DEV_INDEX_SUMMARY.json"

# The index builder already reloads every NPZ and recomputes OC-MERO from m_star.
# Re-reading the complete train+dev dataset here was a redundant third I/O pass.
# The audit consumes the inline fresh-vs-cached errors from the two indexes.
phase "running teacher-component semantic audit from inline fresh OC-MERO checks"
"$PYTHON_BIN" tools/audit_v48_56_teacher_component_semantics.py \
  --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" \
  --output "$PROTOCOL_ROOT/V48_56_STRICT_TEACHER_COMPONENT_SEMANTIC_AUDIT.json"

printf '[v48.56] strict teacher shadow READY for analysis: %s\n' "$PROTOCOL_ROOT"
printf '[v48.56] no performance Main is authorized by this script.\n'
