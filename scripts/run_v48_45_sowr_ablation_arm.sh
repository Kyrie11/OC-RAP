#!/usr/bin/env bash
set -Eeuo pipefail
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
if [[ "$ARM" == D ]]; then
  export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_45_sowr_main}"
else
  export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_45_sowr_ablation_${ARM}}"
fi
export RESUME_AFTER_ADAPTATION=0
unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true

# v48.45 source-run resolution must not depend on the versioned checkout cwd.
# The v48.13 reference checkpoints live under the persistent BASE_OUT used by
# the previous successful v48.44 run.  An explicit SOURCE_RUN still wins.
SOURCE_RUN_BASENAME="${V4845_SOURCE_RUN_BASENAME:-ocrap_v48_13_terra_proxy_4801}"
REBUILT_SOURCE_BASENAME="${V4845_REBUILT_SOURCE_BASENAME:-ocrap_v48_45_source_rebuild_s7}"
if [[ -z "${SOURCE_RUN:-}" ]]; then
  # After historical checkpoint loss, prefer the explicitly reconstructed,
  # hash-sealed common source. Historical v48.13 remains supported when present.
  if [[ -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/SOURCE_REBUILD_COMPLETE.json" && \
        -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
        -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$BASE_OUT/$REBUILT_SOURCE_BASENAME"
  elif [[ -f "$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$BASE_OUT/$SOURCE_RUN_BASENAME"
  elif [[ -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/SOURCE_REBUILD_COMPLETE.json" && \
          -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$REPO/runs/$REBUILT_SOURCE_BASENAME"
  elif [[ -f "$REPO/runs/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$REPO/runs/$SOURCE_RUN_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$REPO/runs/$SOURCE_RUN_BASENAME"
  else
    # Keep a deterministic canonical path so the dedicated source preflight
    # publishes an authoritative RC=30 instead of silently random-initializing.
    SOURCE_RUN="$BASE_OUT/$REBUILT_SOURCE_BASENAME"
  fi
fi
SOURCE_RUN="$(python - "$SOURCE_RUN" <<'PY_SOURCE_RUN'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY_SOURCE_RUN
)"
export SOURCE_RUN

# v48.45 holds the entire v48.44-D selector/ROCT mechanism fixed.  The 2x2
# changes only which paper-matched witness heads are recalibrated before ROCT:
#   A none; B root-probability + recovery-margin witness; C observation kernel;
#   D both.  All three dataset regimes remain evaluation/training strata only.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true
export EVIDENCE_FACTORIZED_HARM_INTERACTION=false
export EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false
export EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false
export EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
# Hold v48.44-D ROCT exactly on in all four arms; no scale/width increase.
export EVIDENCE_ROCT_BENEFIT=true
export EVIDENCE_ROCT_DEPLOYABILITY=true
export EVIDENCE_ROCT_SCALE="${EVIDENCE_ROCT_SCALE:-3.0}"
export EVIDENCE_ROCT_ALPHA="${EVIDENCE_ROCT_ALPHA:-0.20}"
export EVIDENCE_ROCT_BETA="${EVIDENCE_ROCT_BETA:-0.20}"
export EVIDENCE_ROCT_TOP_M="${EVIDENCE_ROCT_TOP_M:-8}"
export EVIDENCE_ROCT_OPTION_TEMPERATURE="${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}"
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true
export EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve
export EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export V4838_RFR_RESERVE_ONLY=1
export V4837_FACTOR_PRESERVING_IDENTITY=0
export V4836_IDENTITY_TRAIN_ALL=0
export V4836_COUPLE_ADMISSION_PRIOR=0
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT=0
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT=0
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT=0
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT=0
export PROPOSAL_TOP_K=5
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction

# SOWR defaults: intentionally short, low-LR, head-only recalibration.
export V4845_SOWR_MARGIN_WITNESS=0
export V4845_SOWR_OBS_KERNEL=0
export SOWR_EPOCHS="${SOWR_EPOCHS:-8}"
export SOWR_PATIENCE="${SOWR_PATIENCE:-3}"
export SOWR_LR="${SOWR_LR:-0.00005}"
export SOWR_BATCH_SIZE="${SOWR_BATCH_SIZE:-72}"

case "$ARM" in
  A)
    export OCRAP_ALGORITHM_VERSION="v48.45-SOWR-ablation-A"
    export OCRAP_IMPLEMENTATION_VERSION="v48.45.2-A-v48.44D-unaltered-witness-rebuilt-source-support"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.45-A-v48.44D-reference"
    ;;
  B)
    export OCRAP_ALGORITHM_VERSION="v48.45-SOWR-ablation-B"
    export OCRAP_IMPLEMENTATION_VERSION="v48.45.2-B-shared-option-margin-witness-recalibration-rebuilt-source-support"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.45-B-margin-witness-recalibration"
    export V4845_SOWR_MARGIN_WITNESS=1
    ;;
  C)
    export OCRAP_ALGORITHM_VERSION="v48.45-SOWR-ablation-C"
    export OCRAP_IMPLEMENTATION_VERSION="v48.45.2-C-observation-kernel-recalibration-rebuilt-source-support"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.45-C-observation-kernel-recalibration"
    export V4845_SOWR_OBS_KERNEL=1
    ;;
  D)
    export OCRAP_ALGORITHM_VERSION="v48.45-SOWR"
    export OCRAP_IMPLEMENTATION_VERSION="v48.45.2-D-shared-option-witness-recalibration-rebuilt-source-support"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.45-D-full-sowr"
    export V4845_SOWR_MARGIN_WITNESS=1
    export V4845_SOWR_OBS_KERNEL=1
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
exec bash scripts/run_v48_36_ocaf_dedicated.sh
