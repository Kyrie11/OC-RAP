#!/usr/bin/env bash
# Direct three-regime evaluation of the frozen deployed owner used by the V48.111 audit chain.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
usage(){ cat <<'EOF'
Usage: scripts/run_ocrap_evaluation.sh [options]
  --model-run DIR       required unless MODEL_RUN is set
  --variants LIST       default balanced,precision
  --out DIR             default runs/ocrap_v48_111_submission_three_regime
  --gpus LIST           default 0,1
  --max-scenarios N     default 0 (all bucket targets)
  --womd-role ROLE      default validation
This evaluator never trains, finetunes, recalibrates, or injects the CNRO audit.
EOF
}
MODEL_RUN="${MODEL_RUN:-}"; VARIANTS="${VARIANTS:-balanced,precision}"; OUT_ROOT="${OUT_ROOT:-runs/ocrap_v48_111_submission_three_regime}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"; MAX_SCENARIOS="${MAX_SCENARIOS:-0}"; WOMD_ROLE="${PRIMARY_WOMD_ROLE:-validation}"
while (($#)); do case "$1" in
 --model-run) MODEL_RUN="$2";shift 2;; --variants) VARIANTS="$2";shift 2;; --out) OUT_ROOT="$2";shift 2;; --gpus) CUDA_DEVICES="$2";shift 2;;
 --max-scenarios) MAX_SCENARIOS="$2";shift 2;; --womd-role) WOMD_ROLE="$2";shift 2;; -h|--help) usage;exit 0;; *) echo "unknown option $1" >&2;usage >&2;exit 2;; esac;done
[[ -n "$MODEL_RUN" ]] || { echo '--model-run is required' >&2; exit 2; }
case "$WOMD_ROLE" in validation|validation_interactive) ;; *) echo "invalid WOMD role: $WOMD_ROLE" >&2; exit 2;; esac
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
MAX_STEPS="${MAX_STEPS:-40}"; NUM_CANDIDATES="${NUM_CANDIDATES:-24}"; NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
mkdir -p "$OUT_ROOT"
IFS=',' read -r -a vv <<< "$VARIANTS"
for v in "${vv[@]}"; do v="$(echo "$v"|xargs)"; [[ -n "$v" ]] || continue
  python tools/check_deployable_stack.py --model-run "$MODEL_RUN" --variant "$v" --output "$OUT_ROOT/V48.111-DEPLOYABLE-STACK-${v}.json"
  MODEL_RUN="$MODEL_RUN" MODEL_VARIANT="$v" CUDA_DEVICES="$CUDA_DEVICES" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" WOMD_ROLE="$WOMD_ROLE" \
    SAFE_WOMD=auto NEAR_WOMD=auto CONTACT_WOMD=auto \
    OUT="$OUT_ROOT/$v" MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    RUN_SAFE=1 RUN_NEAR=1 RUN_CONTACT=1 SKIP_COMPLETE_REGIMES=true FINALIZE_COMPLETE_JOURNALS=true bash scripts/run_ocrap_three_regime_evaluation.sh
done
python tools/summarize_three_regime_results.py --root "$OUT_ROOT" --variants "$VARIANTS" --output-json "$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.json" --output-csv "$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.csv" --output-md "$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.md"
