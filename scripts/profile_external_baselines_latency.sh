#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

usage(){ cat <<'EOF'
Usage: scripts/profile_external_baselines_latency.sh --regime safe|near|contact --source-run DIR [options]

Runs the same one-regime closed-loop suite serially with exactly one process on
one GPU. This is the supported mode for publication latency; the normal 3-jobs/
GPU launcher is a throughput launcher and intentionally does not provide an
uncontended latency measurement.

Options:
  --regime REGIME       safe|near|contact (required)
  --source-run DIR      existing multi-baseline run root, e.g. runs/external_baselines_v48_111
  --out DIR             default: <source-run>_latency_isolated
  --gpu ID              default: 0
  --max-scenarios N     default: 0 (all paired targets)
  --womd-role ROLE      default: validation
EOF
}

REGIME=""; SOURCE_RUN=""; OUT=""; GPU=0; MAX_SCENARIOS=0; WOMD_ROLE=validation
while (($#)); do
  case "$1" in
    --regime) REGIME="$2"; shift 2;;
    --source-run) SOURCE_RUN="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --max-scenarios) MAX_SCENARIOS="$2"; shift 2;;
    --womd-role) WOMD_ROLE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$REGIME" =~ ^(safe|near|contact)$ ]] || { echo "--regime is required" >&2; exit 2; }
[[ -n "$SOURCE_RUN" ]] || { echo "--source-run is required" >&2; exit 2; }
[[ -n "$OUT" ]] || OUT="${SOURCE_RUN%/}_latency_isolated"

# Reuse trained learned checkpoints, but write closed-loop timing artifacts to a
# different directory so throughput and latency runs cannot be mixed.
export CHECKPOINT_ROOT="${SOURCE_RUN%/}/${REGIME}/checkpoints"
# Near-Contact also consumes the already-fitted CPSF calibration artifact.
# Point the isolated run at the source artifact so --test-only remains fail-closed
# without refitting or mixing calibration outputs into the latency directory.
if [[ "$REGIME" == near ]]; then
  export CONFORMAL_CALIBRATION="${SOURCE_RUN%/}/near/conformal_calibration.json"
fi
export CL_PROFILE_TIMING=true
export CL_LATENCY_EXECUTION_CONTRACT=isolated_single_process_single_gpu
export CL_LATENCY_WARMUP_DECISIONS="${LATENCY_WARMUP_DECISIONS:-3}"
export SKIP_COMPLETE_METHODS=false

bash scripts/run_external_baselines.sh \
  --regime "$REGIME" \
  --out "$OUT" \
  --gpus "$GPU" \
  --jobs-per-gpu 1 \
  --max-parallel 1 \
  --max-scenarios "$MAX_SCENARIOS" \
  --test-only \
  --womd-role "$WOMD_ROLE"
