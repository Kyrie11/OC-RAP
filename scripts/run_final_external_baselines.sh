#!/usr/bin/env bash
# Paired external-baseline characterization on the method-independent observation-legal
# target locks used by the frozen final OC-RAP characterization.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"

# Resume-repair requirement / design logic:
# - Keep the frozen closed-loop protocol and output root unchanged.
# - Make force-resume an explicit launcher option instead of accidentally treating
#   a Python-style --set token as the regime positional argument.
# - Environment variables remain authoritative for existing automation.
REGIME="all"
CLI_RESUME_FORCE=""
usage() {
  cat <<'EOF'
Usage: scripts/run_final_external_baselines.sh [safe|near|contact|all] [--resume-force]

  --resume-force      allow verified in-place continuation of a legacy/mismatched
                      resume fingerprint. The runner migrates accepted journal
                      fingerprints before appending new scenes.
  --no-resume-force   explicitly disable force-resume.

Equivalent environment override: CL_RESUME_FORCE=true
EOF
}
while (($#)); do
  case "$1" in
    safe|near|contact|all) REGIME="$1"; shift ;;
    --resume-force) CLI_RESUME_FORCE=true; shift ;;
    --no-resume-force) CLI_RESUME_FORCE=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option/regime: $1" >&2; usage >&2; exit 2 ;;
  esac
done
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final_v2}"
GPU_LIST="${GPU_LIST:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
PROFILE_LATENCY="${PROFILE_LATENCY:-true}"
LATENCY_GPU="${LATENCY_GPU:-0}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
FINAL_MAX_STEPS="${FINAL_MAX_STEPS:-40}"
METRIC_SEMANTICS_VERSION="${METRIC_SEMANTICS_VERSION:-publication_v55_signed_clearance_unclipped_v1}"
# Optional path to a previous external-baseline root. Validated learned
# checkpoints are copied into the final run so only closed-loop testing is redone.
# Missing/invalid methods (e.g. a failed GameFormer training) are trained normally.
PRETRAINED_BASELINE_ROOT="${PRETRAINED_BASELINE_ROOT:-}"
# Main closed-loop runs are publication-fingerprint safe to resume: persistence
# knobs are excluded from the fingerprint, while every result-affecting input is
# checked.  Respect explicit user overrides but default final runs to continuation
# instead of deleting/restarting partial 250-scene artifacts.
FINAL_CL_RESUME="${FINAL_CL_RESUME:-${CL_RESUME:-true}}"
FINAL_CL_RESUME_FORCE="${FINAL_CL_RESUME_FORCE:-${CLI_RESUME_FORCE:-${CL_RESUME_FORCE:-false}}}"
FINAL_SKIP_COMPLETE_METHODS="${FINAL_SKIP_COMPLETE_METHODS:-${SKIP_COMPLETE_METHODS:-true}}"

ensure_target_locks() {
  local rebuild=false
  if [[ ! -s "$OCRAP_OUT/target_keys/safe.json" || ! -s "$OCRAP_OUT/target_keys/near.json" || ! -s "$OCRAP_OUT/target_keys/contact.json" || ! -s "$OCRAP_OUT/contact_anchor/contact_anchor_manifest.json" ]]; then
    rebuild=true
  elif ! python - "$OCRAP_OUT/target_keys/safe.json" "$OCRAP_OUT/target_keys/near.json" "$OCRAP_OUT/target_keys/contact.json" "$OCRAP_OUT/contact_anchor/contact_anchor_manifest.json" "$FINAL_MAX_STEPS" <<'PY'
import hashlib,json,pathlib,sys
safe,near,contact,manifest=map(pathlib.Path,sys.argv[1:5]); horizon=int(sys.argv[5])
try:
    ds=json.loads(safe.read_text()); dn=json.loads(near.read_text()); dc=json.loads(contact.read_text()); dm=json.loads(manifest.read_text())
except Exception:
    raise SystemExit(1)
if ds.get('schema')!='ocrap-observation-legal-target-lock-v1' or dn.get('schema')!='ocrap-observation-legal-target-lock-v1': raise SystemExit(1)
if dc.get('schema')!='ocrap-observation-legal-contact-anchor-target-lock-v1': raise SystemExit(1)
cc=dc.get('contact_anchor_contract') or {}
if cc.get('protocol')!='exact_a0_pretreatment_prelude_v1' or int(cc.get('min_post_steps') or -1)!=horizon or cc.get('state_fingerprint_required') is not True: raise SystemExit(1)
if dm.get('schema')!='ocrap-contact-anchor-manifest-v1' or dm.get('valid') is not True or int(dm.get('min_post_steps') or -1)!=horizon: raise SystemExit(1)
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=cc.get('manifest_sha256'): raise SystemExit(1)
raise SystemExit(0)
PY
  then
    rebuild=true
  fi
  if [[ "$rebuild" == true ]]; then
    echo "[FINAL BASELINE] final fair target locks/Contact anchor missing or stale; rebuilding from frozen bucket+WOMD provenance"
    env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$FINAL_MAX_STEPS" \
      bash scripts/build_final_observation_legal_target_locks.sh
  fi
}
ensure_target_locks

run_one() {
  local r="$1" keyfile="$OCRAP_OUT/target_keys/$1.json"
  [[ -s "$keyfile" ]] || { echo "missing observation-legal target-key lock: $keyfile" >&2; exit 30; }
  if [[ -n "$PRETRAINED_BASELINE_ROOT" && -d "$PRETRAINED_BASELINE_ROOT/$r/checkpoints" ]]; then
    mkdir -p "$OUT/$r/checkpoints"
    cp -a "$PRETRAINED_BASELINE_ROOT/$r/checkpoints/." "$OUT/$r/checkpoints/"
    if [[ "$r" == near && -f "$PRETRAINED_BASELINE_ROOT/near/conformal_calibration.json" ]]; then
      cp -f "$PRETRAINED_BASELINE_ROOT/near/conformal_calibration.json" "$OUT/near/conformal_calibration.json"
    fi
    echo "[FINAL BASELINE] staged validated checkpoint candidates from $PRETRAINED_BASELINE_ROOT/$r/checkpoints"
  fi
  local contact_env=()
  if [[ "$r" == contact ]]; then
    local anchor_manifest="$OCRAP_OUT/contact_anchor/contact_anchor_manifest.json"
    [[ -s "$anchor_manifest" ]] || { echo "missing shared Contact anchor manifest: $anchor_manifest" >&2; exit 30; }
    contact_env=(
      CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true
      CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60
      CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1
      CL_CONTACT_ANCHOR_REQUIRE_FOUND=true
      CL_CONTACT_ANCHOR_MANIFEST_FILE="$anchor_manifest"
    )
  fi
  echo "[FINAL BASELINE] regime=$r target_keys=$keyfile"
  env CL_TARGET_KEYS_FILE="$keyfile" USE_DYNAMIC_SCHEDULER="${USE_DYNAMIC_SCHEDULER:-auto}" \
    CL_RESUME="$FINAL_CL_RESUME" CL_RESUME_FORCE="$FINAL_CL_RESUME_FORCE" SKIP_COMPLETE_METHODS="$FINAL_SKIP_COMPLETE_METHODS" \
    CL_METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    "${contact_env[@]}" \
    RUN_SUPPLEMENTARY_SAFE=true RUN_SUPPLEMENTARY_NEAR=true \
    bash scripts/run_external_baselines.sh \
      --regime "$r" --out "$OUT" --gpus "$GPU_LIST" \
      --jobs-per-gpu "$JOBS_PER_GPU" --max-parallel "$MAX_PARALLEL" \
      --max-scenarios 0 --womd-role "$WOMD_ROLE"

  if [[ "${PROFILE_LATENCY,,}" == true || "${PROFILE_LATENCY,,}" == 1 || "${PROFILE_LATENCY,,}" == yes ]]; then
    env CL_TARGET_KEYS_FILE="$keyfile" RUN_SUPPLEMENTARY_SAFE=true RUN_SUPPLEMENTARY_NEAR=true \
      CL_RESUME="${LATENCY_CL_RESUME:-false}" CL_RESUME_FORCE=false SKIP_COMPLETE_METHODS="${LATENCY_SKIP_COMPLETE_METHODS:-true}" \
      CL_METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
      "${contact_env[@]}" \
      bash scripts/profile_external_baselines_latency.sh \
        --regime "$r" --source-run "$OUT" --out "${OUT}_latency_isolated" \
        --gpu "$LATENCY_GPU" --max-scenarios 0 --womd-role "$WOMD_ROLE"
  fi
}

case "$REGIME" in
  safe|near|contact) run_one "$REGIME" ;;
  all) run_one safe; run_one near; run_one contact ;;
  *) echo "invalid regime: $REGIME" >&2; usage >&2; exit 2 ;;
esac
