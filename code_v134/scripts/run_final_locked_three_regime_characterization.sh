#!/usr/bin/env bash
# V48.124.10.7.4 final immutable characterization (evaluation-contract fix).
# This freezes the current Main for reporting; it does NOT convert the historical
# V48.124 deployment-acceptance Near STOP into GO.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
TERMINAL_CLOSURE="${OCRAP_TERMINAL_CLOSURE_JSON:-$BASE_OUT/OC-RAP-v48.124.10.7.2-TERMINAL-INTERNAL-CLOSURE.json}"
LOCK_JSON="${OCRAP_FINAL_EVALUATION_LOCK_JSON:-$BASE_OUT/OC-RAP-v48.124.10.7.3-FINAL-EVALUATION-LOCK.json}"
OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
MAX_STEPS="${MAX_STEPS:-40}"
RUN_NOMINAL="${RUN_NOMINAL:-1}"
PROFILE_LATENCY="${PROFILE_LATENCY:-true}"
BUILD_TARGET_LOCKS="${BUILD_TARGET_LOCKS:-true}"
LATENCY_GPU="${LATENCY_GPU:-$GPU0}"
OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"
METRIC_SEMANTICS_VERSION="${METRIC_SEMANTICS_VERSION:-publication_v55_signed_clearance_unclipped_v1}"

[[ -s "$TERMINAL_CLOSURE" ]] || { echo "missing terminal closure: $TERMINAL_CLOSURE" >&2; exit 30; }
python tools/create_final_evaluation_lock.py \
  --terminal-closure "$TERMINAL_CLOSURE" \
  --model-run "$MODEL_RUN" \
  --repo "$REPO" \
  --output "$LOCK_JSON"

mkdir -p "$OUT/ocrap/balanced" "$OUT/ocrap/precision" "$OUT/nominal" "$OUT/target_keys" "$OUT/paired_target_keys"

# Freeze the method-independent observation-legal cohort before launching any
# planner. This makes external baselines runnable in parallel with OC-RAP and
# prevents a malformed WOMD sdc_paths record from aborting the full suite.
if [[ "${BUILD_TARGET_LOCKS,,}" == true || "${BUILD_TARGET_LOCKS,,}" == 1 || "${BUILD_TARGET_LOCKS,,}" == yes ]]; then
  env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OUT" WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$MAX_STEPS" \
    bash scripts/build_final_observation_legal_target_locks.sh
else
  for regime in safe near contact; do
    [[ -s "$OUT/target_keys/$regime.json" ]] || { echo "missing target lock with BUILD_TARGET_LOCKS=false: $OUT/target_keys/$regime.json" >&2; exit 30; }
  done
  [[ -s "$OUT/contact_anchor/contact_anchor_manifest.json" ]] || { echo "missing Contact anchor manifest with BUILD_TARGET_LOCKS=false: $OUT/contact_anchor/contact_anchor_manifest.json" >&2; exit 30; }
fi

run_variant() {
  local variant="$1" gpu="$2"
  env WOMD_ROLE="$WOMD_ROLE" MODEL_RUN="$MODEL_RUN" MODEL_VARIANT="$variant" \
    OUT="$OUT/ocrap/$variant" CUDA_DEVICES="$gpu" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
    SAFE_TARGET_KEYS_FILE="$OUT/target_keys/safe.json" \
    NEAR_TARGET_KEYS_FILE="$OUT/target_keys/near.json" \
    CONTACT_TARGET_KEYS_FILE="$OUT/target_keys/contact.json" \
    CONTACT_ANCHOR_PRELUDE_ENABLED=true \
    CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
    CONTACT_ANCHOR_MANIFEST_FILE="$OUT/contact_anchor/contact_anchor_manifest.json" \
    RESUME=false RESUME_FORCE=false SKIP_COMPLETE_REGIMES=false \
    METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    INCLUDE_SCENES_IN_RESULT=false RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics \
    RENDER_SAFE=false RENDER_NEAR=false RENDER_CONTACT=false \
    bash scripts/run_ocrap_three_regime_evaluation.sh
}

set +e
run_variant balanced "$GPU0" & p0=$!
run_variant precision "$GPU1" & p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "final OC-RAP characterization failed balanced=$r0 precision=$r1" >&2; exit 30; }

if [[ "$RUN_NOMINAL" == 1 ]]; then
  env WOMD_ROLE="$WOMD_ROLE" OUT="$OUT/nominal" CUDA_DEVICES="$GPU0,$GPU1" \
    SAFE_TARGET_KEYS_FILE="$OUT/target_keys/safe.json" \
    NEAR_TARGET_KEYS_FILE="$OUT/target_keys/near.json" \
    CONTACT_TARGET_KEYS_FILE="$OUT/target_keys/contact.json" \
    CONTACT_ANCHOR_PRELUDE_ENABLED=true \
    CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
    CONTACT_ANCHOR_MANIFEST_FILE="$OUT/contact_anchor/contact_anchor_manifest.json" \
    RESUME=false RESUME_FORCE=false FORCE_RERUN=true METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
    INCLUDE_SCENES_IN_RESULT=false RESULT_SCENE_DETAIL=metrics \
    bash scripts/run_nominal_three_regime_control.sh
fi


if [[ "${PROFILE_LATENCY,,}" == true || "${PROFILE_LATENCY,,}" == 1 || "${PROFILE_LATENCY,,}" == yes ]]; then
  env BASE_OUT="$BASE_OUT" MODEL_RUN="$MODEL_RUN" OCRAP_FINAL_CHARACTERIZATION_OUT="$OUT" \
    OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT" LATENCY_GPU="$LATENCY_GPU" WOMD_ROLE="$WOMD_ROLE" \
    MAX_STEPS="$MAX_STEPS" METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    CONTACT_ANCHOR_MANIFEST_FILE="$OUT/contact_anchor/contact_anchor_manifest.json" \
    bash scripts/profile_ocrap_latency.sh all
fi

for regime in safe near contact; do
  args=(
    --input "balanced=$OUT/ocrap/balanced/$regime/closed_loop_ocrap.json"
    --input "precision=$OUT/ocrap/precision/$regime/closed_loop_ocrap.json"
  )
  if [[ "$RUN_NOMINAL" == 1 ]]; then
    args+=(--input "nominal=$OUT/nominal/$regime/closed_loop_nominal.json")
  fi
  python tools/export_paired_target_keys.py "${args[@]}" --output "$OUT/paired_target_keys/$regime.json"
  python tools/check_target_key_lock.py \
    --expected "$OUT/target_keys/$regime.json" \
    --observed "$OUT/paired_target_keys/$regime.json"
done

python - "$LOCK_JSON" "$OUT" "$OCRAP_LATENCY_OUT" <<'PY'
import hashlib,json,pathlib,sys
lock=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); latency_root=pathlib.Path(sys.argv[3])
def rec(p):
    p=pathlib.Path(p); return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}
rows={}
for variant in ('balanced','precision'):
    rows[variant]={}
    for regime in ('safe','near','contact'):
        p=root/'ocrap'/variant/regime/'closed_loop_ocrap.json'; rows[variant][regime]=rec(p)
nom={}
for regime in ('safe','near','contact'):
    p=root/'nominal'/regime/'closed_loop_nominal.json'
    if p.is_file(): nom[regime]=rec(p)
keys={r:rec(root/'target_keys'/f'{r}.json') for r in ('safe','near','contact')}
anchor_path=root/'contact_anchor'/'contact_anchor_manifest.json'
latency_path=latency_root/'LATENCY_INDEX.json'
out={
 'schema':'ocrap-v48.124.10.7.4-final-characterization-index-v1',
 'status':'FINAL_LOCKED_CHARACTERIZATION_COMPLETE',
 'evaluation_lock':rec(lock),
 'ocrap':rows,'nominal':nom,'paired_target_keys':keys,
 'contact_anchor_manifest':rec(anchor_path) if anchor_path.is_file() else None,
 'isolated_latency_index':rec(latency_path) if latency_path.is_file() else None,
 'claim_scope':'final locked characterization only; deployment-acceptance Near STOP is unchanged',
 'next':'run paired external baselines on exactly these target-key files, then build comparison tables',
}
(root/'FINAL_CHARACTERIZATION_INDEX.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':out['status'],'index':str(root/'FINAL_CHARACTERIZATION_INDEX.json')}))
PY

cat <<EOF
Final locked OC-RAP characterization complete:
  $OUT/FINAL_CHARACTERIZATION_INDEX.json
Target-key locks for external baselines:
  $OUT/target_keys/safe.json
  $OUT/target_keys/near.json
  $OUT/target_keys/contact.json
Deployment-acceptance freeze remains NO; no further tuning is authorized.
EOF
