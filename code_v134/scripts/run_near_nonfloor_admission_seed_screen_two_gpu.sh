#!/usr/bin/env bash
# V48.124.10.6 diagnostic-only non-floor admission seed screen.
# Runs only the clean pre-first-trigger seed scenes localized by 10.5.  It is a
# privileged falsification screen, not a deployable algorithm or population GO test.
set -Eeuo pipefail
ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_NONFLOOR_SCREEN_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"; export OCRAP_NONFLOOR_SCREEN_RUN_ID="$RUN_ID"
if [[ "${OCRAP_NONFLOOR_SCREEN_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_6_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env OCRAP_NONFLOOR_SCREEN_SNAPSHOT_ACTIVE=1 OCRAP_ORIGIN_REPO="$ORIGIN_REPO" OCRAP_REPO="$SNAPSHOT_REPO" \
    OCRAP_NONFLOOR_SCREEN_RUN_ID="$RUN_ID" OCRAP_CONSTRAINT_AUDIT_MODE=nonfloor_admission_screen \
    BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" bash "$SNAPSHOT_REPO/scripts/run_near_nonfloor_admission_seed_screen_two_gpu.sh"
fi
REPO="${OCRAP_REPO:?snapshot repo missing}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

SUPPORT_ZIP="${OCRAP_V48124105_RESULTS_ZIP:-$BASE_OUT/OC-RAP-v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION-results.zip}"
OUT_ROOT="${OCRAP_NONFLOOR_SCREEN_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_6_nonfloor_admission_seed_screen}"; OUT="$OUT_ROOT/$RUN_ID"; REF="$OUT/reference"
mkdir -p "$OUT" "$REF" "$OUT/screen/balanced" "$OUT/screen/precision" "$OUT/comparisons" "$OUT/provenance" "$OUT/cohort"
SHARED_JAX_CACHE="${OCRAP_NONFLOOR_SCREEN_JAX_CACHE:-$BASE_OUT/.jax_compilation_cache/ocrap_nonfloor_admission_seed_screen}"; mkdir -p "$SHARED_JAX_CACHE"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN-results.zip"
RESULTS_MANIFEST="$OUT/OC-RAP-v48.124.10.6-result-bundle-manifest.json"
FINAL="$OUT/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN.json"
package_results(){ local rc="$1"; python tools/package_near_nonfloor_admission_screen_results.py --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST" --exit-code "$rc"; }
on_exit(){ local rc=$?; trap - EXIT; package_results "$rc" || true; exit "$rc"; }; trap on_exit EXIT

python tools/package_runtime_source_snapshot.py --repo "$REPO" --output "$OUT/provenance/runtime_source_snapshot.zip" --manifest "$OUT/provenance/runtime_source_snapshot_manifest.json"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"
python tools/materialize_near_nonfloor_admission_screen_reference.py --support-results-zip "$SUPPORT_ZIP" --output-dir "$REF"
python tools/check_near_nonfloor_admission_screen_contract.py --repo "$REPO" --reference-source-manifest "$REF/provenance/runtime_source_snapshot_manifest.json" --output "$OUT/provenance/nonfloor_screen_runtime_contract.json"
python tools/build_near_nonfloor_seed_cohort.py \
  --balanced-audit "$REF/audit/balanced/closed_loop_ocrap.json" --precision-audit "$REF/audit/precision/closed_loop_ocrap.json" \
  --adjudication "$REF/OC-RAP-v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION.json" \
  --output "$OUT/cohort/nonfloor_seed_plan.json" --target-keys-output "$OUT/cohort/nonfloor_seed_target_keys.json"

readarray -t META < <(python - \
  "$REF/reference/reference/v48124102/base/balanced/closed_loop_ocrap.json" \
  "$REF/reference/reference/v48124102/base/precision/closed_loop_ocrap.json" \
  "$REF/reference/reference/v48124102/support/near_dataset_support.json" \
  "$REF/reference/reference/v48124102/provenance/frozen_checkpoint_contract.json" <<'PY'
import json,sys
b,p,s,c=[json.load(open(x)) for x in sys.argv[1:]]
assert int(b['num_scenes'])==int(p['num_scenes'])==250 and b['bucket_dataset']==p['bucket_dataset']==s['dataset']
assert s['raw_source_role']=='validation' and c['valid'] and c['attribution_ready']
print(s['womd_pattern']); print(b['bucket_dataset']); print(b['gamma_rec']); print(p['gamma_rec'])
for v in ('balanced','precision'):
 print(c['checkpoints'][v]['actual']['path']); print(c['checkpoints'][v]['expected']['sha256']); print(c['checkpoints'][v]['expected']['size'])
PY
)
WOMD_VAL="${META[0]}"; BUCKET="${META[1]}"; BGAMMA="${META[2]}"; PGAMMA="${META[3]}"
BCKPT="${META[4]}"; BSH="${META[5]}"; BSZ="${META[6]}"; PCKPT="${META[7]}"; PSH="${META[8]}"; PSZ="${META[9]}"
python - "$BCKPT" "$PCKPT" "$BSH" "$PSH" "$BSZ" "$PSZ" "$OUT/provenance/current_checkpoint_contract.json" <<'PY'
import hashlib,json,pathlib,sys
b,p,bsh,psh,bsz,psz,out=sys.argv[1:]; rows={}
for name,path,want_sha,want_size in [('balanced',b,bsh,int(bsz)),('precision',p,psh,int(psz))]:
 q=pathlib.Path(path)
 if not q.is_file(): raise SystemExit(f'missing frozen checkpoint {q}')
 raw=q.read_bytes(); got=hashlib.sha256(raw).hexdigest(); size=len(raw); ok=(got==want_sha and size==want_size)
 rows[name]={'path':str(q.resolve()),'sha256':got,'size':size,'expected_sha256':want_sha,'expected_size':want_size,'match':ok}
 if not ok: raise SystemExit(f'frozen checkpoint mismatch {name}')
d={'schema':'ocrap-v48.124.10.6-current-checkpoint-contract-v1','valid':True,'attribution_ready':True,'checkpoints':rows}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); print(json.dumps(d,indent=2,sort_keys=True))
PY

SUPPORT="$REF/reference/reference/v48124102/support/near_dataset_support.json"; KEYS="$OUT/cohort/nonfloor_seed_target_keys.json"; PLAN="$OUT/cohort/nonfloor_seed_plan.json"
BBASE="$REF/reference/reference/v48124102/base/balanced/closed_loop_ocrap.json"; PBASE="$REF/reference/reference/v48124102/base/precision/closed_loop_ocrap.json"
NOMINAL="$REF/reference/reference/v48124102/nominal/near/closed_loop_nominal.json"
NSEED="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["target_keys"]))' "$KEYS")"
[[ "$NSEED" -ge 1 && "$NSEED" -le 8 ]] || { echo "invalid nonfloor seed scene count=$NSEED" >&2; exit 30; }

run_screen(){
 local variant="$1" ckpt="$2" gamma="$3" gpu="$4" dir="$5"; mkdir -p "$dir"
 env RUN_DIR="$dir" OUTPUT="$dir/closed_loop_ocrap.json" WOMD_VAL="$WOMD_VAL" EXPECTED_WOMD_ROLE=validation \
   CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" MAX_SCENARIOS=0 MAX_STEPS=40 REPLAN_INTERVAL=1 \
   LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 AUDIT_MAX_LABELS=0 AUDIT_AUTO_MAX_LABELS=0 \
   NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 BUCKET_DATASET="$BUCKET" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 \
   TARGET_KEYS_FILE="$KEYS" REQUIRE_TARGET_KEYS=true PREFLIGHT_SUPPORT_JSON="$SUPPORT" \
   RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=full SCENE_JOURNAL_DETAIL=full MEMORY_SCENE_DETAIL=full \
   PARTIAL_WRITE_EVERY_SCENES="$NSEED" PROGRESS_EVERY_STEPS=20 PROFILE_TIMING=true JAX_CACHE_DIR="$SHARED_JAX_CACHE" \
   SCAN_PREFIX_ROLLOUTS=true BATCH_TEACHER_OPTION_ROLLOUTS=true VALIDATE_JIT_PREFIX_ROLLOUT=true VALIDATE_BATCHED_TEACHER_METRICS=true BATCH_TEACHER_VALIDATION_ATOL=1e-6 \
   USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false \
   PRIVILEGED_PCD_ORACLE_CEILING=false PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING=true PRIVILEGED_NONFLOOR_PCD_ORACLE_EPSILON=1e-6 \
   PRIVILEGED_NONFLOOR_RDEP_FLOOR=0.5 PRIVILEGED_NONFLOOR_RDEP_TOL=1e-8 PRIVILEGED_NONFLOOR_SEED_PLAN_FILE="$PLAN" PRIVILEGED_NONFLOOR_REQUIRE_NOMINAL_BEFORE_START=true \
   bash scripts/run_ocrap_closed_loop.sh
}
run_screen balanced "$BCKPT" "$BGAMMA" "$GPU0" "$OUT/screen/balanced" & PB=$!
run_screen precision "$PCKPT" "$PGAMMA" "$GPU1" "$OUT/screen/precision" & PP=$!
set +e; wait "$PB"; RB=$?; wait "$PP"; RP=$?; set -e
[[ $RB == 0 && $RP == 0 ]] || { echo "nonfloor seed screen failed balanced=$RB precision=$RP" >&2; exit 30; }

BS="$OUT/screen/balanced/closed_loop_ocrap.json"; PS="$OUT/screen/precision/closed_loop_ocrap.json"
python tools/audit_observation_legal_route_results.py --result balanced="$BS" --result precision="$PS" --output "$OUT/provenance/route_audit.json"
BN="$OUT/comparisons/balanced_screen_vs_nominal.json"; PN="$OUT/comparisons/precision_screen_vs_nominal.json"
BB="$OUT/comparisons/balanced_screen_vs_base.json"; PBAS="$OUT/comparisons/precision_screen_vs_base.json"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$BS" --bootstrap 5000 --seed 2027 --output "$BN"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$PS" --bootstrap 5000 --seed 2027 --output "$PN"
python tools/compare_paired_closed_loop.py "$BBASE" "$BS" --bootstrap 5000 --seed 2027 --output "$BB"
python tools/compare_paired_closed_loop.py "$PBASE" "$PS" --bootstrap 5000 --seed 2027 --output "$PBAS"
python tools/adjudicate_near_nonfloor_admission_screen.py \
 --reference-contract "$REF/REFERENCE_CONTRACT.json" --runtime-contract "$OUT/provenance/nonfloor_screen_runtime_contract.json" --seed-cohort "$PLAN" \
 --nominal-full "$NOMINAL" --route-audit "$OUT/provenance/route_audit.json" \
 --balanced-result "$BS" --balanced-vs-nominal "$BN" --balanced-vs-base "$BB" \
 --precision-result "$PS" --precision-vs-nominal "$PN" --precision-vs-base "$PBAS" --output "$FINAL"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
cp -f "$FINAL" "$BASE_OUT/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN.json"
package_results 0; trap - EXIT
echo "V48.124.10.6 non-floor admission seed screen complete: $RESULTS_ZIP"
