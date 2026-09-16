#!/usr/bin/env bash
# V48.124.10.7 terminal diagnostic-only one-shot action-realization ceiling.
# Executes exactly one preregistered strongest non-floor seed action per seed scene
# from the untouched exact-nominal trajectory, then returns to exact nominal.
# No model/admission/threshold/recovery mechanism is changed or trained.
set -Eeuo pipefail
ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_ONE_SHOT_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"; export OCRAP_ONE_SHOT_RUN_ID="$RUN_ID"
if [[ "${OCRAP_ONE_SHOT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_7_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env OCRAP_ONE_SHOT_SNAPSHOT_ACTIVE=1 OCRAP_ORIGIN_REPO="$ORIGIN_REPO" OCRAP_REPO="$SNAPSHOT_REPO" \
    OCRAP_ONE_SHOT_RUN_ID="$RUN_ID" OCRAP_CONSTRAINT_AUDIT_MODE=one_shot_action_realization \
    BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" bash "$SNAPSHOT_REPO/scripts/run_near_nonfloor_one_shot_realization_two_gpu.sh"
fi
REPO="${OCRAP_REPO:?snapshot repo missing}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

PREV_ZIP="${OCRAP_V48124106_RESULTS_ZIP:-$BASE_OUT/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN-results.zip}"
OUT_ROOT="${OCRAP_ONE_SHOT_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_7_one_shot_action_realization}"; OUT="$OUT_ROOT/$RUN_ID"; REF="$OUT/reference"
mkdir -p "$OUT" "$REF" "$OUT/one_shot/balanced" "$OUT/one_shot/precision" "$OUT/comparisons" "$OUT/provenance"
SHARED_JAX_CACHE="${OCRAP_ONE_SHOT_JAX_CACHE:-$BASE_OUT/.jax_compilation_cache/ocrap_one_shot_action_realization}"; mkdir -p "$SHARED_JAX_CACHE"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip"
RESULTS_MANIFEST="$OUT/OC-RAP-v48.124.10.7.1-result-bundle-manifest.json"
FINAL="$OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION.json"
package_results(){ local rc="$1"; python tools/package_near_nonfloor_one_shot_results.py --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST" --exit-code "$rc"; }
on_exit(){ local rc=$?; trap - EXIT; package_results "$rc" || true; exit "$rc"; }; trap on_exit EXIT

python tools/package_runtime_source_snapshot.py --repo "$REPO" --output "$OUT/provenance/runtime_source_snapshot.zip" --manifest "$OUT/provenance/runtime_source_snapshot_manifest.json"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"
python - "$PREV_ZIP" "$REF" <<'PY'
import json,sys,zipfile,pathlib,hashlib
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
if not src.is_file(): raise SystemExit(f'missing V48.124.10.6 results zip: {src}')
with zipfile.ZipFile(src) as z: z.extractall(out)
adj=json.load(open(out/'OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN.json'))
assert adj.get('valid') and adj.get('attribution_ready') and adj.get('status')=='NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING'
assert adj.get('algorithm_modified') is False
man=json.load(open(out/'OC-RAP-v48.124.10.6-result-bundle-manifest.json'))
for rel,row in (man.get('files') or {}).items():
    p=out/rel
    if not p.is_file(): raise SystemExit(f'10.6 manifest missing {rel}')
    raw=p.read_bytes()
    if len(raw)!=int(row['size']) or hashlib.sha256(raw).hexdigest()!=row['sha256']: raise SystemExit(f'10.6 manifest mismatch {rel}')
print(json.dumps({'valid':True,'predecessor_status':adj['status'],'verified_files':len(man.get('files') or {})},indent=2))
PY

PLAN="$REF/cohort/nonfloor_seed_plan.json"; KEYS="$REF/cohort/nonfloor_seed_target_keys.json"
NOMINAL="$REF/reference/reference/reference/v48124102/nominal/near/closed_loop_nominal.json"
SUPPORT="$REF/reference/reference/reference/v48124102/support/near_dataset_support.json"
readarray -t META < <(python - \
  "$REF/reference/reference/reference/v48124102/base/balanced/closed_loop_ocrap.json" \
  "$REF/reference/reference/reference/v48124102/base/precision/closed_loop_ocrap.json" \
  "$SUPPORT" "$REF/reference/reference/reference/v48124102/provenance/frozen_checkpoint_contract.json" <<'PY'
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
d={'schema':'ocrap-v48.124.10.7-current-checkpoint-contract-v1','valid':True,'attribution_ready':True,'checkpoints':rows}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
NSEED="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["target_keys"]))' "$KEYS")"
[[ "$NSEED" == 2 ]] || { echo "10.7 requires the exact 2-scene 10.6 seed cohort; got $NSEED" >&2; exit 30; }

run_one_shot(){
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
   PRIVILEGED_PCD_ORACLE_CEILING=false PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING=false PRIVILEGED_NONFLOOR_ONE_SHOT_SEED_REALIZATION=true \
   PRIVILEGED_NONFLOOR_PCD_ORACLE_EPSILON=1e-6 PRIVILEGED_NONFLOOR_RDEP_FLOOR=0.5 PRIVILEGED_NONFLOOR_RDEP_TOL=1e-8 \
   PRIVILEGED_NONFLOOR_SEED_PLAN_FILE="$PLAN" PRIVILEGED_NONFLOOR_REQUIRE_NOMINAL_BEFORE_START=true \
   bash scripts/run_ocrap_closed_loop.sh
}
run_one_shot balanced "$BCKPT" "$BGAMMA" "$GPU0" "$OUT/one_shot/balanced" & PB=$!
run_one_shot precision "$PCKPT" "$PGAMMA" "$GPU1" "$OUT/one_shot/precision" & PP=$!
set +e; wait "$PB"; RB=$?; wait "$PP"; RP=$?; set -e
[[ $RB == 0 && $RP == 0 ]] || { echo "one-shot realization failed balanced=$RB precision=$RP" >&2; exit 30; }

BS="$OUT/one_shot/balanced/closed_loop_ocrap.json"; PS="$OUT/one_shot/precision/closed_loop_ocrap.json"
python tools/audit_observation_legal_route_results.py --result balanced="$BS" --result precision="$PS" --output "$OUT/provenance/route_audit.json"
BN="$OUT/comparisons/balanced_one_shot_vs_nominal.json"; PN="$OUT/comparisons/precision_one_shot_vs_nominal.json"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$BS" --bootstrap 5000 --seed 2027 --output "$BN"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$PS" --bootstrap 5000 --seed 2027 --output "$PN"
python tools/adjudicate_near_nonfloor_one_shot_realization.py \
 --predecessor-adjudication "$REF/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN.json" --seed-cohort "$PLAN" \
 --nominal-full "$NOMINAL" --route-audit "$OUT/provenance/route_audit.json" \
 --balanced-result "$BS" --balanced-vs-nominal "$BN" --precision-result "$PS" --precision-vs-nominal "$PN" --output "$FINAL"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
cp -f "$FINAL" "$BASE_OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION.json"
package_results 0; trap - EXIT
echo "V48.124.10.7.1 one-shot action-realization ceiling complete: $RESULTS_ZIP"
