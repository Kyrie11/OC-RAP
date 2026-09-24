#!/usr/bin/env bash
# Diagnostic-only follow-up to the attribution-ready V48.124.10.2 Near STOP.
# Replays only the fresh observation-legal intervention scenes and labels all
# candidates *after* the frozen base selector has chosen its action.  No policy,
# checkpoint, candidate/recovery library, threshold, horizon, or dynamics change.
set -Eeuo pipefail
ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_CANDIDATE_AUDIT_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"; export OCRAP_CANDIDATE_AUDIT_RUN_ID="$RUN_ID"

if [[ "${OCRAP_CANDIDATE_AUDIT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_3_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env OCRAP_CANDIDATE_AUDIT_SNAPSHOT_ACTIVE=1 OCRAP_ORIGIN_REPO="$ORIGIN_REPO" OCRAP_REPO="$SNAPSHOT_REPO" OCRAP_CANDIDATE_AUDIT_RUN_ID="$RUN_ID" OCRAP_CONSTRAINT_AUDIT_MODE=candidate_quality BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" bash "$SNAPSHOT_REPO/scripts/run_near_candidate_quality_audit_two_gpu.sh"
fi

REPO="${OCRAP_REPO:?snapshot repo missing}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

REFERENCE_ZIP="${OCRAP_V48124102_RESULTS_ZIP:-$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-results.zip}"
OUT_ROOT="${OCRAP_CANDIDATE_AUDIT_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_3_candidate_quality_audit}"
OUT="$OUT_ROOT/$RUN_ID"; REF="$OUT/reference_v48124102"
# Compilation artifacts are execution caches, not scientific outputs.  Keep a
# stable cache outside the per-run UUID so retries and the two same-architecture
# A30 workers can reuse already compiled Waymax/XLA executables.  JAX cache keys
# include executable/device fingerprints, so stale code does not alias silently.
SHARED_JAX_CACHE="${OCRAP_CANDIDATE_JAX_CACHE:-$BASE_OUT/.jax_compilation_cache/ocrap_candidate_quality}"
mkdir -p "$OUT" "$REF" "$OUT/audit" "$OUT/provenance" "$SHARED_JAX_CACHE"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT-results.zip"
RESULTS_MANIFEST="$OUT/OC-RAP-v48.124.10.3-result-bundle-manifest.json"
package_results(){ local rc="$1"; python tools/package_candidate_quality_audit.py --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST" --exit-code "$rc"; }
on_exit(){ local rc=$?; trap - EXIT; package_results "$rc" || true; exit "$rc"; }
trap on_exit EXIT

python tools/package_runtime_source_snapshot.py --repo "$REPO" --output "$OUT/provenance/runtime_source_snapshot.zip" --manifest "$OUT/provenance/runtime_source_snapshot_manifest.json"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"
python tools/materialize_near_candidate_quality_reference.py --results-zip "$REFERENCE_ZIP" --output-dir "$REF"
python tools/check_candidate_quality_diagnostic_contract.py --repo "$REPO" --reference-source-manifest "$REF/provenance/runtime_source_snapshot_manifest.json" --output "$OUT/provenance/candidate_quality_runtime_contract.json"

# Extract immutable data/checkpoint/gamma provenance from the attribution-ready 10.2 reference.
readarray -t META < <(python - "$REF/base/balanced/closed_loop_ocrap.json" "$REF/base/precision/closed_loop_ocrap.json" "$REF/support/near_dataset_support.json" "$REF/provenance/frozen_checkpoint_contract.json" <<'PY'
import json,sys,hashlib,os
b,p,s,c=[json.load(open(x)) for x in sys.argv[1:]]
assert int(b['num_scenes'])==int(p['num_scenes'])==250
assert b['bucket_dataset']==p['bucket_dataset']==s['dataset']
assert s['raw_source_role']=='validation'
assert c['valid'] and c['attribution_ready']
print(s['womd_pattern'])
print(b['bucket_dataset'])
print(b['gamma_rec'])
print(p['gamma_rec'])
print(c['checkpoints']['balanced']['actual']['path'])
print(c['checkpoints']['precision']['actual']['path'])
print(c['checkpoints']['balanced']['expected']['sha256'])
print(c['checkpoints']['precision']['expected']['sha256'])
print(c['checkpoints']['balanced']['expected']['size'])
print(c['checkpoints']['precision']['expected']['size'])
PY
)
WOMD_VAL="${META[0]}"; BUCKET="${META[1]}"; BGAMMA="${META[2]}"; PGAMMA="${META[3]}"
BCKPT="${META[4]}"; PCKPT="${META[5]}"; BSH="${META[6]}"; PSH="${META[7]}"; BSZ="${META[8]}"; PSZ="${META[9]}"
python - "$BCKPT" "$PCKPT" "$BSH" "$PSH" "$BSZ" "$PSZ" "$OUT/provenance/current_checkpoint_contract.json" <<'PY'
import hashlib,json,os,pathlib,sys
b,p,bsh,psh,bsz,psz,out=sys.argv[1:]
rows={}
for name,path,want_sha,want_size in [('balanced',b,bsh,int(bsz)),('precision',p,psh,int(psz))]:
    q=pathlib.Path(path)
    if not q.is_file(): raise SystemExit(f'missing frozen checkpoint {q}')
    raw=q.read_bytes(); got=hashlib.sha256(raw).hexdigest(); size=len(raw)
    rows[name]={'path':str(q.resolve()),'sha256':got,'size':size,'expected_sha256':want_sha,'expected_size':want_size,'match':got==want_sha and size==want_size}
    if not rows[name]['match']: raise SystemExit(f'frozen checkpoint mismatch {name}')
d={'schema':'ocrap-v48.124.10.3-current-checkpoint-contract-v1','valid':True,'attribution_ready':True,'checkpoints':rows}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps(d,indent=2,sort_keys=True))
PY

SUPPORT="$REF/support/near_dataset_support.json"
BKEYS="$REF/cohort/balanced_keys.json"; PKEYS="$REF/cohort/precision_keys.json"
BBASE="$REF/base/balanced/closed_loop_ocrap.json"; PBASE="$REF/base/precision/closed_loop_ocrap.json"

run_audit(){
  local variant="$1" ckpt="$2" gamma="$3" gpu="$4" keys="$5" dir="$6"
  mkdir -p "$dir"
  env RUN_DIR="$dir" OUTPUT="$dir/closed_loop_ocrap.json" WOMD_VAL="$WOMD_VAL" EXPECTED_WOMD_ROLE=validation CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS=40 REPLAN_INTERVAL=1 LABEL_MODE=coverage AUDIT_EVERY_N_STEPS=1 AUDIT_MAX_LABELS=0 AUDIT_AUTO_MAX_LABELS=0 \
    AUDIT_INTERVENTION_ONLY=true AUDIT_CANDIDATE_SCOPE=all AUDIT_STORE_CANDIDATE_RECORDS=true \
    NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 BUCKET_DATASET="$BUCKET" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 TARGET_KEYS_FILE="$keys" REQUIRE_TARGET_KEYS=true \
    PREFLIGHT_SUPPORT_JSON="$SUPPORT" RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true \
    RESULT_SCENE_DETAIL=full SCENE_JOURNAL_DETAIL=full MEMORY_SCENE_DETAIL=full PARTIAL_WRITE_EVERY_SCENES=11 PROGRESS_EVERY_STEPS=20 PROFILE_TIMING=true \
    JAX_CACHE_DIR="$SHARED_JAX_CACHE" SCAN_PREFIX_ROLLOUTS=true BATCH_TEACHER_OPTION_ROLLOUTS=true \
    VALIDATE_JIT_PREFIX_ROLLOUT=true VALIDATE_BATCHED_TEACHER_METRICS=true BATCH_TEACHER_VALIDATION_ATOL=1e-6 \
    USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false \
    bash scripts/run_ocrap_closed_loop.sh
}

# Two A30s in parallel, one frozen robustness variant per GPU.  No same-GPU
# oversubscription: teacher counterfactual labels are the expensive part here.
run_audit balanced "$BCKPT" "$BGAMMA" "$GPU0" "$BKEYS" "$OUT/audit/balanced" & PB=$!
run_audit precision "$PCKPT" "$PGAMMA" "$GPU1" "$PKEYS" "$OUT/audit/precision" & PP=$!
set +e
wait "$PB"; RB=$?; wait "$PP"; RP=$?
set -e
[[ $RB == 0 && $RP == 0 ]] || { echo "candidate-quality audit failed balanced=$RB precision=$RP" >&2; exit 30; }
BAUD="$OUT/audit/balanced/closed_loop_ocrap.json"; PAUD="$OUT/audit/precision/closed_loop_ocrap.json"

python tools/audit_observation_legal_route_results.py --result balanced="$BAUD" --result precision="$PAUD" --output "$OUT/provenance/route_audit.json"
FINAL="$OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json"
python tools/adjudicate_near_candidate_quality_audit.py \
  --reference-adjudication "$REF/OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json" \
  --base-balanced "$BBASE" --base-precision "$PBASE" \
  --audit-balanced "$BAUD" --audit-precision "$PAUD" --output "$FINAL"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
cp -f "$FINAL" "$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json"
package_results 0
trap - EXIT
echo "V48.124.10.3 candidate-quality diagnostic complete: $RESULTS_ZIP"
