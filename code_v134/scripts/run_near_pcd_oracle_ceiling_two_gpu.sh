#!/usr/bin/env bash
# V48.124.10.4 privileged PCD-oracle ceiling diagnostic.
# Purpose: answer a single causal-sufficiency question after the attribution-ready
# V48.124.10.3.1 candidate-quality localization.  This is NOT a deployable method.
# The frozen Base selector is evaluated first at every current state.  Privileged
# teacher labels are constructed only when Base already selects a non-nominal
# action.  The oracle may choose only among Base absolute-admitted candidates and
# only when execution-consistent teacher PCD is strictly better than nominal.
set -Eeuo pipefail

ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_PCD_ORACLE_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_PCD_ORACLE_RUN_ID="$RUN_ID"

if [[ "${OCRAP_PCD_ORACLE_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_4_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" \
    --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env \
    OCRAP_PCD_ORACLE_SNAPSHOT_ACTIVE=1 \
    OCRAP_ORIGIN_REPO="$ORIGIN_REPO" \
    OCRAP_REPO="$SNAPSHOT_REPO" \
    OCRAP_PCD_ORACLE_RUN_ID="$RUN_ID" \
    OCRAP_CONSTRAINT_AUDIT_MODE=pcd_oracle_ceiling \
    BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
    bash "$SNAPSHOT_REPO/scripts/run_near_pcd_oracle_ceiling_two_gpu.sh"
fi

REPO="${OCRAP_REPO:?snapshot repo missing}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

NEAR_ZIP="${OCRAP_V48124102_RESULTS_ZIP:-$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-results.zip}"
CAND_ZIP="${OCRAP_V48124103_RESULTS_ZIP:-$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT-results.zip}"
OUT_ROOT="${OCRAP_PCD_ORACLE_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_4_pcd_oracle_ceiling}"
OUT="$OUT_ROOT/$RUN_ID"
REF="$OUT/reference"
mkdir -p "$OUT" "$REF" "$OUT/oracle/balanced" "$OUT/oracle/precision" "$OUT/full" "$OUT/comparisons" "$OUT/provenance"
SHARED_JAX_CACHE="${OCRAP_PCD_ORACLE_JAX_CACHE:-$BASE_OUT/.jax_compilation_cache/ocrap_pcd_oracle_ceiling}"
mkdir -p "$SHARED_JAX_CACHE"

RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING-results.zip"
RESULTS_MANIFEST="$OUT/OC-RAP-v48.124.10.4-result-bundle-manifest.json"
FINAL="$OUT/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json"
package_results(){
  local rc="$1"
  python tools/package_near_pcd_oracle_ceiling_results.py \
    --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST" --exit-code "$rc"
}
on_exit(){ local rc=$?; trap - EXIT; package_results "$rc" || true; exit "$rc"; }
trap on_exit EXIT

# Provenance is established before any GPU work.
python tools/package_runtime_source_snapshot.py \
  --repo "$REPO" --output "$OUT/provenance/runtime_source_snapshot.zip" \
  --manifest "$OUT/provenance/runtime_source_snapshot_manifest.json"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"
python tools/materialize_near_pcd_oracle_reference.py \
  --near-results-zip "$NEAR_ZIP" \
  --candidate-results-zip "$CAND_ZIP" \
  --output-dir "$REF"
python tools/check_near_pcd_oracle_ceiling_contract.py \
  --repo "$REPO" \
  --reference-source-manifest "$REF/v48124103/provenance/runtime_source_snapshot_manifest.json" \
  --output "$OUT/provenance/pcd_oracle_runtime_contract.json"

# Resolve the exact frozen data/checkpoint/calibration identities from the 10.2
# attribution-ready reference.  Checkpoint bytes are not packaged; their SHA+size
# are revalidated on the machine before any rollout starts.
readarray -t META < <(python - \
  "$REF/v48124102/base/balanced/closed_loop_ocrap.json" \
  "$REF/v48124102/base/precision/closed_loop_ocrap.json" \
  "$REF/v48124102/support/near_dataset_support.json" \
  "$REF/v48124102/provenance/frozen_checkpoint_contract.json" <<'PY'
import json,sys
b,p,s,c=[json.load(open(x)) for x in sys.argv[1:]]
assert int(b['num_scenes'])==int(p['num_scenes'])==250
assert b['bucket_dataset']==p['bucket_dataset']==s['dataset']
assert s['raw_source_role']=='validation'
assert c['valid'] and c['attribution_ready']
print(s['womd_pattern']); print(b['bucket_dataset']); print(b['gamma_rec']); print(p['gamma_rec'])
for v in ('balanced','precision'):
    print(c['checkpoints'][v]['actual']['path'])
    print(c['checkpoints'][v]['expected']['sha256'])
    print(c['checkpoints'][v]['expected']['size'])
PY
)
WOMD_VAL="${META[0]}"; BUCKET="${META[1]}"; BGAMMA="${META[2]}"; PGAMMA="${META[3]}"
BCKPT="${META[4]}"; BSH="${META[5]}"; BSZ="${META[6]}"
PCKPT="${META[7]}"; PSH="${META[8]}"; PSZ="${META[9]}"
python - "$BCKPT" "$PCKPT" "$BSH" "$PSH" "$BSZ" "$PSZ" "$OUT/provenance/current_checkpoint_contract.json" <<'PY'
import hashlib,json,pathlib,sys
b,p,bsh,psh,bsz,psz,out=sys.argv[1:]
rows={}
for name,path,want_sha,want_size in [('balanced',b,bsh,int(bsz)),('precision',p,psh,int(psz))]:
    q=pathlib.Path(path)
    if not q.is_file(): raise SystemExit(f'missing frozen checkpoint {q}')
    raw=q.read_bytes(); got=hashlib.sha256(raw).hexdigest(); size=len(raw)
    ok=(got==want_sha and size==want_size)
    rows[name]={'path':str(q.resolve()),'sha256':got,'size':size,'expected_sha256':want_sha,'expected_size':want_size,'match':ok}
    if not ok: raise SystemExit(f'frozen checkpoint mismatch {name}')
d={'schema':'ocrap-v48.124.10.4-current-checkpoint-contract-v1','valid':True,'attribution_ready':True,'checkpoints':rows}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps(d,indent=2,sort_keys=True))
PY

SUPPORT="$REF/v48124102/support/near_dataset_support.json"
BKEYS="$REF/v48124102/cohort/balanced_keys.json"
PKEYS="$REF/v48124102/cohort/precision_keys.json"
BBASE="$REF/v48124102/base/balanced/closed_loop_ocrap.json"
PBASE="$REF/v48124102/base/precision/closed_loop_ocrap.json"
NOMINAL="$REF/v48124102/nominal/near/closed_loop_nominal.json"

run_oracle(){
  local variant="$1" ckpt="$2" gamma="$3" gpu="$4" keys="$5" dir="$6"
  mkdir -p "$dir"
  env \
    RUN_DIR="$dir" OUTPUT="$dir/closed_loop_ocrap.json" \
    WOMD_VAL="$WOMD_VAL" EXPECTED_WOMD_ROLE=validation \
    CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS=40 REPLAN_INTERVAL=1 \
    LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 AUDIT_MAX_LABELS=0 AUDIT_AUTO_MAX_LABELS=0 \
    NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 \
    BUCKET_DATASET="$BUCKET" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 \
    TARGET_KEYS_FILE="$keys" REQUIRE_TARGET_KEYS=true PREFLIGHT_SUPPORT_JSON="$SUPPORT" \
    RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true \
    RESULT_SCENE_DETAIL=full SCENE_JOURNAL_DETAIL=full MEMORY_SCENE_DETAIL=full \
    PARTIAL_WRITE_EVERY_SCENES=11 PROGRESS_EVERY_STEPS=20 PROFILE_TIMING=true \
    JAX_CACHE_DIR="$SHARED_JAX_CACHE" SCAN_PREFIX_ROLLOUTS=true BATCH_TEACHER_OPTION_ROLLOUTS=true \
    VALIDATE_JIT_PREFIX_ROLLOUT=true VALIDATE_BATCHED_TEACHER_METRICS=true BATCH_TEACHER_VALIDATION_ATOL=1e-6 \
    USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true \
    ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false \
    PRIVILEGED_PCD_ORACLE_CEILING=true PRIVILEGED_PCD_ORACLE_EPSILON=1e-6 \
    bash scripts/run_ocrap_closed_loop.sh
}

# Only the 11 historical Base-intervention scenes are fresh replayed.  Balanced
# and precision run on separate A30s.  The 239 historical zero-intervention
# scenes are later reused exactly by the trigger-gating induction proof.
run_oracle balanced "$BCKPT" "$BGAMMA" "$GPU0" "$BKEYS" "$OUT/oracle/balanced" & PB=$!
run_oracle precision "$PCKPT" "$PGAMMA" "$GPU1" "$PKEYS" "$OUT/oracle/precision" & PP=$!
set +e
wait "$PB"; RB=$?
wait "$PP"; RP=$?
set -e
[[ $RB == 0 && $RP == 0 ]] || { echo "PCD-oracle replay failed balanced=$RB precision=$RP" >&2; exit 30; }

BSUB="$OUT/oracle/balanced/closed_loop_ocrap.json"
PSUB="$OUT/oracle/precision/closed_loop_ocrap.json"
python tools/audit_observation_legal_route_results.py \
  --result balanced="$BSUB" --result precision="$PSUB" \
  --output "$OUT/provenance/route_audit.json"

BFULL="$OUT/full/balanced_closed_loop_oracle.json"
PFULL="$OUT/full/precision_closed_loop_oracle.json"
python tools/merge_trigger_gated_near_subset.py --baseline-full "$BBASE" --diagnostic-subset "$BSUB" --target-keys "$BKEYS" --output "$BFULL"
python tools/merge_trigger_gated_near_subset.py --baseline-full "$PBASE" --diagnostic-subset "$PSUB" --target-keys "$PKEYS" --output "$PFULL"

BNOM="$OUT/comparisons/balanced_oracle_vs_nominal.json"
PNOM="$OUT/comparisons/precision_oracle_vs_nominal.json"
BBASECMP="$OUT/comparisons/balanced_oracle_vs_base.json"
PBASECMP="$OUT/comparisons/precision_oracle_vs_base.json"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$BFULL" --bootstrap 5000 --seed 2027 --output "$BNOM"
python tools/compare_paired_closed_loop.py "$NOMINAL" "$PFULL" --bootstrap 5000 --seed 2027 --output "$PNOM"
python tools/compare_paired_closed_loop.py "$BBASE" "$BFULL" --bootstrap 5000 --seed 2027 --output "$BBASECMP"
python tools/compare_paired_closed_loop.py "$PBASE" "$PFULL" --bootstrap 5000 --seed 2027 --output "$PBASECMP"

python tools/adjudicate_near_pcd_oracle_ceiling.py \
  --reference-contract "$REF/REFERENCE_CONTRACT.json" \
  --runtime-contract "$OUT/provenance/pcd_oracle_runtime_contract.json" \
  --candidate-adjudication "$REF/v48124103/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json" \
  --nominal-full "$NOMINAL" \
  --balanced-subset "$BSUB" --balanced-full "$BFULL" \
  --balanced-vs-nominal "$BNOM" --balanced-vs-base "$BBASECMP" \
  --precision-subset "$PSUB" --precision-full "$PFULL" \
  --precision-vs-nominal "$PNOM" --precision-vs-base "$PBASECMP" \
  --route-audit "$OUT/provenance/route_audit.json" \
  --output "$FINAL"

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
cp -f "$FINAL" "$BASE_OUT/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json"
package_results 0
trap - EXIT
echo "V48.124.10.4 privileged PCD-oracle ceiling complete: $RESULTS_ZIP"
