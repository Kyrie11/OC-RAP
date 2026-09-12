#!/usr/bin/env bash
# Stable OC-RAP audit entrypoint.
# V48.124 OC-FMSA is NOT a new recovery mechanism. V48.123 closed the
# recovery-set representation search. This run evaluates the frozen L80 Main
# under exact same-target nominal controls for coverage, determinism, Safe
# non-interference, Near closed-loop validity, and Contact recovery validity.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_ORIENTATION_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_ORIENTATION_RUN_ID="$RUN_ID"

L80_RUN="${OCRAP_ORIENTATION_MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V123_PIPELINE="${OCRAP_ORIENTATION_V123_PIPELINE:-$BASE_OUT/OC-RAP-v48.123-PIPELINE_COMPLETE.json}"
V123_COMPARE="${OCRAP_ORIENTATION_V123_COMPARE:-$BASE_OUT/OC-RAP-v48.123-DCP-DRFC-BCDE-RIFA-OC-ZBST-comparison.json}"
V123_BALANCED="${OCRAP_ORIENTATION_V123_BALANCED:-$BASE_OUT/OC-RAP-v48.123-ZBST-balanced.json}"
V123_PRECISION="${OCRAP_ORIENTATION_V123_PRECISION:-$BASE_OUT/OC-RAP-v48.123-ZBST-precision.json}"

WORK="$BASE_OUT/ocrap_v48_124_fixed_main_stability"
NOMINAL_OUT="$WORK/nominal"
BALANCED_OUT="$WORK/balanced"
PRECISION_OUT="$WORK/precision"
SENTINEL_DIR="$WORK/sentinels"
COMPARE_DIR="$WORK/comparisons"
KEY_DIR="$WORK/sentinel_keys"
RUNTIME="$BASE_OUT/OC-RAP-v48.124-runtime-code-contract.json"
SENTINEL_INDEX="$BASE_OUT/OC-RAP-v48.124-sentinel-index.json"
ADJUDICATION="$BASE_OUT/OC-RAP-v48.124-fixed-main-adjudication.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.124-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.124-OC-FMSA-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124-OC-FMSA-results.zip"

mkdir -p "$BASE_OUT" "$WORK" "$SENTINEL_DIR" "$COMPARE_DIR" "$KEY_DIR"
rm -f "$RUNTIME" "$SENTINEL_INDEX" "$ADJUDICATION" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before long GPU work if this checkout does not satisfy the fixed-Main contract.
python tools/check_fixed_main_stability_contract.py --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# V48.123 STOP + exact freeze branch is the only scientific license for V48.124.
python - "$V123_PIPELINE" "$V123_COMPARE" "$V123_BALANCED" "$V123_PRECISION" <<'PY'
import hashlib,json,pathlib,sys
p,c,b,q=map(pathlib.Path,sys.argv[1:])
want={
 p:'cd963f76b508f19dbb5140d36d94f76ab815669ad36ec3bbae8431a7e097fe81',
 c:'cb850cbf24fd1c27f00f56805bcb919e2c15267a365ac045dccebed7ddcba067',
 b:'0d4f1281f4278334f2b63fe6893a61e3445a0385baba5c62d755de48a90c9837',
 q:'8e76846b56f90bafff02110cb31e58ca4f4eacf5076a7edeadd37e0914b47538',
}
for path,digest in want.items():
    if not path.is_file(): raise SystemExit(f'missing V48.123 prerequisite {path}')
    got=hashlib.sha256(path.read_bytes()).hexdigest()
    if got!=digest: raise SystemExit(f'authoritative V48.123 SHA mismatch {path.name}: {got}')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
status='ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_STOP'
branch='close_zero_boundary_viability_state_transition_then_freeze_recovery_set_mechanism_family_and_preregister_fixed_main_stability_noninterference_adjudication_no_new_recovery_mechanism_capacity_regime_source_horizon_or_threshold_sweep'
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status')==status):
    raise SystemExit('authoritative V48.123 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status')==status and d.get('next_branch')==branch):
    raise SystemExit('V48.123 did not authorize fixed-Main stability/non-interference adjudication')
PY

resolve_candidate_root() {
  local variant="$1"
  local root="$L80_RUN/candidates/$variant"
  if [[ ! -f "$root/model_v48_trac_sr/best.pt" && -f "$L80_RUN/dedicated_candidates/$variant/model_v48_trac_sr/best.pt" ]]; then
    root="$L80_RUN/dedicated_candidates/$variant"
  fi
  printf '%s\n' "$root"
}
BROOT="$(resolve_candidate_root balanced)"
PROOT="$(resolve_candidate_root precision)"
BCKPT="$BROOT/model_v48_trac_sr/best.pt"
PCKPT="$PROOT/model_v48_trac_sr/best.pt"
BCAL="$BROOT/calibration/gamma_rec_by_bucket_v48.json"
PCAL="$PROOT/calibration/gamma_rec_by_bucket_v48.json"
for f in "$BCKPT" "$PCKPT" "$BCAL" "$PCAL"; do [[ -f "$f" ]] || { echo "missing frozen Main artifact $f" >&2; exit 30; }; done

# Full same-target nominal control. All publication/test buckets are standard
# WOMD validation. Explicit role + bucket provenance disagreement fails closed.
env WOMD_ROLE=validation OUT="$NOMINAL_OUT" CUDA_DEVICES="$GPU0,$GPU1" MAX_SCENARIOS=0 INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics \
  bash scripts/run_nominal_three_regime_control.sh

# Frozen Main, balanced then precision robustness variant. No retraining/recalibration.
run_variant() {
  local variant="$1" out="$2"
  env WOMD_ROLE=validation MODEL_RUN="$L80_RUN" MODEL_VARIANT="$variant" OUT="$out" CUDA_DEVICES="$GPU0,$GPU1" \
    MAX_SCENARIOS=0 INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics \
    bash scripts/run_ocrap_three_regime_evaluation.sh
}
run_variant balanced "$BALANCED_OUT"
run_variant precision "$PRECISION_OUT"

# Canonical paths to the nine full-population results.
NS="$NOMINAL_OUT/safe/closed_loop_nominal.json"; NN="$NOMINAL_OUT/near/closed_loop_nominal.json"; NC="$NOMINAL_OUT/contact/closed_loop_nominal.json"
BS="$BALANCED_OUT/safe/closed_loop_ocrap.json"; BN="$BALANCED_OUT/near/closed_loop_ocrap.json"; BC="$BALANCED_OUT/contact/closed_loop_ocrap.json"
PS="$PRECISION_OUT/safe/closed_loop_ocrap.json"; PN="$PRECISION_OUT/near/closed_loop_ocrap.json"; PC="$PRECISION_OUT/contact/closed_loop_ocrap.json"
for f in "$NS" "$NN" "$NC" "$BS" "$BN" "$BC" "$PS" "$PN" "$PC"; do [[ -s "$f" ]] || { echo "missing full closed-loop result $f" >&2; exit 30; }; done

python tools/build_fixed_main_sentinel_keys.py \
  --nominal-safe "$NS" --balanced-safe "$BS" --precision-safe "$PS" \
  --nominal-near "$NN" --balanced-near "$BN" --precision-near "$PN" \
  --nominal-contact "$NC" --balanced-contact "$BC" --precision-contact "$PC" \
  --key-dir "$KEY_DIR" --output "$SENTINEL_INDEX"

# Fixed paired bootstrap comparisons. Balanced/precision are robustness variants,
# not additional independent population replications.
for variant in balanced precision; do
  if [[ "$variant" == balanced ]]; then VS="$BS"; VN="$BN"; VC="$BC"; else VS="$PS"; VN="$PN"; VC="$PC"; fi
  python tools/compare_paired_closed_loop.py "$NS" "$VS" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_safe_vs_nominal.json"
  python tools/compare_paired_closed_loop.py "$NN" "$VN" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_near_vs_nominal.json"
  python tools/compare_paired_closed_loop.py "$NC" "$VC" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_contact_vs_nominal.json"
done

# Replay the lexicographically first common target per regime exactly once for
# each frozen Main robustness variant. Source/gamma are read from the full run.
run_sentinel() {
  local variant="$1" regime="$2" full="$3" ckpt="$4" gpu="$5"
  local keyfile="$KEY_DIR/$regime.json"
  local outdir="$SENTINEL_DIR/$variant/$regime"
  local output="$outdir/closed_loop_ocrap.json"
  local vals source bucket gamma
  vals="$(python - "$full" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(json.dumps([x.get('source'),x.get('bucket_dataset'),x.get('gamma_rec')]))
PY
)"
  source="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[0])' "$vals")"
  bucket="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[1])' "$vals")"
  gamma="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[2])' "$vals")"
  [[ -n "$source" && "$source" != None && -n "$bucket" && "$bucket" != None && -n "$gamma" && "$gamma" != None ]] || { echo "missing sentinel provenance $variant/$regime" >&2; return 30; }
  mkdir -p "$outdir"
  env RUN_DIR="$outdir" OUTPUT="$output" WOMD_VAL="$source" CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS=40 LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 \
    BUCKET_DATASET="$bucket" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 TARGET_KEYS_FILE="$keyfile" REQUIRE_TARGET_KEYS=true \
    RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics \
    SCENE_JOURNAL_DETAIL=metrics MEMORY_SCENE_DETAIL=metrics bash scripts/run_ocrap_closed_loop.sh
}

# Two GPUs: replay one robustness variant per GPU for each regime.
for regime in safe near contact; do
  case "$regime" in safe) BF="$BS"; PF="$PS";; near) BF="$BN"; PF="$PN";; contact) BF="$BC"; PF="$PC";; esac
  set +e
  run_sentinel balanced "$regime" "$BF" "$BCKPT" "$GPU0" & p0=$!
  run_sentinel precision "$regime" "$PF" "$PCKPT" "$GPU1" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?
  set -e
  [[ $r0 == 0 && $r1 == 0 ]] || { echo "sentinel replay failure $regime balanced=$r0 precision=$r1" >&2; exit 30; }
done

BSS="$SENTINEL_DIR/balanced/safe/closed_loop_ocrap.json"; BNS="$SENTINEL_DIR/balanced/near/closed_loop_ocrap.json"; BCS="$SENTINEL_DIR/balanced/contact/closed_loop_ocrap.json"
PSS="$SENTINEL_DIR/precision/safe/closed_loop_ocrap.json"; PNS="$SENTINEL_DIR/precision/near/closed_loop_ocrap.json"; PCS="$SENTINEL_DIR/precision/contact/closed_loop_ocrap.json"

python tools/adjudicate_fixed_main_stability.py \
  --nominal-safe "$NS" --balanced-safe "$BS" --precision-safe "$PS" \
  --nominal-near "$NN" --balanced-near "$BN" --precision-near "$PN" \
  --nominal-contact "$NC" --balanced-contact "$BC" --precision-contact "$PC" \
  --balanced-safe-comparison "$COMPARE_DIR/balanced_safe_vs_nominal.json" --precision-safe-comparison "$COMPARE_DIR/precision_safe_vs_nominal.json" \
  --balanced-near-comparison "$COMPARE_DIR/balanced_near_vs_nominal.json" --precision-near-comparison "$COMPARE_DIR/precision_near_vs_nominal.json" \
  --balanced-contact-comparison "$COMPARE_DIR/balanced_contact_vs_nominal.json" --precision-contact-comparison "$COMPARE_DIR/precision_contact_vs_nominal.json" \
  --balanced-safe-sentinel "$BSS" --precision-safe-sentinel "$PSS" \
  --balanced-near-sentinel "$BNS" --precision-near-sentinel "$PNS" \
  --balanced-contact-sentinel "$BCS" --precision-contact-sentinel "$PCS" \
  --sentinel-index "$SENTINEL_INDEX" --balanced-checkpoint "$BCKPT" --precision-checkpoint "$PCKPT" \
  --balanced-calibration "$BCAL" --precision-calibration "$PCAL" \
  --v123-pipeline "$V123_PIPELINE" --v123-comparison "$V123_COMPARE" --v123-balanced "$V123_BALANCED" --v123-precision "$V123_PRECISION" \
  --run-id "$RUN_ID" --output "$ADJUDICATION"

python tools/check_fixed_main_stability_pipeline.py \
  --runtime "$RUNTIME" --adjudication "$ADJUDICATION" --sentinel-index "$SENTINEL_INDEX" \
  --v123-pipeline "$V123_PIPELINE" --v123-comparison "$V123_COMPARE" \
  --run-id "$RUN_ID" --output "$COMPLETE"

python tools/package_fixed_main_stability_results.py \
  --pipeline "$COMPLETE" --base-out "$BASE_OUT" --run-id "$RUN_ID" --manifest "$BUNDLE_MANIFEST" --output "$RESULTS_ZIP"

printf 'V48.124 OC-FMSA fixed-Main adjudication bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
