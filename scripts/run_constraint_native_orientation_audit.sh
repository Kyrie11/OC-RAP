#!/usr/bin/env bash
# Stable OC-RAP audit entrypoint.
# V48.124 OC-FMSA is NOT a new recovery mechanism. V48.123 closed the
# recovery-set representation search. This run evaluates the RIFA-conformant frozen L80 Main
# under exact same-target nominal controls for coverage, determinism, Safe
# non-interference, Near closed-loop validity, and Contact recovery validity.
set -Eeuo pipefail

# V48.124.10.7.1 completed the preregistered terminal one-shot ceiling and
# returned MIXED: the original strongest seed action is locally positive in one
# scene and worsens Near physical endpoints in the other. Internal mechanism
# search is therefore closed/frozen. The stable default now performs only an
# offline terminal-closure adjudication; it must not silently launch another GPU
# diagnostic. Historical branches remain explicit for reproduction only.
OCRAP_CONSTRAINT_AUDIT_MODE="${OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure}"
if [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "terminal_internal_closure" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "closure" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "terminal_closure" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_terminal_internal_closure.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "one_shot_action_realization" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "one_shot" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "terminal_ceiling" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_nonfloor_one_shot_realization_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "nonfloor_admission_screen" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "nonfloor_screen" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_nonfloor_admission_seed_screen_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "all_state_support_localization" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "support_localization" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_all_state_support_localization_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "pcd_oracle_ceiling" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "oracle_ceiling" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_pcd_oracle_ceiling_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "candidate_quality" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "candidate_audit" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_candidate_quality_audit_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "route_legal_near_axis" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "near_axis" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "near" || "$OCRAP_CONSTRAINT_AUDIT_MODE" == "diagnostic" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_observation_legal_near_axis_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" == "legacy_near_axis" ]]; then
  REPO_DISPATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  exec bash "$REPO_DISPATCH/scripts/run_near_rifa_system_axis_two_gpu.sh"
elif [[ "$OCRAP_CONSTRAINT_AUDIT_MODE" != "full" ]]; then
  echo "unknown OCRAP_CONSTRAINT_AUDIT_MODE=$OCRAP_CONSTRAINT_AUDIT_MODE (expected terminal_internal_closure, one_shot_action_realization, nonfloor_admission_screen, all_state_support_localization, pcd_oracle_ceiling, candidate_quality, route_legal_near_axis, legacy_near_axis, or full)" >&2
  exit 30
fi

ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_ORIENTATION_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_ORIENTATION_RUN_ID="$RUN_ID"

# The long fixed-Main audit must not observe a mutable worktree.  Build an
# immutable execution snapshot once, then run every full-population phase,
# comparison, sentinel replay, and adjudicator from that same snapshot.  This
# lets the operator continue editing the original repo (for example external
# baselines) without contaminating a running scientific experiment.
if [[ "${OCRAP_EXECUTION_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" \
    --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env \
    OCRAP_EXECUTION_SNAPSHOT_ACTIVE=1 \
    OCRAP_ORIGIN_REPO="$ORIGIN_REPO" \
    OCRAP_REPO="$SNAPSHOT_REPO" \
    OCRAP_ORIENTATION_RUN_ID="$RUN_ID" \
    BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
    bash "$SNAPSHOT_REPO/scripts/run_constraint_native_orientation_audit.sh"
fi

REPO="${OCRAP_REPO:?snapshot repo missing}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

L80_RUN="${OCRAP_ORIENTATION_MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V123_PIPELINE="${OCRAP_ORIENTATION_V123_PIPELINE:-$BASE_OUT/OC-RAP-v48.123-PIPELINE_COMPLETE.json}"
V123_COMPARE="${OCRAP_ORIENTATION_V123_COMPARE:-$BASE_OUT/OC-RAP-v48.123-DCP-DRFC-BCDE-RIFA-OC-ZBST-comparison.json}"
V123_BALANCED="${OCRAP_ORIENTATION_V123_BALANCED:-$BASE_OUT/OC-RAP-v48.123-ZBST-balanced.json}"
V123_PRECISION="${OCRAP_ORIENTATION_V123_PRECISION:-$BASE_OUT/OC-RAP-v48.123-ZBST-precision.json}"

WORK="$BASE_OUT/ocrap_v48_124_contact_anchored_fixed_main"
NOMINAL_OUT="$WORK/nominal"
BALANCED_OUT="$WORK/balanced"
PRECISION_OUT="$WORK/precision"
ANCHOR_DIR="$WORK/contact_anchor"
ANCHOR_MANIFEST="$ANCHOR_DIR/contact_anchor_manifest.json"
ANCHOR_KEYS="$ANCHOR_DIR/contact_anchor_target_keys.json"
SENTINEL_DIR="$WORK/sentinels"
COMPARE_DIR="$WORK/comparisons"
KEY_DIR="$WORK/sentinel_keys"
RUNTIME="$BASE_OUT/OC-RAP-v48.124-runtime-code-contract.json"
SENTINEL_INDEX="$BASE_OUT/OC-RAP-v48.124-sentinel-index.json"
ADJUDICATION="$BASE_OUT/OC-RAP-v48.124-fixed-main-adjudication.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.124-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.124-OC-FMSA-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124-OC-FMSA-results.zip"
PROVENANCE_DIR="$WORK/provenance"
FULL_RUN_RUNTIME="$PROVENANCE_DIR/full_population_runtime_contract.json"
EXECUTION_SNAPSHOT_MANIFEST="$REPO/EXECUTION_SNAPSHOT.json"

mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$SENTINEL_INDEX" "$ADJUDICATION" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"
python tools/check_fixed_main_stability_contract.py --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Reuse is legal only when the completed full-population evidence was generated
# by exactly the same immutable scientific source snapshot.  A mismatched work
# directory is archived by rename (cheap on the same filesystem) rather than
# silently resumed under new code.
if [[ -f "$FULL_RUN_RUNTIME" ]]; then
  if ! python - "$RUNTIME" "$FULL_RUN_RUNTIME" <<'RTEQ'
import json,sys
new=json.load(open(sys.argv[1],encoding='utf-8')); old=json.load(open(sys.argv[2],encoding='utf-8'))
def fp(d):
    rows=d.get('runtime_files') or {}
    return {k:(v or {}).get('sha256') for k,v in rows.items()}
if not (new.get('scientific_version')==old.get('scientific_version') and new.get('scientific_contract')==old.get('scientific_contract') and fp(new)==fp(old)):
    raise SystemExit(30)
RTEQ
  then
    old_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_instance_id") or "unknown")' "$FULL_RUN_RUNTIME" 2>/dev/null || echo unknown)"
    archive="${WORK}.stale-${old_id}"
    rm -rf "$archive"
    mv "$WORK" "$archive"
    echo "archived stale V48.124 workdir with different execution source: $archive" >&2
  fi
fi
mkdir -p "$WORK" "$SENTINEL_DIR" "$COMPARE_DIR" "$KEY_DIR" "$PROVENANCE_DIR" "$ANCHOR_DIR"

# V48.124.9 is evaluation/provenance engineering only. Safe/Near retain exact-a0 controls.
# Contact is formed before treatment by an exact-a0 prelude that stops at the
# first actual Waymax overlap. A scene-disjoint manifest freezes one anchor per
# scene, and all three arms must reproduce the same dynamic-state fingerprint.
FULL_RESULTS_PRESENT=0
for f in \
  "$NOMINAL_OUT/safe/closed_loop_nominal.json" "$NOMINAL_OUT/near/closed_loop_nominal.json" "$NOMINAL_OUT/contact/closed_loop_nominal.json" \
  "$BALANCED_OUT/safe/closed_loop_ocrap.json" "$BALANCED_OUT/near/closed_loop_ocrap.json" "$BALANCED_OUT/contact/closed_loop_ocrap.json" \
  "$PRECISION_OUT/safe/closed_loop_ocrap.json" "$PRECISION_OUT/near/closed_loop_ocrap.json" "$PRECISION_OUT/contact/closed_loop_ocrap.json"; do
  [[ -s "$f" ]] && FULL_RESULTS_PRESENT=1 && break
done
if [[ "$FULL_RESULTS_PRESENT" == 1 && ! -f "$FULL_RUN_RUNTIME" ]]; then
  echo "completed V48.124 artifacts exist without their runtime contract; refuse provenance-unsafe reuse" >&2
  exit 30
fi
[[ -f "$FULL_RUN_RUNTIME" ]] || cp -f "$RUNTIME" "$FULL_RUN_RUNTIME"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

# V48.123 STOP + exact freeze branch is the only scientific license for V48.124.
python - "$V123_PIPELINE" "$V123_COMPARE" "$V123_BALANCED" "$V123_PRECISION" <<'V123PY'
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
    raise SystemExit('V48.123 did not authorize fixed-Main adjudication')
V123PY

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

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
# Phase A uses both A30s concurrently: GPU0 mines treatment-free Contact anchors;
# GPU1 runs exact-a0 Safe/Near controls.
set +e
env WOMD_ROLE=validation OUT="$ANCHOR_DIR" GPU="$GPU0" MIN_POST_STEPS="${CONTACT_MIN_POST_STEPS:-10}" \
  bash scripts/build_contact_anchor_cohort.sh & p_anchor=$!
env WOMD_ROLE=validation OUT="$NOMINAL_OUT" CUDA_DEVICES="$GPU1" MAX_SCENARIOS=0 \
  RUN_SAFE=1 RUN_NEAR=1 RUN_CONTACT=0 INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics \
  bash scripts/run_nominal_three_regime_control.sh & p_nominal=$!
wait "$p_anchor"; r_anchor=$?; wait "$p_nominal"; r_nominal=$?
set -e
[[ $r_anchor == 0 && $r_nominal == 0 ]] || { echo "phase-A failure anchor=$r_anchor nominal_safe_near=$r_nominal" >&2; exit 30; }
[[ -s "$ANCHOR_MANIFEST" && -s "$ANCHOR_KEYS" ]] || { echo "missing Contact anchor manifest/keys" >&2; exit 30; }

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
# Exact-a0 Contact control from the frozen pre-treatment anchor cohort.
env WOMD_ROLE=validation OUT="$NOMINAL_OUT" CUDA_DEVICES="$GPU0" MAX_SCENARIOS=0 \
  RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 CONTACT_TARGET_KEYS_FILE="$ANCHOR_KEYS" \
  CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" \
  CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
  INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics bash scripts/run_nominal_three_regime_control.sh

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
# Frozen Main robustness variants. Each variant gets one GPU; the Contact
# manifest and state fingerprints are immutable and shared read-only.
run_variant() {
  local variant="$1" out="$2" gpu="$3"
  env WOMD_ROLE=validation MODEL_RUN="$L80_RUN" MODEL_VARIANT="$variant" OUT="$out" CUDA_DEVICES="$gpu" \
    MAX_SCENARIOS=0 INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics \
    CONTACT_TARGET_KEYS_FILE="$ANCHOR_KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true \
    CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
    CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
    bash scripts/run_ocrap_three_regime_evaluation.sh
}
set +e
run_variant balanced "$BALANCED_OUT" "$GPU0" & pb=$!
run_variant precision "$PRECISION_OUT" "$GPU1" & pp=$!
wait "$pb"; rb=$?; wait "$pp"; rp=$?
set -e
[[ $rb == 0 && $rp == 0 ]] || { echo "full variant failure balanced=$rb precision=$rp" >&2; exit 30; }

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

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
# Fixed paired bootstrap comparisons. Balanced/precision are robustness variants,
# not additional independent population replications.
for variant in balanced precision; do
  if [[ "$variant" == balanced ]]; then VS="$BS"; VN="$BN"; VC="$BC"; else VS="$PS"; VN="$PN"; VC="$PC"; fi
  python tools/compare_paired_closed_loop.py "$NS" "$VS" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_safe_vs_nominal.json"
  python tools/compare_paired_closed_loop.py "$NN" "$VN" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_near_vs_nominal.json"
  python tools/compare_paired_closed_loop.py "$NC" "$VC" --bootstrap 5000 --seed 2027 --output "$COMPARE_DIR/${variant}_contact_vs_nominal.json"
done

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
# Replay the lexicographically first common target per regime exactly once for
# each frozen Main robustness variant. Source/gamma are read from the full run.
run_sentinel() {
  local variant="$1" regime="$2" full="$3" ckpt="$4" gpu="$5"
  local keyfile="$KEY_DIR/$regime.json"
  local outdir="$SENTINEL_DIR/$variant/$regime"
  local output="$outdir/closed_loop_ocrap.json"
  local support="$(dirname "$full")/closed_loop_dataset_support.json"
  local vals source bucket gamma source_role support_valid
  [[ -f "$support" ]] || { echo "missing sentinel support provenance $variant/$regime: $support" >&2; return 30; }
  vals="$(python - "$full" "$support" <<'PY'
import json,sys
full=json.load(open(sys.argv[1],encoding='utf-8'))
support=json.load(open(sys.argv[2],encoding='utf-8'))
print(json.dumps([
    support.get('womd_pattern'),
    full.get('bucket_dataset'),
    full.get('gamma_rec'),
    support.get('raw_source_role'),
    bool(support.get('schema_supports_closed_loop')),
]))
PY
)"
  source="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[0])' "$vals")"
  bucket="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[1])' "$vals")"
  gamma="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[2])' "$vals")"
  source_role="$(python -c 'import json,sys; print(json.loads(sys.argv[1])[3])' "$vals")"
  support_valid="$(python -c 'import json,sys; print(str(bool(json.loads(sys.argv[1])[4])).lower())' "$vals")"
  [[ -n "$source" && "$source" != None && -n "$bucket" && "$bucket" != None && -n "$gamma" && "$gamma" != None ]] || { echo "missing sentinel provenance $variant/$regime" >&2; return 30; }
  [[ "$source_role" == validation && "$support_valid" == true ]] || { echo "invalid sentinel WOMD provenance $variant/$regime role=$source_role support_valid=$support_valid" >&2; return 30; }
  mkdir -p "$outdir"
  local anchor_env=()
  if [[ "$regime" == contact ]]; then
    anchor_env=(CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" CONTACT_ANCHOR_REQUIRE_FOUND=true CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1)
  fi
  env RUN_DIR="$outdir" OUTPUT="$output" WOMD_VAL="$source" EXPECTED_WOMD_ROLE=validation CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS=40 LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 \
    BUCKET_DATASET="$bucket" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 TARGET_KEYS_FILE="$keyfile" REQUIRE_TARGET_KEYS=true \
    RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics \
    SCENE_JOURNAL_DETAIL=metrics MEMORY_SCENE_DETAIL=metrics "${anchor_env[@]}" bash scripts/run_ocrap_closed_loop.sh
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

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
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
  --full-run-runtime "$FULL_RUN_RUNTIME" --contact-anchor-manifest "$ANCHOR_MANIFEST" \
  --run-id "$RUN_ID" --output "$ADJUDICATION"

python tools/check_fixed_main_stability_pipeline.py \
  --runtime "$RUNTIME" --adjudication "$ADJUDICATION" --sentinel-index "$SENTINEL_INDEX" \
  --v123-pipeline "$V123_PIPELINE" --v123-comparison "$V123_COMPARE" \
  --run-id "$RUN_ID" --output "$COMPLETE"

python tools/package_fixed_main_stability_results.py \
  --pipeline "$COMPLETE" --base-out "$BASE_OUT" --run-id "$RUN_ID" --manifest "$BUNDLE_MANIFEST" --output "$RESULTS_ZIP"

printf 'V48.124 OC-FMSA fixed-Main adjudication bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
