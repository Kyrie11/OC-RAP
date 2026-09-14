#!/usr/bin/env bash
# V48.124.10 diagnostic-only Near system-axis audit.
#
# Scientific scope:
#   * V48.124.9 frozen model / absolute admission / candidate library unchanged.
#   * No training, recalibration, threshold/capacity/horizon/regime sweep.
#   * Two preregistered downstream selector completions are evaluated only on
#     the exact historical V48.124.9 Near intervention cohort (35/250 scenes).
#   * Historical zero-intervention scenes are reused only because the diagnostic
#     selector is intervention-monotone: it cannot create a first intervention.
#   * Any promoted selector MUST subsequently be confirmed by a fresh 250-scene
#     Near run before publication/deployed-main freeze.
set -Eeuo pipefail

ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_NEAR_AXIS_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_NEAR_AXIS_RUN_ID="$RUN_ID"

# Exactly the same immutable-source discipline used by V48.124.9.  This avoids
# another mixed-working-tree failure while allowing the operator to edit the
# original repo during a long GPU run.
if [[ "${OCRAP_NEAR_AXIS_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" \
    --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env \
    OCRAP_NEAR_AXIS_SNAPSHOT_ACTIVE=1 \
    OCRAP_ORIGIN_REPO="$ORIGIN_REPO" \
    OCRAP_REPO="$SNAPSHOT_REPO" \
    OCRAP_NEAR_AXIS_RUN_ID="$RUN_ID" \
    OCRAP_CONSTRAINT_AUDIT_MODE=near_axis \
    BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
    bash "$SNAPSHOT_REPO/scripts/run_near_rifa_system_axis_two_gpu.sh"
fi

REPO="${OCRAP_REPO:?snapshot repo missing}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

L80_RUN="${OCRAP_ORIENTATION_MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
OUT_ROOT="${OCRAP_V4812410_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_near_rifa_system_axis}"
OUT="$OUT_ROOT/$RUN_ID"
REFERENCE="$OUT/reference_v481249"
mkdir -p "$OUT" "$REFERENCE" "$OUT/support" "$OUT/cohort" "$OUT/merged" "$OUT/comparisons" "$OUT/provenance" "$OUT/jax_cache"

RUNTIME="$OUT/provenance/V48.124.10-near-axis-runtime-contract.json"
python tools/check_near_rifa_system_axis_contract.py --repo "$REPO" --output "$RUNTIME"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"

# Prefer the canonical V48.124.9 packaged evidence and verify every manifested
# byte.  Fall back to the original workdir only when that canonical zip is not
# present, while still requiring the attribution-ready Near STOP adjudication.
python tools/materialize_historical_near_reference.py \
  --base-out "$BASE_OUT" --output-dir "$REFERENCE" \
  --historical-workdir "${OCRAP_V481249_WORK:-$BASE_OUT/ocrap_v48_124_contact_anchored_fixed_main}"
REF_CONTRACT="$REFERENCE/V48.124.9-near-reference-contract.json"
HIST_ADJ="$REFERENCE/OC-RAP-v48.124-fixed-main-adjudication.json"
NN="$REFERENCE/OC-RAP-v48.124-nominal-near-closed-loop.json"
BN="$REFERENCE/OC-RAP-v48.124-balanced-near-closed-loop.json"
PN="$REFERENCE/OC-RAP-v48.124-precision-near-closed-loop.json"
BSUP="$REFERENCE/OC-RAP-v48.124-balanced-near-dataset-support.json"
PSUP="$REFERENCE/OC-RAP-v48.124-precision-near-dataset-support.json"

# Fail closed unless V48.124.9 established exactly the branch that licenses this
# diagnostic.  A result-quality change can never be used to bypass this gate.
python - "$HIST_ADJ" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); d=x.get('preregistered_decision') or {}
assert x.get('valid') is True and x.get('attribution_ready') is True
assert x.get('engineering_version') == 'v48.124.9-OC-FMSA-RUNTIME-SNAPSHOT-ENGFIX'
assert (d.get('status') or x.get('status')) == 'FIXED_MAIN_NEAR_VALIDITY_STOP'
assert x.get('recovery_set_mechanism_family_frozen') is True
assert x.get('new_recovery_mechanism_authorized') is False
assert x.get('planner_parameters_trained') == 0
assert x.get('source_parameters_trained') == 0
assert x.get('stage_i_parameters_trained') == 0
assert x.get('root_decoder_parameters_trained') == 0
assert x.get('recalibration_performed') is False
PY

KEYS="$OUT/cohort/near_intervention_target_keys.json"
COHORT_AUDIT="$OUT/cohort/near_intervention_audit.json"
python tools/build_near_intervention_cohort.py \
  --balanced "$BN" --precision "$PN" \
  --target-keys-output "$KEYS" --audit-output "$COHORT_AUDIT"

resolve_candidate_root() {
  local variant="$1"
  local root="$L80_RUN/candidates/$variant"
  if [[ ! -f "$root/model_v48_trac_sr/best.pt" && -f "$L80_RUN/dedicated_candidates/$variant/model_v48_trac_sr/best.pt" ]]; then
    root="$L80_RUN/dedicated_candidates/$variant"
  fi
  printf '%s\n' "$root"
}
BROOT="$(resolve_candidate_root balanced)"; PROOT="$(resolve_candidate_root precision)"
BCKPT="$BROOT/model_v48_trac_sr/best.pt"; PCKPT="$PROOT/model_v48_trac_sr/best.pt"
for f in "$BCKPT" "$PCKPT"; do [[ -s "$f" ]] || { echo "missing frozen L80 checkpoint: $f" >&2; exit 30; }; done

# Resolve the exact historical WOMD/bucket/gamma contract.  Both robustness
# variants must agree on data provenance.  gamma remains checkpoint-specific.
readarray -t META < <(python - "$BN" "$PN" "$BSUP" "$PSUP" <<'PY'
import json,sys
b,p,bs,ps=[json.load(open(x,encoding='utf-8')) for x in sys.argv[1:]]
assert b.get('bucket_dataset') == p.get('bucket_dataset')
assert bs.get('dataset') == ps.get('dataset') == b.get('bucket_dataset')
assert bs.get('womd_pattern') == ps.get('womd_pattern')
assert bs.get('raw_source_role') == ps.get('raw_source_role') == 'validation'
assert bs.get('schema_supports_closed_loop') and ps.get('schema_supports_closed_loop')
assert int(b.get('num_scenes')) == int(p.get('num_scenes')) == 250
print(bs['womd_pattern']); print(b['bucket_dataset']); print(b['gamma_rec']); print(p['gamma_rec'])
PY
)
WOMD_VAL="${META[0]}"; BUCKET="${META[1]}"; BGAMMA="${META[2]}"; PGAMMA="${META[3]}"

# One target-aware data scan is shared by all four jobs.  This is read-only and
# execution-equivalent; it removes four redundant WOMD support scans.
DIAG_SUPPORT="$OUT/support/near_intervention_dataset_support.json"
python tools/check_closed_loop_dataset_support.py \
  --dataset "$BUCKET" --split test --womd-pattern "$WOMD_VAL" \
  --expected-source-role validation --target-keys-file "$KEYS" --require-target-keys \
  --output "$DIAG_SUPPORT"

run_diag() {
  local variant="$1" ckpt="$2" gamma="$3" gpu="$4" arm="$5" config="$6"
  local dir="$OUT/$arm/$variant"
  mkdir -p "$dir"
  # JAX compilation cache is shared between the two sequential arms of one
  # robustness variant only. Compiled executable caching does not share rollout
  # state/RNG and is semantics-preserving; it mainly accelerates arm #2.
  local cache="$OUT/jax_cache/$variant"
  mkdir -p "$cache"
  env RUN_DIR="$dir" OUTPUT="$dir/closed_loop_ocrap.json" \
    WOMD_VAL="$WOMD_VAL" EXPECTED_WOMD_ROLE=validation CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS=40 REPLAN_INTERVAL=1 LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 \
    NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 BUCKET_DATASET="$BUCKET" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 \
    TARGET_KEYS_FILE="$KEYS" REQUIRE_TARGET_KEYS=true PREFLIGHT_SUPPORT_JSON="$DIAG_SUPPORT" \
    CONFIG="$config" RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true \
    RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics MEMORY_SCENE_DETAIL=metrics \
    PARTIAL_WRITE_EVERY_SCENES=35 PROGRESS_EVERY_STEPS=20 PROFILE_TIMING=true JAX_CACHE_DIR="$cache" \
    bash scripts/run_ocrap_closed_loop.sh
}

# Four jobs are causally independent.  GPU0 owns balanced, GPU1 owns precision;
# each GPU runs delta then nested sequentially.  This maximizes two-A30 use while
# avoiding same-GPU process oversubscription and new memory/scheduling variables.
worker() {
  local variant="$1" ckpt="$2" gamma="$3" gpu="$4"
  run_diag "$variant" "$ckpt" "$gamma" "$gpu" delta "$REPO/configs/v48_124_10_near_relative_delta.yaml"
  run_diag "$variant" "$ckpt" "$gamma" "$gpu" nested "$REPO/configs/v48_124_10_near_nested_evidence.yaml"
}
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
set +e
worker balanced "$BCKPT" "$BGAMMA" "$GPU0" & pb=$!
worker precision "$PCKPT" "$PGAMMA" "$GPU1" & pp=$!
wait "$pb"; rb=$?; wait "$pp"; rp=$?
set -e
[[ $rb == 0 && $rp == 0 ]] || { echo "Near diagnostic failure balanced=$rb precision=$rp" >&2; exit 30; }
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

# Reconstruct the full paired 250-scene population exactly by replacing only
# the historical intervention cohort.  These artifacts are INTERNAL DIAGNOSTIC
# evidence and are marked as such; they cannot be used as paper Main results.
for arm in delta nested; do
  for variant in balanced precision; do
    sub="$OUT/$arm/$variant/closed_loop_ocrap.json"
    [[ -s "$sub" ]] || { echo "missing diagnostic subset result $sub" >&2; exit 30; }
    if [[ "$variant" == balanced ]]; then hist="$BN"; else hist="$PN"; fi
    full="$OUT/merged/${arm}_${variant}_full250.json"
    python tools/merge_monotone_near_subset.py \
      --baseline-full "$hist" --diagnostic-subset "$sub" --target-keys "$KEYS" --output "$full"
    python tools/compare_paired_closed_loop.py "$NN" "$full" --bootstrap 5000 --seed 2027 \
      --output "$OUT/comparisons/${arm}_${variant}_vs_nominal.json"
    python tools/compare_paired_closed_loop.py "$hist" "$full" --bootstrap 5000 --seed 2027 \
      --output "$OUT/comparisons/${arm}_${variant}_vs_historical.json"
  done
done

ADJ="$OUT/V48.124.10-near-rifa-system-axis-adjudication.json"
python tools/adjudicate_near_rifa_system_axis.py \
  --historical-adjudication "$HIST_ADJ" --cohort-audit "$COHORT_AUDIT" \
  --runtime-contract "$RUNTIME" --reference-contract "$REF_CONTRACT" \
  --historical-balanced-full "$BN" --historical-precision-full "$PN" --nominal-full "$NN" \
  --delta-balanced-full "$OUT/merged/delta_balanced_full250.json" \
  --delta-precision-full "$OUT/merged/delta_precision_full250.json" \
  --delta-balanced-vs-nominal "$OUT/comparisons/delta_balanced_vs_nominal.json" \
  --delta-precision-vs-nominal "$OUT/comparisons/delta_precision_vs_nominal.json" \
  --delta-balanced-vs-historical "$OUT/comparisons/delta_balanced_vs_historical.json" \
  --delta-precision-vs-historical "$OUT/comparisons/delta_precision_vs_historical.json" \
  --nested-balanced-full "$OUT/merged/nested_balanced_full250.json" \
  --nested-precision-full "$OUT/merged/nested_precision_full250.json" \
  --nested-balanced-vs-nominal "$OUT/comparisons/nested_balanced_vs_nominal.json" \
  --nested-precision-vs-nominal "$OUT/comparisons/nested_precision_vs_nominal.json" \
  --nested-balanced-vs-historical "$OUT/comparisons/nested_balanced_vs_historical.json" \
  --nested-precision-vs-historical "$OUT/comparisons/nested_precision_vs_historical.json" \
  --output "$ADJ"

# Canonical diagnostic package for the next analysis turn.
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10-NEAR-RIFA-SYSTEM-AXIS-results.zip"
RESULTS_MANIFEST="$BASE_OUT/OC-RAP-v48.124.10-NEAR-RIFA-SYSTEM-AXIS-result-manifest.json"
rm -f "$RESULTS_ZIP" "$RESULTS_MANIFEST"
python tools/package_near_rifa_system_axis_results.py \
  --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST"
cp -f "$ADJ" "$BASE_OUT/OC-RAP-v48.124.10-NEAR-RIFA-SYSTEM-AXIS-adjudication.json"

echo "V48.124.10 Near system-axis diagnostic complete"
echo "  adjudication: $ADJ"
echo "  canonical results: $RESULTS_ZIP"
echo "  manifest: $RESULTS_MANIFEST"
