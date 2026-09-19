#!/usr/bin/env bash
# Frozen-checkpoint functional ablations for the cleaned submission stack.
#
# Key properties:
#   * publication metrics come from the v55 exact OBB/TTC/penetration closed-loop core;
#   * latency is observation -> deployable action selection only (profile_timing remains on);
#   * independently isolated accuracy workers dynamically claim jobs; adjustable workers/GPU;
#   * immutable bucket/WOMD provenance is preflighted once per regime and reused safely;
#   * checkpoint and per-bucket gamma calibration stay frozen for every functional knockout.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
OUT_ROOT="${OUT_ROOT:-$BASE_OUT/ocrap_v48_124_final_ablations}"
FULL_RUN_ROOT="${FULL_RUN_ROOT:-$BASE_OUT/ocrap_v48_124_final_characterization/ocrap}"
VARIANTS="${VARIANTS:-balanced,precision}"
CUDA_DEVICES="${CUDA_DEVICES:-${GPU0:-0},${GPU1:-1}}"
# Accuracy throughput only. Independent process-per-job workers may share a GPU;
# no checkpoint, RNG, target set or result is shared between processes. Set 1
# for legacy behavior or increase after measuring *total* jobs/hour. Isolated
# publication latency below deliberately remains serial and uncontended.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] && ((WORKERS_PER_GPU <= 64)) || {
  echo "WORKERS_PER_GPU must be an integer in [1,64]" >&2; exit 2;
}
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
WOMD_NUM_SHARDS="${WOMD_NUM_SHARDS:-150}"
BUCKET_SPLIT="${BUCKET_SPLIT:-test}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
SAFE_BUCKET="${SAFE_BUCKET:-$OCRAP_ROOT/test_safe}"
NEAR_BUCKET="${NEAR_BUCKET:-$OCRAP_ROOT/test_near_contact}"
CONTACT_BUCKET="${CONTACT_BUCKET:-$OCRAP_ROOT/test_contact}"
TARGET_LOCK_CHARACTERIZATION_OUT="${TARGET_LOCK_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
FINAL_TARGET_LOCK_ROOT="${FINAL_TARGET_LOCK_ROOT:-$TARGET_LOCK_CHARACTERIZATION_OUT/target_keys}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
MAX_STEPS="${MAX_STEPS:-40}"
NUM_CANDIDATES="${NUM_CANDIDATES:-24}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-1}"
METRIC_SEMANTICS_VERSION="${METRIC_SEMANTICS_VERSION:-publication_v55_signed_clearance_unclipped_v1}"
CONTACT_ANCHOR_PRELUDE_MAX_STEPS="${CONTACT_ANCHOR_PRELUDE_MAX_STEPS:-60}"
CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL="${CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL:-1}"
CONTACT_ANCHOR_REQUIRE_FOUND="${CONTACT_ANCHOR_REQUIRE_FOUND:-true}"
CONTACT_ANCHOR_MANIFEST_FILE="${CONTACT_ANCHOR_MANIFEST_FILE:-$TARGET_LOCK_CHARACTERIZATION_OUT/contact_anchor/contact_anchor_manifest.json}"
# main = final frozen-stack functional ablations; supplementary adds active-set alignment.
ABLATION_SET="${ABLATION_SET:-main}"
ABLATIONS="${ABLATIONS:-}"
RUN_TAG="${RUN_TAG:-submission_ablation_metrics}"
NATIVE_FULL_REFERENCE_CONFIG="${NATIVE_FULL_REFERENCE_CONFIG:-configs/ablations/submission_native_certified_full.yaml}"
NATIVE_FULL_REFERENCE_ROOT="${NATIVE_FULL_REFERENCE_ROOT:-$OUT_ROOT/_native_full_reference}"

# Resume-aware execution controls. Complete artifacts are reused; --force is unnecessary for scientific ablations.
SKIP_COMPLETE="${SKIP_COMPLETE:-true}"
BUILD_TABLES="${BUILD_TABLES:-true}"
BUILD_TARGET_LOCKS="${BUILD_TARGET_LOCKS:-true}"
TABLE_OUT="${TABLE_OUT:-$OUT_ROOT/submission_tables}"
PARTIAL_WRITE_EVERY_SCENES="${PARTIAL_WRITE_EVERY_SCENES:-64}"
PROGRESS_EVERY_STEPS="${PROGRESS_EVERY_STEPS:-20}"
PROFILE_TIMING="${PROFILE_TIMING:-true}"
# Publication latency follows the same contract as the final characterization:
# one process on one GPU, rerun after accuracy jobs complete.  Set false only
# for diagnostics; paper tables then omit latency rather than mixing contracts.
PROFILE_ISOLATED_LATENCY="${PROFILE_ISOLATED_LATENCY:-true}"
LATENCY_ROOT="${LATENCY_ROOT:-$OUT_ROOT/latency_isolated}"
LATENCY_WARMUP_DECISIONS="${LATENCY_WARMUP_DECISIONS:-3}"
LABEL_MODE="${LABEL_MODE:-fast}"
AUDIT_EVERY_N_STEPS="${AUDIT_EVERY_N_STEPS:-0}"

# Publication ablations intentionally avoid evaluation-only teacher/audit work.
# PROFILE_TIMING must stay enabled because the paper table reports deployment latency.
runtime_bool_true "$PROFILE_TIMING" || { echo "PROFILE_TIMING must be true for publication ablations." >&2; exit 2; }
[[ "$LABEL_MODE" == "fast" ]] || { echo "LABEL_MODE must be fast for publication ablations; teacher/audit labels are evaluation-only." >&2; exit 2; }
[[ "$AUDIT_EVERY_N_STEPS" == 0 ]] || { echo "AUDIT_EVERY_N_STEPS must be 0 for publication ablations." >&2; exit 2; }

mkdir -p "$OUT_ROOT"
ABLATION_RUN_ID="${OCRAP_ABLATION_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_ABLATION_RUN_ID="$ABLATION_RUN_ID"
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" \
  --run-id "$ABLATION_RUN_ID" \
  --output "$OUT_ROOT/cnro-runtime-code-contract.ablations.json"
[[ -f "$NATIVE_FULL_REFERENCE_CONFIG" ]] || { echo "Missing native Full reference config: $NATIVE_FULL_REFERENCE_CONFIG" >&2; exit 30; }

# name|config|safe|near|contact|tier|description
# These are inference-time/frozen-checkpoint functional knockouts.  Every
# submission arm enables the paper-faithful native recovery certificate before
# OC-MERO; physical-semantic knockouts remove one factor from that same native
# path. They never re-introduce the historical learned-AFE selector.
ARMS=(
  "no_obs_consistency|configs/ablations/submission_no_obs_consistency.yaml|0|1|1|main|Disable observation-compatible OC-MERO grouping at inference; frozen checkpoint/heads"
  "mean_tail|configs/ablations/submission_mean_tail.yaml|0|1|1|main|Replace nested lower-tail aggregation by weighted mean at inference; frozen checkpoint/heads"
  "no_actuator_projection|configs/ablations/submission_no_actuator_projection.yaml|0|1|1|main|Disable actuator-envelope projection in executable recovery witness/certification"
  "no_persistent_reentry|configs/ablations/submission_no_persistent_reentry.yaml|0|0|1|main|Remove persistent re-entry alignment from post-contact recovery witness/certification"
  "no_rifa_absolute_admission|configs/ablations/submission_no_rifa_absolute_admission.yaml|1|1|1|main|Remove the absolute deployable-recovery admission gate while retaining hard/harm feasibility and frozen scoring"
  "no_nominal_abstention|configs/ablations/submission_no_nominal_abstention.yaml|1|1|1|main|Keep absolute admission but allow the legacy recovery-first fallback to resurrect an unadmitted intervention"
  "no_route_alignment|configs/ablations/submission_no_route_alignment.yaml|0|1|1|main|Remove executable route-alignment semantics from recovery witness/certification"
  "no_active_set_alignment|configs/ablations/submission_no_active_set_alignment.yaml|0|1|1|supplementary|Remove active-set alignment from executable recovery witness/certification"
)

selected() {
  local name="$1" tier="$2"
  if [[ -n "$ABLATIONS" ]]; then
    [[ ",${ABLATIONS}," == *",${name},"* ]]
    return
  fi
  if [[ "$ABLATION_SET" == "all" ]]; then return 0; fi
  if [[ "$ABLATION_SET" == "main" ]]; then [[ "$tier" == "main" ]]; return; fi
  echo "Unsupported ABLATION_SET=$ABLATION_SET (expected main|all or set ABLATIONS=name1,name2)" >&2
  exit 2
}

bool_true() { runtime_bool_true "$1"; }

is_publication_artifact() {
  local artifact="$1" regime="$2" target_keys="$3" latency_contract="${4:-}"
  [[ -f "$artifact" ]] || return 1
  local args=(
    --output "$artifact" --regime "$regime" --target-keys-file "$target_keys"
    --metric-semantics-version "$METRIC_SEMANTICS_VERSION"
    --max-steps "$MAX_STEPS" --replan-interval "$REPLAN_INTERVAL"
    --num-candidates "$NUM_CANDIDATES" --num-recovery-options "$NUM_RECOVERY_OPTIONS"
    --womd-role "${WOMD_ROLE:-validation}" --require-finite-timing --quiet
  )
  if [[ "$regime" == contact ]]; then
    args+=(--contact-anchor-manifest "$CONTACT_ANCHOR_MANIFEST_FILE")
  fi
  [[ -z "$latency_contract" ]] || args+=(--require-latency-contract "$latency_contract")
  python tools/check_publication_closed_loop_artifact.py "${args[@]}"
}

archive_incompatible_complete_artifact() {
  local run_dir="$1" artifact="$2"
  [[ -f "$artifact" ]] || return 0
  local stamp archive f
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive="$run_dir/incompatible_pre_publication_contract_$stamp"
  mkdir -p "$archive"
  for f in "$artifact" "$artifact.progress.json" "$artifact.partial" "$artifact.scenes.jsonl" "${artifact%.json}.log"; do
    [[ -e "$f" ]] && mv "$f" "$archive/"
  done
  echo "[ARCHIVE] incompatible completed artifact moved to $archive"
}

archive_incompatible_partial_contact_if_needed() {
  local run_dir="$1" artifact="$2" regime="$3"
  [[ "$regime" == contact ]] || return 0
  [[ -f "$artifact" ]] && return 0
  local journal="$artifact.scenes.jsonl"
  [[ -s "$journal" ]] || return 0
  # A current-protocol Contact partial must already carry the frozen anchor
  # identity on every completed scene. Old counterfactual-contact partials did
  # not, so never feed them into a new anchored RESUME run.
  if python - "$journal" <<'PY2'
import json,sys
p=sys.argv[1]
seen=0
with open(p,encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        raw=json.loads(line); scene=raw.get('scene',raw) if isinstance(raw,dict) else None
        if not isinstance(scene,dict):
            continue
        seen += 1
        if scene.get('contact_anchor_protocol') != 'exact_a0_pretreatment_prelude_v1':
            raise SystemExit(30)
        if not scene.get('contact_anchor_fingerprint') or scene.get('contact_anchor_found') is not True:
            raise SystemExit(30)
raise SystemExit(0 if seen else 30)
PY2
  then
    return 0
  fi
  local stamp archive f
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive="$run_dir/incompatible_pre_publication_contract_$stamp"
  mkdir -p "$archive"
  for f in "$artifact.progress.json" "$artifact.partial" "$artifact.scenes.jsonl" "${artifact%.json}.log"; do
    [[ -e "$f" ]] && mv "$f" "$archive/"
  done
  echo "[ARCHIVE] incompatible pre-anchor Contact partial moved to $archive"
}

write_phase() {
  local root="$1" regime="$2" status="$3" rc="$4" started="$5" ended="$6"
  python - "$root/$regime.phase.json" "$regime" "$status" "$rc" "$started" "$ended" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
json.dump({'regime':sys.argv[2],'status':sys.argv[3],'exit_code':int(sys.argv[4]),
           'started_at':sys.argv[5],'ended_at':sys.argv[6]},p.open('w'),indent=2)
PY
}

variant_paths() {
  local variant="$1"
  local root="$MODEL_RUN/candidates/$variant"
  if [[ ! -f "$root/model_v48_trac_sr/best.pt" && -f "$MODEL_RUN/dedicated_candidates/$variant/model_v48_trac_sr/best.pt" ]]; then
    root="$MODEL_RUN/dedicated_candidates/$variant"
  fi
  local checkpoint="$root/model_v48_trac_sr/best.pt"
  local gamma_json="$root/calibration/gamma_rec_by_bucket_v48.json"
  [[ -f "$checkpoint" ]] || { echo "Missing OC-RAP checkpoint: $checkpoint" >&2; return 2; }
  [[ -f "$gamma_json" ]] || { echo "Missing bucket calibration JSON: $gamma_json" >&2; return 2; }
  printf '%s\t%s\n' "$checkpoint" "$gamma_json"
}

read_gammas() {
  local gamma_json="$1"
  python - "$gamma_json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['gamma_rec_by_bucket']
print(x['test_safe'], x['test_near_contact'], x['test_contact'])
PY
}

# Manifest is intentionally path/name compatible with the original launcher.
MANIFEST="$OUT_ROOT/ablation_run_manifest.tsv"
printf 'arm\tconfig\tsafe\tnear\tcontact\ttier\tdescription\n' > "$MANIFEST"
SELECTED_SPECS=()
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
  selected "$arm" "$tier" || continue
  [[ -f "$config" ]] || { echo "Missing ablation config: $config" >&2; exit 30; }
  SELECTED_SPECS+=("$spec")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$config" "$run_safe" "$run_near" "$run_contact" "$tier" "$description" >> "$MANIFEST"
done
((${#SELECTED_SPECS[@]})) || { echo "No ablations selected." >&2; exit 2; }

IFS=',' read -r -a variants <<< "$VARIANTS"
CLEAN_VARIANTS=()
for raw in "${variants[@]}"; do
  v="$(echo "$raw" | xargs)"; [[ -n "$v" ]] && CLEAN_VARIANTS+=("$v")
done
((${#CLEAN_VARIANTS[@]})) || { echo "No variants selected." >&2; exit 2; }

# Use at most two physical devices as in the original experiment. The number
# of independent accuracy processes per device is separately configurable.
IFS=',' read -r -a _gpu_raw <<< "$CUDA_DEVICES"
GPUS=()
for raw in "${_gpu_raw[@]}"; do g="$(echo "$raw" | xargs)"; [[ -n "$g" ]] && GPUS+=("$g"); done
((${#GPUS[@]})) || GPUS=(0)
if ((${#GPUS[@]} > 2)); then GPUS=("${GPUS[0]}" "${GPUS[1]}"); fi
LATENCY_GPU="${LATENCY_GPU:-${GPUS[0]}}"

echo "[Ablation scheduler] GPUs=${GPUS[*]} variants=${CLEAN_VARIANTS[*]} set=$ABLATION_SET"
echo "[Ablation contract] frozen checkpoint + frozen per-bucket gamma; LABEL_MODE=$LABEL_MODE; PROFILE_TIMING=$PROFILE_TIMING"
echo "[Ablation causal Full] native-certified reference=$NATIVE_FULL_REFERENCE_ROOT (historical FULL_RUN_ROOT=$FULL_RUN_ROOT is not used for causal tables unless NATIVE_FULL_REFERENCE_ROOT is pointed there)"

# Resolve bucket provenance once and preflight each immutable regime/WOMD pair
# once.  Every arm/variant reuses the resulting verified JSON contract.
SAFE_WOMD="$(runtime_resolve_bucket_womd_spec "$SAFE_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "${WOMD_ROLE:-validation}")"
NEAR_WOMD="$(runtime_resolve_bucket_womd_spec "$NEAR_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "${WOMD_ROLE:-validation}")"
CONTACT_WOMD="$(runtime_resolve_bucket_womd_spec "$CONTACT_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "${WOMD_ROLE:-validation}")"
# Re-run the same deterministic target-lock builder used by final
# characterization unless a top-level orchestrator has already frozen it.
# BUILD_TARGET_LOCKS=false is safe only when all three locks and the Contact
# anchor manifest already exist; the fail-closed checks below still validate
# the full contract before any ablation job is queued.
if bool_true "$BUILD_TARGET_LOCKS"; then
  env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$TARGET_LOCK_CHARACTERIZATION_OUT" \
    WOMD_ROLE="${WOMD_ROLE:-validation}" FINAL_MAX_STEPS="$MAX_STEPS" \
    CONTACT_ANCHOR_GPU="${CONTACT_ANCHOR_GPU:-${GPUS[0]}}" \
    bash scripts/build_final_observation_legal_target_locks.sh
else
  echo "[Ablation target lock] reuse prebuilt final target locks under $FINAL_TARGET_LOCK_ROOT"
fi
for _r in safe near contact; do
  [[ -s "$FINAL_TARGET_LOCK_ROOT/$_r.json" ]] || { echo "Missing final target lock: $FINAL_TARGET_LOCK_ROOT/$_r.json" >&2; exit 30; }
done
[[ -s "$CONTACT_ANCHOR_MANIFEST_FILE" ]] || { echo "Missing final Contact anchor manifest: $CONTACT_ANCHOR_MANIFEST_FILE" >&2; exit 30; }
# Prove that contact.json, the anchor manifest and this run's treatment horizon
# are one coherent frozen contract before any ablation job is queued.
python - "$FINAL_TARGET_LOCK_ROOT/contact.json" "$CONTACT_ANCHOR_MANIFEST_FILE" "$MAX_STEPS" <<'PY2'
import hashlib,json,pathlib,sys
lock_p=pathlib.Path(sys.argv[1]); manifest_p=pathlib.Path(sys.argv[2]); max_steps=int(sys.argv[3])
lock=json.loads(lock_p.read_text(encoding='utf-8')); manifest=json.loads(manifest_p.read_text(encoding='utf-8'))
keys=set(lock.get('target_keys') or []); mkeys=set(manifest.get('target_keys') or [])
cc=lock.get('contact_anchor_contract') or {}; sha=hashlib.sha256(manifest_p.read_bytes()).hexdigest()
checks={
 'lock_schema': lock.get('schema')=='ocrap-observation-legal-contact-anchor-target-lock-v1',
 'protocol': cc.get('protocol')=='exact_a0_pretreatment_prelude_v1',
 'manifest_sha': cc.get('manifest_sha256')==sha,
 'fingerprint_required': cc.get('state_fingerprint_required') is True,
 'manifest_valid': manifest.get('valid') is True,
 'pre_treatment_policy': manifest.get('pre_treatment_policy')=='exact_a0',
 'target_keys_equal': bool(keys) and keys==mkeys,
 'scene_disjoint': int(manifest.get('num_selected_anchors') or 0)==int(manifest.get('num_selected_scenes') or -1)==len(keys),
 'full_horizon': int(manifest.get('min_post_steps') or 0)>=max_steps,
}
bad=[k for k,v in checks.items() if not v]
if bad: raise SystemExit('invalid final Contact target/anchor contract: '+json.dumps({'failed':bad,'lock':str(lock_p),'manifest':str(manifest_p)}))
print(json.dumps({'event':'ablation_contact_anchor_contract_verified','targets':len(keys),'manifest_sha256':sha,'min_post_steps':manifest.get('min_post_steps')}))
PY2
PREFLIGHT_ROOT="$OUT_ROOT/_shared_preflight_${RUN_TAG}"
mkdir -p "$PREFLIGHT_ROOT"

# The fresh native-certified Full reference is part of the causal ablation
# experiment, so all three immutable strata are always preflighted once.
declare -A NEED_REGIME=( [safe]=1 [near]=1 [contact]=1 )
for spec in "${SELECTED_SPECS[@]}"; do
  IFS='|' read -r _ _ rs rn rc _ _ <<< "$spec"
  [[ "$rs" == 1 ]] && NEED_REGIME[safe]=1
  [[ "$rn" == 1 ]] && NEED_REGIME[near]=1
  [[ "$rc" == 1 ]] && NEED_REGIME[contact]=1
done

preflight_one() {
  # Under `set -u`, do not reference a variable from another assignment in the
  # same `local` command: RHS expansion happens before the assignments become
  # visible, so `$regime` would be considered unbound here.
  local regime="$1"
  local bucket="$2"
  local womd="$3"
  local keyfile="$FINAL_TARGET_LOCK_ROOT/$regime.json"
  local out="$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json"
  if [[ "${NEED_REGIME[$regime]}" != 1 ]]; then return 0; fi
  echo "[PREFLIGHT once] regime=$regime"
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$bucket" --split "$BUCKET_SPLIT" \
    --womd-pattern "$womd" --expected-source-role "${WOMD_ROLE:-validation}" \
    --target-keys-file "$keyfile" --require-target-keys \
    --output "$out"
}
preflight_one safe "$SAFE_BUCKET" "$SAFE_WOMD"
preflight_one near "$NEAR_BUCKET" "$NEAR_WOMD"
preflight_one contact "$CONTACT_BUCKET" "$CONTACT_WOMD"

# Validate each frozen deployed owner once, archive legacy wrong outputs once,
# and materialize independent regime jobs using the original output layout.
QUEUE_ROOT="$OUT_ROOT/.${RUN_TAG}.queue.$$"
mkdir -p "$QUEUE_ROOT/pending" "$QUEUE_ROOT/running" "$QUEUE_ROOT/failed" "$QUEUE_ROOT/done"
job_id=0
ROOTS_TO_FINALIZE=()
for variant in "${CLEAN_VARIANTS[@]}"; do
  python tools/check_deployable_stack.py \
    --model-run "$MODEL_RUN" --variant "$variant" \
    --output "$OUT_ROOT/DEPLOYABLE-STACK-${variant}.json"
  IFS=$'\t' read -r checkpoint gamma_json < <(variant_paths "$variant")
  read -r gamma_safe gamma_near gamma_contact < <(read_gammas "$gamma_json")

  # A paper-faithful ablation needs a Full reference with the same native
  # certification path.  Existing characterization artifacts predate this
  # bridge and are therefore not a valid counterfactual for the three repaired
  # physical-semantic knockouts.  Build/reuse this reference inside OUT_ROOT.
  ref_root="$NATIVE_FULL_REFERENCE_ROOT/$variant"
  mkdir -p "$ref_root/safe" "$ref_root/near" "$ref_root/contact"
  ROOTS_TO_FINALIZE+=("$ref_root")
  for regime in safe near contact; do
    case "$regime" in
      safe) gamma="$gamma_safe"; womd="$SAFE_WOMD"; bucket="$SAFE_BUCKET" ;;
      near) gamma="$gamma_near"; womd="$NEAR_WOMD"; bucket="$NEAR_BUCKET" ;;
      contact) gamma="$gamma_contact"; womd="$CONTACT_WOMD"; bucket="$CONTACT_BUCKET" ;;
    esac
    # Long Contact jobs are claimed first to reduce the final slow-job tail.
    # Job contents and per-scene ordering are untouched.
    case "$regime" in contact) priority=0 ;; near) priority=1 ;; safe) priority=2 ;; esac
    job_id=$((job_id+1)); jf="$QUEUE_ROOT/pending/${priority}.$(printf '%04d' "$job_id").job"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "_native_full_reference" "$variant" "$regime" "$NATIVE_FULL_REFERENCE_CONFIG" "$checkpoint" "$gamma" "$womd" "$bucket" "$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json" "$FINAL_TARGET_LOCK_ROOT/$regime.json" > "$jf"
  done
  python tools/build_ocrap_three_regime_index.py --root "$ref_root" --launcher-exit-code 1 >/dev/null || true

  for spec in "${SELECTED_SPECS[@]}"; do
    IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
        root="$OUT_ROOT/$arm/$variant"
    mkdir -p "$root/safe" "$root/near" "$root/contact"
    ROOTS_TO_FINALIZE+=("$root")

    # Preserve the original three-regime index semantics for intentionally
    # unevaluated strata.
    now="$(runtime_iso_now)"
    [[ "$run_safe" == 1 ]] || write_phase "$root" safe skipped 0 "$now" "$now"
    [[ "$run_near" == 1 ]] || write_phase "$root" near skipped 0 "$now" "$now"
    [[ "$run_contact" == 1 ]] || write_phase "$root" contact skipped 0 "$now" "$now"

    for regime in safe near contact; do
      case "$regime" in
        safe) enabled="$run_safe"; gamma="$gamma_safe"; womd="$SAFE_WOMD"; bucket="$SAFE_BUCKET" ;;
        near) enabled="$run_near"; gamma="$gamma_near"; womd="$NEAR_WOMD"; bucket="$NEAR_BUCKET" ;;
        contact) enabled="$run_contact"; gamma="$gamma_contact"; womd="$CONTACT_WOMD"; bucket="$CONTACT_BUCKET" ;;
      esac
      [[ "$enabled" == 1 ]] || continue
      case "$regime" in contact) priority=0 ;; near) priority=1 ;; safe) priority=2 ;; esac
      job_id=$((job_id+1)); jf="$QUEUE_ROOT/pending/${priority}.$(printf '%04d' "$job_id").job"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$arm" "$variant" "$regime" "$config" "$checkpoint" "$gamma" "$womd" "$bucket" "$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json" "$FINAL_TARGET_LOCK_ROOT/$regime.json" > "$jf"
    done
    python tools/build_ocrap_three_regime_index.py --root "$root" --launcher-exit-code 1 >/dev/null || true
  done
done

echo "[Ablation scheduler] queued regime jobs=$job_id; GPUs=${#GPUS[@]}; accuracy workers/GPU=$WORKERS_PER_GPU; total processes up to $(( ${#GPUS[@]} * WORKERS_PER_GPU ))"

claim_job() {
  local gpu="$1" slot="$2" lock="$QUEUE_ROOT/.claim_lock" f base claimed
  while ! mkdir "$lock" 2>/dev/null; do sleep 0.05; done
  f="$(find "$QUEUE_ROOT/pending" -maxdepth 1 -type f -name '*.job' -print | LC_ALL=C sort | head -n 1 || true)"
  if [[ -z "$f" ]]; then rmdir "$lock"; return 1; fi
  base="$(basename "$f")"; claimed="$QUEUE_ROOT/running/${base%.job}.gpu${gpu}.slot${slot}.job"
  mv "$f" "$claimed"; rmdir "$lock"; printf '%s\n' "$claimed"
}

run_claimed_job() {
  local gpu="$1" slot="$2" jf="$3"
  local arm variant regime config checkpoint gamma womd bucket preflight target_keys
  IFS=$'\t' read -r arm variant regime config checkpoint gamma womd bucket preflight target_keys < "$jf"
  # Same nounset rule as preflight_one(): initialize dependent locals in order.
  local root
  if [[ "$arm" == "_native_full_reference" ]]; then
    root="$NATIVE_FULL_REFERENCE_ROOT/$variant"
  else
    root="$OUT_ROOT/$arm/$variant"
  fi
  local run_dir="$root/$regime"
  local artifact="$run_dir/closed_loop_ocrap.json"
  local started="$(runtime_iso_now)" rc=0
  local base_check=(--output "$artifact" --quiet --target-keys-file "$target_keys" --require-metric-semantics-version "$METRIC_SEMANTICS_VERSION")
  local contact_anchor_env=()
  if [[ "$regime" == contact ]]; then
    contact_anchor_env=(
      CONTACT_ANCHOR_PRELUDE_ENABLED=true
      CONTACT_ANCHOR_PRELUDE_MAX_STEPS="$CONTACT_ANCHOR_PRELUDE_MAX_STEPS"
      CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL="$CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL"
      CONTACT_ANCHOR_REQUIRE_FOUND="$CONTACT_ANCHOR_REQUIRE_FOUND"
      CONTACT_ANCHOR_MANIFEST_FILE="$CONTACT_ANCHOR_MANIFEST_FILE"
    )
  fi

  if bool_true "$SKIP_COMPLETE" && python tools/check_closed_loop_artifact.py "${base_check[@]}" && is_publication_artifact "$artifact" "$regime" "$target_keys"; then
    echo "[REUSE] gpu=$gpu arm=$arm variant=$variant regime=$regime"
    write_phase "$root" "$regime" complete 0 "$started" "$(runtime_iso_now)"
    mv "$jf" "$QUEUE_ROOT/done/$(basename "$jf")"
    return 0
  fi
  # Do not resume an old counterfactual-Contact partial under the new exact-a0
  # anchor protocol. Current-protocol partials remain resumable and the runner
  # still verifies its full result-affecting fingerprint.
  archive_incompatible_partial_contact_if_needed "$run_dir" "$artifact" "$regime"
  # A completed pre-fix artifact must not be fed into resume.
  if [[ -f "$artifact" ]] && ! is_publication_artifact "$artifact" "$regime" "$target_keys"; then
    archive_incompatible_complete_artifact "$run_dir" "$artifact"
  fi

  echo "[RUN] gpu=$gpu slot=$slot arm=$arm variant=$variant regime=$regime -> $artifact"
  write_phase "$root" "$regime" running 0 "$started" ""
  if env \
      RUN_DIR="$run_dir" OUTPUT="$artifact" \
      WOMD_VAL="$womd" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" EXPECTED_WOMD_ROLE="$WOMD_ROLE" \
      CHECKPOINT="$checkpoint" GAMMA_REC="$gamma" GPU="$gpu" \
      MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" REPLAN_INTERVAL="$REPLAN_INTERVAL" \
      METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
      LABEL_MODE="$LABEL_MODE" AUDIT_EVERY_N_STEPS="$AUDIT_EVERY_N_STEPS" \
      NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
      BUCKET_DATASET="$bucket" BUCKET_SPLIT="$BUCKET_SPLIT" MAX_TARGETS_PER_SCENE=1 \
      TARGET_KEYS_FILE="$target_keys" REQUIRE_TARGET_KEYS=true \
      USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true \
      ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false \
      CONFIG="$config" RENDER_TRACE=false SAVE_PARTIAL=true RESUME=true RESUME_FORCE=false \
      PROFILE_TIMING="$PROFILE_TIMING" PREFLIGHT_SUPPORT_JSON="$preflight" \
      JAX_CACHE_DIR="$OUT_ROOT/.jax_compilation_cache/gpu${gpu}/slot${slot}" \
      PARTIAL_WRITE_EVERY_SCENES="$PARTIAL_WRITE_EVERY_SCENES" PROGRESS_EVERY_STEPS="$PROGRESS_EVERY_STEPS" \
      "${contact_anchor_env[@]}" \
      bash scripts/run_ocrap_closed_loop.sh; then
    if python tools/check_closed_loop_artifact.py "${base_check[@]}" && is_publication_artifact "$artifact" "$regime" "$target_keys"; then rc=0; else rc=91; fi
  else
    rc=$?
  fi

  if ((rc==0)); then
    write_phase "$root" "$regime" complete 0 "$started" "$(runtime_iso_now)"
    mv "$jf" "$QUEUE_ROOT/done/$(basename "$jf")"
    echo "[DONE] gpu=$gpu slot=$slot arm=$arm variant=$variant regime=$regime"
    return 0
  fi
  write_phase "$root" "$regime" failed "$rc" "$started" "$(runtime_iso_now)"
  mv "$jf" "$QUEUE_ROOT/failed/$(basename "$jf")"
  echo "[FAILED] gpu=$gpu slot=$slot arm=$arm variant=$variant regime=$regime rc=$rc" >&2
  return 0  # keep this GPU worker alive to finish the remaining independent jobs
}

worker() {
  local gpu="$1" slot="$2" jf
  mkdir -p "$OUT_ROOT/.jax_compilation_cache/gpu${gpu}/slot${slot}"
  while jf="$(claim_job "$gpu" "$slot")"; do
    run_claimed_job "$gpu" "$slot" "$jf"
  done
}

PIDS=()
for gpu in "${GPUS[@]}"; do
  for ((slot=0; slot<WORKERS_PER_GPU; slot++)); do
    worker "$gpu" "$slot" & PIDS+=("$!")
  done
done
for p in "${PIDS[@]}"; do wait "$p"; done

failed_count="$(find "$QUEUE_ROOT/failed" -maxdepth 1 -type f -name '*.job' | wc -l | xargs)"
launcher_rc=0; [[ "$failed_count" == 0 ]] || launcher_rc=1

# Rebuild the exact legacy-compatible per-arm/variant indices after all workers
# have stopped, so no two processes write the same summary concurrently.
for root in "${ROOTS_TO_FINALIZE[@]}"; do
  python tools/build_ocrap_three_regime_index.py --root "$root" --launcher-exit-code "$launcher_rc" >/dev/null || true
done

# Record the execution/scientific contract next to the original manifest.
python - "$OUT_ROOT/ablation_execution_contract.json" "$RUN_TAG" "$LABEL_MODE" "$PROFILE_TIMING" "$failed_count" "$CUDA_DEVICES" "$ABLATION_RUN_ID" "$NATIVE_FULL_REFERENCE_ROOT" "$FULL_RUN_ROOT" "$METRIC_SEMANTICS_VERSION" "$FINAL_TARGET_LOCK_ROOT" "$CONTACT_ANCHOR_MANIFEST_FILE" "$PROFILE_ISOLATED_LATENCY" "$LATENCY_ROOT" "$WORKERS_PER_GPU" <<'PY'
import hashlib,json,sys,datetime,pathlib
(out,tag,label,profile,failed,gpus,run_id,native_full_root,historical_full_root,
 metric_semantics,target_lock_root,contact_manifest,isolated_latency,latency_root,workers_per_gpu)=sys.argv[1:]
mp=pathlib.Path(contact_manifest)
manifest_sha=hashlib.sha256(mp.read_bytes()).hexdigest() if mp.is_file() else None
doc={
  'schema_version': 3,
  'tag': tag,
  'run_instance_id': run_id,
  'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'ablation_type': 'frozen-checkpoint inference-time functional knockout',
  'checkpoint_policy': 'same frozen V48.80 checkpoint within each variant',
  'calibration_policy': 'same frozen per-bucket gamma_rec as the Full native-certified reference; no recalibration',
  'native_recovery_certification': 'enabled before OC-MERO for Full and every submission ablation arm',
  'native_full_reference_root': native_full_root,
  'historical_full_run_root_not_used_for_causal_tables': historical_full_root,
  'metric_semantics_version': metric_semantics,
  'metric_contract': 'v55 exact signed OBB clearance/penetration + swept-SAT CV TTC, t0 included, left-endpoint duration/AUC support, missing metrics fail closed',
  'target_lock_root': target_lock_root,
  'contact_anchor_protocol': 'exact_a0_pretreatment_prelude_v1',
  'contact_anchor_manifest': contact_manifest,
  'contact_anchor_manifest_sha256': manifest_sha,
  'accuracy_timing_is_diagnostic_only': True,
  'publication_latency_enabled': isolated_latency.lower() in {'1','true','yes','on'},
  'publication_latency_root': latency_root if isolated_latency.lower() in {'1','true','yes','on'} else None,
  'publication_latency_execution_contract': 'isolated_single_process_single_gpu' if isolated_latency.lower() in {'1','true','yes','on'} else 'omitted',
  'latency_scope': 'state_history + candidate_features + policy_selection only; teacher/audit/Waymax bookkeeping excluded',
  'label_mode': label,
  'profile_timing': profile.lower() in {'1','true','yes','on'},
  'scheduler': 'dynamic shared queue, configurable independent closed-loop processes per GPU for accuracy; isolated serial single-GPU post-pass for publication latency',
  'cuda_devices': gpus,
  'accuracy_workers_per_gpu': int(workers_per_gpu),
  'failed_job_count': int(failed),
  'interpretation_notes': {
    'no_actuator_projection': 're-rolls executable recovery without actuator-envelope projection and restores the post-hoc control barrier before OC-MERO',
    'no_persistent_reentry': 'removes the persistent re-entry barrier from the native root-option certificate before OC-MERO',
    'no_route_alignment': 'removes the executable route-consistency barrier from the native root-option certificate before OC-MERO',
    'without_obs_or_tail': 'changes OC-MERO inference aggregation while retaining frozen learned heads/representation',
    'no_rifa_absolute_admission': 'removes the absolute deployable-recovery admission predicate while retaining hard/harm feasibility and frozen scoring',
    'no_nominal_abstention': 'keeps absolute admission but re-enables the legacy recovery-first fallback when no candidate is admitted',
    'target_cohort': 'all arms use the exact final method-independent observation-legal target locks; Contact additionally reproduces the same frozen actual-contact state fingerprint per target',
    'causal_full_reference': 'the fresh _native_full_reference has native_recovery_certification enabled; historical FULL_RUN_ROOT is provenance only and is not a one-factor causal reference for physical-semantic knockouts',
  },
}
open(out,'w',encoding='utf-8').write(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY

# Timing-only postprocessing: reads already committed artifacts; it never
# changes the experiment data or blocks independent worker execution.
python tools/analyze_ablation_bottlenecks.py --root "$OUT_ROOT" \
  --full-root "$NATIVE_FULL_REFERENCE_ROOT" \
  --output "$OUT_ROOT/ablation_bottlenecks.csv" || {
    echo "[WARN] Post-hoc timing analysis unavailable (accuracy results are untouched)" >&2
  }

if [[ "$failed_count" != 0 ]]; then
  echo "Ablation workers finished with $failed_count failed regime job(s). Queue retained at: $QUEUE_ROOT" >&2
  exit 1
fi
rm -rf "$QUEUE_ROOT"

# Publication latency is measured with the same execution contract as the final
# characterization. Accuracy workers above may run concurrently on two GPUs;
# those inline timings are useful diagnostics but are never substituted for the
# isolated latency column in a paper table.
run_isolated_latency_one() {
  local arm="$1" variant="$2" regime="$3" config="$4" checkpoint="$5" gamma="$6" womd="$7" bucket="$8" preflight="$9" target_keys="${10}"
  local run_dir="$LATENCY_ROOT/$arm/$variant/$regime"
  local artifact="$run_dir/closed_loop_ocrap.json"
  local contact_anchor_env=()
  if [[ "$regime" == contact ]]; then
    contact_anchor_env=(
      CONTACT_ANCHOR_PRELUDE_ENABLED=true
      CONTACT_ANCHOR_PRELUDE_MAX_STEPS="$CONTACT_ANCHOR_PRELUDE_MAX_STEPS"
      CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL="$CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL"
      CONTACT_ANCHOR_REQUIRE_FOUND="$CONTACT_ANCHOR_REQUIRE_FOUND"
      CONTACT_ANCHOR_MANIFEST_FILE="$CONTACT_ANCHOR_MANIFEST_FILE"
    )
  fi
  mkdir -p "$run_dir" "$LATENCY_ROOT/.jax_compilation_cache/gpu${LATENCY_GPU}"
  local base_check=(--output "$artifact" --quiet --target-keys-file "$target_keys" --require-metric-semantics-version "$METRIC_SEMANTICS_VERSION" --require-latency-contract isolated_single_process_single_gpu)
  if bool_true "$SKIP_COMPLETE" && python tools/check_closed_loop_artifact.py "${base_check[@]}" && is_publication_artifact "$artifact" "$regime" "$target_keys" isolated_single_process_single_gpu; then
    echo "[LATENCY REUSE] gpu=$LATENCY_GPU arm=$arm variant=$variant regime=$regime"
    return 0
  fi
  archive_incompatible_partial_contact_if_needed "$run_dir" "$artifact" "$regime"
  if [[ -f "$artifact" ]] && ! is_publication_artifact "$artifact" "$regime" "$target_keys" isolated_single_process_single_gpu; then
    archive_incompatible_complete_artifact "$run_dir" "$artifact"
  fi
  echo "[LATENCY RUN] gpu=$LATENCY_GPU arm=$arm variant=$variant regime=$regime -> $artifact"
  env \
    RUN_DIR="$run_dir" OUTPUT="$artifact" \
    WOMD_VAL="$womd" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" EXPECTED_WOMD_ROLE="$WOMD_ROLE" \
    CHECKPOINT="$checkpoint" GAMMA_REC="$gamma" GPU="$LATENCY_GPU" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" REPLAN_INTERVAL="$REPLAN_INTERVAL" \
    METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    LABEL_MODE="$LABEL_MODE" AUDIT_EVERY_N_STEPS="$AUDIT_EVERY_N_STEPS" \
    NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    BUCKET_DATASET="$bucket" BUCKET_SPLIT="$BUCKET_SPLIT" MAX_TARGETS_PER_SCENE=1 \
    TARGET_KEYS_FILE="$target_keys" REQUIRE_TARGET_KEYS=true \
    USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true \
    ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false \
    CONFIG="$config" RENDER_TRACE=false SAVE_PARTIAL=true RESUME=true RESUME_FORCE=false \
    PROFILE_TIMING=true LATENCY_EXECUTION_CONTRACT=isolated_single_process_single_gpu \
    LATENCY_WARMUP_DECISIONS="$LATENCY_WARMUP_DECISIONS" PREFLIGHT_SUPPORT_JSON="$preflight" \
    JAX_CACHE_DIR="$LATENCY_ROOT/.jax_compilation_cache/gpu${LATENCY_GPU}" \
    PARTIAL_WRITE_EVERY_SCENES="$PARTIAL_WRITE_EVERY_SCENES" PROGRESS_EVERY_STEPS="$PROGRESS_EVERY_STEPS" \
    "${contact_anchor_env[@]}" \
    bash scripts/run_ocrap_closed_loop.sh
  python tools/check_closed_loop_artifact.py "${base_check[@]}"
  is_publication_artifact "$artifact" "$regime" "$target_keys" isolated_single_process_single_gpu
}

if bool_true "$PROFILE_ISOLATED_LATENCY"; then
  echo "[Ablation latency] isolated single-process/single-GPU profiling on GPU $LATENCY_GPU"
  mkdir -p "$LATENCY_ROOT"
  for variant in "${CLEAN_VARIANTS[@]}"; do
    IFS=$'\t' read -r checkpoint gamma_json < <(variant_paths "$variant")
    read -r gamma_safe gamma_near gamma_contact < <(read_gammas "$gamma_json")
    for regime in safe near contact; do
      case "$regime" in
        safe) gamma="$gamma_safe"; womd="$SAFE_WOMD"; bucket="$SAFE_BUCKET" ;;
        near) gamma="$gamma_near"; womd="$NEAR_WOMD"; bucket="$NEAR_BUCKET" ;;
        contact) gamma="$gamma_contact"; womd="$CONTACT_WOMD"; bucket="$CONTACT_BUCKET" ;;
      esac
      run_isolated_latency_one "_native_full_reference" "$variant" "$regime" "$NATIVE_FULL_REFERENCE_CONFIG" "$checkpoint" "$gamma" "$womd" "$bucket" "$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json" "$FINAL_TARGET_LOCK_ROOT/$regime.json"
    done
    for spec in "${SELECTED_SPECS[@]}"; do
      IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
      for regime in safe near contact; do
        case "$regime" in
          safe) enabled="$run_safe"; gamma="$gamma_safe"; womd="$SAFE_WOMD"; bucket="$SAFE_BUCKET" ;;
          near) enabled="$run_near"; gamma="$gamma_near"; womd="$NEAR_WOMD"; bucket="$NEAR_BUCKET" ;;
          contact) enabled="$run_contact"; gamma="$gamma_contact"; womd="$CONTACT_WOMD"; bucket="$CONTACT_BUCKET" ;;
        esac
        [[ "$enabled" == 1 ]] || continue
        run_isolated_latency_one "$arm" "$variant" "$regime" "$config" "$checkpoint" "$gamma" "$womd" "$bucket" "$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json" "$FINAL_TARGET_LOCK_ROOT/$regime.json"
      done
    done
  done
fi

# Refresh read-only report now that isolated latency artifacts are available.
# Never include stale latency files when the user disabled latency profiling.
latency_report_args=()
if bool_true "$PROFILE_ISOLATED_LATENCY"; then
  latency_report_args=(--latency-root "$LATENCY_ROOT")
fi
python tools/analyze_ablation_bottlenecks.py --root "$OUT_ROOT" \
  --full-root "$NATIVE_FULL_REFERENCE_ROOT" "${latency_report_args[@]}" \
  --output "$OUT_ROOT/ablation_bottlenecks.csv" || {
    echo "[WARN] Final timing analysis unavailable (results are untouched)" >&2
  }

# One final fail-closed sweep over every selected paper-facing artifact. This
# prevents a table from silently dropping a selected arm or accepting a file
# whose per-scene Contact anchor identity changed after the worker completed.
publication_artifact_count=0
verify_ablation_publication_artifact() {
  local arm="$1" variant="$2" regime="$3"
  local acc_root artifact target_keys latency_artifact
  if [[ "$arm" == "_native_full_reference" ]]; then
    acc_root="$NATIVE_FULL_REFERENCE_ROOT/$variant"
  else
    acc_root="$OUT_ROOT/$arm/$variant"
  fi
  artifact="$acc_root/$regime/closed_loop_ocrap.json"
  target_keys="$FINAL_TARGET_LOCK_ROOT/$regime.json"
  is_publication_artifact "$artifact" "$regime" "$target_keys" || {
    echo "Publication ablation audit failed: arm=$arm variant=$variant regime=$regime artifact=$artifact" >&2
    return 30
  }
  publication_artifact_count=$((publication_artifact_count+1))
  if bool_true "$PROFILE_ISOLATED_LATENCY"; then
    latency_artifact="$LATENCY_ROOT/$arm/$variant/$regime/closed_loop_ocrap.json"
    is_publication_artifact "$latency_artifact" "$regime" "$target_keys" isolated_single_process_single_gpu || {
      echo "Publication ablation latency audit failed: arm=$arm variant=$variant regime=$regime artifact=$latency_artifact" >&2
      return 30
    }
  fi
}
for variant in "${CLEAN_VARIANTS[@]}"; do
  for regime in safe near contact; do
    verify_ablation_publication_artifact "_native_full_reference" "$variant" "$regime"
  done
  for spec in "${SELECTED_SPECS[@]}"; do
    IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
    for regime in safe near contact; do
      case "$regime" in
        safe) enabled="$run_safe" ;;
        near) enabled="$run_near" ;;
        contact) enabled="$run_contact" ;;
      esac
      [[ "$enabled" == 1 ]] || continue
      verify_ablation_publication_artifact "$arm" "$variant" "$regime"
    done
  done
done
python - "$OUT_ROOT/ablation_metric_fairness_audit.json" "$publication_artifact_count" "$PROFILE_ISOLATED_LATENCY" "$METRIC_SEMANTICS_VERSION" "$CONTACT_ANCHOR_MANIFEST_FILE" <<'PY'
import hashlib,json,pathlib,sys,datetime
out,count,latency,metric,manifest=sys.argv[1:]
mp=pathlib.Path(manifest)
doc={
  'schema_version': 1,
  'valid': True,
  'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'accuracy_artifact_count': int(count),
  'isolated_latency_artifacts_verified': latency.lower() in {'1','true','yes','on'},
  'metric_semantics_version': metric,
  'contact_anchor_protocol': 'exact_a0_pretreatment_prelude_v1',
  'contact_anchor_manifest': manifest,
  'contact_anchor_manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
  'checks': [
    'exact frozen target-key set',
    'evaluation-contract equality to final publication semantics',
    'complete clearance/TTC/overlap/offroad coverage',
    'observation-legal route contract',
    'Contact exact-a0 manifest and per-scene state fingerprint reproduction',
    '100% observed-contact and post-contact eligibility for Contact',
    'isolated single-process/single-GPU latency contract when enabled',
  ],
}
pathlib.Path(out).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY

# Paper tables are built only against the fresh native-certified Full reference
# produced by this launcher, paired on the same immutable target keys.
if bool_true "$BUILD_TABLES"; then
  full_ready=1
  for variant in "${CLEAN_VARIANTS[@]}"; do
    for regime in safe near contact; do
      artifact="$NATIVE_FULL_REFERENCE_ROOT/$variant/$regime/closed_loop_ocrap.json"
      target_keys="$FINAL_TARGET_LOCK_ROOT/$regime.json"
      if ! python tools/check_closed_loop_artifact.py --output "$artifact" --quiet --target-keys-file "$target_keys" --require-metric-semantics-version "$METRIC_SEMANTICS_VERSION" || ! is_publication_artifact "$artifact" "$regime" "$target_keys"; then
        full_ready=0
      fi
      if bool_true "$PROFILE_ISOLATED_LATENCY"; then
        latency_artifact="$LATENCY_ROOT/_native_full_reference/$variant/$regime/closed_loop_ocrap.json"
        if ! is_publication_artifact "$latency_artifact" "$regime" "$target_keys" isolated_single_process_single_gpu; then
          full_ready=0
        fi
      fi
    done
  done
  if ((full_ready)); then
    tier="$ABLATION_SET"; [[ "$tier" == main || "$tier" == all ]] || tier=all
    table_args=(
      --full-run "$NATIVE_FULL_REFERENCE_ROOT" --ablation-root "$OUT_ROOT"
      --variants "$VARIANTS" --tier "$tier" --output-dir "$TABLE_OUT"
    )
    if bool_true "$PROFILE_ISOLATED_LATENCY"; then
      table_args+=(--latency-root "$LATENCY_ROOT")
    fi
    python tools/build_submission_ablation_tables.py "${table_args[@]}"
  else
    echo "[WARN] Full three-regime OC-RAP artifacts are not all complete under $NATIVE_FULL_REFERENCE_ROOT; skip paired ablation tables for now." >&2
  fi
fi

printf '\nAblations complete.\nRoot: %s\nManifest: %s\nContract: %s\n' \
  "$OUT_ROOT" "$MANIFEST" "$OUT_ROOT/ablation_execution_contract.json"
