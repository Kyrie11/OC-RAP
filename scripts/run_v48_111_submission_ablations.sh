#!/usr/bin/env bash
# Corrected-metric, frozen-checkpoint functional ablations for the V48.111 submission stack.
#
# Key properties:
#   * publication metrics come from the v55 exact OBB/TTC/penetration closed-loop core;
#   * latency is observation -> deployable action selection only (profile_timing remains on);
#   * stale pre-v55 results/journals are archived once before the corrected rerun;
#   * one independent closed-loop worker is bound to each GPU; workers dynamically claim jobs;
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
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
OUT_ROOT="${OUT_ROOT:-$BASE_OUT/ocrap_v48_111_submission_ablations}"
FULL_RUN_ROOT="${FULL_RUN_ROOT:-$BASE_OUT/ocrap_v48_111_submission_three_regime}"
VARIANTS="${VARIANTS:-balanced,precision}"
CUDA_DEVICES="${CUDA_DEVICES:-${GPU0:-0},${GPU1:-1}}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
WOMD_NUM_SHARDS="${WOMD_NUM_SHARDS:-150}"
BUCKET_SPLIT="${BUCKET_SPLIT:-test}"
SAFE_BUCKET="${SAFE_BUCKET:-$OCRAP_ROOT/test_safe}"
NEAR_BUCKET="${NEAR_BUCKET:-$OCRAP_ROOT/test_near_contact}"
CONTACT_BUCKET="${CONTACT_BUCKET:-$OCRAP_ROOT/test_contact}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
MAX_STEPS="${MAX_STEPS:-40}"
NUM_CANDIDATES="${NUM_CANDIDATES:-24}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
# main = five paper-facing ablations; supplementary adds active-set/route knockouts.
ABLATION_SET="${ABLATION_SET:-main}"
ABLATIONS="${ABLATIONS:-}"

# Correctness/performance controls.  RESET_STALE_RESULTS=true archives legacy
# artifacts only once per arm+variant; interrupted corrected runs can then resume.
CORRECTED_RUN_TAG="${CORRECTED_RUN_TAG:-v56_corrected_ablation_metrics}"
RESET_STALE_RESULTS="${RESET_STALE_RESULTS:-true}" # true|false|force
SKIP_COMPLETE_CORRECTED="${SKIP_COMPLETE_CORRECTED:-true}"
BUILD_TABLES="${BUILD_TABLES:-true}"
TABLE_OUT="${TABLE_OUT:-$OUT_ROOT/submission_tables}"
PARTIAL_WRITE_EVERY_SCENES="${PARTIAL_WRITE_EVERY_SCENES:-64}"
PROGRESS_EVERY_STEPS="${PROGRESS_EVERY_STEPS:-20}"
PROFILE_TIMING="${PROFILE_TIMING:-true}"
LABEL_MODE="${LABEL_MODE:-fast}"
AUDIT_EVERY_N_STEPS="${AUDIT_EVERY_N_STEPS:-0}"

# Publication ablations intentionally avoid evaluation-only teacher/audit work.
# PROFILE_TIMING must stay enabled because the paper table reports deployment latency.
v50_bool_true "$PROFILE_TIMING" || { echo "PROFILE_TIMING must be true for publication ablations." >&2; exit 2; }
[[ "$LABEL_MODE" == "fast" ]] || { echo "LABEL_MODE must be fast for publication ablations; teacher/audit labels are evaluation-only." >&2; exit 2; }
[[ "$AUDIT_EVERY_N_STEPS" == 0 ]] || { echo "AUDIT_EVERY_N_STEPS must be 0 for publication ablations." >&2; exit 2; }

mkdir -p "$OUT_ROOT"
python tools/check_v48_111_runtime_code_contract.py \
  --repo "$REPO" \
  --output "$OUT_ROOT/OC-RAP-v48.111-runtime-code-contract.ablations.json"

# name|config|safe|near|contact|tier|description
# These are inference-time/frozen-checkpoint functional knockouts.  In
# particular, actuator projection and persistent re-entry refer to the
# executable recovery witness/certification semantics; they do not replace
# Waymax dynamics or execute a separate low-level recovery controller.
ARMS=(
  "no_obs_consistency|configs/ablations/without_observation_kernel.yaml|0|1|1|main|Disable observation-compatible OC-MERO grouping at inference; frozen checkpoint/heads"
  "mean_tail|configs/ablations/without_lower_tail.yaml|0|1|1|main|Replace nested lower-tail aggregation by weighted mean at inference; frozen checkpoint/heads"
  "no_actuator_projection|configs/ablations/submission_no_actuator_projection.yaml|0|1|1|main|Disable actuator-envelope projection in executable recovery witness/certification"
  "no_persistent_reentry|configs/ablations/submission_no_persistent_reentry.yaml|0|0|1|main|Remove persistent re-entry alignment from post-contact recovery witness/certification"
  "no_rifa_absolute_admission|configs/ablations/submission_no_rifa_absolute_admission.yaml|1|1|1|main|Remove only the RIFA absolute-admission set gate; preserve frozen relative/scoring path"
  "no_active_set_alignment|configs/ablations/submission_no_active_set_alignment.yaml|0|1|1|supplementary|Remove active-set alignment from executable recovery witness/certification"
  "no_route_alignment|configs/ablations/submission_no_route_alignment.yaml|0|1|1|supplementary|Remove executable route-alignment semantics from recovery witness/certification"
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

bool_true() { v50_bool_true "$1"; }

is_corrected_artifact() {
  local artifact="$1"
  [[ -f "$artifact" ]] || return 1
  python - "$artifact" <<'PY2' >/dev/null 2>&1
import json, math, sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
rc=d.get("runtime_contract",{}) or {}
geom=rc.get("publication_geometry_metric","")
if geom != "exact_oriented_box_signed_clearance+penetration+swept_sat_constant_velocity_ttc_v55":
    raise SystemExit(1)
if rc.get("publication_metrics_include_target_state_t0") is not True:
    raise SystemExit(1)
per=((d.get("timing",{}) or {}).get("per_decision_s",{}) or {})
v=per.get("deployed_planner",None)
if v is None or not math.isfinite(float(v)):
    raise SystemExit(1)
PY2
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

archive_stale_arm_variant() {
  local arm="$1" variant="$2" run_safe="$3" run_near="$4" run_contact="$5"
  local root="$OUT_ROOT/$arm/$variant"
  local marker="$root/.${CORRECTED_RUN_TAG}.reset_done"
  mkdir -p "$root/safe" "$root/near" "$root/contact"
  if [[ "$RESET_STALE_RESULTS" == "false" || "$RESET_STALE_RESULTS" == "0" ]]; then
    return 0
  fi
  if [[ "$RESET_STALE_RESULTS" != "force" && -f "$marker" ]]; then
    return 0
  fi
  local stamp archive any=0
  stamp="$(date -u +'%Y%m%dT%H%M%SZ')"
  archive="$OUT_ROOT/_stale_before_${CORRECTED_RUN_TAG}/$stamp/$arm/$variant"
  for pair in "safe:$run_safe" "near:$run_near" "contact:$run_contact"; do
    local regime="${pair%%:*}" enabled="${pair##*:}"
    [[ "$enabled" == 1 ]] || continue
    local d="$root/$regime"
    shopt -s nullglob
    local files=(
      "$d"/closed_loop_ocrap.json
      "$d"/closed_loop_ocrap.json.partial
      "$d"/closed_loop_ocrap.json.progress.json
      "$d"/closed_loop_ocrap.json.scenes.jsonl
      "$d"/closed_loop_ocrap.log
      "$d"/closed_loop_dataset_support.json
      "$d"/womd_spec_validation.json
    )
    shopt -u nullglob
    if ((${#files[@]})); then
      mkdir -p "$archive/$regime"
      for f in "${files[@]}"; do [[ -e "$f" ]] && mv "$f" "$archive/$regime/" && any=1; done
    fi
  done
  for f in "$root"/OCRAP_THREE_REGIME_RUN_INDEX.json "$root"/SUMMARY.json "$root"/safe.phase.json "$root"/near.phase.json "$root"/contact.phase.json; do
    if [[ -e "$f" ]]; then mkdir -p "$archive"; mv "$f" "$archive/"; any=1; fi
  done
  mkdir -p "$root"
  printf 'tag=%s\nreset_at=%s\narchived=%s\n' "$CORRECTED_RUN_TAG" "$(v50_iso_now)" "$any" > "$marker"
  if ((any)); then echo "[RESET] archived stale artifacts -> $archive"; fi
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

# Use at most two physical workers by default, exactly matching the requested
# two-GPU execution.  One process owns one GPU at a time: no unsafe sharing.
IFS=',' read -r -a _gpu_raw <<< "$CUDA_DEVICES"
GPUS=()
for raw in "${_gpu_raw[@]}"; do g="$(echo "$raw" | xargs)"; [[ -n "$g" ]] && GPUS+=("$g"); done
((${#GPUS[@]})) || GPUS=(0)
if ((${#GPUS[@]} > 2)); then GPUS=("${GPUS[0]}" "${GPUS[1]}"); fi

echo "[Ablation scheduler] GPUs=${GPUS[*]} variants=${CLEAN_VARIANTS[*]} set=$ABLATION_SET"
echo "[Ablation contract] frozen checkpoint + frozen per-bucket gamma; LABEL_MODE=$LABEL_MODE; PROFILE_TIMING=$PROFILE_TIMING"

# Resolve bucket provenance once and preflight each immutable regime/WOMD pair
# once.  Every arm/variant reuses the resulting verified JSON contract.
SAFE_WOMD="$(v50_resolve_bucket_womd_spec "$SAFE_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" auto)"
NEAR_WOMD="$(v50_resolve_bucket_womd_spec "$NEAR_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" auto)"
CONTACT_WOMD="$(v50_resolve_bucket_womd_spec "$CONTACT_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" auto)"
PREFLIGHT_ROOT="$OUT_ROOT/_shared_preflight_${CORRECTED_RUN_TAG}"
mkdir -p "$PREFLIGHT_ROOT"

declare -A NEED_REGIME=( [safe]=0 [near]=0 [contact]=0 )
for spec in "${SELECTED_SPECS[@]}"; do
  IFS='|' read -r _ _ rs rn rc _ _ <<< "$spec"
  [[ "$rs" == 1 ]] && NEED_REGIME[safe]=1
  [[ "$rn" == 1 ]] && NEED_REGIME[near]=1
  [[ "$rc" == 1 ]] && NEED_REGIME[contact]=1
done

preflight_one() {
  local regime="$1" bucket="$2" womd="$3" out="$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json"
  if [[ "${NEED_REGIME[$regime]}" != 1 ]]; then return 0; fi
  echo "[PREFLIGHT once] regime=$regime"
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$bucket" --split "$BUCKET_SPLIT" \
    --womd-pattern "$womd" --expected-source-role auto \
    --output "$out"
}
preflight_one safe "$SAFE_BUCKET" "$SAFE_WOMD"
preflight_one near "$NEAR_BUCKET" "$NEAR_WOMD"
preflight_one contact "$CONTACT_BUCKET" "$CONTACT_WOMD"

# Validate each frozen deployed owner once, archive legacy wrong outputs once,
# and materialize independent regime jobs using the original output layout.
QUEUE_ROOT="$OUT_ROOT/.${CORRECTED_RUN_TAG}.queue.$$"
mkdir -p "$QUEUE_ROOT/pending" "$QUEUE_ROOT/running" "$QUEUE_ROOT/failed" "$QUEUE_ROOT/done"
job_id=0
ROOTS_TO_FINALIZE=()
for variant in "${CLEAN_VARIANTS[@]}"; do
  python tools/check_v48_111_deployable_stack.py \
    --model-run "$MODEL_RUN" --variant "$variant" \
    --output "$OUT_ROOT/V48.111-DEPLOYABLE-STACK-${variant}.json"
  IFS=$'\t' read -r checkpoint gamma_json < <(variant_paths "$variant")
  read -r gamma_safe gamma_near gamma_contact < <(read_gammas "$gamma_json")

  for spec in "${SELECTED_SPECS[@]}"; do
    IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
    archive_stale_arm_variant "$arm" "$variant" "$run_safe" "$run_near" "$run_contact"
    root="$OUT_ROOT/$arm/$variant"
    mkdir -p "$root/safe" "$root/near" "$root/contact"
    ROOTS_TO_FINALIZE+=("$root")

    # Preserve the original three-regime index semantics for intentionally
    # unevaluated strata.
    now="$(v50_iso_now)"
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
      job_id=$((job_id+1)); jf="$QUEUE_ROOT/pending/$(printf '%04d' "$job_id").job"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$arm" "$variant" "$regime" "$config" "$checkpoint" "$gamma" "$womd" "$bucket" "$PREFLIGHT_ROOT/$regime.closed_loop_dataset_support.json" > "$jf"
    done
    python tools/build_ocrap_three_regime_index.py --root "$root" --launcher-exit-code 1 >/dev/null || true
  done
done

echo "[Ablation scheduler] queued regime jobs=$job_id; concurrency=${#GPUS[@]} (one job per GPU)"

claim_job() {
  local gpu="$1" lock="$QUEUE_ROOT/.claim_lock" f base claimed
  while ! mkdir "$lock" 2>/dev/null; do sleep 0.05; done
  f="$(find "$QUEUE_ROOT/pending" -maxdepth 1 -type f -name '*.job' -print | LC_ALL=C sort | head -n 1 || true)"
  if [[ -z "$f" ]]; then rmdir "$lock"; return 1; fi
  base="$(basename "$f")"; claimed="$QUEUE_ROOT/running/${base%.job}.gpu${gpu}.job"
  mv "$f" "$claimed"; rmdir "$lock"; printf '%s\n' "$claimed"
}

run_claimed_job() {
  local gpu="$1" jf="$2"
  local arm variant regime config checkpoint gamma womd bucket preflight
  IFS=$'\t' read -r arm variant regime config checkpoint gamma womd bucket preflight < "$jf"
  local root="$OUT_ROOT/$arm/$variant" run_dir="$root/$regime" artifact="$run_dir/closed_loop_ocrap.json"
  local started="$(v50_iso_now)" rc=0

  if bool_true "$SKIP_COMPLETE_CORRECTED" && python tools/check_closed_loop_artifact.py --output "$artifact" --quiet && is_corrected_artifact "$artifact"; then
    echo "[REUSE corrected] gpu=$gpu arm=$arm variant=$variant regime=$regime"
    write_phase "$root" "$regime" complete 0 "$started" "$(v50_iso_now)"
    mv "$jf" "$QUEUE_ROOT/done/$(basename "$jf")"
    return 0
  fi

  echo "[RUN] gpu=$gpu arm=$arm variant=$variant regime=$regime -> $artifact"
  write_phase "$root" "$regime" running 0 "$started" ""
  if env \
      RUN_DIR="$run_dir" OUTPUT="$artifact" \
      WOMD_VAL="$womd" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" \
      CHECKPOINT="$checkpoint" GAMMA_REC="$gamma" GPU="$gpu" \
      MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
      LABEL_MODE="$LABEL_MODE" AUDIT_EVERY_N_STEPS="$AUDIT_EVERY_N_STEPS" \
      NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
      BUCKET_DATASET="$bucket" BUCKET_SPLIT="$BUCKET_SPLIT" MAX_TARGETS_PER_SCENE=1 \
      CONFIG="$config" RENDER_TRACE=false SAVE_PARTIAL=true RESUME=true RESUME_FORCE=false \
      PROFILE_TIMING="$PROFILE_TIMING" PREFLIGHT_SUPPORT_JSON="$preflight" \
      JAX_CACHE_DIR="$OUT_ROOT/.jax_compilation_cache/gpu${gpu}" \
      PARTIAL_WRITE_EVERY_SCENES="$PARTIAL_WRITE_EVERY_SCENES" PROGRESS_EVERY_STEPS="$PROGRESS_EVERY_STEPS" \
      bash scripts/run_ocrap_closed_loop_optimized.sh; then
    if python tools/check_closed_loop_artifact.py --output "$artifact" --quiet && is_corrected_artifact "$artifact"; then rc=0; else rc=91; fi
  else
    rc=$?
  fi

  if ((rc==0)); then
    write_phase "$root" "$regime" complete 0 "$started" "$(v50_iso_now)"
    mv "$jf" "$QUEUE_ROOT/done/$(basename "$jf")"
    echo "[DONE] gpu=$gpu arm=$arm variant=$variant regime=$regime"
    return 0
  fi
  write_phase "$root" "$regime" failed "$rc" "$started" "$(v50_iso_now)"
  mv "$jf" "$QUEUE_ROOT/failed/$(basename "$jf")"
  echo "[FAILED] gpu=$gpu arm=$arm variant=$variant regime=$regime rc=$rc" >&2
  return 0  # keep this GPU worker alive to finish the remaining independent jobs
}

worker() {
  local gpu="$1" jf
  mkdir -p "$OUT_ROOT/.jax_compilation_cache/gpu${gpu}"
  while jf="$(claim_job "$gpu")"; do
    run_claimed_job "$gpu" "$jf"
  done
}

PIDS=()
for gpu in "${GPUS[@]}"; do worker "$gpu" & PIDS+=("$!"); done
for p in "${PIDS[@]}"; do wait "$p"; done

failed_count="$(find "$QUEUE_ROOT/failed" -maxdepth 1 -type f -name '*.job' | wc -l | xargs)"
launcher_rc=0; [[ "$failed_count" == 0 ]] || launcher_rc=1

# Rebuild the exact legacy-compatible per-arm/variant indices after all workers
# have stopped, so no two processes write the same summary concurrently.
for root in "${ROOTS_TO_FINALIZE[@]}"; do
  python tools/build_ocrap_three_regime_index.py --root "$root" --launcher-exit-code "$launcher_rc" >/dev/null || true
done

# Record the execution/scientific contract next to the original manifest.
python - "$OUT_ROOT/ablation_execution_contract.json" "$CORRECTED_RUN_TAG" "$LABEL_MODE" "$PROFILE_TIMING" "$failed_count" "$CUDA_DEVICES" <<'PY'
import json,sys,datetime
out,tag,label,profile,failed,gpus=sys.argv[1:]
doc={
  'schema_version': 1,
  'tag': tag,
  'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'ablation_type': 'frozen-checkpoint inference-time functional knockout',
  'checkpoint_policy': 'same frozen V48.80 checkpoint within each variant',
  'calibration_policy': 'same frozen per-bucket gamma_rec as the full model; no recalibration',
  'metric_contract': 'v55 exact publication geometry, t0 included, missing metrics not zero-filled',
  'latency_contract': 'state_history + candidate_features + policy_selection only; teacher/audit/Waymax bookkeeping excluded',
  'label_mode': label,
  'profile_timing': profile.lower() in {'1','true','yes','on'},
  'scheduler': 'dynamic shared queue, one closed-loop process per GPU',
  'cuda_devices': gpus,
  'failed_job_count': int(failed),
  'interpretation_notes': {
    'no_actuator_projection': 'removes actuator projection from executable recovery witness/certification; does not replace Waymax dynamics or directly execute a separate recovery controller',
    'no_persistent_reentry': 'removes persistent re-entry alignment from the recovery witness/certification semantics',
    'without_obs_or_tail': 'changes OC-MERO inference aggregation while retaining frozen learned heads/representation',
    'no_rifa_absolute_admission': 'removes only the absolute-admission set gate while retaining hard/harm feasibility and frozen scoring path',
  },
}
open(out,'w',encoding='utf-8').write(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY

if [[ "$failed_count" != 0 ]]; then
  echo "Ablation workers finished with $failed_count failed regime job(s). Queue retained at: $QUEUE_ROOT" >&2
  exit 1
fi
rm -rf "$QUEUE_ROOT"

# Paper tables are built only against a complete corrected Full OC-RAP run and
# remain paired by target key.  Missing Full results are a warning, not a reason
# to invalidate successfully completed ablation trajectories.
if bool_true "$BUILD_TABLES"; then
  full_ready=1
  for variant in "${CLEAN_VARIANTS[@]}"; do
    for regime in safe near contact; do
      if ! python tools/check_closed_loop_artifact.py --output "$FULL_RUN_ROOT/$variant/$regime/closed_loop_ocrap.json" --quiet || ! is_corrected_artifact "$FULL_RUN_ROOT/$variant/$regime/closed_loop_ocrap.json"; then
        full_ready=0
      fi
    done
  done
  if ((full_ready)); then
    tier="$ABLATION_SET"; [[ "$tier" == main || "$tier" == all ]] || tier=all
    python tools/build_submission_ablation_tables.py \
      --full-run "$FULL_RUN_ROOT" --ablation-root "$OUT_ROOT" \
      --variants "$VARIANTS" --tier "$tier" --output-dir "$TABLE_OUT"
  else
    echo "[WARN] Full corrected three-regime OC-RAP artifacts are not all complete under $FULL_RUN_ROOT; skip paired ablation tables for now." >&2
  fi
fi

printf '\nCorrected ablations complete.\nRoot: %s\nManifest: %s\nContract: %s\n' \
  "$OUT_ROOT" "$MANIFEST" "$OUT_ROOT/ablation_execution_contract.json"
