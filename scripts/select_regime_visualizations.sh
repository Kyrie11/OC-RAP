#!/usr/bin/env bash
# Build fail-closed, metric-only qualitative selections from the *current* full
# OC-RAP run and the three external-baseline result roots.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${OCRAP_RESULTS_ROOT:?set OCRAP_RESULTS_ROOT to the frozen OC-RAP full-metric variant root (contains safe/near/contact)}"
: "${OCRAP_MODEL_RUN:?set OCRAP_MODEL_RUN to the frozen OC-RAP checkpoint/calibration owner}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${WOMD_NUM_SHARDS:=150}"
: "${MODEL_VARIANT:=balanced}"
: "${TARGET_LOCK_ROOT:?set TARGET_LOCK_ROOT to final observation-legal target_keys root}"
: "${SAFE_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/safe}"
: "${NEAR_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/near}"
: "${CONTACT_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/contact}"
: "${OUT:=/home/senzeyu2/code/OC-RAP/runs/regime_visualization}"
: "${NUM_SCENES:=5}"
: "${SAFE_NUM_SCENES:=$NUM_SCENES}"
: "${NEAR_NUM_SCENES:=$NUM_SCENES}"
: "${CONTACT_NUM_SCENES:=$NUM_SCENES}"
: "${SELECTION_DIR:=$OUT/selection}"
: "${MIN_VIDEO_DURATION_S:=5.0}"
: "${FALLBACK_MIN_VIDEO_DURATION_S:=4.0}"
: "${CONTACT_FALLBACK_MIN_VIDEO_DURATION_S:=4.0}"
: "${VIS_CONTACT_ANCHOR_MANIFEST_FILE:=}"
: "${CONTACT_ALLOWED_TARGET_KEYS_FILE:=}"
: "${MAX_SELECTED_TIER_RANK:=1}"  # 0=strict hardest-baseline win only; 1 allows majority-material strong evidence.
: "${ALLOW_FEWER_SCENES:=false}"

mkdir -p "$OUT/provenance" "$SELECTION_DIR"

python tools/check_regime_visualization_inputs.py \
  --ocrap-results-root "$OCRAP_RESULTS_ROOT" \
  --ocrap-model-run "$OCRAP_MODEL_RUN" \
  --ocrap-root "$OCRAP_ROOT" \
  --womd-root "$WOMD_ROOT" \
  --womd-shards "$WOMD_NUM_SHARDS" \
  --variant "$MODEL_VARIANT" \
  --target-lock-root "$TARGET_LOCK_ROOT" \
  --safe-external-root "$SAFE_EXTERNAL_ROOT" \
  --near-external-root "$NEAR_EXTERNAL_ROOT" \
  --contact-external-root "$CONTACT_EXTERNAL_ROOT" \
  --output "$OUT/provenance/VISUALIZATION_INPUT_CONTRACT.json"

# Explicit method lists are intentionally duplicated from provenance.py here so
# a stale visualization script cannot silently substitute legacy baselines.
safe_methods=(gameformer_lite plantf pluto pdm_closed pdm_hybrid idm diffusion_planner)
near_methods=(marc_lite racp_lite robust_scenario_mpc predictive_safety_filter dr_cvar_safety_filter conformal_predictive_safety_filter flow_planner plan_r1 betopnet)
contact_methods=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)

select_one() {
  local regime="$1" ext_root="$2" requested="$3"; shift 3
  local -a methods=("$@") args=()
  local m
  for m in "${methods[@]}"; do
    args+=(--baseline "$m=$ext_root/closed_loop_${m}.json.scenes.jsonl")
  done
  local -a filter_args=()
  local fallback_duration="$FALLBACK_MIN_VIDEO_DURATION_S"
  if [[ "$regime" == contact ]]; then
    [[ -s "$VIS_CONTACT_ANCHOR_MANIFEST_FILE" ]] || { echo "missing frozen Contact visualization anchor manifest: $VIS_CONTACT_ANCHOR_MANIFEST_FILE" >&2; exit 30; }
    filter_args+=(--contact-anchor-manifest "$VIS_CONTACT_ANCHOR_MANIFEST_FILE")
    fallback_duration="$CONTACT_FALLBACK_MIN_VIDEO_DURATION_S"
    # Backward-compatible optional filter.  The final v125 path deliberately
    # leaves this unset so Contact selection stays on the full locked cohort.
    if [[ -n "$CONTACT_ALLOWED_TARGET_KEYS_FILE" ]]; then
      [[ -s "$CONTACT_ALLOWED_TARGET_KEYS_FILE" ]] || { echo "missing Contact visualization target filter: $CONTACT_ALLOWED_TARGET_KEYS_FILE" >&2; exit 30; }
      filter_args+=(--allowed-target-keys-file "$CONTACT_ALLOWED_TARGET_KEYS_FILE")
    fi
  fi
  local -a count_args=()
  [[ "$ALLOW_FEWER_SCENES" == true ]] && count_args+=(--allow-fewer-scenes)
  python tools/select_regime_visualization_scenes.py \
    --regime "$regime" \
    --ocrap-scenes "$OCRAP_RESULTS_ROOT/$regime/closed_loop_ocrap.json.scenes.jsonl" \
    "${args[@]}" "${filter_args[@]}" "${count_args[@]}" \
    --output "$SELECTION_DIR/${regime}_selection.json" \
    --target-keys-output "$SELECTION_DIR/${regime}_target_keys.json" \
    --num-scenes "$requested" \
    --min-duration-s "$MIN_VIDEO_DURATION_S" \
    --fallback-min-duration-s "$fallback_duration" \
    --max-selected-tier-rank "$MAX_SELECTED_TIER_RANK"
}

select_one safe "$SAFE_EXTERNAL_ROOT" "$SAFE_NUM_SCENES" "${safe_methods[@]}"
select_one near "$NEAR_EXTERNAL_ROOT" "$NEAR_NUM_SCENES" "${near_methods[@]}"
select_one contact "$CONTACT_EXTERNAL_ROOT" "$CONTACT_NUM_SCENES" "${contact_methods[@]}"

python - "$SELECTION_DIR" <<'PY'
import json, pathlib, sys
selection_dir=pathlib.Path(sys.argv[1])
summary={"event":"regime_visualization_selection_index","regimes":{}}
for regime in ("safe","near","contact"):
    p=selection_dir/f"{regime}_selection.json"
    d=json.loads(p.read_text())
    summary["regimes"][regime]={
        "selection":str(p),
        "num_selected":len(d["selected"]),
        "global_strongest_external_method":d.get("global_strongest_external_method"),
        "selected":[{
            "rank":x.get("category_rank"),"target_key":x["target_key"],"tier":x["selection_tier"],
            "primary_external_method":x.get("primary_external_method"),"best_external_method":x.get("best_external_method"),
            "worst_external_method":x.get("worst_external_method"),"score":x.get("score"),
        } for x in d["selected"]],
    }
(selection_dir/"SELECTION_INDEX.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY

echo "Selection complete: $SELECTION_DIR/SELECTION_INDEX.json"
