#!/usr/bin/env bash
# Build fail-closed, metric-only qualitative selections from the *current* full
# OC-RAP run and the three independently rerun external-baseline roots.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${OCRAP_RESULTS_ROOT:?set OCRAP_RESULTS_ROOT to the frozen V48.111 submission full-metric variant root (contains safe/near/contact)}"
: "${OCRAP_MODEL_RUN:?set OCRAP_MODEL_RUN to the frozen V48.80 checkpoint/calibration owner}"
: "${MODEL_VARIANT:=balanced}"
: "${SAFE_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/safe_external}"
: "${NEAR_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/near_external}"
: "${CONTACT_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/contact_external}"
: "${OUT:=/home/senzeyu2/code/OC-RAP/runs/submission_visualization_v52}"
: "${NUM_SCENES:=5}"
: "${MIN_VIDEO_DURATION_S:=5.0}"
: "${FALLBACK_MIN_VIDEO_DURATION_S:=3.0}"
: "${MAX_SELECTED_TIER_RANK:=1}"  # 0=strict hardest-baseline win only; 1 allows majority-material strong evidence.

mkdir -p "$OUT/provenance" "$OUT/selection"

python tools/check_regime_visualization_inputs.py \
  --ocrap-results-root "$OCRAP_RESULTS_ROOT" \
  --ocrap-model-run "$OCRAP_MODEL_RUN" \
  --variant "$MODEL_VARIANT" \
  --safe-external-root "$SAFE_EXTERNAL_ROOT" \
  --near-external-root "$NEAR_EXTERNAL_ROOT" \
  --contact-external-root "$CONTACT_EXTERNAL_ROOT" \
  --output "$OUT/provenance/VISUALIZATION_INPUT_CONTRACT.json"

# Explicit method lists are intentionally duplicated from provenance.py here so
# a stale visualization script cannot silently substitute legacy baselines.
safe_methods=(gameformer_lite plantf pluto pdm_closed pdm_hybrid idm)
near_methods=(marc_lite racp_lite robust_scenario_mpc predictive_safety_filter dr_cvar_safety_filter conformal_predictive_safety_filter)
contact_methods=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)

select_one() {
  local regime="$1" ext_root="$2"; shift 2
  local -a methods=("$@") args=()
  local m
  for m in "${methods[@]}"; do
    args+=(--baseline "$m=$ext_root/closed_loop_${m}.json.scenes.jsonl")
  done
  python tools/select_regime_visualization_scenes.py \
    --regime "$regime" \
    --ocrap-scenes "$OCRAP_RESULTS_ROOT/$regime/closed_loop_ocrap.json.scenes.jsonl" \
    "${args[@]}" \
    --output "$OUT/selection/${regime}_selection.json" \
    --target-keys-output "$OUT/selection/${regime}_target_keys.json" \
    --num-scenes "$NUM_SCENES" \
    --min-duration-s "$MIN_VIDEO_DURATION_S" \
    --fallback-min-duration-s "$FALLBACK_MIN_VIDEO_DURATION_S" \
    --max-selected-tier-rank "$MAX_SELECTED_TIER_RANK"
}

select_one safe "$SAFE_EXTERNAL_ROOT" "${safe_methods[@]}"
select_one near "$NEAR_EXTERNAL_ROOT" "${near_methods[@]}"
select_one contact "$CONTACT_EXTERNAL_ROOT" "${contact_methods[@]}"

python - "$OUT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
summary={"event":"submission_visualization_selection_index_v52","regimes":{}}
for regime in ("safe","near","contact"):
    p=root/"selection"/f"{regime}_selection.json"
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
(root/"selection"/"SELECTION_INDEX.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY

echo "Selection complete: $OUT/selection/SELECTION_INDEX.json"
