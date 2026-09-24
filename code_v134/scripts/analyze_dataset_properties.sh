#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
OUT="${OUT:-$REPO/runs/dataset_properties_v48_111}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DATASETS="${DATASETS:-train_safe,train_near_contact,train_contact,val_safe,val_near_contact,val_contact,calibration_safe,calibration_near_contact,calibration_contact,test_safe,test_near_contact,test_contact}"
mkdir -p "$OUT"
IFS=',' read -r -a names <<< "$DATASETS"
reports=()
for raw in "${names[@]}"; do
  name="$(echo "$raw" | xargs)"; [[ -n "$name" ]] || continue
  root="$OCRAP_ROOT/$name"
  if [[ ! -d "$root" ]]; then
    echo "[WARN] skip missing dataset: $root" >&2
    continue
  fi
  args=(python tools/export_dataset_properties.py --dataset "$root" --output "$OUT/$name.json")
  [[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
  "${args[@]}"
  reports+=("$OUT/$name.json")
done
((${#reports[@]})) || { echo "no dataset reports produced" >&2; exit 2; }
python - "$OUT" "${reports[@]}" <<'PY'
import csv,json,sys
from pathlib import Path
out=Path(sys.argv[1]); paths=[Path(x) for x in sys.argv[2:]]
rows=[]; combined={}
for p in paths:
    d=json.loads(p.read_text(encoding='utf-8')); name=Path(d['dataset']).name; combined[name]=d
    x=d['diagnostics']; c=d['construction']; manifest=c.get('manifest') or {}; recon=c.get('reconstructed_build_parameters') or {}; scan=c.get('scan_scope_evidence') or {}
    rows.append({
        'dataset':name,'samples':x.get('num_samples'),'scenes':x.get('num_scenes'),'groups':x.get('num_scene_time_groups'),
        'negative_deployable_fraction':(x.get('recovery_labels') or {}).get('negative_deployable_fraction'),
        'artifact_fraction':(x.get('recovery_labels') or {}).get('artifact_fraction'),
        'oracle_recoverable_fraction':(x.get('recovery_labels') or {}).get('oracle_recoverable_fraction'),
        'alias_conflict_fraction':(x.get('roots_and_observation') or {}).get('incompatible_alias_pair_fraction'),
        'valid_root_mean':((x.get('roots_and_observation') or {}).get('valid_root_count') or {}).get('mean'),
        'option_count_mean':((x.get('recovery_labels') or {}).get('option_count') or {}).get('mean'),
        'ocmero_max_abs_error':(x.get('schema') or {}).get('ocmero_recompute_max_abs_error'),
        'womd_roles':json.dumps(manifest.get('womd_source_role_counts',{}),sort_keys=True),
        'womd_patterns':str(recon.get('womd_patterns')),
        'num_candidates':recon.get('num_candidate_prefixes'),'num_roots':recon.get('num_roots'),'num_options':recon.get('num_recovery_options'),
        'num_targeted_futures':recon.get('num_targeted_futures'),'targeted_future_kinds':json.dumps(recon.get('targeted_future_kinds'),ensure_ascii=False),
        'scenario_start_index':scan.get('scenario_start_index',recon.get('scenario_start_index')),
        'scenario_stride':scan.get('scenario_stride',recon.get('scenario_stride')),
        'scenario_worker_index':scan.get('scenario_worker_index',recon.get('scenario_worker_index')),
        'source_max_scenarios':scan.get('source_max_scenarios'),
        'raw_scenarios_seen':scan.get('raw_scenarios_seen'),
        'failures':len(x.get('failures') or []),'warnings':len(x.get('warnings') or []),
    })
(out/'ALL_DATASET_PROPERTIES.json').write_text(json.dumps(combined,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
fields=list(rows[0])
with (out/'ALL_DATASET_PROPERTIES.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps({'datasets':len(rows),'json':str(out/'ALL_DATASET_PROPERTIES.json'),'csv':str(out/'ALL_DATASET_PROPERTIES.csv')},ensure_ascii=False))
PY
(
  cd "$OUT"
  rm -f dataset_properties_bundle.zip
  zip -qr dataset_properties_bundle.zip . -x dataset_properties_bundle.zip
)
echo "Dataset audit bundle: $OUT/dataset_properties_bundle.zip"
