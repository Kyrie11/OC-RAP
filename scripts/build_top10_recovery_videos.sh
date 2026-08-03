#!/usr/bin/env bash
set -euo pipefail

# Exactly ten positive, auditable recovery videos: 5 Near + 5 Contact.
# Closed-loop runs must have closed_loop.render_trace=true and identical target sets.
: "${NEAR_OCRAP_SCENES:?set NEAR_OCRAP_SCENES to OC-RAP near .scenes.jsonl or result JSON}"
: "${NEAR_BASELINE_SCENES:?set NEAR_BASELINE_SCENES to paired near baseline scenes}"
: "${CONTACT_OCRAP_SCENES:?set CONTACT_OCRAP_SCENES to OC-RAP contact scenes}"
: "${CONTACT_BASELINE_SCENES:?set CONTACT_BASELINE_SCENES to paired contact baseline scenes}"
: "${OUT:=runs/external_comparison/top10_recovery_videos}"
: "${NEAR_BASELINE_NAME:=Best external baseline}"
: "${CONTACT_BASELINE_NAME:=Best external baseline}"
: "${FPS:=10}"

mkdir -p "$OUT/near" "$OUT/contact"
PYTHONPATH=src python tools/select_critical_scenes_v48_34.py \
  --method-scenes "$NEAR_OCRAP_SCENES" --control-scenes "$NEAR_BASELINE_SCENES" \
  --regime near --num-positive 5 --num-failure 0 --max-per-scene 1 \
  --output "$OUT/near_selection.json"
PYTHONPATH=src python tools/render_critical_scenes_v48_34.py \
  --method-scenes "$NEAR_OCRAP_SCENES" --control-scenes "$NEAR_BASELINE_SCENES" \
  --selection "$OUT/near_selection.json" --output-dir "$OUT/near" --fps "$FPS" --format mp4 \
  --method-name "OC-RAP" --control-name "$NEAR_BASELINE_NAME"

PYTHONPATH=src python tools/select_critical_scenes_v48_34.py \
  --method-scenes "$CONTACT_OCRAP_SCENES" --control-scenes "$CONTACT_BASELINE_SCENES" \
  --regime contact --num-positive 5 --num-failure 0 --max-per-scene 1 \
  --output "$OUT/contact_selection.json"
PYTHONPATH=src python tools/render_critical_scenes_v48_34.py \
  --method-scenes "$CONTACT_OCRAP_SCENES" --control-scenes "$CONTACT_BASELINE_SCENES" \
  --selection "$OUT/contact_selection.json" --output-dir "$OUT/contact" --fps "$FPS" --format mp4 \
  --method-name "OC-RAP" --control-name "$CONTACT_BASELINE_NAME"

python - "$OUT" <<'PY'
import json, pathlib, sys
out=pathlib.Path(sys.argv[1]); videos=[]
for regime in ("near","contact"):
    p=out/regime/"VIDEO_INDEX.json"
    d=json.load(p.open())
    videos.extend([{**x,"regime":regime} for x in d.get("videos",[])])
if len(videos)!=10:
    raise SystemExit(f"expected exactly 10 positive videos, got {len(videos)}; inspect eligibility scores")
json.dump({"event":"top10_recovery_videos","num_videos":10,"videos":videos},(out/"TOP10_VIDEO_INDEX.json").open("w"),indent=2)
print({"event":"top10_recovery_videos","num_videos":10,"output":str(out)})
PY
