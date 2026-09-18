from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_master_suite_cleans_only_named_final_run_roots_and_requests_6s_visualization() -> None:
    text = (ROOT / "scripts/run_final_submission_suite.sh").read_text()
    for name in (
        "ocrap_v48_124_latency_isolated",
        "ocrap_v48_124_final_characterization",
        "external_baselines_v48_124_final_v2_latency_isolated",
        "external_baselines_v48_124_final_v2",
    ):
        assert f'$BASE_OUT/{name}' in text
    assert 'safe_rm_tree "$ABLATION_OUT"' not in text
    assert 'VIS_TRACE_STEPS="${VIS_TRACE_STEPS:-60}"' in text
    assert 'VIS_MIN_DURATION_S="${VIS_MIN_DURATION_S:-6.0}"' in text
    assert '--trace-steps "$VIS_TRACE_STEPS"' in text
    assert '--min-duration-s "$VIS_MIN_DURATION_S"' in text
    assert 'PROFILE_LATENCY=false' in text
    assert 'PROFILE_ISOLATED_LATENCY=false' in text
    assert 'isolated single-process/single-GPU profiling' in text
    assert 'package_final_submission_results.py' in text
    assert 'final_results.zip' in text


def test_visualization_pipeline_enforces_continuous_full_length_trace() -> None:
    build = (ROOT / "scripts/build_regime_visualizations.sh").read_text()
    generate = (ROOT / "scripts/generate_selected_regime_traces.sh").read_text()
    assert 'MIN_VIDEO_DURATION_S="${MIN_VIDEO_DURATION_S:-6.0}"' in build
    assert 'VIS_CONTACT_MIN_POST_STEPS="${VIS_CONTACT_MIN_POST_STEPS:-$TRACE_MAX_STEPS}"' in build
    assert 'build_contact_anchor_manifest.py' in build
    assert '--min-post-steps "$VIS_CONTACT_MIN_POST_STEPS"' in build
    assert 'CONTACT_ANCHOR_PRELUDE_ENABLED=true' in generate
    assert 'CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true' in generate
    assert 'len(trace) < required_steps + 1' in generate


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def test_ablation_export_keeps_six_authoritative_split_tables(tmp_path: Path) -> None:
    src = tmp_path / "tables"
    out = tmp_path / "runs"
    for variant in ("balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            fields = ["method", f"{regime}_metric", "decision_latency_ms"]
            _write_csv(src / variant / regime / f"{regime}_comparison.csv", fields,
                       [{"method": "Full OC-RAP", f"{regime}_metric": "1", "decision_latency_ms": "2"}])
    subprocess.run([
        sys.executable, str(ROOT / "tools/export_final_ablation_csvs.py"),
        "--submission-table-root", str(src), "--output-root", str(out),
    ], check=True, cwd=ROOT)
    for variant in ("balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            assert (out / f"ablation_{variant}_{regime}.csv").is_file()
    combined = list(csv.DictReader((out / "ablation_results.csv").open(encoding="utf-8")))
    assert len(combined) == 6
    assert {r["regime"] for r in combined} == {"safe", "near", "contact"}
    index = json.loads((out / "ablation_results_index.json").read_text())
    assert index["authoritative_layout"] == "split_by_variant_and_regime"


def test_result_packager_includes_tables_and_rendered_outputs_but_not_large_trace_intermediates(tmp_path: Path) -> None:
    runs = tmp_path / "runs"; vis = runs / "regime_visualization_v48_124_final"
    runs.mkdir(); (vis / "videos").mkdir(parents=True); (vis / "paper_figures").mkdir(); (vis / "selective_traces").mkdir()
    required = [
        "safe_compare.csv", "near_compare.csv", "contact_compare.csv", "ablation_results.csv",
        "ablation_balanced_safe.csv", "ablation_balanced_near.csv", "ablation_balanced_contact.csv",
        "ablation_precision_safe.csv", "ablation_precision_near.csv", "ablation_precision_contact.csv",
    ]
    for name in required:
        (runs / name).write_text("method,value\na,1\n", encoding="utf-8")
    (vis / "videos" / "REGIME_VIDEO_INDEX.json").write_text("{}\n", encoding="utf-8")
    (vis / "videos" / "clip.mp4").write_bytes(b"fake-mp4")
    (vis / "paper_figures" / "fig.png").write_bytes(b"fake-png")
    (vis / "selective_traces" / "huge.jsonl").write_text("x\n", encoding="utf-8")
    out = runs / "final_results.zip"
    subprocess.run([
        sys.executable, str(ROOT / "tools/package_final_submission_results.py"),
        "--runs-root", str(runs), "--visualization-root", str(vis), "--output", str(out),
    ], check=True, cwd=ROOT)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "tables/safe_compare.csv" in names
    assert "visualization/videos/clip.mp4" in names
    assert "visualization/paper_figures/fig.png" in names
    assert not any("selective_traces" in name for name in names)
    assert "MANIFEST.json" in names
