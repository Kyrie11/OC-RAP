#!/usr/bin/env python3
"""Watch paper-figure/video rendering progress without modifying any artifacts."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def _files(root: Path, suffixes: tuple[str, ...]):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)


def _size_mb(paths):
    return sum(p.stat().st_size for p in paths if p.exists()) / 1048576.0


def _selected_count(root: Path):
    total=0
    per={}
    for regime in ('safe','near','contact'):
        p=root/'selection'/f'{regime}_selection.json'
        if p.is_file():
            try:
                n=len((json.loads(p.read_text(encoding='utf-8')).get('selected') or []))
            except Exception:
                n=0
        else:
            n=0
        per[regime]=n; total+=n
    return total,per


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--interval-s', type=float, default=10.0)
    ap.add_argument('--once', action='store_true')
    args=ap.parse_args()
    total, per = _selected_count(args.root)
    expected_fig_files=total*4  # pair PNG/PDF + all-method PNG/PDF
    expected_videos=total*2     # primary pair + all-method montage
    while True:
        figs=_files(args.root/'paper_figures',('.png','.pdf'))
        vids=_files(args.root/'videos',('.mp4','.gif'))
        fig_index=(args.root/'paper_figures'/'PAPER_FIGURE_INDEX.json').is_file()
        vid_index=(args.root/'videos'/'REGIME_VIDEO_INDEX.json').is_file()
        print(json.dumps({
            'event':'regime_visualization_render_progress_v128',
            'selected_scenes':per,
            'figure_files':len(figs),'expected_figure_files':expected_fig_files,'figure_size_mib':round(_size_mb(figs),1),'figure_index_ready':fig_index,
            'video_files':len(vids),'expected_video_files':expected_videos,'video_size_mib':round(_size_mb(vids),1),'video_index_ready':vid_index,
            'latest_outputs':[str(p) for p in (figs+vids)[-6:]],
        }, ensure_ascii=False), flush=True)
        if args.once or (fig_index and vid_index and len(figs)>=expected_fig_files and len(vids)>=expected_videos):
            return 0
        time.sleep(max(args.interval_s,1.0))

if __name__=='__main__':
    raise SystemExit(main())
