#!/usr/bin/env python3
"""Render paired selected closed-loop traces as side-by-side MP4/GIF videos.

The renderer intentionally consumes only the selective trace reruns.  It uses a
shared ego-centric viewport, draws a clearance circle when available, marks the
first overlap/contact location, and labels held terminal frames when one policy
finishes earlier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms


def _load_scenes(path: Path):
    if path.suffix=='.json' and not path.name.endswith('.scenes.jsonl'):
        alt=Path(str(path)+'.scenes.jsonl')
        if alt.is_file(): path=alt
    out={}; by_scene_time={}; by_scene={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            e=json.loads(line); s=e.get('scene',e)
            key=str(s.get('target_key') or e.get('resume_key') or '')
            if key.startswith('target:'): key=key[len('target:'):]
            sid=str(s.get('scene_id') or ''); tv=s.get('target_time_index'); ti=str(tv if tv is not None else '')
            if not key: key=f'{sid}:t{ti}' if sid and ti else sid
            if not key: raise ValueError(f'scene without target key in {path}')
            if key in out: raise ValueError(f'duplicate target key {key} in {path}')
            out[key]=s
            if sid:
                by_scene.setdefault(sid,[]).append(s)
                if ti: by_scene_time[(sid,ti)]=s
    return out,by_scene_time,by_scene


def _resolve_scene(item,direct,by_scene_time,by_scene):
    key=str(item['target_key']); scene=direct.get(key)
    if scene is not None: return scene
    sid=str(item.get('scene_id') or ''); ti=str(item.get('target_time_index') if item.get('target_time_index') is not None else '')
    if sid and ti: scene=by_scene_time.get((sid,ti))
    if scene is None and sid and len(by_scene.get(sid,[]))==1: scene=by_scene[sid][0]
    return scene


def _sdc(frame):
    for a in frame.get('agents',[]):
        if a.get('is_sdc'):
            try: return float(a['x']),float(a['y'])
            except Exception: return None
    return None


def _frame(trace,i): return trace[min(i,len(trace)-1)]


def _shared_center(ct,mt,i):
    pts=[p for p in (_sdc(_frame(ct,i)),_sdc(_frame(mt,i))) if p is not None]
    if not pts: return 0.0,0.0
    return sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)


def _contact_marker(trace, regime):
    for frame in trace:
        m=frame.get('metrics',{}) or {}
        try: overlap=float(m.get('overlap',m.get('overlap_any',0.0)))
        except Exception: overlap=0.0
        if overlap>0.5:
            return _sdc(frame), 'observed overlap'
    # Post-contact targets may start after the initiating collision.  The first
    # simulated SDC state is therefore the causal contact anchor even when the
    # rollout begins separated.  Mark it explicitly instead of pretending no
    # contact context exists.
    if regime == 'contact' and trace:
        return _sdc(trace[0]), 'causal contact anchor'
    return None, None


def _metric_float(frame,key):
    try:
        v=float((frame.get('metrics',{}) or {}).get(key)); return v if math.isfinite(v) else None
    except Exception: return None


def _draw_roadgraph(ax, context, center, radius):
    cx,cy=center
    for poly in (context or {}).get('roadgraph_polylines',[]):
        xy=poly.get('xy') or []
        if len(xy)<2: continue
        pts=[]
        for point in xy:
            try:
                x,y=float(point[0]),float(point[1])
            except Exception:
                continue
            if abs(x-cx)<=radius+5.0 and abs(y-cy)<=radius+5.0: pts.append((x,y))
        if len(pts)>=2:
            ax.plot([p[0] for p in pts],[p[1] for p in pts],linewidth=0.65,alpha=0.35,zorder=0)


def _draw_frame(ax,trace,i,title,center,radius,contact_xy,contact_label,context=None):
    held=i>=len(trace); frame=_frame(trace,i); cx,cy=center
    ax.clear(); ax.set_aspect('equal',adjustable='box'); ax.set_xlim(cx-radius,cx+radius); ax.set_ylim(cy-radius,cy+radius)
    _draw_roadgraph(ax,context,center,radius)
    ax.set_title(title+(' · final state held' if held else '')); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(alpha=0.15)
    trail=[_sdc(x) for x in trace[:min(i,len(trace)-1)+1]]; trail=[x for x in trail if x]
    if len(trail)>=2: ax.plot([x[0] for x in trail],[x[1] for x in trail],linewidth=1.8,alpha=0.9,zorder=2)
    if contact_xy is not None:
        ax.scatter([contact_xy[0]],[contact_xy[1]],marker='x',s=80,linewidths=2,zorder=5)
        ax.annotate(contact_label or 'contact', contact_xy, xytext=(6, 6), textcoords='offset points',
                    fontsize=7, bbox={'boxstyle':'round','facecolor':'white','alpha':0.75}, zorder=7)
    sdc_xy=_sdc(frame)
    clearance=_metric_float(frame,'min_clearance_m')
    if sdc_xy and clearance is not None and 0.0<clearance<radius:
        ax.add_patch(patches.Circle(sdc_xy,clearance,fill=False,linestyle='--',linewidth=1.1,alpha=0.65,zorder=1))
    overlap=(_metric_float(frame,'overlap') or 0.0)>0.5; offroad=(_metric_float(frame,'offroad') or 0.0)>0.5
    for a in frame.get('agents',[]):
        try:
            x,y=float(a['x']),float(a['y']); length=max(float(a['length']),0.1); width=max(float(a['width']),0.1); yaw=float(a['yaw'])
        except Exception: continue
        if abs(x-cx)>radius+length or abs(y-cy)>radius+length: continue
        is_sdc=bool(a.get('is_sdc')); edge='red' if is_sdc and overlap else ('orange' if is_sdc and offroad else ('tab:blue' if is_sdc else '0.35'))
        rect=patches.Rectangle((x-length/2,y-width/2),length,width,fill=False,linewidth=2.4 if is_sdc else 0.8,edgecolor=edge,zorder=4 if is_sdc else 3)
        rect.set_transform(transforms.Affine2D().rotate_around(x,y,yaw)+ax.transData); ax.add_patch(rect)
        if is_sdc: ax.text(x,y,'SDC',fontsize=8,ha='center',va='center',zorder=6)
    lines=[f"t={frame.get('time_index')} macro={frame.get('selected_macro','')}",f"candidate={frame.get('selected_candidate_index')} reason={frame.get('selection_reason','')}"]
    for key,label,unit in (('ttc_s','TTC','s'),('min_clearance_m','clearance','m'),('overlap','overlap',''),('offroad','offroad','')):
        value=_metric_float(frame,key)
        if value is not None: lines.append(f'{label}={value:.3f}{unit}')
    ax.text(0.01,0.99,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=8,bbox={'boxstyle':'round','facecolor':'white','alpha':0.82},zorder=10)


def _delta_text(item,regime):
    t=item.get('terms',{}) or {}
    keys=(('ttc_p05_s','ΔTTCp05','s'),('clearance_p05_m','Δclearancep05','m'),('critical_ttc_exposure_s','Δcritical exposure','s')) if regime=='near' else (('post_contact_terminal_clearance_m','Δterminal clearance','m'),('post_contact_free_space_auc_normalized_m','Δfree-space AUC','m'),('post_contact_clearance_gain_m','Δclearance gain','m'),('post_contact_overlap_duration_s','Δoverlap duration','s'),('new_stable_stop_quality_event','Δnew stable stop',''),('recontact_event','Δrecontact',''))
    parts=[]
    for key,label,unit in keys:
        try:
            v=float(t[key])
            if math.isfinite(v): parts.append(f'{label}={v:+.2f}{unit}')
        except Exception: pass
    return ' | '.join(parts)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--method-scenes',type=Path,required=True); ap.add_argument('--control-scenes',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--fps',type=int,default=10); ap.add_argument('--format',choices=('auto','mp4','gif'),default='auto'); ap.add_argument('--method-name',default='OC-RAP'); ap.add_argument('--control-name',default='Comparator'); ap.add_argument('--view-radius-m',type=float,default=35.0)
    args=ap.parse_args()
    if args.fps<=0 or args.view_radius_m<=5: raise SystemExit('fps must be positive and view radius >5m')
    method,method_st,method_s=_load_scenes(args.method_scenes); control,control_st,control_s=_load_scenes(args.control_scenes); selection=json.loads(args.selection.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True,exist_ok=True); index=[]; csv_rows=[]; used=set()
    use_mp4=args.format=='mp4' or (args.format=='auto' and shutil.which('ffmpeg') is not None)
    if args.format=='mp4' and shutil.which('ffmpeg') is None: raise SystemExit('--format mp4 requested but ffmpeg is unavailable')
    for item in selection.get('selected',[]):
        key=str(item['target_key']); ms=_resolve_scene(item,method,method_st,method_s); cs=_resolve_scene(item,control,control_st,control_s)
        if ms is None or cs is None: raise SystemExit(f'selected scene {key} missing from paired traces')
        mt=ms.get('render_trace') or []; ct=cs.get('render_trace') or []
        if not mt or not ct: raise SystemExit(f'scene {key} has no render_trace; run the selective trace stage with closed_loop.render_trace=true')
        frames=max(len(mt),len(ct)); fig,axes=plt.subplots(1,2,figsize=(12.8,6.4),dpi=100); regime=str(selection.get('regime')); cc,cc_label=_contact_marker(ct,regime); mc,mc_label=_contact_marker(mt,regime)
        def update(i):
            center=_shared_center(ct,mt,i)
            _draw_frame(axes[0],ct,i,args.control_name,center,args.view_radius_m,cc,cc_label,cs.get('render_context')); _draw_frame(axes[1],mt,i,args.method_name,center,args.view_radius_m,mc,mc_label,ms.get('render_context'))
            delta=_delta_text(item,regime)
            tier=str(item.get('selection_tier') or 'strict_material_improvement')
            improvements=', '.join(item.get('material_improvements') or []) or 'positive non-regressive score'
            fig.suptitle(f'{regime} | rank {item.get("category_rank")} | {tier} | {key}\n{improvements} | {delta}',fontsize=10.5)
            return []
        ani=animation.FuncAnimation(fig,update,frames=frames,interval=1000/args.fps,blit=False)
        sid=str(ms.get('scene_id') or 'scene'); ti=ms.get('target_time_index'); kh=hashlib.sha1(key.encode()).hexdigest()[:8]; stem=f'{regime}_{item.get("category_rank",0)}_{sid}_t{ti}_{kh}'
        if stem in used: raise SystemExit(f'duplicate output stem for {key}')
        used.add(stem)
        if use_mp4:
            out=args.output_dir/f'{stem}.mp4'; writer=animation.FFMpegWriter(fps=args.fps,bitrate=2400,extra_args=['-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart'])
        else:
            out=args.output_dir/f'{stem}.gif'; writer=animation.PillowWriter(fps=args.fps)
        ani.save(out,writer=writer); plt.close(fig)
        record={'target_key':key,'scene_id':ms.get('scene_id'),'target_time_index':ms.get('target_time_index'),'category':item.get('category'),'category_rank':item.get('category_rank'),'selection_tier':item.get('selection_tier'),'selection_score':item.get('score'),'material_improvements':item.get('material_improvements'),'selection_terms':item.get('terms'),'video':str(out),'num_control_frames':len(ct),'num_method_frames':len(mt),'fps':args.fps,'view_radius_m':args.view_radius_m}
        index.append(record); csv_rows.append({k:record.get(k) for k in ('target_key','scene_id','target_time_index','category','category_rank','selection_tier','selection_score','video','num_control_frames','num_method_frames','fps','view_radius_m')})
    doc={'event':'critical_scene_recovery_videos_v50','exploratory_qualitative_only':True,'paper_population_claim_allowed':False,'rendering_scope':'nearby WOMD roadgraph polylines, agent boxes, ego trail, clearance circle and observed-overlap/causal-contact anchor; no raster image dependency','method_scenes':str(args.method_scenes),'control_scenes':str(args.control_scenes),'selection':str(args.selection),'method_name':args.method_name,'control_name':args.control_name,'videos':index}
    (args.output_dir/'VIDEO_INDEX.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (args.output_dir/'VIDEO_INDEX.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['target_key','scene_id','target_time_index','category','category_rank','selection_tier','selection_score','video','num_control_frames','num_method_frames','fps','view_radius_m']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(csv_rows)
    print(json.dumps({'event':doc['event'],'num_videos':len(index),'output_dir':str(args.output_dir)})); return 0
if __name__=='__main__': raise SystemExit(main())
