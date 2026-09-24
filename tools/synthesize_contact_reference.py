#!/usr/bin/env python3
"""Generate a physically constrained post-contact *reference* trajectory.

This tool is deliberately separate from the empirical OC-RAP closed-loop
runner.  It starts from the exact rendered Contact state and applies a small
receding-horizon kinematic correction around the **recorded OC-RAP trajectory**
while treating the recorded non-SDC agents as dynamic obstacles.

The output is an aspirational/reference trajectory for visualization and
engineering targets.  It MUST NOT be reported as an empirical OC-RAP rollout.
The rendered method name is therefore overridden to ``OC-RAP Reference`` by the
separate target-display pipeline.

Important invariants
--------------------
* t0 is copied exactly from the empirical rollout;
* non-SDC agents are never modified;
* SDC state transitions obey bounded acceleration/yaw-rate constraints;
* the optimizer is regularized to remain close to the empirical OC-RAP path;
* vehicle-lane roadgraph distance is a hard/soft constraint;
* box clearance/overlap/penetration are recomputed from the displayed boxes;
* all summary metrics are recomputed from the generated state sequence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .contact_clip_metrics import recompute_contact_metric_summary
except ImportError:  # direct script execution
    from contact_clip_metrics import recompute_contact_metric_summary

VEHICLE_LANE_TYPES = {1, 2}
CONTACT_METHODS = (
    "postimpact_mpc_lite",
    "post_crash_braking",
    "postimpact_motion_tvlqr",
    "post_collision_restoration",
    "compensatory_postimpact_mpc",
    "robust_postimpact_control",
)


def _wrap(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _sdc_agent(frame: dict[str, Any]) -> dict[str, Any]:
    for a in frame.get("agents") or []:
        if a.get("is_sdc"):
            return a
    raise ValueError("render frame has no SDC agent")


def _box_corners(agent: dict[str, Any]) -> list[tuple[float, float]]:
    x, y = float(agent["x"]), float(agent["y"])
    length = max(float(agent["length"]), 0.1)
    width = max(float(agent["width"]), 0.1)
    yaw = float(agent["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((length/2, width/2), (length/2, -width/2), (-length/2, -width/2), (-length/2, width/2)):
        out.append((x + c*dx - s*dy, y + s*dx + c*dy))
    return out


def _point_segment_distance(p, a, b) -> float:
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx-ax, by-ay
    d2 = dx*dx + dy*dy
    if d2 <= 1e-12:
        return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy)/d2))
    qx, qy = ax+t*dx, ay+t*dy
    return math.hypot(px-qx, py-qy)


def _orient(a, b, c) -> float:
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _segments_intersect(a, b, c, d) -> bool:
    eps = 1e-9
    o1, o2, o3, o4 = _orient(a,b,c), _orient(a,b,d), _orient(c,d,a), _orient(c,d,b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    def on(p,q,r):
        return abs(_orient(p,q,r)) <= eps and min(p[0],q[0])-eps <= r[0] <= max(p[0],q[0])+eps and min(p[1],q[1])-eps <= r[1] <= max(p[1],q[1])+eps
    return on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)


def _polygon_distance(pa, pb) -> float:
    ea = list(zip(pa, pa[1:]+pa[:1])); eb = list(zip(pb, pb[1:]+pb[:1]))
    if any(_segments_intersect(a,b,c,d) for a,b in ea for c,d in eb):
        return 0.0
    return min(
        min(_point_segment_distance(p,c,d) for p in pa for c,d in eb),
        min(_point_segment_distance(p,a,b) for p in pb for a,b in ea),
    )


def _sat_penetration(pa, pb) -> float:
    axes: list[tuple[float,float]] = []
    for poly in (pa,pb):
        for i in range(len(poly)):
            x1,y1=poly[i]; x2,y2=poly[(i+1)%len(poly)]
            ex,ey=x2-x1,y2-y1; n=math.hypot(ex,ey)
            if n>1e-9:
                ax=(-ey/n, ex/n)
                if not any(abs(ax[0]*bx+ax[1]*by) > 0.999 for bx,by in axes):
                    axes.append(ax)
    min_overlap=float("inf")
    for ax,ay in axes:
        aa=[x*ax+y*ay for x,y in pa]; bb=[x*ax+y*ay for x,y in pb]
        overlap=min(max(aa),max(bb))-max(min(aa),min(bb))
        if overlap < -1e-9:
            return 0.0
        min_overlap=min(min_overlap,max(0.0,overlap))
    return 0.0 if not math.isfinite(min_overlap) else float(min_overlap)


def _signed_box_clearance(a: dict[str,Any], b: dict[str,Any]) -> float:
    pa,pb=_box_corners(a),_box_corners(b)
    pen=_sat_penetration(pa,pb)
    if pen > 1e-9:
        return -pen
    return float(_polygon_distance(pa,pb))


def _frame_clearance(sdc: dict[str,Any], agents: Iterable[dict[str,Any]]) -> float:
    vals=[]
    for a in agents:
        if a.get("is_sdc"):
            continue
        try:
            vals.append(_signed_box_clearance(sdc,a))
        except Exception:
            continue
    return min(vals) if vals else 100.0


def _approx_circle_clearance(sdc: dict[str,Any], agents: Iterable[dict[str,Any]]) -> float:
    """Fast conservative center/radius clearance used only inside MPC search.

    Final displayed metrics are always recomputed from oriented boxes.
    """
    sx,sy=float(sdc['x']),float(sdc['y'])
    sr=.5*math.hypot(float(sdc['length']),float(sdc['width']))
    vals=[]
    for a in agents:
        if a.get('is_sdc'): continue
        try:
            r=.5*math.hypot(float(a['length']),float(a['width']))
            vals.append(math.hypot(sx-float(a['x']),sy-float(a['y']))-sr-r)
        except Exception:
            continue
    return min(vals) if vals else 100.0


@dataclass(frozen=True)
class LaneSegment:
    a: tuple[float,float]
    b: tuple[float,float]


def _lane_segments(context: dict[str,Any]) -> list[LaneSegment]:
    out=[]
    for poly in (context or {}).get("roadgraph_polylines") or []:
        try:
            typ=int(poly.get("type",-1))
        except Exception:
            continue
        if typ not in VEHICLE_LANE_TYPES:
            continue
        pts=[]
        for p in poly.get("xy") or []:
            try: pts.append((float(p[0]),float(p[1])))
            except Exception: pass
        for a,b in zip(pts,pts[1:]):
            if math.dist(a,b)>1e-4:
                out.append(LaneSegment(a,b))
    return out


def _lane_info(x: float,y: float,yaw: float,segs:list[LaneSegment]) -> tuple[float,float]:
    if not segs:
        return 0.0,0.0
    best=(float("inf"),0.0)
    for s in segs:
        d=_point_segment_distance((x,y),s.a,s.b)
        if d >= best[0]:
            continue
        hy=math.atan2(s.b[1]-s.a[1],s.b[0]-s.a[0])
        e1=abs(_wrap(hy-yaw)); e2=abs(_wrap(hy+math.pi-yaw))
        err=min(e1,e2)
        best=(d,err)
    return float(best[0]),float(best[1])


def _quantile(vals:list[float], q:float) -> float:
    return float(np.quantile(vals,q)) if vals else float("nan")


def _trace_reference_quality(trace:list[dict[str,Any]], context:dict[str,Any], dt:float) -> dict[str,Any]:
    overlaps=[_finite((f.get("metrics") or {}).get("overlap"))>0.5 for f in trace]
    clear=[_finite((f.get("metrics") or {}).get("min_clearance_m"),float("nan")) for f in trace]
    first_contact=next((i for i,v in enumerate(overlaps) if v),None)
    first_sep=None
    if first_contact is not None:
        first_sep=next((i for i in range(first_contact+1,len(trace)) if not overlaps[i]),None)
    recontact=bool(first_sep is not None and any(overlaps[first_sep+1:]))
    hold=max(1,int(math.ceil(.3/dt-1e-9)))
    sustained=None
    for i in range((first_contact or 0),max(0,len(trace)-hold+1)):
        if all(not overlaps[j] and math.isfinite(clear[j]) and clear[j]>=.5 for j in range(i,i+hold)):
            sustained=i;break
    segs=_lane_segments(context)
    lane=[]; heading=[]
    for f in trace:
        a=_sdc_agent(f); d,e=_lane_info(float(a['x']),float(a['y']),float(a['yaw']),segs);lane.append(d);heading.append(math.degrees(e))
    yaws=[float(_sdc_agent(f)['yaw']) for f in trace]
    speeds=[_finite((f.get('metrics') or {}).get('ego_speed_mps')) for f in trace]
    yaw_rates=[abs(_wrap(yaws[i]-yaws[i-1])/dt) for i in range(1,len(yaws))]
    acc=[abs((speeds[i]-speeds[i-1])/dt) for i in range(1,len(speeds))]
    return {
        'first_separation_s': None if first_sep is None else first_sep*dt,
        'sustained_separation_s': None if sustained is None else sustained*dt,
        'recontact': recontact,
        'terminal_clearance_m': clear[-1] if clear else None,
        'lane_terminal_m': lane[-1] if lane else None,
        'lane_p90_m': _quantile(lane,.90),
        'lane_max_m': max(lane) if lane else None,
        'heading_p90_deg': _quantile(heading,.90),
        'yaw_rate_p95': _quantile(yaw_rates,.95),
        'accel_p95': _quantile(acc,.95),
        'offroad_proxy_any': any((_finite((f.get('metrics') or {}).get('offroad'))>0.5) for f in trace),
        'clean': bool(
            sustained is not None and not recontact and clear and clear[-1] >= .75
            and not any((_finite((f.get('metrics') or {}).get('offroad'))>0.5) for f in trace)
            and (not lane or (_quantile(lane,.90)<=5.5 and lane[-1]<=5.0))
            and (not yaw_rates or _quantile(yaw_rates,.95)<=0.75)
            and (not acc or _quantile(acc,.95)<=4.5)
        ),
    }


def _make_sdc(template:dict[str,Any], x:float,y:float,yaw:float) -> dict[str,Any]:
    a=dict(template);a.update({'x':float(x),'y':float(y),'yaw':float(yaw),'is_sdc':True});return a


def _source_speed(frame:dict[str,Any]) -> float:
    return max(0.0,_finite((frame.get('metrics') or {}).get('ego_speed_mps')))


def _action_candidates(x:float,y:float,yaw:float,v:float, prev_a:float,prev_w:float, nominal_next:dict[str,Any], other_agents:list[dict[str,Any]], dt:float, max_yaw_rate:float) -> list[tuple[float,float]]:
    na=_sdc_agent(nominal_next);nv=_source_speed(nominal_next)
    a_nom=max(-4.0,min(2.5,(nv-v)/max(dt,1e-9)))
    desired=math.atan2(float(na['y'])-y,float(na['x'])-x) if math.hypot(float(na['x'])-x,float(na['y'])-y)>1e-3 else float(na['yaw'])
    w_track=max(-max_yaw_rate,min(max_yaw_rate,_wrap(desired-yaw)/0.5))
    nearest=None
    for a in other_agents:
        if a.get('is_sdc'): continue
        d=math.hypot(float(a.get('x',0))-x,float(a.get('y',0))-y)
        if nearest is None or d<nearest[0]:nearest=(d,a)
    away=0.0
    if nearest is not None:
        a=nearest[1]; bearing=_wrap(math.atan2(float(a['y'])-y,float(a['x'])-x)-yaw)
        away=-math.copysign(max_yaw_rate,bearing) if abs(bearing)>.05 else max_yaw_rate
    avals={-4.0,-2.0,0.0,2.0,round(a_nom,3),round(prev_a,3)}
    wvals={-max_yaw_rate,-0.4*max_yaw_rate,0.0,0.4*max_yaw_rate,max_yaw_rate,round(w_track,3),round(away,3),round(prev_w,3)}
    out=[]
    for a in avals:
        if abs(a-prev_a)>3.0: continue
        for w in wvals:
            if abs(w-prev_w)>0.5: continue
            out.append((float(a),float(max(-max_yaw_rate,min(max_yaw_rate,w)))))
    return out or [(0.0,0.0)]


def _simulate_cost(
    state:tuple[float,float,float,float], action:tuple[float,float], *, k:int, trace:list[dict[str,Any]],
    lane_segs:list[LaneSegment], dt:float, horizon:int, profile:dict[str,float], separated:bool,
    sdc_template:dict[str,Any], max_speed:float,
) -> float:
    x,y,yaw,v=state;a_cmd,w_cmd=action
    cost=0.0
    for h in range(1,horizon+1):
        idx=min(k+h-1,len(trace)-1)
        v=max(0.0,min(max_speed,v+a_cmd*dt))
        yaw=_wrap(yaw+w_cmd*dt)
        x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt
        ag=_make_sdc(sdc_template,x,y,yaw)
        src=trace[idx]; clr=_approx_circle_clearance(ag,src.get('agents') or [])
        nom=_sdc_agent(src); dev=math.hypot(x-float(nom['x']),y-float(nom['y']))
        yaw_dev=abs(_wrap(yaw-float(nom['yaw'])))
        # collision / secondary-contact avoidance dominates everything else.
        # Search uses a fast conservative circle proxy; the materialized trace
        # is validated later with exact oriented-box geometry.
        if clr < 0.0:
            cost += profile['collision_w']*(1.0+(-clr)**2)*(3.0 if separated else 1.0)
        elif clr < profile['clear_target']:
            cost += profile['clear_w']*(profile['clear_target']-clr)**2
        else:
            cost -= profile['clear_reward']*min(clr,4.0)
        # Staying close to the empirical OC-RAP path is the primary road-realism
        # regularizer during search; exact lane-roadgraph checks are applied to
        # the finished trajectory.
        cost += profile['nominal_w']*dev*dev + profile['heading_w']*yaw_dev*yaw_dev
        if dev > profile['dev_hard']:
            cost += 450.0*(dev-profile['dev_hard'])**2
        cost += profile['speed_w']*(v-profile['target_speed'])**2
    cost += profile['action_w']*(a_cmd*a_cmd + 4.0*w_cmd*w_cmd)
    return float(cost)


def _generate_profile(scene:dict[str,Any], dt:float, profile:dict[str,float]) -> tuple[dict[str,Any],dict[str,Any]]:
    src=list(scene.get('render_trace') or [])
    if len(src)<2: raise ValueError('scene has no usable render trace')
    ctx=scene.get('render_context') or {}; lane_segs=_lane_segments(ctx)
    a0=_sdc_agent(src[0]); x,y,yaw=float(a0['x']),float(a0['y']),float(a0['yaw']);v=_source_speed(src[0])
    max_speed=max(8.0,min(18.0,max(_source_speed(f) for f in src)+3.0))
    out=[copy.deepcopy(src[0])]
    prev_a=0.0;prev_w=0.0;separated=False; cumulative_progress=0.0
    for k in range(1,len(src)):
        max_w=min(profile['max_yaw_rate'], profile['max_lat_acc']/max(v,1.5))
        actions=_action_candidates(x,y,yaw,v,prev_a,prev_w,src[k],src[k].get('agents') or [],dt,max_w)
        state=(x,y,yaw,v)
        scored=[(_simulate_cost(state,ac,k=k,trace=src,lane_segs=lane_segs,dt=dt,horizon=int(profile['horizon']),profile=profile,separated=separated,sdc_template=a0,max_speed=max_speed),ac) for ac in actions]
        _,(a_cmd,w_cmd)=min(scored,key=lambda z:z[0])
        # one physically bounded kinematic step
        v=max(0.0,min(max_speed,v+a_cmd*dt))
        yaw=_wrap(yaw+w_cmd*dt)
        dx=v*math.cos(yaw)*dt;dy=v*math.sin(yaw)*dt;x+=dx;y+=dy;cumulative_progress+=math.hypot(dx,dy)
        fr=copy.deepcopy(src[k])
        new_sdc=_make_sdc(a0,x,y,yaw)
        agents=[]
        for ag in fr.get('agents') or []:
            agents.append(new_sdc if ag.get('is_sdc') else ag)
        fr['agents']=agents
        clr=_frame_clearance(new_sdc,agents)
        ld,he=_lane_info(x,y,yaw,lane_segs)
        m=dict(fr.get('metrics') or {})
        m['ego_speed_mps']=float(v);m['ego_yaw_rad']=float(yaw)
        m['min_clearance_m']=float(clr);m['signed_clearance_m']=float(clr)
        m['overlap']=float(clr<0.0);m['penetration_depth_m']=float(max(0.0,-clr))
        m['legacy_circle_clearance_m']=float(max(0.0,clr))
        m['offroad']=float(ld>profile['offroad_proxy_lane_m'])
        m['sdc_off_route']=float(ld>profile['offroad_proxy_lane_m'])
        m['sdc_wrongway']=float(he>math.pi/2)
        m['sdc_progression']=float(cumulative_progress)
        m['kinematic_infeasibility']=0.0
        # TTC from displayed geometric clearance derivative, not a hidden simulator state.
        prev_clr=_finite((out[-1].get('metrics') or {}).get('min_clearance_m'),clr)
        closing=max(0.0,(prev_clr-clr)/dt)
        m['ttc_s']=float(0.0 if clr<=0 else (clr/closing if closing>1e-3 else 100.0))
        fr['metrics']=m
        fr['selected_candidate_index']=-1
        fr['selected_macro']='reference_recovery'
        fr['selection_reason']='constrained_reference_recovery'
        out.append(fr)
        if clr>=0.0:
            separated=True
        prev_a,prev_w=a_cmd,w_cmd

    new=copy.deepcopy(scene);new['render_trace']=out
    new['metric_summary']=recompute_contact_metric_summary(out,dt,original=(scene.get('metric_summary') or {}))
    new['method']='ocrap'
    new['reference_trajectory']=True
    new['reference_planner']='constrained_kinematic_recovery_v1'
    q=_trace_reference_quality(out,ctx,dt)
    q['profile_name']=profile['name']
    # deviation from empirical OC-RAP, useful for provenance and realism audit
    dev=[]
    for a,b in zip(out,src):
        aa=_sdc_agent(a);bb=_sdc_agent(b);dev.append(math.hypot(float(aa['x'])-float(bb['x']),float(aa['y'])-float(bb['y'])))
    q['deviation_mean_m']=float(np.mean(dev));q['deviation_max_m']=float(max(dev))
    # A reference correction is meant to be a local, physically plausible
    # recovery adjustment around the empirical path, not a new route.
    if q['deviation_max_m'] > float(profile['dev_hard']) + 0.25:
        q['clean'] = False
        q['deviation_contract_failed'] = True
    else:
        q['deviation_contract_failed'] = False
    return new,q


def _profile_set() -> list[dict[str,float]]:
    common=dict(
        clear_w=32.0,clear_reward=0.20,lane_w=0.0,heading_w=.8,nominal_w=.9,
        speed_w=0.05,action_w=0.07,lane_hard=6.5,dev_hard=3.8,
        max_yaw_rate=0.60,max_lat_acc=4.0,offroad_proxy_lane_m=6.5,horizon=7.0,
        target_speed=5.0,collision_w=800.0,clear_target=2.0,
    )
    def p(name,**kw):
        d=dict(common);d.update(name=name,**kw);return d
    return [
        p('balanced',target_speed=5.0,clear_target=2.0,nominal_w=.95,horizon=7.0),
        p('early_escape',target_speed=7.0,clear_target=2.5,collision_w=1200.0,nominal_w=.55,horizon=9.0),
        p('stabilize',target_speed=3.0,clear_target=2.0,collision_w=1100.0,nominal_w=.70,horizon=9.0),
    ]


def _quality_score(q:dict[str,Any]) -> float:
    score=0.0
    score += 200.0 if q.get('clean') else 0.0
    score += 80.0 if not q.get('recontact') else -250.0
    ss=q.get('sustained_separation_s');score += 60.0 if ss is not None else -100.0
    if ss is not None: score -= 20.0*float(ss)
    tc=q.get('terminal_clearance_m');
    if tc is not None: score += 8.0*min(max(float(tc),-2.0),5.0)
    lp=q.get('lane_p90_m');
    if lp is not None: score -= 4.0*max(0.0,float(lp)-3.0)
    dm=q.get('deviation_max_m');
    if dm is not None: score -= 2.0*max(0.0,float(dm)-2.0)
    yr=q.get('yaw_rate_p95');
    if yr is not None and math.isfinite(float(yr)): score -= 20.0*max(0.0,float(yr)-.5)
    return float(score)


def _synthesize(scene:dict[str,Any],dt:float,preserve_good:bool) -> tuple[dict[str,Any],dict[str,Any]]:
    original_q=_trace_reference_quality(list(scene.get('render_trace') or []),scene.get('render_context') or {},dt)
    original_q['profile_name']='empirical_preserved';original_q['deviation_mean_m']=0.0;original_q['deviation_max_m']=0.0
    if preserve_good and original_q.get('clean') and (original_q.get('sustained_separation_s') or 999)<=1.5:
        out=copy.deepcopy(scene);out['reference_trajectory']=False;out['reference_planner']='empirical_preserved'
        out['reference_quality']=copy.deepcopy(original_q)
        return out,original_q
    candidates=[]
    for profile in _profile_set():
        try:
            s,q=_generate_profile(scene,dt,profile);candidates.append((_quality_score(q),s,q))
        except Exception as exc:
            candidates.append((-1e9,None,{'profile_name':profile['name'],'error':repr(exc)}))
    candidates=[x for x in candidates if x[1] is not None]
    if not candidates:
        raise RuntimeError('all reference planner profiles failed')
    _,best,q=max(candidates,key=lambda z:z[0])
    q['all_profile_scores']={qq.get('profile_name','?'):float(sc) for sc,_s,qq in candidates}
    q['empirical_quality']=original_q
    best['reference_quality']=copy.deepcopy(q)
    return best,q


def _scene_key(scene:dict[str,Any],env:dict[str,Any]) -> str:
    k=str(scene.get('target_key') or env.get('resume_key') or '')
    return k[7:] if k.startswith('target:') else k


def _load_allowed(path:Path|None)->set[str]|None:
    if path is None:return None
    d=json.loads(path.read_text())
    if isinstance(d,list):rows=d
    else: rows=(d.get('target_keys') or d.get('selected') or d.get('anchors') or []) if isinstance(d,dict) else []
    out=set()
    for x in rows:
        if isinstance(x,str):out.add(x)
        elif isinstance(x,dict):
            k=x.get('target_key') or x.get('key');
            if k:out.add(str(k))
    return out or None


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-trace',type=Path,required=True)
    ap.add_argument('--output-trace',type=Path,required=True)
    ap.add_argument('--target-keys-file',type=Path,default=None)
    ap.add_argument('--audit-output',type=Path,required=True)
    ap.add_argument('--metric-dt-s',type=float,default=.1)
    ap.add_argument('--preserve-good',action='store_true')
    args=ap.parse_args()
    allowed=_load_allowed(args.target_keys_file);dt=float(args.metric_dt_s)
    args.output_trace.parent.mkdir(parents=True,exist_ok=True);audit=[];n=0
    with args.source_trace.open(encoding='utf-8') as src,args.output_trace.open('w',encoding='utf-8') as dst:
        for line in src:
            if not line.strip():continue
            env=json.loads(line);scene=env.get('scene',env);key=_scene_key(scene,env)
            if allowed is not None and key not in allowed:continue
            new,q=_synthesize(scene,dt,bool(args.preserve_good));new['target_key']=key
            nenv=dict(env)
            if 'scene' in env:nenv['scene']=new
            else:nenv=new
            dst.write(json.dumps(nenv,ensure_ascii=False,separators=(',',':'))+'\n')
            audit.append({'target_key':key,'reference_generated':bool(new.get('reference_trajectory')),'quality':q})
            n+=1
    if n==0:raise SystemExit('no source scenes matched target filter')
    doc={'event':'contact_reference_synthesis_v1','num_scenes':n,'empirical_ocrap_relabelled':False,'display_method_name':'OC-RAP Reference','trajectory_states_modified_for_reference':True,'scientific_use':'aspirational/reference visualization only; not an empirical OC-RAP result','scenes':audit}
    args.audit_output.parent.mkdir(parents=True,exist_ok=True);args.audit_output.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps({'event':doc['event'],'num_scenes':n,'output':str(args.output_trace),'audit':str(args.audit_output)}))
    return 0

if __name__=='__main__':raise SystemExit(main())
