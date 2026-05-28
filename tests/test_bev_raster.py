import numpy as np
from ocrap.raster.geometry import world_to_ego, ego_to_bev_pixel, bev_pixel_to_ego
from ocrap.raster.bev_builder import BEVBuilder, HistoryBuffer, REQUIRED_CHANNELS
from ocrap.utils.datatypes import BEVSpec, EgoState, ActorState, MapFeatures, RouteInfo


def fixture_scene():
    spec=BEVSpec(H=160,W=160,history_steps=5)
    ego=EgoState(0,0,0,v=5)
    actor=ActorState('a',10,0,0,vx=10,vy=0)
    drivable=np.array([[-40,-5],[40,-5],[40,5],[-40,5]],np.float32)
    center=np.stack([np.linspace(-40,40,50),np.zeros(50)],axis=-1).astype(np.float32)
    mf=MapFeatures([drivable],[center],[center+[0,1.8],center+[0,-1.8]],[],13.9)
    route=RouteInfo.straight(40,20,13.9)
    return spec,ego,[actor],mf,route


def test_ego_center_heading_up():
    spec,ego,actors,mf,route=fixture_scene(); builder=BEVBuilder(spec); out=builder.build_from_state(ego,actors,mf,route,HistoryBuffer(5))
    pix=ego_to_bev_pixel(np.array([[0,0]],np.float32),spec)[0]
    assert abs(pix[0]-spec.H/2)<1 and abs(pix[1]-spec.W/2)<1
    ahead=ego_to_bev_pixel(np.array([[10,0]],np.float32),spec)[0]
    assert ahead[0] < pix[0]


def test_channel_order_matches_config():
    spec,ego,actors,mf,route=fixture_scene(); builder=BEVBuilder(spec)
    assert builder.channel_names[:len(REQUIRED_CHANNELS)] == REQUIRED_CHANNELS


def test_dynamic_velocity_in_ego_frame():
    spec,ego,actors,mf,route=fixture_scene(); builder=BEVBuilder(spec); out=builder.build_from_state(ego,actors,mf,route,HistoryBuffer(5))
    frame=out['bev'][-1].astype(np.float32); vx=frame[builder.channel_index['dyn_vx_t0']]
    assert vx.max() > 0.4


def test_history_not_current_frame_copy():
    spec,ego,actors,mf,route=fixture_scene(); hist=HistoryBuffer(5)
    for h in range(5):
        hist.push(EgoState(-h,0,0,v=5), [ActorState('a',10-h,0,0,vx=10,vy=0)])
    builder=BEVBuilder(spec); out=builder.build_from_state(ego,actors,mf,route,hist)
    frame=out['bev'][-1].astype(np.float32)
    hnames=[n for n in builder.channel_names if n.startswith('dyn_occ_hist_')]
    assert hnames and not np.allclose(frame[builder.channel_index[hnames[0]]], frame[builder.channel_index['dyn_occ_t0']])


def test_route_corridor_alignment():
    spec,ego,actors,mf,route=fixture_scene(); builder=BEVBuilder(spec); out=builder.build_from_state(ego,actors,mf,route,HistoryBuffer(5))
    frame=out['bev'][-1].astype(np.float32)
    route_ch=frame[builder.channel_index['route_corridor']]; center=frame[builder.channel_index['lane_centerline']]
    assert (route_ch*center).sum() > 0


def test_affordance_channels_nonempty_when_available():
    spec,ego,actors,mf,route=fixture_scene(); builder=BEVBuilder(spec); out=builder.build_from_state(ego,actors,mf,route,HistoryBuffer(5))
    frame=out['bev'][-1].astype(np.float32)
    assert frame[builder.channel_index['affordance_stop']].max() > 0


def test_world_to_ego_and_pixel_inverse():
    spec=BEVSpec(H=160,W=160); pts=np.array([[5,2],[-3,1]],np.float32); pix=ego_to_bev_pixel(pts,spec); back=bev_pixel_to_ego(pix,spec)
    assert np.max(np.abs(pts-back)) <= max(spec.resolution_x,spec.resolution_y)
