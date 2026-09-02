import importlib.util, pathlib, numpy as np, torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load():
 sp=importlib.util.spec_from_file_location('m',ROOT/'tools/run_v48_84_stage_i_action_observability_probe.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def test_permutation_is_group_local_and_cyclic():
 m=load();r=[{'group':(1,'a',1),'candidate':1,'delta':np.array([1.])},{'group':(1,'a',1),'candidate':2,'delta':np.array([2.])},{'group':(2,'b',2),'candidate':1,'delta':np.array([7.])},{'group':(2,'b',2),'candidate':2,'delta':np.array([8.])}];o=m.permute_within_group(r,'delta').reshape(-1);assert np.allclose(o,[2,1,8,7])
def test_stats_zero_on_invalid():
 m=load();v=torch.randn(2,3,4);w=torch.ones(2,3);valid=torch.zeros(2,3,dtype=torch.bool);z=m._stats(v,w,valid);assert z.shape==(2,16);assert torch.isfinite(z).all();assert torch.count_nonzero(z)==0
def test_probe_shape():
 m=load();p=m.Probe(13); assert tuple(p(torch.zeros(5,13)).shape)==(5,3)
