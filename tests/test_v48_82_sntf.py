from __future__ import annotations
from dataclasses import asdict
import torch
from ocrap.models.data import OPTION_FEATURE_DIM
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

def layout(): return FlatFeatureLayout(feature_max_agents=2)
def model(signed=False):
 L=layout(); return OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_active_set_alignment=True,direct_recovery_semantic_witness_path_stop_alignment=False,direct_recovery_semantic_witness_route_alignment=True,direct_recovery_semantic_witness_reentry_alignment=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_root_tail_source=True,direct_recovery_semantic_witness_tail_localization=True,direct_recovery_semantic_witness_structured_tail_field=True,direct_recovery_semantic_witness_signed_tail_channels=signed)

def test_v4882_zero_init_and_shapes():
 a=model(False); b=model(True)
 assert a.direct_absolute_root_tail_source_scale is None and b.direct_absolute_root_tail_source_scale is None
 assert tuple(a.direct_absolute_structured_tail_field_weight.shape)==(1,16)
 assert tuple(b.direct_absolute_structured_tail_field_weight.shape)==(2,16)
 assert torch.count_nonzero(a.direct_absolute_structured_tail_field_weight)==0
 assert torch.count_nonzero(b.direct_absolute_structured_tail_field_weight)==0

def test_v4882_signed_requires_structured():
 L=layout()
 try:
  OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_root_tail_source=True,direct_recovery_semantic_witness_signed_tail_channels=True)
 except ValueError as e: assert 'signed tail channels' in str(e)
 else: raise AssertionError('expected fail closed')
