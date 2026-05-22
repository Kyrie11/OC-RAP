import torch, pytest
from recap.models.care import CARE


def batch():
    B,K,L,M=1,2,3,4
    return dict(bev=torch.zeros(B,5,24,32,32),ego_info=torch.zeros(B,11),route_command=torch.zeros(B,20,6),actions=torch.zeros(B,K,11,6),options=torch.zeros(B,K,L,26,6),action_mask=torch.ones(B,K,dtype=torch.bool),option_mask=torch.ones(B,K,L,dtype=torch.bool))

def test_inference_batch_does_not_contain_labels():
    b=batch(); assert not {'R_star','Y_option','mode_seed_params'}.intersection(b.keys())

def test_model_forward_rejects_mode_seed_params():
    model=CARE(C_bev=24,H_h=5,M=4)
    b=batch()
    with pytest.raises(ValueError):
        model(**b, mode_seed_params=torch.zeros(1,4,10))

def test_eval_decision_does_not_read_R_star_or_Y_option():
    model=CARE(C_bev=24,H_h=5,M=4); b=batch(); out=model(**b)
    assert 'R_star' not in out and 'Y_option' not in out
