import torch
from ocrap.models.mero import compute_profiles, MEROParams, existential_option_aggregate


def pred(P=0.2,G=0.8,C=0.8,U=0.1,K=0.1,L=3):
    B,Kc,M=1,1,2
    return {"P":torch.full((B,Kc,L,M),P),"G":torch.full((B,Kc,L,M),G),"C":torch.full((B,Kc,L,M),C),"U":torch.full((B,Kc,M),U),"H":torch.zeros(B,Kc,M),"Kdef":torch.full((B,Kc,L,M),K),"mode_probs":torch.ones(B,M)/M}

def masks(L=3): return {"action_mask":torch.ones(1,1,dtype=torch.bool),"option_mask":torch.ones(1,1,L,dtype=torch.bool)}

def R(p): return compute_profiles(p,masks(p['P'].shape[2]),MEROParams()).get('R').item()

def test_increasing_P_cannot_increase_R():
    assert R(pred(P=0.8)) <= R(pred(P=0.1)) + 1e-6

def test_increasing_G_or_C_cannot_decrease_R():
    assert R(pred(G=0.9,C=0.9)) >= R(pred(G=0.2,C=0.2)) - 1e-6

def test_increasing_U_or_K_cannot_increase_R():
    assert R(pred(U=0.8,K=0.8)) <= R(pred(U=0.1,K=0.1)) + 1e-6

def test_H_does_not_change_R():
    p1=pred(); p2=pred(); p2['H']+=1.0
    assert abs(R(p1)-R(p2)) < 1e-8

def test_existential_option_beats_mean_option_when_one_witness_succeeds():
    p=pred(L=4); p['P'][:]=0.9; p['G'][:]=0.1; p['C'][:]=0.1; p['Kdef'][:]=0.9
    p['P'][:,:,0,:]=0.0; p['G'][:,:,0,:]=1.0; p['C'][:,:,0,:]=1.0; p['Kdef'][:,:,0,:]=0.0
    ex=compute_profiles(p,masks(4),MEROParams(mean_over_options=False))['R'].item()
    mn=compute_profiles(p,masks(4),MEROParams(mean_over_options=True))['R'].item()
    assert ex > mn

def test_masked_invalid_options_do_not_affect_logsumexp():
    v=torch.tensor([[[[1.0],[10.0]]]])  # [1,1,2,1]
    m=torch.tensor([[[True,False]]])
    out=existential_option_aggregate(v,m,tau_R=0.25,c_R=0.0)
    v2=torch.tensor([[[[1.0]]]])
    m2=torch.tensor([[[True]]])
    out2=existential_option_aggregate(v2,m2,tau_R=0.25,c_R=0.0)
    assert torch.allclose(out,out2,atol=1e-6)

def test_duplicate_valid_option_does_not_change_normalized_softmaxmax():
    v1=torch.ones(1,1,1,2)
    v2=torch.ones(1,1,2,2)
    o1=existential_option_aggregate(v1,torch.ones(1,1,1,dtype=torch.bool))
    o2=existential_option_aggregate(v2,torch.ones(1,1,2,dtype=torch.bool))
    assert torch.allclose(o1,o2,atol=1e-5)
