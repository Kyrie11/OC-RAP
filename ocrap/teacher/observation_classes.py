from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def _as_vec(sig) -> np.ndarray:
    if isinstance(sig, dict):
        vals=[]
        for key in sorted(sig.keys()):
            v=np.asarray(sig[key], dtype=np.float32).reshape(-1)
            vals.append(v)
        return np.concatenate(vals) if vals else np.zeros(1, dtype=np.float32)
    return np.asarray(sig, dtype=np.float32).reshape(-1)


def post_prefix_observation_signature(trace, root_obj=None, mode=None, params=None) -> dict:
    """Observable-only signature after the prefix; no future labels/success data."""
    Hp = int(getattr(trace, "stage_boundary_idx", 0))
    ego = np.asarray(trace.ego_states[min(Hp, len(trace.ego_states)-1), :6], dtype=np.float32)
    actor = getattr(trace, "actor_states", None)
    if actor is not None and np.asarray(actor).size:
        arr = np.asarray(actor)
        actor_summary = arr[min(Hp, arr.shape[0]-1)].reshape(-1)[:32].astype(np.float32)
    else:
        actor_summary = np.zeros(1, dtype=np.float32)
    return {"ego": ego, "visible_actor_summary": actor_summary}


def obs_distance(sig1, sig2, weights=None) -> float:
    a, b = _as_vec(sig1), _as_vec(sig2)
    n = max(len(a), len(b))
    aa = np.zeros(n, dtype=np.float32); bb = np.zeros(n, dtype=np.float32)
    aa[:len(a)] = a; bb[:len(b)] = b
    w = np.ones(n, dtype=np.float32) if weights is None else np.resize(np.asarray(weights, dtype=np.float32), n)
    return float(np.sqrt(np.mean(w * (aa - bb) ** 2)))


def build_obs_equivalence(signatures: Iterable, eps_o: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    sigs=list(signatures); M=len(sigs)
    obs_equiv=np.eye(M, dtype=bool)
    for i in range(M):
        for j in range(i+1, M):
            eq = obs_distance(sigs[i], sigs[j]) <= eps_o
            obs_equiv[i,j]=obs_equiv[j,i]=eq
    # Union-find for transitive classes.
    parent=list(range(M))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(M):
        for j in range(M):
            if obs_equiv[i,j]: union(i,j)
    roots={}; cls=np.zeros(M, dtype=np.int64)
    for i in range(M):
        r=find(i); roots.setdefault(r, len(roots)); cls[i]=roots[r]
    # Make matrix class-consistent after transitive closure.
    obs_equiv = cls[:,None] == cls[None,:]
    return cls, obs_equiv


def beta_from_obs_equiv(mode_probs: np.ndarray, obs_equiv: np.ndarray) -> np.ndarray:
    p=np.asarray(mode_probs, dtype=np.float32).reshape(-1)
    p=p/np.clip(p.sum(), 1e-8, None)
    eq=np.asarray(obs_equiv, dtype=bool)
    beta=np.zeros_like(eq, dtype=np.float32)
    for m in range(eq.shape[0]):
        denom=float(p[eq[m]].sum())
        if denom <= 1e-8:
            beta[m,m]=1.0
        else:
            beta[m,eq[m]]=p[eq[m]]/denom
    return beta


def class_consistent_witness(Y_option: np.ndarray, margin_option: np.ndarray, mode_probs: np.ndarray, obs_class: np.ndarray):
    Y=np.asarray(Y_option, dtype=np.float32)  # [L,M]
    margins=np.asarray(margin_option, dtype=np.float32)
    p=np.asarray(mode_probs, dtype=np.float32); p=p/np.clip(p.sum(), 1e-8, None)
    obs_class=np.asarray(obs_class, dtype=np.int64)
    L,M=Y.shape
    witness_class={}
    witness_oc=np.zeros(M, dtype=np.int64)
    for c in np.unique(obs_class):
        modes=np.where(obs_class==c)[0]
        score=(Y[:,modes]*p[modes][None,:]).sum(axis=1)
        mean_margin=margins[:,modes].mean(axis=1)
        # lexsort last key primary: score, then margin, then lower index.
        order=np.lexsort((np.arange(L), -mean_margin, -score))
        j=int(order[0])
        witness_class[int(c)]=j
        witness_oc[modes]=j
    Y_oc=Y[witness_oc, np.arange(M)]
    return witness_oc, Y_oc, witness_class
