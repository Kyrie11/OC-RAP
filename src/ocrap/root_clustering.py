from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import CounterfactualFuture


@dataclass
class RootClusteringResult:
    assignments: np.ndarray
    root_probs: np.ndarray
    root_signature: np.ndarray
    root_valid: np.ndarray
    representative_indices: np.ndarray
    future_to_root_weight: np.ndarray


def robust_scale(signatures: np.ndarray) -> np.ndarray:
    med = np.median(signatures, axis=0, keepdims=True)
    mad = np.median(np.abs(signatures - med), axis=0, keepdims=True)
    return (signatures - med) / np.maximum(1.4826 * mad, 1e-3)


def _cluster_distance(X: np.ndarray, ca: list[int], cb: list[int]) -> float:
    vals = []
    for i in ca:
        for j in cb:
            vals.append(np.mean(np.abs(X[i] - X[j])))
    return float(np.mean(vals)) if vals else 0.0


def agglomerative_cluster(signatures: np.ndarray, target_k: int, eps_sig: float) -> list[list[int]]:
    X = robust_scale(signatures.astype(np.float64))
    clusters = [[i] for i in range(X.shape[0])]
    while len(clusters) > target_k:
        best = None
        best_d = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _cluster_distance(X, clusters[a], clusters[b])
                if d < best_d:
                    best_d = d
                    best = (a, b)
        if best is None:
            break
        a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    # optional threshold pass: merge very close clusters even if below target_k not required.
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        best = None
        best_d = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _cluster_distance(X, clusters[a], clusters[b])
                if d < best_d:
                    best_d, best = d, (a, b)
        if best is not None and best_d <= eps_sig and len(clusters) > 1:
            a, b = best
            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]
            changed = True
    return clusters


def build_recovery_signature(M_future: np.ndarray, futures: list[CounterfactualFuture], cfg: dict) -> np.ndarray:
    clip = float(cfg.get("margin_clip", 5.0))
    margins = np.clip(M_future, -clip, clip)
    indicators = []
    for f in futures:
        meta = f.metadata
        indicators.append(
            [
                float(meta.get("contact_surrogate", False)),
                float(meta.get("secondary_threat", False)),
                float(meta.get("control_envelope_uncertain", False)),
                float(meta.get("hidden_emergence", False)),
                float(meta.get("route_blocked", False)),
                float(meta.get("lateral_escape_blocked", False)),
                float(meta.get("rejoin_corridor_available", True)),
                float(meta.get("friction_factor", 1.0)),
            ]
        )
    return np.concatenate([margins, np.asarray(indicators, dtype=np.float32)], axis=1).astype(np.float32)


def _medoid_index(signatures: np.ndarray, idxs: list[int]) -> int:
    if len(idxs) == 1:
        return idxs[0]
    X = signatures[idxs]
    D = np.abs(X[:, None, :] - X[None, :, :]).mean(axis=-1)
    return idxs[int(np.argmin(D.sum(axis=1)))]


def cluster_roots(M_future: np.ndarray, future_probs: np.ndarray, futures: list[CounterfactualFuture], cfg: dict) -> RootClusteringResult:
    K = int(cfg.get("num_roots", 8))
    signatures = build_recovery_signature(M_future, futures, cfg)
    clusters = agglomerative_cluster(signatures, target_k=K, eps_sig=float(cfg.get("eps_signature", 0.3)))
    # If threshold merging resulted in fewer clusters, pad; if somehow more, merge smallest into nearest.
    while len(clusters) > K:
        sizes = np.array([sum(future_probs[c]) for c in clusters])
        small = int(np.argmin(sizes))
        X = robust_scale(signatures)
        best, best_d = None, float("inf")
        for i in range(len(clusters)):
            if i == small:
                continue
            d = _cluster_distance(X, clusters[small], clusters[i])
            if d < best_d:
                best, best_d = i, d
        clusters[best].extend(clusters[small])
        del clusters[small]

    J, D = signatures.shape
    assignments = np.full((J,), -1, dtype=np.int64)
    root_probs = np.zeros((K,), dtype=np.float32)
    root_signature = np.zeros((K, D), dtype=np.float32)
    root_valid = np.zeros((K,), dtype=bool)
    representative = np.full((K,), -1, dtype=np.int64)
    f2r = np.zeros((J, K), dtype=np.float32)
    probs = np.asarray(future_probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)
    for k, idxs in enumerate(clusters[:K]):
        idxs = list(idxs)
        root_valid[k] = True
        representative[k] = _medoid_index(signatures, idxs)
        p = probs[idxs]
        p_sum = max(float(p.sum()), 1e-8)
        root_probs[k] = p_sum
        root_signature[k] = np.average(signatures[idxs], axis=0, weights=p / p_sum).astype(np.float32)
        for j in idxs:
            assignments[j] = k
            f2r[j, k] = 1.0
    if root_probs.sum() > 0:
        root_probs = root_probs / root_probs.sum()
    return RootClusteringResult(assignments, root_probs.astype(np.float32), root_signature, root_valid, representative, f2r)


def aggregate_root_margins(M_future: np.ndarray, assignments: np.ndarray, future_probs: np.ndarray, K: int) -> np.ndarray:
    J, L = M_future.shape
    M = np.full((K, L), -1e6, dtype=np.float32)
    probs = np.asarray(future_probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)
    for k in range(K):
        idx = np.where(assignments == k)[0]
        if len(idx) == 0:
            continue
        w = probs[idx]
        w = w / max(float(w.sum()), 1e-8)
        M[k] = np.average(M_future[idx], axis=0, weights=w).astype(np.float32)
    return M


def future_trajectory_signature(futures: list[CounterfactualFuture], assignments: np.ndarray, future_probs: np.ndarray, K: int, width: int = 32) -> np.ndarray:
    J = len(futures)
    feats = np.zeros((J, width), dtype=np.float32)
    for j, f in enumerate(futures):
        st = f.agent_states
        val = f.agent_valid
        # Endpoint and mid-point features for ego plus nearest valid agents, enough for full-future-cluster ablation.
        parts = []
        for tt in [min(st.shape[0] - 1, st.shape[0] // 2), st.shape[0] - 1]:
            valid_agents = np.where(val[tt])[0][:4]
            for a in valid_agents:
                parts.extend([float(st[tt, a, 0]), float(st[tt, a, 1]), float(st[tt, a, 3]), float(st[tt, a, 4])])
        arr = np.asarray(parts, dtype=np.float32)
        feats[j, : min(width, len(arr))] = arr[:width]
    out = np.zeros((K, width), dtype=np.float32)
    probs = np.asarray(future_probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)
    for k in range(K):
        idx = np.where(assignments == k)[0]
        if len(idx) == 0:
            continue
        w = probs[idx]
        w = w / max(float(w.sum()), 1e-8)
        out[k] = np.average(feats[idx], axis=0, weights=w).astype(np.float32)
    return out
