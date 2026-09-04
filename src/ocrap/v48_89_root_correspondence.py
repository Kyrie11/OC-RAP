from __future__ import annotations

"""V48.89 counterfactual root-correspondence / response-identifiability tools.

This module is audit-only.  It does not alter the planner, the teacher labels,
or the datasets.  It answers a prerequisite question exposed by V48.88:

    Can a candidate root be matched to the corresponding nominal root, and do
    the stored teacher sidecars identify the candidate-induced physical-margin
    change on the exact nested OC-MERO lower tail?

The correspondence uses shared counterfactual-future semantics, not root slot
indices.  The physical response is represented as a conservative interval by
inverting the current ordered structural teacher operators at each root-option
cell and differencing matched candidate/nominal cells.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.algorithms.ocmero import sparsify_compatibility
from ocrap.v48_79_truth_contract import weighted_lcvar_influence_np
from ocrap.v48_81_switch_inverse_truth_contract import (
    _BOUND,
    _cell_preimage,
    structural_root_option_reason_profile,
)

_IDENTITY_FIELDS = (
    "reactive_variant",
    "rollout_variant",
    "targeted_type",
    "artifact_branch",
    "hidden_intent",
    "visible_branch",
    "ego_after_prefix_accel",
    "contact_surrogate",
    "secondary_collision_approach",
    "low_friction",
    "control_delay_noise",
    "scenario_augmented",
    "natural_hidden_candidate",
)


@dataclass(frozen=True)
class RootCorrespondenceRecord:
    valid: bool
    error: str | None
    shared_future_mass_candidate: float
    shared_future_mass_nominal: float
    semantic_identity_fallback_fraction_candidate: float
    semantic_identity_fallback_fraction_nominal: float
    exact_candidate_root_fraction: float
    exact_candidate_root_probability_mass: float
    mean_soft_root_purity: float
    nested_tail_exact_correspondence_mass: float
    nested_tail_soft_correspondence_mass: float
    candidate_tail_influence_sum: float
    nominal_tail_influence_sum: float
    candidate_to_nominal_root: list[int]
    candidate_root_purity: list[float]
    branch_key_collision_count_candidate: int
    branch_key_collision_count_nominal: int
    candidate_r_dep_recomputed: float
    candidate_r_dep_stored: float
    candidate_r_dep_abs_error: float
    nominal_r_dep_recomputed: float
    nominal_r_dep_stored: float
    nominal_r_dep_abs_error: float
    matched_tail_informative_response_mass: float
    matched_tail_sign_identifiable_mass: float
    matched_tail_positive_response_mass: float
    matched_tail_negative_response_mass: float
    matched_tail_ambiguous_response_mass: float
    matched_tail_point_identifiable_mass: float
    matched_tail_signed_mass_score: float
    matched_tail_finite_midpoint_score: float | None
    slot_tail_informative_response_mass: float
    slot_tail_sign_identifiable_mass: float
    slot_tail_signed_mass_score: float
    branch_vs_slot_mapping_disagreement_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            value = arr.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value


def _string_vector(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out: list[str] = []
    for x in arr.reshape(-1):
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="ignore")
        out.append(str(x))
    return out


def _stable_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return str(value)


def semantic_future_branch_keys(sample: dict[str, Any]) -> tuple[list[str], np.ndarray, int]:
    """Return unique semantic branch keys and weak-fallback flags.

    Root slots are candidate-specific, but the generated counterfactual branches
    carry stable semantics such as replay/reactive variant/targeted type.  A
    deterministic occurrence suffix handles repeated stress branches.  The
    suffix is marked as a weak order fallback when metadata does not distinguish
    otherwise identical branches; the audit reports its mass instead of hiding
    it.
    """
    sources = _string_vector(sample.get("future_sources", []))
    metas = _json_scalar(sample.get("future_metadata"), [])
    if not isinstance(metas, list):
        metas = []
    n = max(len(sources), len(metas), int(np.asarray(sample.get("future_probs", [])).size))
    if len(sources) < n:
        sources += [""] * (n - len(sources))
    bases: list[str] = []
    weak = np.zeros(n, dtype=bool)
    for i in range(n):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        payload: dict[str, Any] = {"source": sources[i]}
        informative_fields = 0
        for k in _IDENTITY_FIELDS:
            if k in meta:
                payload[k] = _stable_scalar(meta[k])
                informative_fields += 1
        # The replay branch is intrinsically unique.  Repeated reactive/targeted
        # branches without semantic metadata require an order fallback.
        weak[i] = bool(informative_fields == 0 and sources[i] not in {"replay", "closed_loop_feature_only"})
        bases.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    multiplicity: dict[str, int] = {}
    for base in bases:
        multiplicity[base] = multiplicity.get(base, 0) + 1
    seen: dict[str, int] = {}
    keys: list[str] = []
    collisions = 0
    for i, base in enumerate(bases):
        ordinal = seen.get(base, 0)
        if ordinal:
            collisions += 1
        seen[base] = ordinal + 1
        # An occurrence suffix is deterministic, but it is only a weak identity
        # when two otherwise identical semantic branches coexist.  Mark every
        # member of such a duplicate class, not just metadata-poor branches, so
        # the preregistered gate cannot silently treat array order as semantics.
        weak[i] = bool(weak[i] or multiplicity[base] > 1)
        keys.append(f"{base}#occ={ordinal}")
    return keys, weak, int(collisions)


def nested_tail_influence(
    sample: dict[str, Any], *, alpha: float = 0.2, beta: float = 0.2, top_m: int = 8
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Exact stable-sort/fractional-tail nested OC-MERO cell influence."""
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    K, L = M.shape
    p = np.asarray(sample.get("root_probs", np.zeros(K)), dtype=np.float64).reshape(-1)[:K]
    rv = np.asarray(sample.get("root_valid", np.ones(K)), dtype=bool).reshape(-1)[:K]
    if p.size < K:
        p = np.pad(p, (0, K - p.size))
    if rv.size < K:
        rv = np.pad(rv, (0, K - rv.size), constant_values=False)
    p = normalize_weights(np.where(rv, p, 0.0))
    C = np.asarray(sample.get("c_star", np.eye(K)), dtype=np.float64)
    if C.shape != (K, K):
        raise ValueError(f"c_star shape mismatch {C.shape} != {(K, K)}")
    C_eff = sparsify_compatibility(C, int(top_m))
    ov = np.asarray(sample.get("option_valid", np.ones(L)), dtype=bool).reshape(-1)[:L]
    if ov.size < L:
        ov = np.pad(ov, (0, L - ov.size), constant_values=False)

    q = np.full((K, L), -1.0e9, dtype=np.float64)
    inner: list[list[np.ndarray | None]] = [[None for _ in range(L)] for _ in range(K)]
    for i in range(K):
        w = normalize_weights(C_eff[i] * p)
        for l in range(L):
            if not ov[l]:
                continue
            q[i, l] = weighted_lcvar(M[:, l], w, float(beta))
            inner[i][l] = weighted_lcvar_influence_np(M[:, l], w, float(beta))
    best = np.argmax(q, axis=1)
    r = q[np.arange(K), best]
    outer = weighted_lcvar_influence_np(r, p, float(alpha))
    r_dep = float(weighted_lcvar(r, p, float(alpha)))
    mass = np.zeros((K, L), dtype=np.float64)
    for i in range(K):
        if outer[i] <= 0.0:
            continue
        l = int(best[i])
        ii = inner[i][l]
        if ii is None:
            continue
        mass[:, l] += float(outer[i]) * ii
    stored = float(np.asarray(sample.get("r_dep_star", np.nan)).reshape(-1)[0])
    return mass, r_dep, stored, p


def root_option_physical_intervals(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative pre-structural physical intervals for every root-option cell."""
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    any_bits, all_bits, complete = structural_root_option_reason_profile(sample)
    if any_bits.shape != M.shape or all_bits.shape != M.shape or complete.shape != M.shape:
        raise ValueError("structural profile shape mismatch")
    lo = np.empty_like(M)
    hi = np.empty_like(M)
    exact = np.zeros_like(M, dtype=bool)
    informative = np.zeros_like(M, dtype=bool)
    for k in range(M.shape[0]):
        for l in range(M.shape[1]):
            a, b, e, _mixed, _incomplete, _contradiction = _cell_preimage(
                float(M[k, l]), int(any_bits[k, l]), int(all_bits[k, l]), bool(complete[k, l])
            )
            lo[k, l], hi[k, l], exact[k, l] = float(a), float(b), bool(e)
            informative[k, l] = bool(a > -0.5 * _BOUND or b < 0.5 * _BOUND)
    return lo, hi, exact, informative


def _root_branch_sets(keys: list[str], assignments: np.ndarray, K: int) -> list[set[str]]:
    out = [set() for _ in range(K)]
    for i, key in enumerate(keys):
        if i >= assignments.size:
            break
        k = int(assignments[i])
        if 0 <= k < K:
            out[k].add(key)
    return out


def _response_stats(
    cand_mass: np.ndarray,
    mapping: np.ndarray,
    c_lo: np.ndarray,
    c_hi: np.ndarray,
    c_exact: np.ndarray,
    n_lo: np.ndarray,
    n_hi: np.ndarray,
    n_exact: np.ndarray,
) -> dict[str, float | None]:
    total = float(cand_mass.sum())
    if total <= 1e-12:
        return {
            "informative": 0.0,
            "sign": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "ambiguous": 0.0,
            "point": 0.0,
            "signed_score": 0.0,
            "midpoint": None,
        }
    inf = sign = pos = neg = amb = point = 0.0
    mid_num = mid_den = 0.0
    Kc, L = cand_mass.shape
    for kc in range(Kc):
        kn = int(mapping[kc]) if kc < mapping.size else -1
        if kn < 0 or kn >= n_lo.shape[0]:
            continue
        for l in range(L):
            m = float(cand_mass[kc, l])
            if m <= 0.0 or l >= n_lo.shape[1]:
                continue
            clo, chi = float(c_lo[kc, l]), float(c_hi[kc, l])
            nlo, nhi = float(n_lo[kn, l]), float(n_hi[kn, l])
            lo_finite = clo > -0.5 * _BOUND and nhi < 0.5 * _BOUND
            hi_finite = chi < 0.5 * _BOUND and nlo > -0.5 * _BOUND
            rlo = clo - nhi if lo_finite else -_BOUND
            rhi = chi - nlo if hi_finite else _BOUND
            if lo_finite or hi_finite:
                inf += m
            if lo_finite and rlo > 0.0:
                sign += m
                pos += m
            elif hi_finite and rhi < 0.0:
                sign += m
                neg += m
            else:
                amb += m
            if bool(c_exact[kc, l] and n_exact[kn, l]):
                point += m
            if lo_finite and hi_finite:
                mid_num += m * 0.5 * (rlo + rhi)
                mid_den += m
    return {
        "informative": float(inf / total),
        "sign": float(sign / total),
        "positive": float(pos / total),
        "negative": float(neg / total),
        "ambiguous": float(amb / total),
        "point": float(point / total),
        "signed_score": float((pos - neg) / total),
        "midpoint": float(mid_num / mid_den) if mid_den > 1e-12 else None,
    }


def audit_candidate_nominal_pair(
    candidate: dict[str, Any],
    nominal: dict[str, Any],
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    top_m: int = 8,
    recompute_tolerance: float = 1.0e-5,
) -> RootCorrespondenceRecord:
    """Audit one candidate against its same-scene-time nominal action."""
    try:
        cm = np.asarray(candidate.get("m_star"), dtype=np.float64)
        nm = np.asarray(nominal.get("m_star"), dtype=np.float64)
        if cm.ndim != 2 or nm.ndim != 2 or cm.shape[1] != nm.shape[1]:
            raise ValueError("candidate/nominal m_star option shape mismatch")
        Kc, _L = cm.shape
        Kn = nm.shape[0]
        ca = np.asarray(candidate.get("root_assignments", []), dtype=np.int64).reshape(-1)
        na = np.asarray(nominal.get("root_assignments", []), dtype=np.int64).reshape(-1)
        cp = np.asarray(candidate.get("future_probs", []), dtype=np.float64).reshape(-1)
        np_ = np.asarray(nominal.get("future_probs", []), dtype=np.float64).reshape(-1)
        cv = np.asarray(candidate.get("future_valid", np.ones_like(cp)), dtype=bool).reshape(-1)
        nv = np.asarray(nominal.get("future_valid", np.ones_like(np_)), dtype=bool).reshape(-1)
        ckeys, cweak, ccoll = semantic_future_branch_keys(candidate)
        nkeys, nweak, ncoll = semantic_future_branch_keys(nominal)
        nc = min(len(ckeys), ca.size, cp.size, cv.size)
        nn = min(len(nkeys), na.size, np_.size, nv.size)
        c_mask = cv[:nc]
        n_mask = nv[:nn]
        ckeys = [k for k, keep in zip(ckeys[:nc], c_mask.tolist()) if keep]
        nkeys = [k for k, keep in zip(nkeys[:nn], n_mask.tolist()) if keep]
        cweak = cweak[:nc][c_mask]
        nweak = nweak[:nn][n_mask]
        ca = ca[:nc][c_mask]
        na = na[:nn][n_mask]
        cp = normalize_weights(cp[:nc][c_mask])
        np_ = normalize_weights(np_[:nn][n_mask])
        ci = {k: i for i, k in enumerate(ckeys)}
        ni = {k: i for i, k in enumerate(nkeys)}
        shared = sorted(set(ci).intersection(ni))
        shared_c_mass = float(sum(cp[ci[k]] for k in shared))
        shared_n_mass = float(sum(np_[ni[k]] for k in shared))
        overlap = np.zeros((Kc, Kn), dtype=np.float64)
        for key in shared:
            ic, inn = ci[key], ni[key]
            kc, kn = int(ca[ic]), int(na[inn])
            if 0 <= kc < Kc and 0 <= kn < Kn:
                overlap[kc, kn] += min(float(cp[ic]), float(np_[inn]))

        csets = _root_branch_sets(ckeys, ca, Kc)
        nsets = _root_branch_sets(nkeys, na, Kn)
        mapping = np.full(Kc, -1, dtype=np.int64)
        purity = np.zeros(Kc, dtype=np.float64)
        exact = np.zeros(Kc, dtype=bool)
        for kc in range(Kc):
            row = overlap[kc]
            rs = float(row.sum())
            if rs <= 1e-12:
                continue
            kn = int(np.argmax(row))
            mapping[kc] = kn
            purity[kc] = float(row[kn] / rs)
            # Exact correspondence is semantic set equality, not root-slot equality.
            exact[kc] = bool(csets[kc] and csets[kc] == nsets[kn])
        # Require mutual uniqueness: two candidate roots cannot both claim one nominal root.
        for kn in range(Kn):
            claim = [kc for kc in range(Kc) if exact[kc] and int(mapping[kc]) == kn]
            if len(claim) != 1:
                for kc in claim:
                    exact[kc] = False
        exact_map = np.where(exact, mapping, -1)

        ctail, cr, cs, croot_p = nested_tail_influence(candidate, alpha=alpha, beta=beta, top_m=top_m)
        ntail, nr, ns, _nroot_p = nested_tail_influence(nominal, alpha=alpha, beta=beta, top_m=top_m)
        csum = float(ctail.sum())
        nsum = float(ntail.sum())
        tail_root_mass = ctail.sum(axis=1)
        exact_tail = float(tail_root_mass[exact].sum() / max(csum, 1e-12))
        soft_tail = float(np.sum(tail_root_mass * purity) / max(csum, 1e-12))

        clo, chi, cex, _cinf = root_option_physical_intervals(candidate)
        nlo, nhi, nex, _ninf = root_option_physical_intervals(nominal)
        branch_stats = _response_stats(ctail, exact_map, clo, chi, cex, nlo, nhi, nex)
        slot_map = np.arange(Kc, dtype=np.int64)
        slot_map[slot_map >= Kn] = -1
        slot_stats = _response_stats(ctail, slot_map, clo, chi, cex, nlo, nhi, nex)
        valid_roots = [kc for kc in range(Kc) if csets[kc]]
        disagree = (
            float(np.mean([int(exact_map[k] != slot_map[k]) for k in valid_roots]))
            if valid_roots
            else 0.0
        )
        valid = bool(
            np.isfinite(cr)
            and np.isfinite(nr)
            and np.isfinite(cs)
            and np.isfinite(ns)
            and abs(cr - cs) <= recompute_tolerance
            and abs(nr - ns) <= recompute_tolerance
            and csum > 0.999 - 1e-5
            and nsum > 0.999 - 1e-5
        )
        weak_c_mass = float(np.sum(cp[cweak])) if cweak.size else 0.0
        weak_n_mass = float(np.sum(np_[nweak])) if nweak.size else 0.0
        exact_root_frac = float(np.mean(exact[valid_roots])) if valid_roots else 0.0
        exact_root_prob = float(np.sum(croot_p[exact]))
        mean_purity = float(np.mean(purity[valid_roots])) if valid_roots else 0.0
        return RootCorrespondenceRecord(
            valid=valid,
            error=None if valid else "OC-MERO recomputation or tail influence contract failed",
            shared_future_mass_candidate=shared_c_mass,
            shared_future_mass_nominal=shared_n_mass,
            semantic_identity_fallback_fraction_candidate=weak_c_mass,
            semantic_identity_fallback_fraction_nominal=weak_n_mass,
            exact_candidate_root_fraction=exact_root_frac,
            exact_candidate_root_probability_mass=exact_root_prob,
            mean_soft_root_purity=mean_purity,
            nested_tail_exact_correspondence_mass=exact_tail,
            nested_tail_soft_correspondence_mass=soft_tail,
            candidate_tail_influence_sum=csum,
            nominal_tail_influence_sum=nsum,
            candidate_to_nominal_root=[int(x) for x in exact_map.tolist()],
            candidate_root_purity=[float(x) for x in purity.tolist()],
            branch_key_collision_count_candidate=ccoll,
            branch_key_collision_count_nominal=ncoll,
            candidate_r_dep_recomputed=cr,
            candidate_r_dep_stored=cs,
            candidate_r_dep_abs_error=abs(cr - cs),
            nominal_r_dep_recomputed=nr,
            nominal_r_dep_stored=ns,
            nominal_r_dep_abs_error=abs(nr - ns),
            matched_tail_informative_response_mass=float(branch_stats["informative"]),
            matched_tail_sign_identifiable_mass=float(branch_stats["sign"]),
            matched_tail_positive_response_mass=float(branch_stats["positive"]),
            matched_tail_negative_response_mass=float(branch_stats["negative"]),
            matched_tail_ambiguous_response_mass=float(branch_stats["ambiguous"]),
            matched_tail_point_identifiable_mass=float(branch_stats["point"]),
            matched_tail_signed_mass_score=float(branch_stats["signed_score"]),
            matched_tail_finite_midpoint_score=branch_stats["midpoint"],
            slot_tail_informative_response_mass=float(slot_stats["informative"]),
            slot_tail_sign_identifiable_mass=float(slot_stats["sign"]),
            slot_tail_signed_mass_score=float(slot_stats["signed_score"]),
            branch_vs_slot_mapping_disagreement_fraction=disagree,
        )
    except Exception as exc:
        return RootCorrespondenceRecord(
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
            shared_future_mass_candidate=0.0,
            shared_future_mass_nominal=0.0,
            semantic_identity_fallback_fraction_candidate=1.0,
            semantic_identity_fallback_fraction_nominal=1.0,
            exact_candidate_root_fraction=0.0,
            exact_candidate_root_probability_mass=0.0,
            mean_soft_root_purity=0.0,
            nested_tail_exact_correspondence_mass=0.0,
            nested_tail_soft_correspondence_mass=0.0,
            candidate_tail_influence_sum=0.0,
            nominal_tail_influence_sum=0.0,
            candidate_to_nominal_root=[],
            candidate_root_purity=[],
            branch_key_collision_count_candidate=0,
            branch_key_collision_count_nominal=0,
            candidate_r_dep_recomputed=float("nan"),
            candidate_r_dep_stored=float("nan"),
            candidate_r_dep_abs_error=float("inf"),
            nominal_r_dep_recomputed=float("nan"),
            nominal_r_dep_stored=float("nan"),
            nominal_r_dep_abs_error=float("inf"),
            matched_tail_informative_response_mass=0.0,
            matched_tail_sign_identifiable_mass=0.0,
            matched_tail_positive_response_mass=0.0,
            matched_tail_negative_response_mass=0.0,
            matched_tail_ambiguous_response_mass=1.0,
            matched_tail_point_identifiable_mass=0.0,
            matched_tail_signed_mass_score=0.0,
            matched_tail_finite_midpoint_score=None,
            slot_tail_informative_response_mass=0.0,
            slot_tail_sign_identifiable_mass=0.0,
            slot_tail_signed_mass_score=0.0,
            branch_vs_slot_mapping_disagreement_fraction=0.0,
        )
