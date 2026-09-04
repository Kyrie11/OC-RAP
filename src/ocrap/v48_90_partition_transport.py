from __future__ import annotations

"""V48.90 counterfactual equivalence-class / root-partition transport audit.

This module is audit-only.  It does not change the deployed planner, teacher
labels, datasets, Stage I, OC-MERO, RIFA, ranking, thresholds, or trainable
state.  It follows the fail-closed V48.89 branch after individual-branch root
correspondence STOP.

The key distinction is between *future instances* and *counterfactual future
classes*.  Dataset construction can intentionally materialize repeated stress
instances with the same branch recipe.  Those copies are exchangeable and do
not have a meaningful individual identity.  Conversely, some augmented
branches use candidate-dependent random realization (hidden spawn, visible
actor choice, contact impulse); matching them only by a coarse semantic label
would fabricate causal correspondence.

V48.90 therefore:
  1) quotients exchangeable instances by a candidate-independent branch recipe;
  2) optionally refines the class with exogenous-realization fields;
  3) induces a probability-mass coupling between candidate/nominal root
     partitions rather than assuming a root-ID bijection;
  4) measures exact nested-tail transport coverage/purity and conservative
     transport-coupled physical-response identifiability.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights
from ocrap.v48_89_root_correspondence import (
    _BOUND,
    _IDENTITY_FIELDS,
    _json_scalar,
    _stable_scalar,
    _string_vector,
    nested_tail_influence,
    root_option_physical_intervals,
)

# Fields that identify the *realized external perturbation*, not merely the
# semantic recipe.  They are only used offline to decide whether two branches
# can be interpreted as the same exogenous counterfactual across candidate and
# nominal actions.
_REALIZATION_FIELDS = (
    "hidden_spawn_xy",
    "hidden_spawn_cell",
    "hidden_actor_object_index",
    "visible_actor_object_index",
    "yaw_rate_impulse",
    "lateral_velocity_impulse",
    "injected_agent_slot",
    "hidden_start_step",
)


@dataclass(frozen=True)
class PartitionTransportMetrics:
    valid: bool
    error: str | None
    recipe_shared_mass_candidate: float
    recipe_shared_mass_nominal: float
    recipe_matched_transport_mass: float
    recipe_unresolved_semantic_mass_candidate: float
    recipe_unresolved_semantic_mass_nominal: float
    exchangeable_duplicate_mass_candidate: float
    exchangeable_duplicate_mass_nominal: float
    duplicate_root_homogeneity_mass_candidate: float
    duplicate_root_homogeneity_mass_nominal: float
    exogenous_shared_mass_candidate: float
    exogenous_shared_mass_nominal: float
    exogenous_matched_transport_mass: float
    exogenous_unresolved_mass_candidate: float
    exogenous_unresolved_mass_nominal: float
    recipe_tail_transport_coverage: float
    recipe_tail_transport_purity: float
    recipe_tail_partition_stability: float
    exogenous_tail_transport_coverage: float
    exogenous_tail_transport_purity: float
    exogenous_tail_partition_stability: float
    exogenous_tail_split_merge_mass: float
    exogenous_tail_unmatched_mass: float
    exogenous_transport_sign_identifiable_mass: float
    exogenous_transport_informative_response_mass: float
    exogenous_transport_point_identifiable_mass: float
    exogenous_transport_positive_response_mass: float
    exogenous_transport_negative_response_mass: float
    exogenous_transport_ambiguous_response_mass: float
    exogenous_transport_signed_response_score: float
    candidate_root_probability_recompute_error: float
    nominal_root_probability_recompute_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_obj(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_stable_obj(v) for v in list(value)]
    return _stable_scalar(value)


def _future_arrays(sample: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    sources = _string_vector(sample.get("future_sources", []))
    metas = _json_scalar(sample.get("future_metadata"), [])
    if not isinstance(metas, list):
        metas = []
    probs = np.asarray(sample.get("future_probs", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(sample.get("future_valid", np.ones_like(probs)), dtype=bool).reshape(-1)
    assignments = np.asarray(sample.get("root_assignments", []), dtype=np.int64).reshape(-1)
    n = probs.size
    if not (len(sources) == len(metas) == valid.size == assignments.size == n):
        raise ValueError(
            "future sidecar length mismatch: "
            f"sources/meta/probs/valid/assign={len(sources)}/{len(metas)}/{n}/{valid.size}/{assignments.size}"
        )
    if any(not isinstance(m, dict) for m in metas):
        raise ValueError("future_metadata must be a list of dicts")
    mask = valid
    if np.any(~np.isfinite(probs[mask])) or np.any(probs[mask] < 0.0):
        raise ValueError("future probabilities must be finite and nonnegative")
    sources = [s for s, keep in zip(sources, mask.tolist()) if keep]
    metas = [m for m, keep in zip(metas, mask.tolist()) if keep]
    assignments = assignments[mask]
    probs = normalize_weights(probs[mask])
    return sources, metas, probs, assignments, mask


def _recipe_payload(source: str, meta: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    payload: dict[str, Any] = {"source": str(source)}
    informative = 0
    for k in _IDENTITY_FIELDS:
        if k in meta:
            payload[k] = _stable_obj(meta[k])
            informative += 1
    # replay/feature-only anchors are intrinsically unique semantic recipes.
    unresolved = bool(informative == 0 and source not in {"replay", "closed_loop_feature_only"})
    return payload, unresolved


def _requires_realization_fingerprint(meta: dict[str, Any]) -> bool:
    # Require a realization fingerprint only when the branch construction
    # actually contains candidate-sensitive/stochastic external-world choices.
    # `contact_surrogate` is also used by the deterministic
    # secondary-collision-approach recipe, so that flag alone is not evidence
    # of a random impulse realization.
    targeted = str(meta.get("targeted_type", ""))
    stochastic_contact_impulse = bool(
        targeted == "waymax_contact_impulse_surrogate"
        or ("yaw_rate_impulse" in meta)
        or ("lateral_velocity_impulse" in meta)
    )
    return bool(
        meta.get("scenario_augmented", False)
        or meta.get("hidden_emergence", False)
        or meta.get("visible_perturbation", False)
        or meta.get("natural_hidden_candidate", False)
        or stochastic_contact_impulse
    )


def _realization_payload(meta: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not _requires_realization_fingerprint(meta):
        return {}, False
    payload: dict[str, Any] = {}
    for k in _REALIZATION_FIELDS:
        if k in meta:
            payload[k] = _stable_obj(meta[k])
    # Natural-hidden provenance can be stored with prefixed fields.  Preserve
    # only realization-like fields, not diagnostic/runtime strings.
    for k, v in meta.items():
        sk = str(k)
        if sk.startswith("natural_hidden_spawn") or sk.startswith("natural_hidden_actor"):
            payload[sk] = _stable_obj(v)
    unresolved = len(payload) == 0
    return payload, unresolved


def future_class_keys(sample: dict[str, Any], *, exogenous: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return quotient class keys, unresolved flags, and duplicate flags.

    Duplicate instances with the same recipe share one key; unlike V48.89 they
    do not receive an occurrence suffix.  In exogenous mode a branch that is
    generated from candidate-sensitive augmentation additionally requires the
    realized perturbation fingerprint to match.
    """
    sources, metas, _p, _a, _mask = _future_arrays(sample)
    recipe_serialized: list[str] = []
    recipe_unresolved: list[bool] = []
    realization_serialized: list[str | None] = []
    realization_unresolved: list[bool] = []
    for source, meta in zip(sources, metas):
        rec, rec_unres = _recipe_payload(source, meta)
        recipe_serialized.append(json.dumps(rec, sort_keys=True, separators=(",", ":")))
        recipe_unresolved.append(rec_unres)
        real, real_unres = _realization_payload(meta)
        realization_serialized.append(
            json.dumps(real, sort_keys=True, separators=(",", ":")) if real else None
        )
        realization_unresolved.append(real_unres)
    mult: dict[str, int] = {}
    for k in recipe_serialized:
        mult[k] = mult.get(k, 0) + 1
    duplicate = np.asarray([mult[k] > 1 for k in recipe_serialized], dtype=bool)
    unresolved = np.asarray(recipe_unresolved, dtype=bool)
    keys: list[str] = []
    for rec, real, real_unres in zip(recipe_serialized, realization_serialized, realization_unresolved):
        if exogenous and real_unres:
            unresolved[len(keys)] = True
        if exogenous and real is not None:
            keys.append(f"{rec}|real={real}")
        else:
            keys.append(rec)
    return keys, unresolved, duplicate


def _class_root_table(
    sample: dict[str, Any], *, exogenous: bool
) -> tuple[dict[str, np.ndarray], dict[str, float], set[str], float, float, float, np.ndarray]:
    sources, metas, probs, assignments, _mask = _future_arrays(sample)
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("m_star must be [K,L]")
    K = M.shape[0]
    if np.any((assignments < 0) | (assignments >= K)):
        raise ValueError("valid future assigned to invalid root")
    keys, unresolved_flags, duplicate_flags = future_class_keys(sample, exogenous=exogenous)
    if len(keys) != len(probs):
        raise ValueError("quotient key/probability length mismatch")
    tables: dict[str, np.ndarray] = {}
    totals: dict[str, float] = {}
    unresolved_keys: set[str] = set()
    multiplicity: dict[str, int] = {}
    for key in keys:
        multiplicity[key] = multiplicity.get(key, 0) + 1
    for j, (key, p, k) in enumerate(zip(keys, probs, assignments)):
        tables.setdefault(key, np.zeros(K, dtype=np.float64))[int(k)] += float(p)
        totals[key] = totals.get(key, 0.0) + float(p)
        if bool(unresolved_flags[j]):
            unresolved_keys.add(key)
    unresolved_mass = float(sum(totals[k] for k in unresolved_keys))
    duplicate_mass = float(sum(float(p) for p, dup in zip(probs, duplicate_flags) if bool(dup)))
    duplicate_homogeneous_mass = 0.0
    if duplicate_mass > 1e-12:
        good = 0.0
        for key, vec in tables.items():
            if multiplicity.get(key, 0) <= 1:
                continue
            total = float(vec.sum())
            if total > 0 and int(np.count_nonzero(vec > 1e-12)) == 1:
                good += total
        duplicate_homogeneous_mass = float(good / duplicate_mass)
    else:
        duplicate_homogeneous_mass = 1.0
    derived_root = np.zeros(K, dtype=np.float64)
    for vec in tables.values():
        derived_root += vec
    stored = np.asarray(sample.get("root_probs", np.zeros(K)), dtype=np.float64).reshape(-1)
    if stored.size < K:
        stored = np.pad(stored, (0, K - stored.size))
    stored = normalize_weights(stored[:K])
    root_error = float(np.max(np.abs(derived_root - stored))) if K else 0.0
    return tables, totals, unresolved_keys, unresolved_mass, duplicate_mass, duplicate_homogeneous_mass, derived_root


def _coupling(
    candidate: dict[str, Any], nominal: dict[str, Any], *, exogenous: bool
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    ct, ctot, cunres, cunres_mass, cdup, chomo, croot = _class_root_table(candidate, exogenous=exogenous)
    nt, ntot, nunres, nunres_mass, ndup, nhomo, nroot = _class_root_table(nominal, exogenous=exogenous)
    Kc, Kn = len(croot), len(nroot)
    P = np.zeros((Kc, Kn), dtype=np.float64)
    shared = sorted((set(ct) - cunres).intersection(set(nt) - nunres))
    c_shared_support = float(sum(ctot[k] for k in shared))
    n_shared_support = float(sum(ntot[k] for k in shared))
    for key in shared:
        cv, nv = ct[key], nt[key]
        cs, ns = float(cv.sum()), float(nv.sum())
        if cs <= 1e-12 or ns <= 1e-12:
            continue
        shared_mass = min(cs, ns)
        P += shared_mass * np.outer(cv / cs, nv / ns)
    meta = {
        "shared_mass_candidate": c_shared_support,
        "shared_mass_nominal": n_shared_support,
        "matched_transport_mass": float(P.sum()),
        "unresolved_mass_candidate": float(cunres_mass),
        "unresolved_mass_nominal": float(nunres_mass),
        "duplicate_mass_candidate": float(cdup),
        "duplicate_mass_nominal": float(ndup),
        "duplicate_homogeneity_candidate": float(chomo),
        "duplicate_homogeneity_nominal": float(nhomo),
    }
    return P, meta, croot, nroot


def _tail_transport_metrics(
    tail_mass: np.ndarray, coupling: np.ndarray, croot: np.ndarray
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    root_tail = np.asarray(tail_mass, dtype=np.float64).sum(axis=1)
    row = coupling.sum(axis=1)
    coverage = np.zeros_like(row)
    ok = croot > 1e-12
    coverage[ok] = np.minimum(1.0, row[ok] / croot[ok])
    purity = np.zeros_like(row)
    rok = row > 1e-12
    if coupling.size:
        purity[rok] = np.max(coupling[rok], axis=1) / row[rok]
    cov = float(np.sum(root_tail * coverage))
    stability = float(np.sum(root_tail * coverage * purity))
    pur = float(stability / cov) if cov > 1e-12 else 0.0
    split = float(max(0.0, cov - stability))
    unmatched = float(max(0.0, 1.0 - cov))
    return cov, pur, stability, split, unmatched, coverage, purity


def _transport_response_stats(
    tail_mass: np.ndarray,
    coupling: np.ndarray,
    croot: np.ndarray,
    clo: np.ndarray,
    chi: np.ndarray,
    cexact: np.ndarray,
    nlo: np.ndarray,
    nhi: np.ndarray,
    nexact: np.ndarray,
    common_option_valid: np.ndarray,
) -> dict[str, float]:
    total = float(tail_mass.sum())
    if total <= 1e-12:
        return {k: 0.0 for k in ("informative", "sign", "point", "positive", "negative", "ambiguous", "signed_score")}
    row = coupling.sum(axis=1)
    coverage = np.zeros_like(row)
    ok = croot > 1e-12
    coverage[ok] = np.minimum(1.0, row[ok] / croot[ok])
    inf = sign = point = pos = neg = amb = 0.0
    for kc in range(tail_mass.shape[0]):
        support = np.where(coupling[kc] > 1e-12)[0]
        if support.size == 0 or coverage[kc] <= 1e-12:
            continue
        for l in range(tail_mass.shape[1]):
            if l >= common_option_valid.size or not bool(common_option_valid[l]):
                continue
            base_mass = float(tail_mass[kc, l]) * float(coverage[kc])
            if base_mass <= 0.0:
                continue
            rlos: list[float] = []
            rhis: list[float] = []
            any_bound = False
            all_point = bool(cexact[kc, l])
            for kn in support.tolist():
                c_lo, c_hi = float(clo[kc, l]), float(chi[kc, l])
                n_lo, n_hi = float(nlo[kn, l]), float(nhi[kn, l])
                lo_finite = c_lo > -0.5 * _BOUND and n_hi < 0.5 * _BOUND
                hi_finite = c_hi < 0.5 * _BOUND and n_lo > -0.5 * _BOUND
                if lo_finite:
                    rlos.append(c_lo - n_hi)
                    any_bound = True
                else:
                    rlos.append(-_BOUND)
                if hi_finite:
                    rhis.append(c_hi - n_lo)
                    any_bound = True
                else:
                    rhis.append(_BOUND)
                all_point = bool(all_point and nexact[kn, l])
            lo = min(rlos) if rlos else -_BOUND
            hi = max(rhis) if rhis else _BOUND
            if any_bound:
                inf += base_mass
            if lo > 0.0:
                sign += base_mass
                pos += base_mass
            elif hi < 0.0:
                sign += base_mass
                neg += base_mass
            else:
                amb += base_mass
            if all_point and lo > -0.5 * _BOUND and hi < 0.5 * _BOUND and abs(hi - lo) <= 1e-6:
                point += base_mass
    # Unmatched tail mass is response-ambiguous by construction.
    matched = float(np.sum(tail_mass.sum(axis=1) * coverage))
    amb += max(0.0, total - matched)
    return {
        "informative": float(inf / total),
        "sign": float(sign / total),
        "point": float(point / total),
        "positive": float(pos / total),
        "negative": float(neg / total),
        "ambiguous": float(min(total, amb) / total),
        "signed_score": float((pos - neg) / total),
    }


def audit_partition_transport_pair(
    candidate: dict[str, Any],
    nominal: dict[str, Any],
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    top_m: int = 8,
    root_prob_tolerance: float = 2.0e-6,
) -> PartitionTransportMetrics:
    try:
        cm = np.asarray(candidate.get("m_star"), dtype=np.float64)
        nm = np.asarray(nominal.get("m_star"), dtype=np.float64)
        if cm.ndim != 2 or nm.ndim != 2 or cm.shape[1] != nm.shape[1]:
            raise ValueError("candidate/nominal m_star option shape mismatch")
        modes_c = _string_vector(candidate.get("recovery_modes", []))
        modes_n = _string_vector(nominal.get("recovery_modes", []))
        if modes_c != modes_n or len(modes_c) != cm.shape[1]:
            raise ValueError("recovery option identity/order mismatch")
        cov_c = np.asarray(candidate.get("option_valid", np.ones(cm.shape[1])), dtype=bool).reshape(-1)[: cm.shape[1]]
        cov_n = np.asarray(nominal.get("option_valid", np.ones(nm.shape[1])), dtype=bool).reshape(-1)[: nm.shape[1]]
        if cov_c.size < cm.shape[1] or cov_n.size < nm.shape[1]:
            raise ValueError("option_valid shape mismatch")
        common_option_valid = cov_c & cov_n

        ctail, cr, cs, _cp = nested_tail_influence(candidate, alpha=alpha, beta=beta, top_m=top_m)
        ntail, nr, ns, _np = nested_tail_influence(nominal, alpha=alpha, beta=beta, top_m=top_m)
        if not (np.isfinite(cr) and np.isfinite(nr) and abs(cr - cs) <= 1e-5 and abs(nr - ns) <= 1e-5):
            raise ValueError("OC-MERO recomputation mismatch")
        if abs(float(ctail.sum()) - 1.0) > 2e-5 or abs(float(ntail.sum()) - 1.0) > 2e-5:
            raise ValueError("nested tail influence does not sum to one")

        Pr, mr, croot_r, nroot_r = _coupling(candidate, nominal, exogenous=False)
        Pe, me, croot_e, nroot_e = _coupling(candidate, nominal, exogenous=True)
        if croot_r.shape != croot_e.shape or nroot_r.shape != nroot_e.shape:
            raise ValueError("recipe/exogenous root shape mismatch")
        cstored = np.asarray(candidate.get("root_probs", np.zeros_like(croot_e)), dtype=np.float64).reshape(-1)[: croot_e.size]
        nstored = np.asarray(nominal.get("root_probs", np.zeros_like(nroot_e)), dtype=np.float64).reshape(-1)[: nroot_e.size]
        cstored = normalize_weights(cstored)
        nstored = normalize_weights(nstored)
        cerr = float(np.max(np.abs(croot_e - cstored))) if croot_e.size else 0.0
        nerr = float(np.max(np.abs(nroot_e - nstored))) if nroot_e.size else 0.0
        if cerr > root_prob_tolerance or nerr > root_prob_tolerance:
            raise ValueError(f"future/root probability recomputation mismatch candidate/nominal={cerr}/{nerr}")

        rcov, rpur, rstab, _rsplit, _runmatched, _rc, _rp = _tail_transport_metrics(ctail, Pr, croot_r)
        ecov, epur, estab, esplit, eunmatched, _ec, _ep = _tail_transport_metrics(ctail, Pe, croot_e)

        clo, chi, cex, _ = root_option_physical_intervals(candidate)
        nlo, nhi, nex, _ = root_option_physical_intervals(nominal)
        resp = _transport_response_stats(
            ctail, Pe, croot_e, clo, chi, cex, nlo, nhi, nex, common_option_valid
        )

        return PartitionTransportMetrics(
            valid=True,
            error=None,
            recipe_shared_mass_candidate=mr["shared_mass_candidate"],
            recipe_shared_mass_nominal=mr["shared_mass_nominal"],
            recipe_matched_transport_mass=mr["matched_transport_mass"],
            recipe_unresolved_semantic_mass_candidate=mr["unresolved_mass_candidate"],
            recipe_unresolved_semantic_mass_nominal=mr["unresolved_mass_nominal"],
            exchangeable_duplicate_mass_candidate=mr["duplicate_mass_candidate"],
            exchangeable_duplicate_mass_nominal=mr["duplicate_mass_nominal"],
            duplicate_root_homogeneity_mass_candidate=mr["duplicate_homogeneity_candidate"],
            duplicate_root_homogeneity_mass_nominal=mr["duplicate_homogeneity_nominal"],
            exogenous_shared_mass_candidate=me["shared_mass_candidate"],
            exogenous_shared_mass_nominal=me["shared_mass_nominal"],
            exogenous_matched_transport_mass=me["matched_transport_mass"],
            exogenous_unresolved_mass_candidate=me["unresolved_mass_candidate"],
            exogenous_unresolved_mass_nominal=me["unresolved_mass_nominal"],
            recipe_tail_transport_coverage=rcov,
            recipe_tail_transport_purity=rpur,
            recipe_tail_partition_stability=rstab,
            exogenous_tail_transport_coverage=ecov,
            exogenous_tail_transport_purity=epur,
            exogenous_tail_partition_stability=estab,
            exogenous_tail_split_merge_mass=esplit,
            exogenous_tail_unmatched_mass=eunmatched,
            exogenous_transport_sign_identifiable_mass=resp["sign"],
            exogenous_transport_informative_response_mass=resp["informative"],
            exogenous_transport_point_identifiable_mass=resp["point"],
            exogenous_transport_positive_response_mass=resp["positive"],
            exogenous_transport_negative_response_mass=resp["negative"],
            exogenous_transport_ambiguous_response_mass=resp["ambiguous"],
            exogenous_transport_signed_response_score=resp["signed_score"],
            candidate_root_probability_recompute_error=cerr,
            nominal_root_probability_recompute_error=nerr,
        )
    except Exception as exc:
        return PartitionTransportMetrics(
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
            recipe_shared_mass_candidate=0.0,
            recipe_shared_mass_nominal=0.0,
            recipe_matched_transport_mass=0.0,
            recipe_unresolved_semantic_mass_candidate=1.0,
            recipe_unresolved_semantic_mass_nominal=1.0,
            exchangeable_duplicate_mass_candidate=0.0,
            exchangeable_duplicate_mass_nominal=0.0,
            duplicate_root_homogeneity_mass_candidate=0.0,
            duplicate_root_homogeneity_mass_nominal=0.0,
            exogenous_shared_mass_candidate=0.0,
            exogenous_shared_mass_nominal=0.0,
            exogenous_matched_transport_mass=0.0,
            exogenous_unresolved_mass_candidate=1.0,
            exogenous_unresolved_mass_nominal=1.0,
            recipe_tail_transport_coverage=0.0,
            recipe_tail_transport_purity=0.0,
            recipe_tail_partition_stability=0.0,
            exogenous_tail_transport_coverage=0.0,
            exogenous_tail_transport_purity=0.0,
            exogenous_tail_partition_stability=0.0,
            exogenous_tail_split_merge_mass=0.0,
            exogenous_tail_unmatched_mass=1.0,
            exogenous_transport_sign_identifiable_mass=0.0,
            exogenous_transport_informative_response_mass=0.0,
            exogenous_transport_point_identifiable_mass=0.0,
            exogenous_transport_positive_response_mass=0.0,
            exogenous_transport_negative_response_mass=0.0,
            exogenous_transport_ambiguous_response_mass=1.0,
            exogenous_transport_signed_response_score=0.0,
            candidate_root_probability_recompute_error=float("inf"),
            nominal_root_probability_recompute_error=float("inf"),
        )
