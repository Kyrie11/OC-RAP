from __future__ import annotations

"""V48.91 common-exogenous future-level physical-margin identifiability.

Audit-only follow-up to V48.90 OC-CEPT.  V48.90 establishes candidate/nominal
counterfactual *partition* transport, but its physical-response sidecar is
constructed by inverting already aggregated/structurally transformed root
margins.  That is conservative and can remain non-identifiable even when the
counterfactual correspondence is correct.

V48.91 deliberately moves the physical-response audit to the minimal causal
unit at which it is available exactly during teacher construction: a common
exogenous future instance/class and recovery option, *before* root aggregation
and before structural floors/overrides.  No value from this module is a model
input and this module contains no trainable state.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights
from ocrap.v48_79_truth_contract import weighted_lcvar_influence_np
from ocrap.v48_89_root_correspondence import nested_tail_influence
from ocrap.v48_90_partition_transport import future_class_keys


ENGINEERING_VERSION = "v48.91.6-OC-CEPMI-RECIPELOCK"


@dataclass(frozen=True)
class FuturePhysicalResponseMetrics:
    valid: bool
    error: str | None
    common_exogenous_tail_coverage: float
    response_informative_mass: float
    response_sign_identifiable_mass: float
    response_point_identifiable_mass: float
    response_positive_mass: float
    response_negative_mass: float
    response_ambiguous_mass: float
    signed_response_score: float
    duplicate_physical_homogeneity_mass_candidate: float
    duplicate_physical_homogeneity_mass_nominal: float
    future_tail_influence_sum: float
    future_tail_reconstruction_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def physical_margin_from_teacher_diag(diag: Any) -> float:
    """Return the exact pre-structural physical active-constraint minimum.

    ``TeacherDiagnostics.component_margins`` is produced before the ordered
    structural floor/cap/override that creates the stored teacher scalar.  The
    active mask determines which components own the physical margin.  This is
    intentionally independent of the final structural label.
    """
    active = getattr(diag, "active", None)
    comps = getattr(diag, "component_margins", None)
    if not isinstance(active, dict) or not isinstance(comps, dict):
        raise ValueError("teacher diagnostic lacks active/component_margins")
    vals: list[float] = []
    for key, enabled in active.items():
        if bool(enabled) and key in comps:
            v = float(comps[key])
            if not np.isfinite(v):
                raise ValueError(f"non-finite physical component {key}={v}")
            vals.append(v)
    if not vals:
        raise ValueError("no active physical teacher component")
    return float(min(vals))


def future_physical_matrix(teacher_diags: list[list[Any]], option_valid: np.ndarray | None = None) -> np.ndarray:
    if not teacher_diags or not teacher_diags[0]:
        raise ValueError("empty teacher diagnostics")
    F, L = len(teacher_diags), len(teacher_diags[0])
    if any(len(row) != L for row in teacher_diags):
        raise ValueError("ragged teacher diagnostics")
    valid = np.ones(L, dtype=bool) if option_valid is None else np.asarray(option_valid, dtype=bool).reshape(-1)
    if valid.size != L:
        raise ValueError(f"option_valid length {valid.size} != {L}")
    out = np.full((F, L), -1.0e9, dtype=np.float64)
    for f, row in enumerate(teacher_diags):
        for l, diag in enumerate(row):
            if valid[l]:
                out[f, l] = physical_margin_from_teacher_diag(diag)
    return out


def future_nested_tail_influence(
    sample: dict[str, Any],
    m_future_structural: np.ndarray,
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    intra_root_alpha: float = 0.2,
    top_m: int = 8,
) -> tuple[np.ndarray, float]:
    """Project exact nested OC-MERO influence from root-option to future-option.

    The production root margin is an intra-root lower-tail of future-option
    margins.  ``nested_tail_influence`` gives the exact influence of the final
    deployable score on root-option cells.  Multiplying by the exact intra-root
    fractional-tail influence yields the influence of each future-option cell.
    """
    mf = np.asarray(m_future_structural, dtype=np.float64)
    if mf.ndim != 2 or mf.size == 0:
        raise ValueError("m_future_structural must be [F,L]")
    probs = np.asarray(sample.get("future_probs", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(sample.get("future_valid", np.ones_like(probs)), dtype=bool).reshape(-1)
    assign = np.asarray(sample.get("root_assignments", []), dtype=np.int64).reshape(-1)
    if not (probs.size == valid.size == assign.size == mf.shape[0]):
        raise ValueError("future arrays / structural margin length mismatch")
    root_mass, _rd, _stored, _p = nested_tail_influence(sample, alpha=alpha, beta=beta, top_m=top_m)
    K, L = root_mass.shape
    if mf.shape[1] != L:
        raise ValueError("future/root option dimension mismatch")
    out = np.zeros_like(mf, dtype=np.float64)
    for k in range(K):
        idx = np.where(valid & (assign == k))[0]
        if not len(idx):
            continue
        w = normalize_weights(probs[idx])
        for l in range(L):
            rm = float(root_mass[k, l])
            if rm <= 0.0:
                continue
            local = weighted_lcvar_influence_np(mf[idx, l], w, float(intra_root_alpha))
            if local.size != idx.size:
                raise ValueError("intra-root influence length mismatch")
            out[idx, l] += rm * np.asarray(local, dtype=np.float64)
    root_sum = float(root_mass.sum())
    future_sum = float(out.sum())
    return out, abs(future_sum - root_sum)


def _valid_future_view(sample: dict[str, Any], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    probs = np.asarray(sample.get("future_probs", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(sample.get("future_valid", np.ones_like(probs)), dtype=bool).reshape(-1)
    if matrix.shape[0] != probs.size or valid.size != probs.size:
        raise ValueError("future view length mismatch")
    keys, unresolved, _dup = future_class_keys(sample, exogenous=True)
    if len(keys) != int(valid.sum()):
        # future_class_keys internally filters invalid futures.
        raise ValueError("exogenous class key count mismatch")
    return normalize_weights(probs[valid]), np.asarray(matrix, dtype=np.float64)[valid], keys, unresolved


def _class_ranges(sample: dict[str, Any], physical: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    probs, phys, keys, unresolved = _valid_future_view(sample, physical)
    table: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if bool(unresolved[i]):
            continue
        table.setdefault(key, []).append(i)
    ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    homogeneous_mass = 0.0
    for key, idx_list in table.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        vals = phys[idx]
        lo = np.full(vals.shape[1], np.nan, dtype=np.float64)
        hi = np.full(vals.shape[1], np.nan, dtype=np.float64)
        for l in range(vals.shape[1]):
            finite = vals[:, l][np.isfinite(vals[:, l])]
            if finite.size:
                lo[l] = float(np.min(finite))
                hi[l] = float(np.max(finite))
        ranges[key] = (lo, hi)
        # Physical homogeneity is descriptive; exact equality is not required for
        # response validity because the class response is conservatively bounded.
        finite_width = np.isfinite(lo) & np.isfinite(hi)
        if np.any(finite_width) and np.max(np.abs(hi[finite_width] - lo[finite_width])) <= 1.0e-6:
            homogeneous_mass += float(probs[idx].sum())
    return ranges, float(homogeneous_mass)


def audit_future_physical_response(
    candidate: dict[str, Any],
    nominal: dict[str, Any],
    candidate_structural: np.ndarray,
    nominal_structural: np.ndarray,
    candidate_physical: np.ndarray,
    nominal_physical: np.ndarray,
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    intra_root_alpha: float = 0.2,
    top_m: int = 8,
) -> FuturePhysicalResponseMetrics:
    try:
        cs = np.asarray(candidate_structural, dtype=np.float64)
        ns = np.asarray(nominal_structural, dtype=np.float64)
        cp = np.asarray(candidate_physical, dtype=np.float64)
        np_ = np.asarray(nominal_physical, dtype=np.float64)
        if cs.shape != cp.shape or ns.shape != np_.shape or cs.shape[1] != ns.shape[1]:
            raise ValueError("candidate/nominal structural/physical shape mismatch")

        cmass_full, cerr = future_nested_tail_influence(
            candidate, cs, alpha=alpha, beta=beta,
            intra_root_alpha=intra_root_alpha, top_m=top_m,
        )
        cvalid = np.asarray(candidate.get("future_valid", np.ones(cs.shape[0])), dtype=bool).reshape(-1)
        cmass = cmass_full[cvalid]
        cprobs = normalize_weights(np.asarray(candidate.get("future_probs", []), dtype=np.float64).reshape(-1)[cvalid])
        ckeys, cunres, _ = future_class_keys(candidate, exogenous=True)
        nprobs, nphys, nkeys, nunres = _valid_future_view(nominal, np_)
        cphys = cp[cvalid]
        if cmass.shape[0] != len(ckeys):
            raise ValueError("candidate influence / class key mismatch")

        cranges, chomo = _class_ranges(candidate, cp)
        nranges, nhomo = _class_ranges(nominal, np_)
        shared = (set(cranges) - {k for k,u in zip(ckeys,cunres) if bool(u)}).intersection(
            set(nranges) - {k for k,u in zip(nkeys,nunres) if bool(u)}
        )
        # Candidate/nominal class probability mass determines how much candidate
        # tail influence has genuine common-exogenous support.
        ctot: dict[str,float] = {}
        ntot: dict[str,float] = {}
        for i,k in enumerate(ckeys):
            if not bool(cunres[i]): ctot[k] = ctot.get(k,0.0) + float(cprobs[i])
        for i,k in enumerate(nkeys):
            if not bool(nunres[i]): ntot[k] = ntot.get(k,0.0) + float(nprobs[i])

        total = float(cmass.sum())
        if total <= 1.0e-12:
            raise ValueError("zero candidate future-tail influence")
        covered=informative=sign=point=pos=neg=amb=0.0
        for i,key in enumerate(ckeys):
            for l in range(cmass.shape[1]):
                m=float(cmass[i,l])
                if m<=0.0: continue
                if bool(cunres[i]) or key not in shared or ctot.get(key,0.0)<=1e-12:
                    amb += m
                    continue
                frac=min(ctot[key],ntot[key])/ctot[key]
                mm=m*max(0.0,min(1.0,float(frac)))
                uncovered=m-mm
                if uncovered>0: amb += uncovered
                if mm<=0: continue
                covered += mm
                clo,chi=cranges[key][0][l],cranges[key][1][l]
                nlo,nhi=nranges[key][0][l],nranges[key][1][l]
                rlo=float(clo-nhi); rhi=float(chi-nlo)
                informative += mm
                if abs(float(chi-clo))<=1e-6 and abs(float(nhi-nlo))<=1e-6:
                    point += mm
                if rlo>0.0:
                    sign+=mm; pos+=mm
                elif rhi<0.0:
                    sign+=mm; neg+=mm
                else:
                    amb+=mm
        return FuturePhysicalResponseMetrics(
            valid=True,error=None,
            common_exogenous_tail_coverage=float(covered/total),
            response_informative_mass=float(informative/total),
            response_sign_identifiable_mass=float(sign/total),
            response_point_identifiable_mass=float(point/total),
            response_positive_mass=float(pos/total),
            response_negative_mass=float(neg/total),
            response_ambiguous_mass=float(amb/total),
            signed_response_score=float((pos-neg)/total),
            duplicate_physical_homogeneity_mass_candidate=chomo,
            duplicate_physical_homogeneity_mass_nominal=nhomo,
            future_tail_influence_sum=total,
            future_tail_reconstruction_error=float(cerr),
        )
    except Exception as exc:
        return FuturePhysicalResponseMetrics(
            valid=False,error=str(exc),common_exogenous_tail_coverage=0.0,
            response_informative_mass=0.0,response_sign_identifiable_mass=0.0,
            response_point_identifiable_mass=0.0,response_positive_mass=0.0,
            response_negative_mass=0.0,response_ambiguous_mass=1.0,signed_response_score=0.0,
            duplicate_physical_homogeneity_mass_candidate=0.0,
            duplicate_physical_homogeneity_mass_nominal=0.0,
            future_tail_influence_sum=0.0,future_tail_reconstruction_error=float("inf"),
        )
