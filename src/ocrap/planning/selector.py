from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    selected_index: int
    reason: str
    admitted: np.ndarray


def crisp_select(utility: np.ndarray, r_dep: np.ndarray, hard: np.ndarray, harm: np.ndarray, feasible: np.ndarray, gamma_rec: float = 0.0, gamma_H: float = 0.0, gamma_D: float = 0.0, nominal_index: int = 0) -> SelectionResult:
    utility = np.asarray(utility, dtype=float)
    r_dep = np.asarray(r_dep, dtype=float)
    hard = np.asarray(hard, dtype=float)
    harm = np.asarray(harm, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    admitted = feasible & (r_dep >= gamma_rec) & (hard <= gamma_H) & (harm <= gamma_D)
    if 0 <= nominal_index < len(utility) and admitted[nominal_index]:
        return SelectionResult(int(nominal_index), "nominal_admitted", admitted)
    if admitted.any():
        idxs = np.where(admitted)[0]
        best = int(idxs[np.argmax(utility[idxs])])
        return SelectionResult(best, "best_admitted_utility", admitted)
    # Lexicographic fallback: minimize hard violation, then harm, then maximize r_dep, then utility.
    order = sorted(range(len(utility)), key=lambda i: (not feasible[i], hard[i], harm[i], -r_dep[i], -utility[i]))
    return SelectionResult(int(order[0]), "lexicographic_fallback", admitted)

def _as_1d_float(x: np.ndarray | None, n: int, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full((n,), float(default), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=float(default))
    return np.where(np.isfinite(arr[:n]), arr[:n], float(default))


def _split_name_set(value) -> set[str]:
    """Parse a comma/space separated macro-name config into a normalized set.

    The selector normally operates on numeric certificates.  v18 showed that
    certificate-only relative recovery can admit `perturb_nominal` and `keep` as
    if they were post-contact recovery maneuvers.  Name sets let the experiment
    explicitly define which non-nominal macro families are allowed to use the
    recovery-feasibility pool, without affecting ordinary scalar admission.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw = []
        for x in value:
            raw.extend(str(x).replace(";", ",").replace("|", ",").split(","))
    else:
        raw = str(value).replace(";", ",").replace("|", ",").replace(" ", ",").split(",")
    return {str(x).strip().lower() for x in raw if str(x).strip()}


def _as_macro_names(candidate_macro_names, n: int) -> list[str]:
    if candidate_macro_names is None:
        return [""] * int(n)
    try:
        vals = list(candidate_macro_names)
    except TypeError:
        vals = [candidate_macro_names]
    out = [str(x).strip().lower() for x in vals[:n]]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def constrained_lcb_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.0,
    nominal_slack: float = 0.0,
    nominal_slack_gap_limit: float = 0.5,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    fallback_rec_weight: float = 0.10,
    fallback_lcb_margin: float = 0.05,
    fallback_gap_margin: float = 0.25,
    nominal_fallback_lcb_slack: float = 0.05,
) -> SelectionResult:
    """Calibrated constrained selector used by OC-RAP in closed loop.

    It uses a conservative recovery lower-bound proxy
    ``r_lcb = r_dep - beta * max(0, oracle_deployable_gap)`` and adds
    explicit intervention/deviation costs so OC-RAP does not over-intervene when
    the nominal prefix is only slightly below the calibrated threshold.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))
    admitted = safe & (rec_lcb >= float(gamma_rec))
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    score = utility - float(intervention_penalty) * intervention - float(deviation_penalty) * dev + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
    if 0 <= nominal_index < n and safe[nominal_index]:
        nom_margin = rec_lcb[nominal_index] - float(gamma_rec)
        nom_gap = gap[nominal_index]
        if admitted[nominal_index]:
            return SelectionResult(int(nominal_index), "nominal_lcb_admitted", admitted)
        if nom_margin >= -float(nominal_slack) and nom_gap <= float(nominal_slack_gap_limit):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_slack_lcb_admitted", admitted)
    if admitted.any():
        cand = np.where(admitted)[0]
        return SelectionResult(int(cand[np.argmax(score[cand])]), "best_admitted_lcb_score", admitted)
    # No candidate satisfies the calibrated recovery constraint.  This branch is
    # deliberately recovery-first.  Earlier versions used a utility-dominated
    # soft fallback, which could pick high-utility prefixes with lower predicted
    # deployable recoverability and larger oracle--deployable gap.  That defeats
    # the purpose of OC-RAP exactly in low-headroom regimes.  We therefore:
    #   1. keep hard/harm safety as the first gate;
    #   2. restrict fallback candidates to the near-best recovery LCB set;
    #   3. within that set, reject candidates with much larger predicted gap;
    #   4. use utility/intervention/deviation only as a final tie-breaker.
    if safe.any():
        fallback_pool = safe.copy()
    elif feasible.any():
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        fallback_pool = near_hard & (harm <= min_harm + 1e-6)
    else:
        fallback_pool = np.ones((n,), dtype=bool)

    cand = np.where(fallback_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_lcb = float(np.max(rec_lcb[cand])) if cand.size else -np.inf

    if 0 <= nominal_index < n and fallback_pool[nominal_index]:
        nom_lcb = float(rec_lcb[nominal_index])
        nom_gap = float(gap[nominal_index])
        best_gap = float(np.min(gap[cand])) if cand.size else nom_gap
        if nom_lcb >= best_lcb - float(nominal_fallback_lcb_slack) and nom_gap <= best_gap + float(fallback_gap_margin):
            return SelectionResult(int(nominal_index), "nominal_recovery_guarded_fallback", admitted)

    near_best_lcb = fallback_pool & (rec_lcb >= best_lcb - float(fallback_lcb_margin))
    cand = np.where(near_best_lcb)[0]
    if cand.size:
        best_gap = float(np.min(gap[cand]))
        gap_guarded = near_best_lcb & (gap <= best_gap + float(fallback_gap_margin))
        if gap_guarded.any():
            cand = np.where(gap_guarded)[0]
    else:
        cand = np.where(fallback_pool)[0]

    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    fallback_score = score - float(fallback_rec_weight) * rec_shortfall - 10.0 * np.maximum(0.0, hard - float(gamma_H)) - 2.0 * np.maximum(0.0, harm - float(gamma_D))
    return SelectionResult(int(cand[np.argmax(fallback_score[cand])]) if cand.size else 0, "recovery_guarded_fallback", admitted)


def calibrated_constrained_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.10,
    shortfall_penalty: float = 1.0,
    gap_penalty: float = 0.05,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    admission_bonus: float = 0.02,
    nominal_slack: float = 0.03,
    nominal_slack_gap_limit: float = 0.50,
    nominal_utility_slack: float = 0.05,
    safe_nominal_slack: float = 0.12,
    regime_name: str | None = None,
    intervention_budget_rate: float | None = None,
    intervention_budget_used: float | None = None,
    intervention_budget_steps: float | None = None,
    intervention_budget_penalty: float = 0.25,
    prefer_admitted: bool = False,
    switch_score_margin: float = 0.0,
    safe_switch_score_margin: float = 0.10,
    safe_min_rec_lcb_gain: float = 0.05,
    safe_min_gap_reduction: float = 0.15,
    budget_preserve_nominal: bool = True,
    budget_nominal_slack: float = 0.08,
    pred_drs: np.ndarray | None = None,
    deployability_bonus: float = 0.0,
    contact_deployability_bonus: float = 0.0,
    contact_gap_penalty: float = 0.0,
    safe_hard_nominal_guard: bool = True,
    safe_nominal_max_gap: float = 0.20,
    safe_override_require_both: bool = True,
    safe_min_drs_gain: float = 0.10,
    safe_force_nominal_when_feasible: bool = False,
    safe_force_nominal_mode: str = "feasible",
    stress_preserve_nominal_min_drs_drop: float = -1.0,
    require_admitted_intervention: bool = False,
    unadmitted_fallback_to_nominal: bool = True,
    intervention_macro_allowlist=None,
    intervention_macro_blocklist=None,
    intervention_require_macro: bool = False,
    intervention_budget_hard: bool = False,
    intervention_budget_hard_min_rec_gain: float = 0.0,
    intervention_budget_hard_min_drs_gain: float = 0.0,
    intervention_budget_hard_min_gap_reduction: float = 0.0,
    intervention_min_pred_drs: float = -1.0,
    intervention_max_pred_gap: float = -1.0,
    safe_cert_min_pred_drs: float = 0.95,
    safe_cert_max_pred_gap: float = 0.20,
    safe_cert_rec_slack: float = 0.15,
    stress_nominal_anchor: bool = False,
    stress_anchor_drs_floor: float = 0.90,
    stress_anchor_max_gap: float = 0.30,
    stress_anchor_rec_slack: float = 0.10,
    stress_anchor_min_drs_gain: float = 0.06,
    stress_anchor_min_rec_gain: float = 0.08,
    stress_anchor_min_gap_reduction: float = 0.05,
    require_intervention_evidence: bool = False,
    intervention_min_rec_lcb_gain: float = 0.00,
    intervention_min_drs_gain: float = 0.00,
    intervention_min_gap_reduction: float = 0.00,
    option_drs_certificate: bool = False,
    option_drs_certificate_threshold: float = 1.01,
    option_drs_certificate_max_gap: float = -1.0,
    option_drs_certificate_rec_slack: float = 0.0,
    option_drs_certificate_min_rec_lcb: float = -1.0e9,
    option_drs_certificate_counts_as_evidence: bool = True,
    relative_recovery_certificate: bool = False,
    relative_recovery_nominal_rec_lcb_max: float = -1.0e9,
    relative_recovery_nominal_gap_min: float = 1.0e9,
    relative_recovery_nominal_drs_max: float = -1.0,
    relative_recovery_min_rec_gain: float = 0.10,
    relative_recovery_min_drs: float = 0.70,
    relative_recovery_min_drs_gain: float = -1.0,
    relative_recovery_max_drs_drop: float = -1.0,
    relative_recovery_max_rec_lcb_drop: float = -1.0,
    relative_recovery_min_improvement_axes: int = 1,
    relative_recovery_max_gap: float = -1.0,
    relative_recovery_max_gap_increase: float = 0.20,
    relative_recovery_min_gap_reduction: float = -1.0,
    relative_recovery_gate: str = "rec_gain",
    relative_recovery_use_recovery_pool: bool = False,
    recovery_cert_max_hard: float = 0.0,
    recovery_cert_max_harm: float | None = None,
    relative_recovery_bonus: float = 0.0,
    relative_recovery_counts_as_evidence: bool = True,
    candidate_macro_names=None,
    relative_recovery_macro_allowlist=None,
    relative_recovery_macro_blocklist=None,
    relative_recovery_require_macro: bool = False,
    relative_recovery_max_intervention_score_gain: float = -1.0,
    protective_macro_certificate: bool = False,
    protective_macro_allowlist=None,
    protective_macro_blocklist=None,
    protective_macro_nominal_rec_lcb_max: float = -1.0e9,
    protective_macro_nominal_gap_min: float = 1.0e9,
    protective_macro_nominal_drs_max: float = -1.0,
    protective_macro_min_drs: float = 0.60,
    protective_macro_min_rec_lcb: float = -1.0e9,
    protective_macro_min_rec_gain: float = -1.0,
    protective_macro_min_gap_reduction: float = -1.0,
    protective_macro_min_drs_gain: float = -1.0,
    protective_macro_max_drs_drop: float = 0.05,
    protective_macro_max_rec_lcb_drop: float = 0.10,
    protective_macro_max_gap: float = -1.0,
    protective_macro_max_gap_increase: float = 0.20,
    protective_macro_max_hard: float = 0.0,
    protective_macro_max_harm: float = 5.0,
    protective_macro_min_improvement_axes: int = 1,
    protective_macro_gate: str = "axes",
    protective_macro_score_min_gain: float = 0.00,
    protective_macro_score_rec_weight: float = 0.30,
    protective_macro_score_drs_weight: float = 0.55,
    protective_macro_score_gap_weight: float = 0.15,
    protective_macro_bonus: float = 0.0,
    protective_macro_counts_as_evidence: bool = True,
    # v23: contact-specific brake rescue channel. This is intentionally not a
    # generic threshold relaxation: it only applies to the physically protective
    # brake macro in contact buckets, after the semantic intervention firewall,
    # and only when nominal itself has failed calibrated admission.  The gate is
    # designed for the v22 failure mode where teacher-audited brake improves PCD
    # but the scalar rec-LCB is under-confident because braking raises the
    # predicted oracle/deployability gap.
    brake_rescue_certificate: bool = False,
    brake_rescue_macro_name: str = "brake",
    brake_rescue_min_pred_drs: float = 0.65,
    brake_rescue_min_pred_r_dep: float = -0.65,
    brake_rescue_min_candidate_gap: float = 0.12,
    brake_rescue_max_candidate_gap: float = 0.34,
    brake_rescue_max_hard: float = 1.0,
    brake_rescue_max_harm: float = 0.70,
    brake_rescue_require_nominal_unadmitted: bool = True,
    brake_rescue_nominal_rec_lcb_max: float = 0.60,
    brake_rescue_nominal_gap_min: float = 0.02,
    brake_rescue_nominal_drs_max: float = 1.01,
    brake_rescue_counts_as_evidence: bool = True,
) -> SelectionResult:
    """Soft calibrated OC-RAP selector.

    ``constrained_lcb_select`` is deliberately hard-gated: candidates below the
    calibrated recovery threshold all fall into a recovery-first fallback.  That
    is useful for stress tests, but in natural/safe scenes it can over-intervene
    because a candidate that misses gamma by a tiny amount is treated like one
    that misses it by a large amount.  This selector keeps hard/harm feasibility
    as a constraint, then turns recovery-threshold shortfall into a continuous
    penalty.  It also has an optional intervention budget and a stronger
    nominal-preservation rule for safe/background buckets.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    if n <= 0:
        return SelectionResult(0, "empty_calibrated_selector", np.zeros((0,), dtype=bool))
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    drs_proxy = np.clip(_as_1d_float(pred_drs, n, default=0.0), 0.0, 1.0)
    macro_names = _as_macro_names(candidate_macro_names, n)

    # v22: separate candidate-family coverage from recovery-maneuver semantics.
    # Scalar/option certificates are purely numerical, so without a global
    # intervention macro gate they can still execute perturb_nominal, lane_shift,
    # or other exploration families as if they were paper-valid recoveries.
    # Nominal is always allowed; non-nominal interventions must satisfy this
    # semantic mask whenever configured.
    intervention_macro_allow = _split_name_set(intervention_macro_allowlist)
    intervention_macro_block = _split_name_set(intervention_macro_blocklist)
    intervention_macro_mask = np.ones((n,), dtype=bool)
    if intervention_macro_allow:
        intervention_macro_mask &= np.asarray([m in intervention_macro_allow for m in macro_names], dtype=bool)
    if intervention_macro_block:
        intervention_macro_mask &= ~np.asarray([m in intervention_macro_block for m in macro_names], dtype=bool)
    if bool(intervention_require_macro):
        intervention_macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)
    if 0 <= int(nominal_index) < n:
        intervention_macro_mask[int(nominal_index)] = True

    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))

    # CRISP's nominal admission uses hard-rule/harm feasibility.  In near-contact
    # and post-contact regimes, however, the *recovery* prefix may deliberately
    # violate a nominal lane/comfort rule while reducing deployable recovery loss
    # (e.g. stabilize after contact, pull over, or brake/yield into a low-utility
    # state).  Keep scalar admission strict, but optionally allow relative
    # recovery certificates to use a regime-specific recovery-feasibility pool.
    recovery_hard_limit = float(gamma_H)
    if bool(relative_recovery_use_recovery_pool):
        recovery_hard_limit = max(float(gamma_H), float(recovery_cert_max_hard))
    recovery_harm_limit = float(gamma_D) if recovery_cert_max_harm is None else float(recovery_cert_max_harm)
    recovery_safe = feasible & (hard <= recovery_hard_limit) & (harm <= recovery_harm_limit)

    scalar_admitted = safe & (rec_lcb >= float(gamma_rec))

    # v15 dual certificate.  The scalar OC-MERO LCB can be overly pessimistic
    # under the closed-loop distribution shift observed in v14, while the
    # observation-consistent shared-option head can still identify a deployable
    # recovery option.  A non-nominal prefix may therefore be certified either by
    # the calibrated scalar recovery LCB, or by a directly predicted shared
    # recovery-option success certificate with gap/LCB guards.  This keeps the
    # paper claim intact: interventions still require an observation-consistent
    # deployability certificate; they are no longer blocked solely because the
    # aggregate scalar head is conservative.
    option_certified = np.zeros((n,), dtype=bool)
    if bool(option_drs_certificate):
        option_certified = safe & (drs_proxy >= float(option_drs_certificate_threshold))
        if float(option_drs_certificate_max_gap) >= 0.0:
            option_certified &= gap <= float(option_drs_certificate_max_gap)
        option_certified &= rec_lcb >= float(gamma_rec) - float(option_drs_certificate_rec_slack)
        option_certified &= rec_lcb >= float(option_drs_certificate_min_rec_lcb)

    # v16 relative-recovery certificate.  The v15 result showed that a single
    # absolute scalar/DRS threshold can still abstain throughout contact scenes
    # where nominal is clearly low-headroom but one of the current candidates is
    # relatively better.  This certificate is deliberately counterfactual: it is
    # enabled only in a nominal-opportunity state, and it admits non-nominal
    # prefixes that improve predicted deployable recovery over nominal while
    # satisfying DRS and gap guards.  This keeps the method certifiable without
    # returning to the v13 soft fallback.
    relative_certified = np.zeros((n,), dtype=bool)
    protective_certified = np.zeros((n,), dtype=bool)
    brake_rescue_certified = np.zeros((n,), dtype=bool)
    rec_gain_vs_nom = np.zeros((n,), dtype=float)
    drs_gain_vs_nom = np.zeros((n,), dtype=float)
    gap_reduction_vs_nom = np.zeros((n,), dtype=float)
    if 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        rec_gain_vs_nom = rec_lcb - rec_lcb[ni]
        drs_gain_vs_nom = drs_proxy - drs_proxy[ni]
        gap_reduction_vs_nom = gap[ni] - gap

    if bool(relative_recovery_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        nominal_opportunity = (
            rec_lcb[ni] <= float(relative_recovery_nominal_rec_lcb_max)
            or gap[ni] >= float(relative_recovery_nominal_gap_min)
            or (float(relative_recovery_nominal_drs_max) >= 0.0 and drs_proxy[ni] <= float(relative_recovery_nominal_drs_max))
        )
        if bool(nominal_opportunity):
            # v17: use a dominance-style relative certificate instead of only an
            # absolute rec-LCB gain.  The v16 audits showed that nominal can be
            # falsely overconfident in contact scenes: recoverable alternatives
            # exist, but many are blocked because the scalar head is conservative
            # or because the maneuver has a nominal hard-rule flag.  A relative
            # recovery certificate is valid when it is observation-consistent,
            # has enough predicted shared-option success, does not create an
            # excessive oracle--deployability gap, and improves at least one
            # deployability axis over nominal (recovery LCB, DRS proxy, or gap).
            base_pool = recovery_safe if bool(relative_recovery_use_recovery_pool) else safe
            rec_ok = rec_gain_vs_nom >= float(relative_recovery_min_rec_gain)
            drs_gain_ok = np.zeros((n,), dtype=bool)
            if float(relative_recovery_min_drs_gain) >= 0.0:
                drs_gain_ok = drs_gain_vs_nom >= float(relative_recovery_min_drs_gain)
            gap_ok_gain = np.zeros((n,), dtype=bool)
            if float(relative_recovery_min_gap_reduction) >= 0.0:
                gap_ok_gain = gap_reduction_vs_nom >= float(relative_recovery_min_gap_reduction)
            gate = str(relative_recovery_gate or "rec_gain").strip().lower()
            # v18: Pareto lower-envelope certificate.  v17's rec_or_gap gate
            # improved contact audits, but it could certify a recovery prefix that
            # merely reduced predicted gap while lowering the shared deployability
            # proxy.  That created offline false admissions and one bad contact
            # pull-over.  The certificate below separates *which axis shows
            # opportunity* from *which axes must not deteriorate*: a candidate may
            # be admitted by recovery, DRS, or gap improvement, but configurable
            # non-inferiority guards keep it on the deployability Pareto frontier
            # relative to nominal.
            improve_count = rec_ok.astype(int) + drs_gain_ok.astype(int) + gap_ok_gain.astype(int)
            if gate in {"any", "any_gain", "dominance", "or"}:
                improvement_ok = improve_count >= max(1, int(relative_recovery_min_improvement_axes))
            elif gate in {"two", "two_of_three", "2of3"}:
                improvement_ok = improve_count >= max(2, int(relative_recovery_min_improvement_axes))
            elif gate in {"rec_or_gap", "headroom_or_gap"}:
                improvement_ok = rec_ok | gap_ok_gain
            elif gate in {"drs", "drs_gain", "shared_option"}:
                improvement_ok = drs_gain_ok
            elif gate in {"drs_or_gap", "shared_or_gap"}:
                improvement_ok = drs_gain_ok | gap_ok_gain
            elif gate in {"pareto", "pareto_lcb", "pareto_frontier"}:
                improvement_ok = improve_count >= max(1, int(relative_recovery_min_improvement_axes))
            else:
                improvement_ok = rec_ok
            macro_allow = _split_name_set(relative_recovery_macro_allowlist)
            macro_block = _split_name_set(relative_recovery_macro_blocklist)
            macro_mask = np.ones((n,), dtype=bool)
            if macro_allow:
                macro_mask &= np.asarray([m in macro_allow for m in macro_names], dtype=bool)
            if macro_block:
                macro_mask &= ~np.asarray([m in macro_block for m in macro_names], dtype=bool)
            if bool(relative_recovery_require_macro):
                macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)

            relative_certified = base_pool & improvement_ok & macro_mask
            relative_certified &= drs_proxy >= float(relative_recovery_min_drs)
            # DRS non-inferiority can now be enforced for every gate.  Setting the
            # gain threshold negative preserves the old behavior.
            if float(relative_recovery_min_drs_gain) >= 0.0:
                relative_certified &= drs_gain_vs_nom >= float(relative_recovery_min_drs_gain)
            if float(relative_recovery_max_drs_drop) >= 0.0:
                relative_certified &= drs_gain_vs_nom >= -float(relative_recovery_max_drs_drop)
            if float(relative_recovery_max_rec_lcb_drop) >= 0.0:
                relative_certified &= rec_gain_vs_nom >= -float(relative_recovery_max_rec_lcb_drop)
            if float(relative_recovery_max_gap) >= 0.0:
                relative_certified &= gap <= float(relative_recovery_max_gap)
            if float(relative_recovery_max_gap_increase) >= 0.0:
                relative_certified &= gap <= gap[ni] + float(relative_recovery_max_gap_increase)
            # v19: relative recovery is not an aggressive trigger.  When enabled,
            # reject candidates whose final learned score advantage is so large
            # that the decision is effectively a utility/deployability bonus rather
            # than a recovery-certificate decision.  The score itself is computed
            # below, so this guard is applied later after score construction.
            relative_certified[ni] = False

    # v20 protective-macro certificate.  v19 correctly blocked nominal-like
    # perturbations, but it also became too conservative and kept missing contact
    # cases whose audited best candidate was a brake/stabilize maneuver.  This
    # certificate is a second, narrower channel: only explicitly protective macro
    # families may use it, and they must be in a low-headroom nominal state while
    # satisfying non-inferiority guards.  It does not certify generic lane/utility
    # changes and therefore keeps the paper claim that interventions are semantic
    # recovery maneuvers, not aggressive trigger actions.
    if bool(protective_macro_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        protective_opportunity = (
            rec_lcb[ni] <= float(protective_macro_nominal_rec_lcb_max)
            or gap[ni] >= float(protective_macro_nominal_gap_min)
            or (float(protective_macro_nominal_drs_max) >= 0.0 and drs_proxy[ni] <= float(protective_macro_nominal_drs_max))
        )
        if bool(protective_opportunity):
            macro_allow = _split_name_set(protective_macro_allowlist)
            macro_block = _split_name_set(protective_macro_blocklist)
            macro_mask = np.ones((n,), dtype=bool)
            if macro_allow:
                macro_mask &= np.asarray([m in macro_allow for m in macro_names], dtype=bool)
            if macro_block:
                macro_mask &= ~np.asarray([m in macro_block for m in macro_names], dtype=bool)
            # An empty macro name should never pass this semantic certificate.
            macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)

            protective_pool = feasible & (hard <= float(protective_macro_max_hard)) & (harm <= float(protective_macro_max_harm))
            rec_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_rec_gain) >= 0.0:
                rec_ok = rec_gain_vs_nom >= float(protective_macro_min_rec_gain)
            drs_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_drs_gain) >= 0.0:
                drs_ok = drs_gain_vs_nom >= float(protective_macro_min_drs_gain)
            gap_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_gap_reduction) >= 0.0:
                gap_ok = gap_reduction_vs_nom >= float(protective_macro_min_gap_reduction)
            improve_count = rec_ok.astype(int) + drs_ok.astype(int) + gap_ok.astype(int)
            # v21: support a calibrated deployability-vector gate for protective
            # macros.  Brake/stabilize can legitimately trade geometric gap for
            # shared-option robustness immediately after contact, so requiring a
            # pure gap reduction (v20) can suppress exactly the macro we want to
            # test.  The score gate still requires semantic macro eligibility,
            # hard/harm feasibility, absolute DRS, and non-inferiority guards.
            protective_score_gain = (
                float(protective_macro_score_rec_weight) * rec_gain_vs_nom
                + float(protective_macro_score_drs_weight) * drs_gain_vs_nom
                + float(protective_macro_score_gap_weight) * gap_reduction_vs_nom
            )
            gate = str(protective_macro_gate or "axes").lower()
            if gate == "score":
                improvement_ok = protective_score_gain >= float(protective_macro_score_min_gain)
            elif gate == "drs_or_score":
                improvement_ok = drs_ok | (protective_score_gain >= float(protective_macro_score_min_gain))
            elif gate == "any":
                improvement_ok = rec_ok | drs_ok | gap_ok
            else:
                improvement_ok = improve_count >= max(1, int(protective_macro_min_improvement_axes))

            protective_certified = protective_pool & macro_mask & improvement_ok
            protective_certified &= drs_proxy >= float(protective_macro_min_drs)
            protective_certified &= rec_lcb >= float(protective_macro_min_rec_lcb)
            if float(protective_macro_max_drs_drop) >= 0.0:
                protective_certified &= drs_gain_vs_nom >= -float(protective_macro_max_drs_drop)
            if float(protective_macro_max_rec_lcb_drop) >= 0.0:
                protective_certified &= rec_gain_vs_nom >= -float(protective_macro_max_rec_lcb_drop)
            if float(protective_macro_max_gap) >= 0.0:
                protective_certified &= gap <= float(protective_macro_max_gap)
            if float(protective_macro_max_gap_increase) >= 0.0:
                protective_certified &= gap <= gap[ni] + float(protective_macro_max_gap_increase)
            protective_certified[ni] = False

    # v23 brake rescue certificate.  Unlike v21/v22 protective_macro_score, this
    # does not trust the scalar recovery LCB to rank contact braking.  It uses a
    # macro-specific physical prior plus calibrated shared-option evidence: a
    # brake rescue is admissible only in contact-like low-headroom states, only
    # if nominal is not already calibrated-admitted, and only when the brake
    # candidate lies in a moderate uncertainty band (large enough gap to indicate
    # recovery ambiguity, not so large that it is an unbounded artifact).
    if bool(brake_rescue_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        brake_name = str(brake_rescue_macro_name or "brake").strip().lower()
        macro_arr = np.asarray([str(m).strip().lower() for m in macro_names], dtype=object)
        nominal_gate = (
            rec_lcb[ni] <= float(brake_rescue_nominal_rec_lcb_max)
            or gap[ni] >= float(brake_rescue_nominal_gap_min)
            or drs_proxy[ni] <= float(brake_rescue_nominal_drs_max)
        )
        if bool(brake_rescue_require_nominal_unadmitted):
            nominal_gate = nominal_gate and (not bool(scalar_admitted[ni] or option_certified[ni] or relative_certified[ni] or protective_certified[ni]))
        brake_rescue_certified = (
            feasible
            & (hard <= float(brake_rescue_max_hard))
            & (harm <= float(brake_rescue_max_harm))
            & (macro_arr == brake_name)
            & (drs_proxy >= float(brake_rescue_min_pred_drs))
            & (r_dep >= float(brake_rescue_min_pred_r_dep))
            & (gap >= float(brake_rescue_min_candidate_gap))
            & (gap <= float(brake_rescue_max_candidate_gap))
            & bool(nominal_gate)
        )
        # Keep the global semantic firewall as the final arbiter.
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            brake_rescue_certified &= intervention_macro_mask

    admitted = scalar_admitted | option_certified | relative_certified | protective_certified | brake_rescue_certified
    if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
        admitted = admitted & intervention_macro_mask
        if 0 <= int(nominal_index) < n and (scalar_admitted[int(nominal_index)] or safe[int(nominal_index)]):
            admitted[int(nominal_index)] = True
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    regime = (regime_name or "").lower()
    is_safe_regime = "safe" in regime or "normal" in regime or "background" in regime
    is_contact_regime = "contact" in regime

    # Continuous recovery shortfall; finite guard prevents NaNs from dominating.
    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    rec_shortfall = np.where(np.isfinite(rec_shortfall), rec_shortfall, 1.0e6)
    score = (
        utility
        - float(shortfall_penalty) * rec_shortfall
        - float(gap_penalty) * gap
        - float(intervention_penalty) * intervention
        - float(deviation_penalty) * dev
        + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
        + float(admission_bonus) * admitted.astype(float)
        + float(deployability_bonus) * drs_proxy
    )
    if bool(relative_recovery_certificate) and float(relative_recovery_bonus) != 0.0:
        rel_adv = np.maximum(0.0, rec_gain_vs_nom) + 0.25 * np.maximum(0.0, drs_gain_vs_nom) + 0.10 * np.maximum(0.0, gap_reduction_vs_nom)
        score = score + float(relative_recovery_bonus) * rel_adv * relative_certified.astype(float)
    if bool(protective_macro_certificate) and float(protective_macro_bonus) != 0.0:
        prot_adv = (
            float(protective_macro_score_rec_weight) * np.maximum(0.0, rec_gain_vs_nom)
            + float(protective_macro_score_drs_weight) * np.maximum(0.0, drs_gain_vs_nom)
            + float(protective_macro_score_gap_weight) * np.maximum(0.0, gap_reduction_vs_nom)
        )
        score = score + float(protective_macro_bonus) * prot_adv * protective_certified.astype(float)
    if is_contact_regime:
        score = score + float(contact_deployability_bonus) * drs_proxy - float(contact_gap_penalty) * gap

    if bool(relative_recovery_certificate) and float(relative_recovery_max_intervention_score_gain) >= 0.0 and 0 <= int(nominal_index) < n:
        # Cap the admission frontier, not the final ranking pool.  A candidate that
        # only looks attractive because of a large learned-score bonus should not
        # bypass the nominal-preserving abstention rule.
        relative_certified &= (score - score[int(nominal_index)]) <= float(relative_recovery_max_intervention_score_gain)
        admitted = scalar_admitted | option_certified | relative_certified | protective_certified | brake_rescue_certified
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            admitted = admitted & intervention_macro_mask
            if 0 <= int(nominal_index) < n and (scalar_admitted[int(nominal_index)] or safe[int(nominal_index)]):
                admitted[int(nominal_index)] = True

    # If the current rollout already exceeded the intervention budget, increase
    # the marginal cost of deviating from nominal.  This is intentionally a soft
    # budget because emergency recovery should still be selectable.
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            used = float(intervention_budget_used or 0.0)
            steps = max(1.0, float(intervention_budget_steps or 1.0))
            target = float(intervention_budget_rate)
            over = max(0.0, (used / steps) - target)
            if over > 0.0:
                score = score - float(intervention_budget_penalty) * over * intervention
        except Exception:
            pass

    if safe.any():
        pool = safe.copy()
    elif feasible.any():
        # Preserve the original hard/harm safety priority when no fully safe
        # candidate exists.
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1.0e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        pool = near_hard & (harm <= min_harm + 1.0e-6)
    else:
        pool = np.ones((n,), dtype=bool)

    # Safe/NUP gate.  In benign regimes the paper claim is nominal utility
    # preservation, not recovery maximization.  The learned recovery heads can be
    # pessimistic on safe_v2 because the stress regimes dominate negative
    # deployability.  Therefore, when the nominal prefix is dynamically feasible
    # and the active regime is safe/background, allow an explicit hard lock that
    # bypasses noisy recovery-head shortfall.
    if bool(safe_force_nominal_when_feasible) and is_safe_regime and 0 <= nominal_index < n:
        # In safe/background regimes, nominal preservation should be a certifiable
        # behavior, not just an unconditional heuristic.  `always` keeps the old
        # hard lock for ablation/papercheck; `certified` locks nominal only when
        # the learned heads and feasibility metadata agree that nominal is already
        # deployable and low-gap.
        force_mode = str(safe_force_nominal_mode or "feasible").strip().lower()
        certified = (
            pool[nominal_index]
            and drs_proxy[nominal_index] >= float(safe_cert_min_pred_drs)
            and gap[nominal_index] <= float(safe_cert_max_pred_gap)
            and rec_lcb[nominal_index] >= float(gamma_rec) - float(safe_cert_rec_slack)
        )
        allow_force = (
            force_mode in {"always", "present", "nominal", "hard"}
            or (pool[nominal_index] and force_mode in {"", "feasible", "safe"})
            or (certified and force_mode in {"cert", "certified", "learned", "model", "guarded"})
        )
        if allow_force:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            reason = "nominal_safe_certified_locked" if force_mode in {"cert", "certified", "learned", "model", "guarded"} else "nominal_safe_force_locked"
            return SelectionResult(int(nominal_index), reason, admitted)

    # Stress-regime option: when calibrated-admitted candidates exist, rank only
    # within them.  The default remains soft-constrained for backward
    # compatibility, but near-contact/contact experiments should set
    # prefer_admitted_by_bucket=true so DRS is not sacrificed to high utility.
    rank_pool = pool & admitted if bool(prefer_admitted) and bool((pool & admitted).any()) else pool

    cand = np.where(rank_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_idx = int(cand[np.argmax(score[cand])])

    # Admitted-intervention abstention.  OC-RAP's claim is deployable recovery,
    # so executing a recovery prefix that the calibrated selector itself did not
    # admit is hard to defend in paper experiments.  When enabled, intervention
    # is allowed only if a candidate passes calibrated admission and optional
    # predicted shared-action DRS/gap guards.  If none is certified, preserve
    # nominal rather than taking a high-utility but uncertified recovery action.
    if bool(require_admitted_intervention):
        certified_intervention = admitted.copy()
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            certified_intervention &= intervention_macro_mask
        if float(intervention_min_pred_drs) >= 0.0:
            certified_intervention &= drs_proxy >= float(intervention_min_pred_drs)
        if float(intervention_max_pred_gap) >= 0.0:
            certified_intervention &= gap <= float(intervention_max_pred_gap)
        # For relative recovery in post-contact/near-contact regimes, a certified
        # recovery maneuver may lie outside the nominal hard-rule pool (e.g. a
        # stabilization/pull-over action).  Do not discard such a candidate after
        # certifying it via the explicit recovery pool above.
        certified_intervention &= (pool | relative_certified | protective_certified | brake_rescue_certified)
        certified_non_nom = certified_intervention.copy()
        if 0 <= nominal_index < n:
            certified_non_nom[int(nominal_index)] = False

        # v14: certified intervention must not merely pass an absolute
        # threshold; it should have some predicted evidence that switching away
        # from nominal improves deployable recoverability or closes the
        # oracle--deployability gap.  This keeps OC-RAP from reducing FRA/ODG by
        # taking low-DRS recovery prefixes and turns abstention into an explicit
        # learned/certified decision.
        if bool(require_intervention_evidence) and 0 <= nominal_index < n:
            rec_gain = rec_lcb - rec_lcb[int(nominal_index)]
            drs_gain = drs_proxy - drs_proxy[int(nominal_index)]
            gap_reduction = gap[int(nominal_index)] - gap
            evidence = (
                (rec_gain >= float(intervention_min_rec_lcb_gain))
                | (drs_gain >= float(intervention_min_drs_gain))
                | (gap_reduction >= float(intervention_min_gap_reduction))
            )
            if bool(option_drs_certificate_counts_as_evidence):
                evidence = evidence | option_certified
            if bool(relative_recovery_counts_as_evidence):
                evidence = evidence | relative_certified
            if bool(protective_macro_counts_as_evidence):
                evidence = evidence | protective_certified
            if bool(brake_rescue_counts_as_evidence):
                evidence = evidence | brake_rescue_certified
            certified_non_nom &= evidence

        if bool(intervention_budget_hard) and intervention_budget_rate is not None and intervention_budget_steps not in {None, 0} and 0 <= nominal_index < n:
            try:
                used = float(intervention_budget_used or 0.0)
                steps = max(1.0, float(intervention_budget_steps or 1.0))
                budget_exceeded_now = (used / steps) >= float(intervention_budget_rate)
            except Exception:
                budget_exceeded_now = False
            if budget_exceeded_now:
                rec_gain = rec_lcb - rec_lcb[int(nominal_index)]
                drs_gain = drs_proxy - drs_proxy[int(nominal_index)]
                gap_reduction = gap[int(nominal_index)] - gap
                hard_budget_evidence = (
                    (rec_gain >= float(intervention_budget_hard_min_rec_gain))
                    | (drs_gain >= float(intervention_budget_hard_min_drs_gain))
                    | (gap_reduction >= float(intervention_budget_hard_min_gap_reduction))
                    | protective_certified
                    | relative_certified
                    | brake_rescue_certified
                )
                certified_non_nom &= hard_budget_evidence

        if bool(certified_non_nom.any()):
            cc = np.where(certified_non_nom)[0]
            best_idx = int(cc[np.argmax(score[cc])])
        elif 0 <= nominal_index < n and bool(unadmitted_fallback_to_nominal):
            # Important: do not require nominal to be in the temporary hard/harm
            # pool.  In closed-loop feature-only runs the nominal metadata can
            # be noisy; executing an uncertified recovery is harder to defend
            # than abstaining to nominal.  This fixes the v13 failure where
            # require_admitted_intervention=true still produced
            # best_calibrated_soft_constraint interventions.
            admitted = admitted.copy()
            admitted[int(nominal_index)] = True
            return SelectionResult(int(nominal_index), "nominal_no_certified_intervention_preserved", admitted)
        elif not admitted[best_idx]:
            admitted_pool = np.where(pool & admitted)[0]
            if admitted_pool.size > 0:
                best_idx = int(admitted_pool[np.argmax(score[admitted_pool])])
            elif 0 <= nominal_index < n:
                admitted = admitted.copy()
                admitted[int(nominal_index)] = True
                return SelectionResult(int(nominal_index), "nominal_no_certified_intervention_preserved", admitted)

    # Stress DRS guard.  A recurring failure mode in the current results is that
    # learned OC-RAP lowers FRA/ODG but sacrifices executed DRS.  When configured,
    # do not switch away from a feasible nominal prefix to a candidate whose
    # predicted shared-option success is materially worse than nominal.  Keep the
    # default disabled because early recovery may deliberately trade nominal
    # behavior for safety.
    if (not is_safe_regime) and float(stress_preserve_nominal_min_drs_drop) >= 0.0 and 0 <= nominal_index < n and pool[nominal_index]:
        if drs_proxy[best_idx] + float(stress_preserve_nominal_min_drs_drop) < drs_proxy[nominal_index]:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_stress_drs_guard", admitted)

    # Stress-regime nominal anchor.  Near-contact scenes are still benign until
    # impact, and contact scenes often have a high-quality nominal recovery
    # baseline.  Avoid switching away from a predicted-good nominal prefix unless
    # the learned candidate brings a material deployability/gap advantage.  This
    # is intentionally optional and should be reported as a cautious-selector
    # ablation, not hidden as a universal rule.
    if bool(stress_nominal_anchor) and (not is_safe_regime) and 0 <= nominal_index < n and pool[nominal_index]:
        nominal_anchor_ok = (
            drs_proxy[nominal_index] >= float(stress_anchor_drs_floor)
            and gap[nominal_index] <= float(stress_anchor_max_gap)
            and rec_lcb[nominal_index] >= float(gamma_rec) - float(stress_anchor_rec_slack)
        )
        if nominal_anchor_ok and best_idx != int(nominal_index):
            drs_gain = float(drs_proxy[best_idx] - drs_proxy[nominal_index])
            rec_gain = float(rec_lcb[best_idx] - rec_lcb[nominal_index])
            gap_reduction = float(gap[nominal_index] - gap[best_idx])
            if (
                drs_gain < float(stress_anchor_min_drs_gain)
                and rec_gain < float(stress_anchor_min_rec_gain)
                and gap_reduction < float(stress_anchor_min_gap_reduction)
            ):
                admitted = admitted.copy()
                admitted[nominal_index] = True
                return SelectionResult(int(nominal_index), "nominal_stress_anchor_preserved", admitted)

    # Nominal-preserving calibration.  In safe/background regimes, recovery
    # shortfall by itself should not trigger large behavior changes unless the
    # alternative is clearly better in calibrated score and materially improves
    # deployable-recovery LCB or oracle-deployable gap.
    nominal_extra_slack = float(safe_nominal_slack if is_safe_regime else nominal_slack)
    budget_exceeded = False
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            budget_exceeded = (float(intervention_budget_used or 0.0) / max(1.0, float(intervention_budget_steps or 1.0))) >= float(intervention_budget_rate)
        except Exception:
            budget_exceeded = False

    if 0 <= nominal_index < n and pool[nominal_index]:
        nom_gap_ok = gap[nominal_index] <= float(nominal_slack_gap_limit)
        nom_near_gamma = rec_lcb[nominal_index] >= float(gamma_rec) - nominal_extra_slack
        nom_near_score = score[nominal_index] >= score[best_idx] - float(nominal_utility_slack)
        if admitted[nominal_index]:
            return SelectionResult(int(nominal_index), "nominal_calibrated_admitted", admitted)
        if bool(budget_preserve_nominal) and budget_exceeded and nom_gap_ok and rec_lcb[nominal_index] >= float(gamma_rec) - float(budget_nominal_slack):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_budget_preserved", admitted)
        if is_safe_regime and nom_gap_ok and nom_near_gamma:
            score_gain = float(score[best_idx] - score[nominal_index])
            rec_gain = float(rec_lcb[best_idx] - rec_lcb[nominal_index])
            gap_reduction = float(gap[nominal_index] - gap[best_idx])
            drs_gain = float(drs_proxy[best_idx] - drs_proxy[nominal_index])
            material_recovery_gain = rec_gain >= float(safe_min_rec_lcb_gain)
            material_gap_gain = gap_reduction >= float(safe_min_gap_reduction)
            material_drs_gain = drs_gain >= float(safe_min_drs_gain)
            if bool(safe_hard_nominal_guard) and gap[nominal_index] <= float(safe_nominal_max_gap):
                # Safe_v2 is the nominal-preservation regime.  Do not switch for
                # a single noisy headroom advantage; require a score improvement
                # plus consistent recovery/gap evidence.
                enough_evidence = (material_recovery_gain and material_gap_gain) if bool(safe_override_require_both) else (material_recovery_gain or material_gap_gain or material_drs_gain)
                if (not enough_evidence) or score_gain < float(safe_switch_score_margin):
                    admitted = admitted.copy()
                    admitted[nominal_index] = True
                    return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
            # Safe/background is a nominal-preserving operating region.  Switch
            # away from nominal only when the alternative is not just higher
            # scoring, but meaningfully improves calibrated recoverability or
            # reduces oracle-deployable gap.
            if (not material_recovery_gain and not material_gap_gain and not material_drs_gain) or score_gain < float(safe_switch_score_margin):
                admitted = admitted.copy()
                admitted[nominal_index] = True
                return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
        if nom_gap_ok and nom_near_gamma and nom_near_score:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_calibrated_preserved", admitted)
        if (not is_safe_regime) and float(score[best_idx] - score[nominal_index]) < float(switch_score_margin) and nom_gap_ok and nom_near_gamma:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_switch_margin_preserved", admitted)

    reason = "best_calibrated_admitted_score" if admitted[best_idx] else "best_calibrated_soft_constraint"
    if admitted[best_idx] and brake_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
        reason = "best_brake_rescue_certified_score"
    elif admitted[best_idx] and protective_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx]:
        reason = "best_protective_macro_recovery_certified_score"
    elif admitted[best_idx] and relative_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx]:
        reason = "best_relative_recovery_certified_score"
    elif admitted[best_idx] and option_certified[best_idx] and not scalar_admitted[best_idx]:
        reason = "best_option_drs_certified_score"
    admitted_rank_pool = pool | relative_certified | protective_certified | brake_rescue_certified
    if bool(prefer_admitted) and bool((admitted_rank_pool & admitted).any()) and admitted[best_idx]:
        if brake_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
            reason = "best_brake_rescue_certified_prefer_admitted"
        elif protective_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx]:
            reason = "best_protective_macro_recovery_certified_prefer_admitted"
        elif relative_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx]:
            reason = "best_relative_recovery_certified_prefer_admitted"
        elif option_certified[best_idx] and not scalar_admitted[best_idx]:
            reason = "best_option_drs_certified_prefer_admitted"
        else:
            reason = "best_calibrated_prefer_admitted"
    return SelectionResult(best_idx, reason, admitted)

