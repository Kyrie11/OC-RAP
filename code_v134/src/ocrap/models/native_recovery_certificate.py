from __future__ import annotations

import torch


def semantic_witness_physical_viability(
    semantic_witness_features: torch.Tensor,
    option_features: torch.Tensor,
    *,
    path_stop_alignment: bool,
    active_set_alignment: bool,
    control_projection: bool,
    route_alignment: bool,
    reentry_alignment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the observation-legal executable-recovery certificate.

    This is the common native certificate used by both the historical semantic
    witness diagnostic head and the frozen-module ablation bridge.  Inputs are
    option-resolved, observation-only continuation features.  The returned
    viability is the non-compensatory minimum in the tanh-bounded certificate
    coordinate; positive values are reserve and negative values are debt.

    Returns ``(physical_viability, limiting_constraint, barrier_stack)`` with
    shapes ``[B,L]``, ``[B,L]`` and ``[B,L,Q]``.
    """
    f = semantic_witness_features
    if f.ndim != 3 or f.shape[-1] < 12:
        raise RuntimeError(
            "native recovery certification requires semantic witness features "
            f"[B,L,>=12], got {tuple(f.shape)}"
        )
    of = option_features.to(device=f.device, dtype=f.dtype)
    if of.ndim != 3 or of.shape[:2] != f.shape[:2] or of.shape[-1] < 8:
        raise RuntimeError(
            "native recovery certification requires option features [B,L,>=8], "
            f"got semantic={tuple(f.shape)} option={tuple(of.shape)}"
        )

    (
        h_min,
        h_terminal,
        h_gain,
        h_stop_legacy,
        h_control,
        h_stab_min,
        h_stab_terminal,
        h_stab_gain,
        h_clear_floor_gain,
        h_stab_floor_gain,
        h_path_stop,
        stability_active_obs,
    ) = [f[..., i] for i in range(12)]
    h_route = f[..., 12] if f.shape[-1] >= 14 else torch.ones_like(h_min)
    h_reentry = f[..., 13] if f.shape[-1] >= 14 else torch.ones_like(h_min)

    clear_recovery = torch.minimum(h_terminal, h_gain)
    clear_recovery_ok = (clear_recovery > 0.0) & (h_clear_floor_gain >= 0.0)
    clearance_barrier = torch.where(clear_recovery_ok, clear_recovery, h_min)

    stab_recovery = torch.minimum(h_stab_terminal, h_stab_gain)
    stab_recovery_ok = (stab_recovery > 0.0) & (h_stab_floor_gain >= 0.0)
    raw_stability_barrier = torch.where(stab_recovery_ok, stab_recovery, h_stab_min)

    # Recovery mode one-hot occupies the first eight option coordinates.
    # Stop-like modes: stop, brake_lane, yield_rejoin, pull_over.
    stop_active = (
        (of[..., 0] > 0.5)
        | (of[..., 1] > 0.5)
        | (of[..., 3] > 0.5)
        | (of[..., 4] > 0.5)
    )
    chosen_stop = h_path_stop if path_stop_alignment else h_stop_legacy
    stop_barrier = torch.where(stop_active, chosen_stop, torch.ones_like(chosen_stop))

    if active_set_alignment:
        stability_barrier = torch.where(
            stability_active_obs > 0.5,
            raw_stability_barrier,
            torch.ones_like(raw_stability_barrier),
        )
    else:
        stability_barrier = raw_stability_barrier

    # With projection ON the rollout commands already satisfy magnitude/rate/
    # jerk envelopes by construction, so control is not vetoed a second time.
    # The knockout turns projection OFF and restores the historical post-hoc
    # controller-envelope barrier, making the factor a genuine construction
    # ablation rather than a diagnostic-side-head switch.
    effective_control_barrier = (
        torch.ones_like(h_control) if control_projection else h_control
    )

    barriers = [
        clearance_barrier,
        stop_barrier,
        effective_control_barrier,
        stability_barrier,
    ]
    if route_alignment:
        if f.shape[-1] < 14:
            raise RuntimeError("route alignment requires semantic witness schema with route coordinate")
        barriers.append(h_route)
    if reentry_alignment:
        if f.shape[-1] < 14:
            raise RuntimeError("persistent re-entry requires semantic witness schema with re-entry coordinate")
        barriers.append(h_reentry)

    barrier_stack = torch.stack(barriers, dim=-1)
    physical_viability, limiting_constraint = barrier_stack.min(dim=-1)
    return physical_viability, limiting_constraint, barrier_stack


def native_certified_root_option_margins(
    learned_margins: torch.Tensor,
    physical_viability: torch.Tensor,
    *,
    option_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservatively attach the native executable certificate before OC-MERO.

    Semantic witness coordinates are tanh(normalized signed reserve).  ``atanh``
    therefore maps the non-compensatory option certificate back into the same
    normalized signed-margin coordinate used by the learned ``M_{k,l}`` target.
    The executable observation-only certificate is an additional active
    constraint, so the root-option margin is the minimum of the learned latent
    margin and that certificate.  It can reject an over-optimistic root-option
    prediction, but can never invent positive reserve absent from the learned
    root-specific margin.
    """
    if learned_margins.ndim != 3:
        raise RuntimeError(f"learned margins must be [B,K,L], got {tuple(learned_margins.shape)}")
    if physical_viability.ndim != 2:
        raise RuntimeError(
            f"physical viability must be [B,L], got {tuple(physical_viability.shape)}"
        )
    if learned_margins.shape[0] != physical_viability.shape[0] or learned_margins.shape[2] != physical_viability.shape[1]:
        raise RuntimeError(
            "native certificate/margin shape mismatch: "
            f"margins={tuple(learned_margins.shape)} viability={tuple(physical_viability.shape)}"
        )

    # Keep a finite inverse-tanh at the exact +/-1 construction sentinels.
    eps = max(float(torch.finfo(learned_margins.dtype).eps) * 16.0, 1.0e-7)
    bounded = physical_viability.to(dtype=learned_margins.dtype).clamp(
        min=-1.0 + eps, max=1.0 - eps
    )
    certificate_margin = torch.atanh(bounded)
    certified = torch.minimum(learned_margins, certificate_margin.unsqueeze(1))

    if option_valid is not None:
        ov = option_valid.to(device=learned_margins.device, dtype=torch.bool)
        if ov.ndim == 1:
            ov = ov.unsqueeze(0).expand(learned_margins.shape[0], -1)
        if ov.shape != physical_viability.shape:
            raise RuntimeError(
                f"option_valid shape mismatch: expected {tuple(physical_viability.shape)}, got {tuple(ov.shape)}"
            )
        certified = torch.where(ov.unsqueeze(1), certified, learned_margins)

    return certified, certificate_margin
