from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from ocrap.utils.datatypes import RolloutTrace
from .margins import compute_margins

SPEC_NAMES = ["no_contact", "minimum_risk", "post_contact"]
G_DIM = 9


def option_margin_from_specs(spec_margins: np.ndarray) -> Tuple[float, int]:
    spec = np.asarray(spec_margins, dtype=np.float32).reshape(-1)
    if spec.size != 3:
        raise ValueError("spec_margins must contain [G_no, G_mr, G_post]")
    spec_id = int(np.nanargmax(spec))
    return float(spec[spec_id]), spec_id


def compute_spec_margins(trace: RolloutTrace, token=None, params: dict | None = None) -> Dict[str, object]:
    """Compute OC-RAP recovery-specification signed margins.

    Positive means the specification is satisfied.  The reported option margin is
    the largest available specification margin, never the minimum across mutually
    exclusive regimes.
    """
    m = compute_margins(trace, params)
    fc = int(trace.first_contact_idx)
    # No-contact: the full prefix+recovery path must remain contact-free and returnable.
    G_no = min(float(m.get("M_path_raw", 1.0)), float(m.get("M_ctrl", 1.0)), float(m.get("M_return", -1.0)))
    # Minimum-risk: eligible only when contact is already present/imminent, not
    # as a loose alternative for ordinary no-contact driving.
    coll = getattr(trace, "collision_margin", None)
    near_contact = False
    if coll is not None and len(coll):
        near_contact = bool(np.nanmin(np.asarray(coll)[: max(1, int(trace.stage_boundary_idx) + 2)]) < 0.25)
    mr_eligible = bool(fc >= 0 or near_contact)
    G_mr = -1.0 if not mr_eligible else min(float(m.get("M_path_pre_no_first_contact", 1.0)), float(m.get("M_ctrl", 1.0)), float(m.get("M_secondary", 1.0)), float(m.get("M_return", -1.0)))
    # Post-contact: only meaningful after contact, and uses post-contact stability.
    G_post = -1.0 if fc < 0 else min(float(m.get("M_ctrl", 1.0)), float(m.get("M_secondary", 1.0)), float(m.get("M_post", -1.0)), float(m.get("M_return", -1.0)))
    spec = np.array([G_no, G_mr, G_post], dtype=np.float32)
    margin, spec_id = option_margin_from_specs(spec)
    g_vector = np.array([
        m.get("M_path_raw", 1.0),
        m.get("M_path_rec", 1.0),
        m.get("M_path_pre_no_first_contact", 1.0),
        m.get("M_secondary", 1.0),
        m.get("M_return", -1.0),
        m.get("M_ctrl", 1.0),
        m.get("M_post", 1.0),
        G_no,
        max(G_mr, G_post),
    ], dtype=np.float32)
    return {
        **m,
        "G_no": float(G_no),
        "G_mr": float(G_mr),
        "G_post": float(G_post),
        "spec_margin_star": spec,
        "spec_id_star": spec_id,
        "margin_option": margin,
        "g_vector": g_vector,
        "y_star": float(margin >= 0.0),
        "k_post": float(m.get("K_star", 0.0)),
        # Rule/hard-shell violation exposed to CRISP should not punish pure
        # control effort that is handled by the physical feasibility shell.
        "c_rule": float(max(0.0, -float(m.get("M_path_raw", 1.0)))),
    }
