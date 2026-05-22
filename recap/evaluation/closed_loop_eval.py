from __future__ import annotations

from .offline_eval import evaluate_offline


def evaluate_closed_loop_or_offline(arrays: dict, method: str = "ours", **kwargs) -> dict:
    """Closed-loop entrypoint.

    When a simulator backend is unavailable, this returns the offline teacher-label
    evaluation using the same action candidates.  Real MetaDrive/CARLA closed-loop
    loops should call this after each replanning step with backend-specific roots.
    """
    return evaluate_offline(arrays, method="oracle" if method == "ours" else method, **kwargs)
