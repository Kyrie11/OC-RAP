from __future__ import annotations

from .offline_eval import evaluate_offline


def evaluate_closed_loop_or_offline(arrays: dict, method: str = "ours", allow_offline_fallback: bool = True, **kwargs) -> dict:
    """Closed-loop entrypoint with explicit offline-fallback guard.

    A real paper-final closed loop must connect a simulator backend that executes
    the first prefix segment and replans.  This repository can still run the
    same-candidate offline selector smoke test, but it is now explicitly marked
    and can be forbidden with ``allow_offline_fallback=False`` so offline numbers
    are not accidentally reported as simulator closed-loop metrics.
    """
    if not allow_offline_fallback:
        raise RuntimeError(
            "No live MetaDrive/CARLA closed-loop backend is connected in this portable package. "
            "Use offline_eval for diagnostic same-candidate metrics, or attach a simulator runner "
            "before requesting paper-final closed-loop results."
        )
    res = evaluate_offline(arrays, method=method, **kwargs)
    res["closed_loop_backend"] = "offline_same_candidate_fallback"
    res["paper_final_closed_loop"] = False
    res["oracle_selector_used_for_ours"] = False
    res["closed_loop_warning"] = "Diagnostic only: no simulator execution/replanning was performed."
    return res
