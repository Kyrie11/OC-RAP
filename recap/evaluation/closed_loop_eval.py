from __future__ import annotations

from .offline_eval import evaluate_offline


def evaluate_closed_loop_or_offline(arrays: dict, method: str = "ours", **kwargs) -> dict:
    """Evaluate a closed-loop run or fall back to same-candidate offline replay.

    The fallback intentionally keeps `method="ours"` as OC-RAP/CRISP.  Older code
    replaced it with oracle teacher selection, which made closed-loop smoke tests
    non-deployable and inflated recovery metrics.
    """
    res = evaluate_offline(arrays, method=method, **kwargs)
    res["closed_loop_backend"] = "offline_same_candidate_fallback"
    res["oracle_selector_used_for_ours"] = False
    return res
