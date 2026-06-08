from __future__ import annotations

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, SceneHistory

from .priors import normalize_future_priors
from .reactive import reactive_future, ReactivePolicy
from .replay import replay_future
from .targeted import TARGETED_KINDS, mine_artifact_futures, targeted_perturbation


def generate_counterfactual_futures(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> list[CounterfactualFuture]:
    total_steps = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10))))
    futures: list[CounterfactualFuture] = []
    priors = cfg.get("future_priors", {})
    replay_prior = float(priors.get("replay", 0.25))
    reactive_total = float(priors.get("reactive", 0.35))
    targeted_total = float(priors.get("targeted", 0.40))
    futures.append(replay_future(history, prefix, total_steps, replay_prior))
    n_reactive = int(cfg.get("num_reactive_futures", 4))
    for i in range(n_reactive):
        futures.append(reactive_future(history, prefix, total_steps, len(futures), reactive_total / max(n_reactive, 1), cfg))
    n_targeted = int(cfg.get("num_targeted_futures", 8))
    targeted_added = 0
    mined = mine_artifact_futures(history, prefix, total_steps, len(futures), targeted_total / max(n_targeted, 1), cfg) if cfg.get("artifact", {}).get("force_mine", True) else []
    for f in mined:
        futures.append(f)
        targeted_added += 1
    kind_cursor = 0
    while targeted_added < n_targeted:
        kind = TARGETED_KINDS[kind_cursor % len(TARGETED_KINDS)]
        kind_cursor += 1
        # Avoid duplicating the mined pair too heavily; stress futures create non-degenerate observations.
        if mined and kind in {"hidden_vehicle_yields", "hidden_vehicle_accelerates"}:
            continue
        fut = targeted_perturbation(history, prefix, total_steps, kind, len(futures), targeted_total / max(n_targeted, 1), cfg)
        if fut is not None:
            futures.append(fut)
            targeted_added += 1
        if kind_cursor > len(TARGETED_KINDS) * 4 and targeted_added == 0:
            break
    normalize_future_priors(futures)
    return futures
