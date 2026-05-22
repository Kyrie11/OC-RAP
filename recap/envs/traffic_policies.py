from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LatentTrafficContext:
    mode_id: int
    semantic: str
    reaction_delay: float
    aggressiveness: float
    desired_speed_scale: float
    lateral_noise: float
    braking_noise: float
    rng_seed: int

    def deterministic_subseed(self, root_id: str, channel: str) -> int:
        import hashlib
        blob = f"{root_id}:{self.mode_id}:{self.rng_seed}:{channel}".encode("utf-8")
        return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def apply_latent_context(env, context: LatentTrafficContext) -> None:
    """Best-effort hook for MetaDrive traffic perturbations.

    The function intentionally modifies policy parameters, not open-loop actor
    trajectories.  If the installed simulator does not expose a field, the call is
    a no-op; metadata records the intended context.
    """
    engine = getattr(env, "engine", None)
    manager = getattr(engine, "traffic_manager", None)
    if manager is not None:
        for attr, val in [
            ("reaction_delay", context.reaction_delay),
            ("aggressiveness", context.aggressiveness),
            ("desired_speed_scale", context.desired_speed_scale),
        ]:
            if hasattr(manager, attr):
                try:
                    setattr(manager, attr, val)
                except Exception:
                    pass
