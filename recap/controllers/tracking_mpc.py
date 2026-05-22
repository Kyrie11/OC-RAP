from __future__ import annotations

from .pure_pursuit_pid import PurePursuitPID


class KinematicTrackingMPC(PurePursuitPID):
    """Interface-compatible MPC placeholder.

    The default implementation delegates to PurePursuitPID so all methods and
    baselines share a controller unless users explicitly replace this class with a
    constrained optimizer.  It intentionally does not read CARE/MERO scores.
    """
    pass
