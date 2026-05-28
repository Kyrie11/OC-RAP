"""ReCAP: recoverability-centered planning for MetaDrive/CARLA BEV roots.

The package intentionally keeps simulator-specific access behind adapters.  The
core CARE + MERO + calibrated selector code is pure PyTorch/NumPy so it can be
unit-tested without MetaDrive or CARLA installed.
"""

__version__ = "0.1.0"
