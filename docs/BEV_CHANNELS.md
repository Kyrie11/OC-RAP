# BEV channels

This document is part of the ReCAP codebase generated from the paper algorithm and the implementation guide.

Key invariants:

- Main input is privileged BEV + ego scalars + route command; no structured neighbor tokens in the main method.
- Root scenes are split by `root_scene_id`.
- Prefix/recovery stages are fixed by horizons, not by collision or event time.
- H is prefix-level first-contact harm exposure and is excluded from recoverability R.
- MERO uses monotone option calibration, existential option aggregation, and lower-tail LCVaR over root-shared semantic modes.

The fixed compact channel prefix is: drivable area, lane boundary, lane centerline, route corridor, speed limit, traffic control stop, static obstacle, occlusion mask, free-space pocket, affordance stop/lane/route/escape, ego current/history, dynamic occupancy/current velocity, then compact history occupancy.
