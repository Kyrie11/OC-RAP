# Dataset schema

This document is part of the ReCAP codebase generated from the paper algorithm and the implementation guide.

Key invariants:

- Main input is privileged BEV + ego scalars + route command; no structured neighbor tokens in the main method.
- Root scenes are split by `root_scene_id`.
- Prefix/recovery stages are fixed by horizons, not by collision or event time.
- H is prefix-level first-contact harm exposure and is excluded from recoverability R.
- MERO uses monotone option calibration, existential option aggregation, and lower-tail LCVaR over root-shared semantic modes.

Required arrays include `bev`, `ego_info`, `route_command`, action/option tensors, `mode_probs`, `mode_seed_params` for debug only, masks, margins, evidence labels, `R_star`, and witness labels. Invalid padded actions/options are kept and masked.
