# DPM-Solver notice

`dpm_solver_pytorch.py` is the PyTorch DPM-Solver / DPM-Solver++ implementation
used by the uploaded Diffusion Planner source.  The upstream DPM-Solver project
(LuChengTHU/dpm-solver) is distributed under the MIT License and explicitly
supports copying this module into downstream projects.  It is vendored here so
OC-RAP's Diffusion Planner adapter can execute the source 10-step, second-order,
multistep DPM-Solver++ sampling path without depending on a separately installed
Diffusion Planner checkout.

Upstream project: https://github.com/LuChengTHU/dpm-solver
