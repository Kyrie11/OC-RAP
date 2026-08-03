from __future__ import annotations

"""Literature/code provenance and fidelity contracts for external baselines.

The registry is intentionally explicit: an OC-RAP candidate-lattice adaptation
must never be reported as an author's official implementation.  `core_retained`
describes the paper mechanisms implemented in this repository; `known_gaps`
lists components that remain outside the fair executable-candidate protocol.
"""

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BaselineProvenance:
    canonical_name: str
    aliases: tuple[str, ...]
    regimes: tuple[str, ...]
    paper_title: str
    paper_year: int | None
    paper_url: str | None
    official_code_url: str | None
    implementation_kind: str
    fidelity: str
    core_retained: tuple[str, ...]
    known_gaps: tuple[str, ...]
    reporting_name: str

    def to_dict(self) -> dict:
        return asdict(self)


REGISTRY: tuple[BaselineProvenance, ...] = (
    BaselineProvenance(
        "nominal_replay", ("nominal", "nominal_replay", "log_replay"), ("safe",),
        "Logged trajectory replay (dataset control; not a paper implementation)", None, None, None,
        "dataset control", "exact",
        ("replays the logged nominal candidate",), (), "Logged nominal replay",
    ),
    BaselineProvenance(
        "wayformer_bc", ("route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer"), ("safe",),
        "Wayformer: Motion Forecasting via Simple & Efficient Attention Networks", 2022,
        "https://arxiv.org/abs/2207.05844", None,
        "native candidate-policy adaptation", "paper-core adaptation",
        ("factorized candidate tokens", "Transformer scene fusion", "latent-query compression and cross-attention", "route-conditioned candidate scoring"),
        ("not the authors' forecasting data pipeline", "no official released checkpoint/backend", "candidate classification replaces the original multimodal trajectory output"),
        "Wayformer-style route BC",
    ),
    BaselineProvenance(
        "gameformer_lite", ("gameformer", "gameformer_lite", "gameformer_levelk"), ("safe", "near"),
        "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2303.05760", "https://github.com/MCZhi/GameFormer-Planner",
        "native paper-core reproduction on OC-RAP tensors", "high paper-core fidelity",
        ("agent/history encoder", "multimodal initial decoder", "level-k interaction decoder", "previous-level future encoding", "deep trajectory supervision", "candidate-level planning decoder"),
        ("OC-RAP/WOMD preprocessing replaces the authors' nuPlan pipeline", "not checkpoint-compatible with the official repository", "finite candidate prefixes replace continuous trajectory sampling"),
        "GameFormer level-k adapter",
    ),
    BaselineProvenance(
        "betopnet_lite", ("betop", "betop_lite", "betopnet", "betopnet_lite"), ("safe",),
        "Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving", 2024,
        "https://arxiv.org/abs/2409.18031", "https://github.com/OpenDriveLab/BeTop",
        "native paper-core adaptation", "paper-core adaptation",
        ("actor and map topology encoders", "iterative topology decoding", "topology-guided sparse attention", "topology auxiliary supervision", "candidate policy scoring"),
        ("not checkpoint-compatible with official code", "the public repository releases the WOMD prediction pipeline while its nuPlan planning release remains marked TODO", "OC-RAP topology labels replace official preprocessing", "candidate selector replaces the unreleased full planning stack"),
        "BeTop topology-aware adapter",
    ),
    BaselineProvenance(
        "marc_lite", ("marc", "marc_lite", "marc_contingency"), ("near",),
        "MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2308.12021", None,
        "finite-lattice contingency reproduction", "paper-core adaptation",
        ("semantic multi-policy families", "dynamic non-anticipative shared prefix", "scenario-tail branching", "expected/CVaR risk tolerance", "constrained policy-family representative selection"),
        ("continuous nonlinear optimization is replaced by executable candidate prefixes", "no author code/checkpoint was identified", "actor prediction uses the shared observation-only mode bank"),
        "MARC candidate-lattice contingency",
    ),
    BaselineProvenance(
        "racp_lite", ("racp", "racp_lite", "risk_aware_contingency"), ("near",),
        "RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions", 2025,
        "https://arxiv.org/abs/2402.17387", "https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions",
        "finite-lattice belief/risk reproduction", "paper-core adaptation",
        ("belief-weighted multimodal prediction", "non-anticipative shared prefix", "branch risk with expected/CVaR mixture", "chance constraint", "utility-risk optimization"),
        ("continuous branch MPC is replaced by candidate enumeration", "not API/checkpoint compatible with official code", "belief update is prior-predictive until new online evidence arrives"),
        "RACP candidate-lattice planner",
    ),
    BaselineProvenance(
        "expected_risk_filter", ("expected_risk", "expected_risk_filter", "expected_risk_planner"), ("near",),
        "Expected-risk constrained planning (generic control baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("multimodal collision loss expectation", "risk-threshold admission", "utility-risk selection"), (), "Expected-risk filter",
    ),
    BaselineProvenance(
        "cvar_risk_filter", ("cvar_risk", "cvar_risk_filter", "cvar_planner"), ("near",),
        "CVaR-constrained planning (generic risk-sensitive baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("weighted upper-tail CVaR", "risk-threshold admission", "utility-risk selection"), (), "CVaR risk filter",
    ),
    BaselineProvenance(
        "dro_cvar_filter", ("dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter"), ("near",),
        "Distributionally robust CVaR planning (generic baseline)", None, None, None,
        "observation-only robust-risk surrogate", "explicit surrogate",
        ("CVaR tail risk", "ambiguity-radius dispersion penalty", "constrained selection"),
        ("dispersion penalty is a fast Wasserstein-inspired surrogate rather than a full inner ambiguity optimization",),
        "DRO-CVaR filter",
    ),
    BaselineProvenance(
        "predictive_safety_filter", ("predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter"), ("near",),
        "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems", 2021,
        "https://arxiv.org/abs/1812.05506", None,
        "finite-lattice predictive safety filter", "paper-core adaptation",
        ("minimal intervention", "input feasibility", "stage backup-set margin", "terminal backup-set margin", "predictive barrier condition"),
        ("candidate enumeration replaces online nonlinear MPC", "geometric stopping set replaces a learned/verified terminal controller"),
        "Predictive safety filter",
    ),
    BaselineProvenance(
        "oracle_recovery_filter", ("oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"), ("near",),
        "OC-RAP teacher-only oracle upper bound (not an external baseline)", None, None, None,
        "non-deployable audit upper bound", "exact teacher audit",
        ("branch-wise existential recovery option", "oracle order: option maximization before latent-root aggregation"),
        ("uses teacher tensors and must never be reported as deployable",), "Teacher oracle upper bound",
    ),
    BaselineProvenance(
        "postimpact_mpc_lite", ("postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc"), ("contact",),
        "Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles", 2023,
        "https://doi.org/10.1109/TIV.2023.3236150", None, "finite-lattice post-impact objective", "paper-core adaptation",
        ("post-impact yaw/adhesion stability", "secondary-collision risk", "terminal motion objective", "control effort constraints"),
        ("no impact impulse/damage-state estimator", "continuous vehicle-dynamics MPC replaced by candidate costs"),
        "Post-impact MPC adapter",
    ),
    BaselineProvenance(
        "post_crash_braking", ("post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"), ("contact",),
        "Post-crash braking / stable-stop control (rule baseline)", None, None, None,
        "deterministic rule baseline", "exact protocol definition",
        ("braking/stabilization macro restriction", "terminal-speed and yaw-rate gates", "secondary-risk gate"), (),
        "Post-crash stable stop",
    ),
    BaselineProvenance(
        "post_collision_restoration", ("post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"), ("contact",),
        "Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions", 2026,
        "https://arxiv.org/abs/2602.08444", None, "finite-lattice restoration objective", "paper-core adaptation",
        ("trajectory re-alignment", "yaw stabilization", "progress preservation", "clearance/risk objective", "Ackermann-feasibility proxies"),
        ("paper is recent/preprint and no official implementation was identified", "online state/impact estimator is not reproduced", "continuous restoration optimization replaced by candidate selection"),
        "Post-collision restoration",
    ),
    BaselineProvenance(
        "severity_minimization", ("severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"), ("contact",),
        "Motion planning for autonomous vehicles with the inclusion of post-impact motions for minimising collision risk", 2023,
        "https://doi.org/10.1080/00423114.2022.2088396", None, "finite-lattice severity objective", "paper-core adaptation",
        ("relative-speed severity proxy", "collision probability", "penetration/clearance penalty", "post-impact stability and controllability"),
        ("detailed occupant injury/damage models are not available in WOMD", "continuous crash configuration optimization replaced by executable prefixes"),
        "Unavoidable-collision severity minimization",
    ),
)


def find_provenance(name: str) -> BaselineProvenance | None:
    key = str(name).strip().lower()
    for item in REGISTRY:
        if key == item.canonical_name or key in item.aliases:
            return item
    return None


def registry_dict(regimes: Iterable[str] | None = None) -> list[dict]:
    selected = set(str(x).lower() for x in regimes) if regimes is not None else None
    return [x.to_dict() for x in REGISTRY if selected is None or selected.intersection(x.regimes)]
