from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class CarlaRootMetadata:
    backend: str = "carla"
    carla_version: str = "unknown"
    map_name: str = "unknown"
    traffic_manager_seed: Optional[int] = None
    recorder_file: str = ""
    root_frame: int = 0
    fork_support: bool = False
    usage: str = "bev_pretraining_or_replay_eval_only_unless_fork_support_true"


class CarlaRecorderAdapter:
    """Schema-compatible CARLA recorder boundary.

    CARLA recorder/log data may reconstruct root scenes or replay to a root tick.
    It must not feed recorder packets, actor log tables, or future trajectories to
    CARE.  It can generate MERO teacher labels only when the scenario can be
    forked under counterfactual ego prefixes with the same Traffic Manager latent
    context.  Otherwise, the exported roots are marked `fork_support=false` and
    are valid only for BEV pretraining or replay evaluation.
    """

    def __init__(self, carla_client: Any | None = None):
        self.client = carla_client

    def inspect_recorder(self, recorder_file: str) -> Dict[str, Any]:
        p = Path(recorder_file)
        if not p.exists():
            raise FileNotFoundError(recorder_file)
        info = {"recorder_file": str(p), "bytes": p.stat().st_size}
        if self.client is not None and hasattr(self.client, "show_recorder_file_info"):
            try:
                info["raw_info"] = self.client.show_recorder_file_info(str(p), True)
            except TypeError:
                info["raw_info"] = self.client.show_recorder_file_info(str(p))
        return info

    def export_root_stub(self, recorder_file: str, root_frame: int, output: str, map_name: str = "unknown", carla_version: str = "unknown", traffic_manager_seed: int | None = None, fork_support: bool = False) -> str:
        meta = CarlaRootMetadata(carla_version=carla_version, map_name=map_name, traffic_manager_seed=traffic_manager_seed, recorder_file=str(recorder_file), root_frame=int(root_frame), fork_support=bool(fork_support))
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "metadata": meta.__dict__,
            "schema_note": "Convert this root to the same BEV/action/option/mode/label schema before model use.",
            "model_input_policy": "BEV + ego_info + route_command only; no future trajectory or recorder packet input.",
        }
        out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return str(out)
