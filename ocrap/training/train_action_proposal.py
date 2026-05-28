from __future__ import annotations

from pathlib import Path
import json


def train_action_proposal(dataset: str, output: str) -> str:
    """Bootstrap placeholder: stores lattice imitation metadata for final head training."""
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    meta = {"status": "bootstrap_lattice_targets_recorded", "dataset": dataset, "final_requires_neural_proposal": True}
    (p / "metadata.json").write_text(json.dumps(meta, indent=2))
    return str(p / "metadata.json")
