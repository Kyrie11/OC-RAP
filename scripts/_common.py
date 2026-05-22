from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
