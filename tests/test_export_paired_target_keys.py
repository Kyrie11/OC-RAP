from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _result(path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenes": [{"target_key": k} for k in keys]}))


def test_export_paired_target_keys(tmp_path: Path) -> None:
    a=tmp_path/'a.json'; b=tmp_path/'b.json'; out=tmp_path/'keys.json'
    _result(a,['x','y']); _result(b,['y','x'])
    subprocess.run([sys.executable, str(ROOT/'tools/export_paired_target_keys.py'), '--input', f'a={a}', '--input', f'b={b}', '--output', str(out)], check=True, cwd=ROOT)
    d=json.loads(out.read_text())
    assert d['target_keys']==['x','y'] and d['num_target_keys']==2
