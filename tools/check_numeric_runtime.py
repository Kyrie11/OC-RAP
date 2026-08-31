#!/usr/bin/env python3
"""Diagnose the NumPy/PyTorch runtime used by OC-RAP without importing it here."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _probe(code: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    return proc.returncode == 0, proc.stdout.strip()


def main() -> int:
    numpy_ok, numpy_out = _probe(
        "import numpy as np; "
        "assert hasattr(np, 'ndarray'), 'numpy.ndarray is missing'; "
        "print(np.__version__, np.__file__, np.ndarray.__module__)"
    )
    torch_ok, torch_out = _probe(
        "import torch; import numpy as np; "
        "assert hasattr(np, 'ndarray'), 'numpy.ndarray is missing after torch import'; "
        "print(torch.__version__, np.__version__)"
    )
    repo = Path.cwd()
    shadow = [str(p) for p in repo.rglob("numpy.py") if ".git" not in p.parts]
    doc = {
        "event": "ocrap_numeric_runtime_check",
        "python": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "numpy_distribution_version": _version("numpy"),
        "torch_distribution_version": _version("torch"),
        "numpy_import_ok": numpy_ok,
        "numpy_probe": numpy_out,
        "torch_import_ok": torch_ok,
        "torch_probe": torch_out,
        "repo_numpy_shadow_files": shadow,
        "healthy": bool(numpy_ok and torch_ok and not shadow),
    }
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0 if doc["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
