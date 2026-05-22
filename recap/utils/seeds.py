from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def numpy_state_to_jsonable(state):
    name, keys, pos, has_gauss, cached_gaussian = state
    return {"name": name, "keys": keys.tolist(), "pos": pos, "has_gauss": has_gauss, "cached_gaussian": cached_gaussian}


def numpy_state_from_jsonable(obj):
    return (obj["name"], np.asarray(obj["keys"], dtype=np.uint32), obj["pos"], obj["has_gauss"], obj["cached_gaussian"])


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": numpy_state_to_jsonable(np.random.get_state())}
    try:
        import torch
        state["torch"] = torch.get_rng_state().cpu().numpy().tolist()
        if torch.cuda.is_available():
            state["torch_cuda"] = [x.cpu().numpy().tolist() for x in torch.cuda.get_rng_state_all()]
    except Exception:
        state["torch"] = None
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(numpy_state_from_jsonable(state["numpy"]))
    try:
        import torch
        if state.get("torch") is not None:
            torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    except Exception:
        pass
