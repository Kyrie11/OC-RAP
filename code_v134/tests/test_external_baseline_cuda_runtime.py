from __future__ import annotations

import torch

from ocrap.external_baselines.runtime import configure_cuda_runtime, resolve_amp_dtype


def test_amp_auto_falls_back_to_fp16_without_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp_dtype({"amp_dtype": "auto"}, torch.device("cuda")) == torch.float16
    assert resolve_amp_dtype({"amp_dtype": "bfloat16"}, torch.device("cuda")) == torch.float16


def test_amp_auto_uses_bf16_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp_dtype({"amp_dtype": "auto"}, torch.device("cuda")) == torch.bfloat16


def test_safe_sdpa_disables_only_cudnn(monkeypatch) -> None:
    calls: dict[str, bool] = {}
    for key, attr in {
        "cudnn": "enable_cudnn_sdp",
        "flash": "enable_flash_sdp",
        "mem_efficient": "enable_mem_efficient_sdp",
        "math": "enable_math_sdp",
    }.items():
        monkeypatch.setattr(torch.backends.cuda, attr, lambda enabled, key=key: calls.__setitem__(key, bool(enabled)))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    configure_cuda_runtime({"sdpa_backend": "safe", "amp_dtype": "auto"}, torch.device("cuda"))
    assert calls == {"cudnn": False, "flash": True, "mem_efficient": True, "math": True}


def test_math_sdpa_is_emergency_compatibility_mode(monkeypatch) -> None:
    calls: dict[str, bool] = {}
    for key, attr in {
        "cudnn": "enable_cudnn_sdp",
        "flash": "enable_flash_sdp",
        "mem_efficient": "enable_mem_efficient_sdp",
        "math": "enable_math_sdp",
    }.items():
        monkeypatch.setattr(torch.backends.cuda, attr, lambda enabled, key=key: calls.__setitem__(key, bool(enabled)))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    configure_cuda_runtime({"sdpa_backend": "math"}, torch.device("cuda"))
    assert calls == {"cudnn": False, "flash": False, "mem_efficient": False, "math": True}
