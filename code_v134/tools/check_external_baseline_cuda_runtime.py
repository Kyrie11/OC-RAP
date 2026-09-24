#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ocrap.external_baselines.runtime import configure_cuda_runtime, resolve_amp_dtype


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OC-RAP external-baseline Transformer SDPA/AMP runtime")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sdpa-backend", default="safe", choices=["safe", "auto", "math", "flash"])
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this process")

    cfg = {"sdpa_backend": args.sdpa_backend, "amp_dtype": args.amp_dtype, "amp": True}
    runtime = configure_cuda_runtime(cfg, device, log=True)
    dtype = resolve_amp_dtype(cfg, device)

    layer = nn.TransformerEncoderLayer(
        d_model=args.d_model,
        nhead=args.num_heads,
        dim_feedforward=4 * args.d_model,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    try:
        model = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False).to(device)
    except TypeError:
        model = nn.TransformerEncoder(layer, num_layers=2).to(device)
    model.train()
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device, requires_grad=True)
    mask = torch.zeros(args.batch_size, args.seq_len, dtype=torch.bool, device=device)
    if args.seq_len > 1:
        mask[:, -1] = True

    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
        y = model(x, src_key_padding_mask=mask)
        loss = y.float().square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    doc = {
        "ok": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False,
        "runtime": runtime,
        "shape": [args.batch_size, args.seq_len, args.d_model],
        "loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(x.grad).all()),
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
