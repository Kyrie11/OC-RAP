from __future__ import annotations

import argparse
from pathlib import Path

from ocrap.scripts.calibrate import calibrate
from ocrap.ocrap.config import deep_update, load_config, parse_cli_overrides
from ocrap.data.dataset_builder import build_dataset
from ocrap.scripts.deploy import deploy
from ocrap.scripts.diagnose import diagnose_dataset
from ocrap.scripts.evaluate import evaluate
from ocrap.ocrap.io import write_json
from ocrap.scripts.train import train


def _cfg(args):
    cfg = load_config(args.config)
    overrides = parse_cli_overrides(getattr(args, "set", None))
    if overrides:
        cfg = deep_update(cfg, overrides)
    # Ablation command-line switches.
    ab = cfg.setdefault("ablation", {})
    for name in [
        "without_observation_kernel",
        "without_lower_tail",
        "without_calibration",
        "without_anti_oracle",
        "full_future_roots",
        "no_occlusion_bev",
    ]:
        if getattr(args, name, False):
            ab[name] = True
    if ab.get("no_occlusion_bev", False):
        cfg.setdefault("model", {})["no_occlusion_bev"] = True
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ocrap", description="OC-RAP dataset/model/evaluation pipeline")
    parser.add_argument("--config", default=None, help="YAML config path. Defaults to configs/default.yaml in the repo.")
    parser.add_argument("--set", action="append", help="Override config with dotted.path=value. Can be repeated.")
    # Global ablation toggles.
    parser.add_argument("--without-observation-kernel", action="store_true", help="Ablation: branch-wise evaluation by disabling observation compatibility in OC-MERO.")
    parser.add_argument("--without-lower-tail", action="store_true", help="Ablation: replace LCVaR with weighted mean.")
    parser.add_argument("--without-calibration", action="store_true", help="Ablation: use fixed threshold instead of calibration JSON.")
    parser.add_argument("--without-anti-oracle", action="store_true", help="Ablation: remove anti-oracle loss.")
    parser.add_argument("--full-future-roots", action="store_true", help="Ablation: train root signature head against future-trajectory signatures.")
    parser.add_argument("--no-occlusion-bev", action="store_true", help="Ablation: ignore BEV occlusion input in the model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-dataset")
    p.add_argument("--output", required=True)

    p = sub.add_parser("diagnose")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)

    p = sub.add_parser("papercheck", help="Fast paper-level dataset sanity check for oracle/deployable recoverability labels.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("calibrate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--calibration", default=None)

    p = sub.add_parser("deploy")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--time-index", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", default=None)
    p.add_argument("--delta", type=float, default=None)

    args = parser.parse_args(argv)
    cfg = _cfg(args)
    if args.cmd == "build-dataset":
        result = build_dataset(args.output, cfg)
    elif args.cmd == "diagnose":
        result = diagnose_dataset(args.dataset, args.output, args.max_samples)
    elif args.cmd == "train":
        result = train(args.dataset, args.output, cfg)
    elif args.cmd == "calibrate":
        result = calibrate(args.dataset, args.checkpoint, args.output)
    elif args.cmd == "evaluate":
        result = evaluate(args.dataset, args.checkpoint, args.output, split=args.split, calibration_json=args.calibration)
    elif args.cmd == "deploy":
        result = deploy(args.dataset, args.checkpoint, args.scene_id, args.time_index, args.output, calibration_json=args.calibration, delta=args.delta)
    else:
        raise AssertionError(args.cmd)
    print(result)


if __name__ == "__main__":
    main()
