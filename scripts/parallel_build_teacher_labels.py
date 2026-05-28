#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ocrap.utils.progress import tqdm


def _split_ids(root_dir: Path, split: str, max_roots: int | None) -> list[str]:
    splits = root_dir / "splits.json"
    if splits.exists() and split != "all":
        ids = json.loads(splits.read_text()).get(split, [])
    else:
        ids = sorted(p.stem for p in root_dir.glob("*.json") if p.name not in ("metadata.json", "splits.json"))
    return ids[:max_roots] if max_roots else ids


def _tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _count_progress(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _part_ranges(n: int, parts: int) -> list[tuple[int, int, int]]:
    parts = max(1, int(parts))
    out: list[tuple[int, int, int]] = []
    for i in range(parts):
        start = i * n // parts
        end = (i + 1) * n // parts
        if start < end:
            out.append((i, start, end))
    return out


def _bool_arg(flag: str, enabled: bool) -> list[str]:
    return [flag] if enabled else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel teacher-label builder with one parent tqdm progress bar.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--bev-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parts", "--num-workers", dest="parts", type=int, default=None, help="Number of build_teacher_labels subprocesses. Defaults to min(num_roots, cpu_count//2).")
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--rollout-backend", choices=["auto", "synthetic", "metadrive"], default="auto")
    ap.add_argument("--scenario-dir", default=None)
    ap.add_argument("--metadrive-reactive-traffic", default="true")
    ap.add_argument("--allow-temporal-root-rollout", action="store_true")
    ap.add_argument("--disable-root-alignment-check", action="store_true")
    ap.add_argument("--disable-root-time-replay", action="store_true")
    ap.add_argument("--disable-root-state-restore", action="store_true")
    ap.add_argument("--root-state-restore-max-m", type=float, default=25.0)
    ap.add_argument("--alignment-tolerance-m", type=float, default=5.0)
    ap.add_argument("--shard-size", type=int, default=1, help="Per-worker output shard size.")
    ap.add_argument("--merge-shard-size", type=int, default=4)
    ap.add_argument("--compress-shards", action="store_true")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--poll-seconds", type=float, default=1.0)
    ap.add_argument("--omp-num-threads", type=int, default=2)
    ap.add_argument("--mkl-num-threads", type=int, default=1)
    ap.add_argument("--cuda-visible-devices", default="", help="Default empty string keeps MetaDrive label generation CPU-only.")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    output = Path(args.output)
    part_root = output.parent / f"{output.stem}_parts"
    log_dir = Path(args.log_dir) if args.log_dir else output.parent / "logs"
    progress_dir = log_dir / f"progress_{args.split}"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    ids = _split_ids(root_dir, args.split, args.max_roots)
    n = len(ids)
    if n == 0:
        raise ValueError(f"split={args.split!r} has no roots under {root_dir}")
    parts = int(args.parts or max(1, min(n, (os.cpu_count() or 2) // 2)))
    ranges = _part_ranges(n, parts)
    if not ranges:
        raise ValueError(f"no non-empty part ranges for split={args.split!r}, roots={n}, parts={parts}")

    if part_root.exists():
        import shutil
        shutil.rmtree(part_root)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    part_root.mkdir(parents=True, exist_ok=True)
    for p in progress_dir.glob("*.jsonl"):
        p.unlink()

    procs: list[tuple[int, subprocess.Popen, Path, Path]] = []
    build_script = Path(__file__).resolve().parent / "build_teacher_labels.py"
    for part_idx, start, end in ranges:
        part_out = part_root / f"part_{part_idx:06d}.zarr"
        progress_file = progress_dir / f"part_{part_idx:06d}.jsonl"
        log_file = log_dir / f"build_{args.split}_part_{part_idx:06d}.log"
        cmd = [
            sys.executable,
            str(build_script),
            "--split", args.split,
            "--root-dir", str(root_dir),
            "--bev-dir", str(args.bev_dir),
            "--output", str(part_out),
            "--rollout-backend", args.rollout_backend,
            "--metadrive-reactive-traffic", str(args.metadrive_reactive_traffic),
            "--alignment-tolerance-m", str(args.alignment_tolerance_m),
            "--root-state-restore-max-m", str(args.root_state_restore_max_m),
            "--root-start", str(start),
            "--root-end", str(end),
            "--root-stride", "1",
            "--shard-size", str(args.shard_size),
            "--progress-file", str(progress_file),
        ]
        if args.config:
            cmd += ["--config", str(args.config)]
        if args.scenario_dir:
            cmd += ["--scenario-dir", str(args.scenario_dir)]
        cmd += _bool_arg("--allow-temporal-root-rollout", args.allow_temporal_root_rollout)
        cmd += _bool_arg("--disable-root-alignment-check", args.disable_root_alignment_check)
        cmd += _bool_arg("--disable-root-time-replay", args.disable_root_time_replay)
        cmd += _bool_arg("--disable-root-state-restore", args.disable_root_state_restore)
        cmd += _bool_arg("--compress-shards", args.compress_shards)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
        env["MKL_NUM_THREADS"] = str(args.mkl_num_threads)
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        with log_file.open("w", encoding="utf-8") as f:
            f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log_handle = log_file.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        # Keep the handle alive through Popen by attaching it to the process object.
        proc._recap_log_handle = log_handle  # type: ignore[attr-defined]
        procs.append((part_idx, proc, log_file, progress_file))

    last_done = 0
    failed: list[tuple[int, int, Path]] = []
    with tqdm(total=n, desc=f"build_labels[{args.split}]", unit="root") as pbar:
        while True:
            done = sum(_count_progress(pf) for _, _, _, pf in procs)
            if done > last_done:
                pbar.update(done - last_done)
                last_done = done
            running = False
            for part_idx, proc, log_file, _ in procs:
                rc = proc.poll()
                if rc is None:
                    running = True
                elif rc != 0 and not any(x[0] == part_idx for x in failed):
                    failed.append((part_idx, rc, log_file))
            if failed or not running:
                break
            time.sleep(max(0.1, float(args.poll_seconds)))
        done = sum(_count_progress(pf) for _, _, _, pf in procs)
        if done > last_done:
            pbar.update(done - last_done)

    for _, proc, _, _ in procs:
        try:
            getattr(proc, "_recap_log_handle", None).close()
        except Exception:
            pass
    if failed:
        for part_idx, rc, log_file in failed:
            print(f"ERROR: split={args.split} part={part_idx} failed with exit code {rc}; log={log_file}", file=sys.stderr)
            tail = _tail(log_file)
            if tail:
                print(f"--- tail {log_file} ---\n{tail}\n--- end tail ---", file=sys.stderr)
        raise SystemExit(1)

    merge_script = Path(__file__).resolve().parent / "merge_sharded_datasets.py"
    inputs = [str(part_root / f"part_{part_idx:06d}.zarr") for part_idx, _, _ in ranges]
    merge_cmd = [sys.executable, str(merge_script), "--inputs", *inputs, "--output", str(output), "--shard-size", str(args.merge_shard_size)]
    if args.compress_shards:
        merge_cmd.append("--compress-shards")
    subprocess.check_call(merge_cmd)

    if not args.keep_parts:
        import shutil
        shutil.rmtree(part_root, ignore_errors=True)
    print(json.dumps({"split": args.split, "roots": n, "parts": len(ranges), "output": str(output), "logs": str(log_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
